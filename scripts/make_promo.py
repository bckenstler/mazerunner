"""MazeRunner promo reel: real attempts, real paths, real verdicts.

Every frame comes from the actual run — the trajectories are the coordinates
models submitted, drawn at the speed they'd be dragged, and the WIN/FAIL stamp
is the scorer's verdict, not decoration.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import sys
sys.path.insert(0, str(Path("/Users/bradley.kenstler/projects/mazerunner/src")))
from mazerunner.analysis.stats import pass_at_k

ROOT = Path("/Users/bradley.kenstler/projects/mazerunner")
OUT = Path("/private/tmp/claude-502/-Users-bradley-kenstler-projects-mazerunner/ab5eb1e4-059e-4a58-8268-7e0ec2e27363/scratchpad/frames")
W = H = 1080
FPS = 30

BG = (9, 11, 18)
PANEL = (16, 20, 30)
CYAN = (34, 211, 238)
AMBER = (245, 158, 11)
GREEN = (52, 211, 153)
RED = (248, 68, 84)
WHITE = (237, 242, 247)
DIM = (110, 122, 140)

MONO = "/System/Library/Fonts/Menlo.ttc"
HELV = "/System/Library/Fonts/HelveticaNeue.ttc"

NAMES = {
    "gpt-xhigh": "GPT-5.6 SOL · XHIGH", "openai": "GPT-5.6 SOL",
    "gemini": "GEMINI 3.6 FLASH", "kimi": "KIMI K3",
    "anthropic": "CLAUDE OPUS 5", "muse-spark": "MUSE SPARK 1.1", "inkling": "INKLING",
}


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


F_TITLE = font(HELV, 128, 1)
F_SUB = font(MONO, 26)
F_MODEL = font(MONO, 34)
F_META = font(MONO, 22)
F_STAMP = font(HELV, 96, 1)
F_LB = font(MONO, 30)
F_LBH = font(MONO, 22)
F_MARK = font(MONO, 20)


def text(d, xy, s, f, fill, anchor="la", spacing=0):
    if spacing:
        x, y = xy
        for ch in s:
            d.text((x, y), ch, font=f, fill=fill, anchor="la")
            x += d.textlength(ch, font=f) + spacing
        return
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def glow(img, box, color, radius=18, alpha=120):
    layer = Image.new("RGB", img.size, (0, 0, 0))
    ImageDraw.Draw(layer).rectangle(box, fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    return Image.blend(img, Image.blend(img, layer, alpha / 255), 0.85)


def base_frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # faint grid, so the frame reads as an instrument rather than a slide
    for x in range(0, W, 60):
        d.line([(x, 0), (x, H)], fill=(14, 17, 26), width=1)
    for y in range(0, H, 60):
        d.line([(0, y), (W, y)], fill=(14, 17, 26), width=1)
    return img, d


def wordmark(d, y=H - 42):
    text(d, (46, y), "MAZE", F_MARK, CYAN)
    x = 46 + d.textlength("MAZE", font=F_MARK)
    text(d, (x, y), "RUNNER", F_MARK, WHITE)


def title_frames(n=75):
    frames = []
    for i in range(n):
        img, d = base_frame()
        t = i / n
        # wordmark assembles, then a scan sweep
        img = glow(img, (140, 470, 940, 620), (10, 40, 60), 40, 150)
        d = ImageDraw.Draw(img)
        reveal = min(1.0, t * 2.2)
        full = "MAZERUNNER"
        shown = full[: max(1, int(len(full) * reveal))]
        wpx = d.textlength(full, font=F_TITLE)
        x0 = (W - wpx) / 2
        d.text((x0, 430), shown[:4], font=F_TITLE, fill=CYAN)
        d.text((x0 + d.textlength("MAZE", font=F_TITLE), 430), shown[4:], font=F_TITLE, fill=WHITE)
        if t > 0.45:
            a = min(1.0, (t - 0.45) * 4)
            c = tuple(int(DIM[k] * a) for k in range(3))
            text(d, (W / 2, 590), "CAN A MODEL DRAW ITS WAY OUT?", F_SUB, c, anchor="ma")
        if t > 0.62:
            a = min(1.0, (t - 0.62) * 4)
            c = tuple(int(AMBER[k] * a) for k in range(3))
            text(d, (W / 2, 636), "1,000 MAZES · 7 MODELS · ONE CONTINUOUS DRAG", F_META, c, anchor="ma")
        # sweep line
        sy = int(t * H * 1.4) - 100
        if 0 < sy < H:
            d.line([(0, sy), (W, sy)], fill=(30, 60, 76), width=2)
        frames.append(img)
    return frames


def fit(img, box_w, box_h):
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS), s


def walk(points, frac):
    """Points along the polyline up to `frac` of its length."""
    if len(points) < 2:
        return points
    segs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(segs) or 1.0
    target = total * frac
    out = [points[0]]
    run = 0.0
    for i, seg in enumerate(segs):
        if run + seg >= target:
            t = (target - run) / seg if seg else 0
            a, b = points[i], points[i + 1]
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            return out
        run += seg
        out.append(points[i + 1])
    return out


def clip_frames(rec, tasks, n_draw=40, n_hold=22, n_in=6):
    provider, task_id, trial, family, archetype, tier, success, points_norm, collision = rec
    task = tasks[task_id]
    src = Image.open(Path(task["_dir"]) / task["image_file"]).convert("RGB")
    canvas_w, canvas_h = 880, 700
    maze, scale = fit(src, canvas_w, canvas_h)
    ox = (W - maze.width) // 2
    oy = 200

    pts = [(p["x"] * (task["width"] - 1) * scale + ox, p["y"] * (task["height"] - 1) * scale + oy)
           for p in points_norm]
    col = None
    if collision:
        col = (collision["x_px"] * scale + ox, collision["y_px"] * scale + oy)

    frames = []
    total = n_in + n_draw + n_hold
    for i in range(total):
        img, d = base_frame()
        img.paste(maze, (ox, oy))
        d = ImageDraw.Draw(img)
        d.rectangle([ox - 2, oy - 2, ox + maze.width + 1, oy + maze.height + 1], outline=(38, 46, 62), width=2)

        # HUD
        text(d, (46, 74), NAMES.get(provider, provider.upper()), F_MODEL, WHITE)
        text(d, (46, 122), f"{family.upper()} · {tier.upper()} · {archetype.replace('-', ' ').upper()}",
             F_META, DIM)
        text(d, (W - 46, 74), "ATTEMPT", F_META, DIM, anchor="ra")
        text(d, (W - 46, 104), f"#{trial + 1}", F_MODEL, CYAN, anchor="ra")

        if i >= n_in:
            frac = min(1.0, (i - n_in) / max(1, n_draw - 1))
            drawn = walk(pts, frac)
            if len(drawn) >= 2:
                d.line(drawn, fill=(255, 255, 255), width=9, joint="curve")
                d.line(drawn, fill=CYAN if frac < 1 or success else RED, width=5, joint="curve")
                hx, hy = drawn[-1]
                d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=WHITE)
            sx, sy = pts[0]
            d.ellipse([sx - 11, sy - 11, sx + 11, sy + 11], outline=WHITE, width=3)

        done = i >= n_in + n_draw - 1
        if done and not success and col:
            d.ellipse([col[0] - 26, col[1] - 26, col[0] + 26, col[1] + 26], outline=RED, width=6)
            d.line([(col[0] - 17, col[1] - 17), (col[0] + 17, col[1] + 17)], fill=RED, width=7)
            d.line([(col[0] - 17, col[1] + 17), (col[0] + 17, col[1] - 17)], fill=RED, width=7)

        if done:
            k = (i - (n_in + n_draw - 1)) / max(1, n_hold)
            pop = 1.0 + 0.14 * max(0.0, 1 - k * 6)
            label = "WIN" if success else "FAIL"
            colr = GREEN if success else RED
            f = ImageFont.truetype(HELV, int(96 * pop), index=1)
            bw = d.textlength(label, font=f)
            bx, by = (W - bw) / 2, oy + maze.height + 28
            img = glow(img, (bx - 30, by, bx + bw + 30, by + 96), colr, 26, 110)
            d = ImageDraw.Draw(img)
            d.text((bx, by), label, font=f, fill=colr)

        wordmark(d)
        frames.append(img)
    return frames


def leaderboard_frames(rank, n=170):
    frames = []
    for i in range(n):
        img, d = base_frame()
        t = i / n
        text(d, (W / 2, 96), "LEADERBOARD", F_TITLE if False else ImageFont.truetype(HELV, 62, index=1),
             WHITE, anchor="ma")
        text(d, (W / 2, 176), "100 MAZES · 8 ATTEMPTS EACH · 5,600 TRIES", F_META, DIM, anchor="ma")
        text(d, (100, 244), "MODEL", F_LBH, DIM)
        text(d, (712, 244), "PASS@1", F_LBH, DIM, anchor="ra")
        text(d, (980, 244), "PASS@8", F_LBH, DIM, anchor="ra")
        d.line([(100, 274), (980, 274)], fill=(38, 46, 62), width=2)

        for r, (name, p1, p8) in enumerate(rank):
            appear = 0.05 + r * 0.055
            if t < appear:
                continue
            a = min(1.0, (t - appear) * 9)
            y = 300 + r * 74
            lead = r == 0
            col = WHITE if lead else (200, 210, 224)
            bar_w = int(560 * (p1 / 100) * a)
            d.rectangle([100, y + 44, 100 + bar_w, y + 52], fill=CYAN if lead else (30, 90, 108))
            text(d, (100, y), f"{r + 1}", F_LB, AMBER if lead else DIM)
            text(d, (150, y), name, F_LB, tuple(int(c * a) for c in col))
            text(d, (712, y), f"{p1:.1f}%", F_LB, tuple(int(c * a) for c in (CYAN if lead else col)), anchor="ra")
            text(d, (980, y), f"{p8:.0f}%", F_LB, tuple(int(c * a) for c in (DIM if not lead else WHITE)), anchor="ra")

        if t > 0.72:
            a = min(1.0, (t - 0.72) * 5)
            text(d, (W / 2, 910), "THE BEST MODEL STILL FAILS 39% OF THE TIME",
                 F_SUB, tuple(int(AMBER[k] * a) for k in range(3)), anchor="ma")
        wordmark(d)
        frames.append(img)
    return frames


def feedback_frames(n=140):
    """Punchline 1: closed-loop feedback made every model worse."""
    rows = [("GPT-5.6 SOL", -24), ("GEMINI 3.6 FLASH", -16),
            ("KIMI K3", -6), ("CLAUDE OPUS 5", -5), ("GPT-5.6 SOL · XHIGH", -3)]
    frames = []
    for i in range(n):
        img, d = base_frame()
        t = i / n
        f_h = ImageFont.truetype(HELV, 52, index=1)
        text(d, (W / 2, 120), "WE SHOWED EACH MODEL", F_SUB, DIM, anchor="ma")
        d.text((W / 2, 164), "ITS OWN MISTAKE", font=f_h, fill=WHITE, anchor="ma")
        text(d, (W / 2, 246), "THE FAILED PATH · THE EXACT WALL IT HIT · TRY AGAIN",
             F_META, (78, 90, 106), anchor="ma")

        for r, (name, delta) in enumerate(rows):
            appear = 0.12 + r * 0.075
            if t < appear:
                continue
            a = min(1.0, (t - appear) * 8)
            y = 360 + r * 78
            text(d, (110, y), name, F_LB, tuple(int(c * a) for c in (200, 210, 224)))
            # bar runs leftward from centre: every model is negative
            bw = int(abs(delta) * 15 * a)
            d.rectangle([760 - bw, y + 6, 760, y + 34], fill=RED if delta < -10 else (150, 52, 62))
            text(d, (980, y), f"{delta:+d}pp", F_LB, tuple(int(c * a) for c in RED), anchor="ra")

        if t > 0.68:
            a = min(1.0, (t - 0.68) * 5)
            f_p = ImageFont.truetype(HELV, 46, index=1)
            d.text((W / 2, 800), "EVERY MODEL GOT WORSE", font=f_p,
                   fill=tuple(int(RED[k] * a) for k in range(3)), anchor="ma")
            text(d, (W / 2, 872), "SEEING YOUR OWN ERROR IS WORSE THAN GUESSING AGAIN",
                 F_META, tuple(int(DIM[k] * a) for k in range(3)), anchor="ma")
        wordmark(d)
        frames.append(img)
    return frames


def scaling_frames(n=170):
    """Punchline 2: only GPT converts test-time compute into accuracy."""
    series = [
        ("GPT-5.6 SOL", [36, 56, 63, 73], CYAN),
        ("GEMINI 3.6 FLASH", [32, 32, 32, None], (168, 178, 196)),
        ("CLAUDE OPUS 5", [8, 7, 13, 12], (118, 128, 148)),
    ]
    labels = ["LOW", "MEDIUM", "HIGH", "XHIGH"]
    x0, x1, y0, y1 = 190, 940, 700, 300
    frames = []
    for i in range(n):
        img, d = base_frame()
        t = i / n
        f_h = ImageFont.truetype(HELV, 52, index=1)
        d.text((W / 2, 116), "TEST-TIME COMPUTE", font=f_h, fill=WHITE, anchor="ma")
        text(d, (W / 2, 190), "SAME MODELS · MORE THINKING · DOES IT HELP?",
             F_META, (78, 90, 106), anchor="ma")

        # axes
        d.line([(x0, y0), (x1, y0)], fill=(46, 56, 74), width=2)
        d.line([(x0, y0), (x0, y1 - 30)], fill=(46, 56, 74), width=2)
        for k, lab in enumerate(labels):
            x = x0 + (x1 - x0) * k / 3
            text(d, (x, y0 + 18), lab, F_LBH, (86, 98, 116), anchor="ma")
        for pct in (0, 25, 50, 75):
            y = y0 - (y0 - y1) * pct / 75
            d.line([(x0 - 8, y), (x0, y)], fill=(46, 56, 74), width=2)
            text(d, (x0 - 20, y - 12), f"{pct}", F_LBH, (86, 98, 116), anchor="ra")

        grow = min(1.0, max(0.0, (t - 0.08) * 1.9))
        for name, values, colr in series:
            pts = []
            for k, v in enumerate(values):
                if v is None:
                    continue
                x = x0 + (x1 - x0) * k / 3
                y = y0 - (y0 - y1) * v / 75
                pts.append((x, y))
            shown = walk(pts, grow) if grow < 1 else pts
            if len(shown) >= 2:
                d.line(shown, fill=colr, width=7 if colr == CYAN else 5, joint="curve")
            for px, py in shown[:-1] if len(shown) > 1 else []:
                d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=colr)
            if shown:
                hx, hy = shown[-1]
                d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=colr)
            if grow > 0.98:
                lx, ly = pts[-1]
                text(d, (lx + 22, ly - 14), f"{values[len([v for v in values if v is not None]) - 1]}%",
                     F_LB, colr)

        if t > 0.5:
            a = min(1.0, (t - 0.5) * 5)
            for r, (name, values, colr) in enumerate(series):
                y = 762 + r * 40
                d.rectangle([190, y + 8, 226, y + 14],
                            fill=tuple(int(c * a) for c in colr))
                text(d, (242, y), name, F_META, tuple(int(c * a) for c in colr))
                final = [v for v in values if v is not None][-1]
                text(d, (600, y), f"{final}%", F_META,
                     tuple(int(c * a) for c in colr), anchor="ra")
        if t > 0.58:
            a = min(1.0, (t - 0.58) * 8)
            f_p = ImageFont.truetype(HELV, 44, index=1)
            d.text((W / 2, 906), "GPT SCALES.", font=f_p,
                   fill=tuple(int(CYAN[k] * a) for k in range(3)), anchor="ma")
            d.text((W / 2, 954), "CLAUDE AND GEMINI DON'T.", font=f_p,
                   fill=tuple(int(WHITE[k] * a) for k in range(3)), anchor="ma")
        text(d, (W - 46, H - 42), "EFFORT SWEEP · 25 TASKS × 3 TRIALS", F_MARK,
             (58, 68, 84), anchor="ra")
        wordmark(d)
        frames.append(img)
    return frames


def end_frames(n=60):
    frames = []
    for i in range(n):
        img, d = base_frame()
        img = glow(img, (200, 440, 880, 600), (10, 40, 60), 44, 140)
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(HELV, 104, index=1)
        wpx = d.textlength("MAZERUNNER", font=f)
        x0 = (W - wpx) / 2
        d.text((x0, 452), "MAZE", font=f, fill=CYAN)
        d.text((x0 + d.textlength("MAZE", font=f), 452), "RUNNER", font=f, fill=WHITE)
        text(d, (W / 2, 596), "A CONTINUOUS-CONTROL BENCHMARK FOR MULTIMODAL MODELS",
             F_META, DIM, anchor="ma")
        text(d, (W / 2, 646), "OPEN DATASET · FULL TRACES · PRE-REGISTERED", F_META, (70, 82, 98), anchor="ma")
        frames.append(img)
    return frames


def main():
    rows = [json.loads(l) for l in (ROOT / "results/main/merged/attempts.jsonl").read_text().splitlines() if l.strip()]
    index = {json.loads(l)["task_id"]: json.loads(l)
             for l in (ROOT / "datasets/v1/dev/index.jsonl").read_text().splitlines() if l.strip()}

    wins, fails = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("error") or not r.get("submission"):
            continue
        ev = r.get("evaluation") or {}
        meta = index.get(r["maze"])
        pts = (r["submission"].get("points") or [])
        if not meta or len(pts) < 8:
            continue
        rp = (r.get("derived") or {}).get("route_progress", 0)
        rec = (r["provider"], r["maze"], r["trial"], meta["family"], meta["archetype"],
               meta["tier"], bool(ev.get("success")), pts, ev.get("first_collision"))
        if ev.get("success"):
            wins[(r["provider"], meta["archetype"])].append((ev.get("efficiency", 0), rec))
        elif ev.get("first_collision") and 0.3 < rp < 0.9:
            fails[(r["provider"], meta["archetype"])].append((rp, rec))

    rng = random.Random(7)
    picked, used_tasks, used_arch = [], set(), []
    # alternate WIN / FAIL, spread across models and archetypes
    win_keys = sorted(wins, key=lambda k: -max(w[0] for w in wins[k]))
    fail_keys = sorted(fails, key=lambda k: -max(f[0] for f in fails[k]))
    strong = ["gpt-xhigh", "openai", "gemini", "kimi", "anthropic", "muse-spark"]

    def take(keys, pool, want_provider):
        for k in keys:
            if k[0] != want_provider or k[1] in used_arch[-4:]:
                continue
            for _score, rec in sorted(pool[k], key=lambda x: -x[0]):
                if rec[1] in used_tasks:
                    continue
                used_tasks.add(rec[1])
                used_arch.append(k[1])
                return rec
        return None

    seq = [("w", "gpt-xhigh"), ("f", "gemini"), ("w", "gemini"), ("f", "anthropic"),
           ("w", "openai"), ("f", "kimi"), ("w", "kimi"), ("f", "openai"),
           ("w", "anthropic"), ("f", "gpt-xhigh"), ("w", "gpt-xhigh"), ("f", "muse-spark"),
           ("w", "openai"), ("f", "gemini")]
    for kind, prov in seq:
        rec = take(win_keys if kind == "w" else fail_keys,
                   wins if kind == "w" else fails, prov)
        if rec:
            picked.append(rec)
    print(f"selected {len(picked)} clips")

    tasks = {}
    for rec in picked:
        tid = rec[1]
        if tid not in tasks:
            d = Path(index[tid]["dir"])
            t = json.loads((d / "task.json").read_text())
            t["_dir"] = str(d)
            tasks[tid] = t

    means = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r.get("error"):
            means[r["provider"]][r["maze"]].append(bool((r.get("evaluation") or {}).get("success")))
    board = []
    for p, tsk in means.items():
        per = [sum(v) / len(v) for v in tsk.values()]
        p1 = 100 * sum(per) / len(per)
        # Match the published leaderboard: the unbiased pass@k estimator over
        # tasks that actually have 8 scored attempts.
        full = [v for v in tsk.values() if len(v) >= 8]
        p8 = 100 * sum(pass_at_k(len(v), sum(v), 8) for v in full) / len(full)
        board.append((NAMES.get(p, p), p1, p8))
    board.sort(key=lambda x: -x[1])

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    frames = title_frames()
    for rec in picked:
        frames += clip_frames(rec, tasks)
    frames += leaderboard_frames(board)
    frames += feedback_frames()
    frames += scaling_frames()
    frames += end_frames()

    for i, f in enumerate(frames):
        f.save(OUT / f"f{i:05d}.png")
    print(f"rendered {len(frames)} frames ({len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
