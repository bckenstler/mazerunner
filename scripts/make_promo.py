"""MazeRunner promo reel: real attempts, real paths, real verdicts, and an
8-bit racing soundtrack rendered by scripts/make_music.py.

Every frame comes from the actual run — the trajectories are the coordinates
models submitted, drawn at the speed they'd be dragged, and the WIN/FAIL stamp
is the scorer's verdict, not decoration.

A failure clip ends on the scorer's own evidence. The zoom lands on the
first point where the pointer touched a wall and switches to a true-scale
view: the path drawn at the pointer's real width, and every wall pixel it
touched lit up. The drawn line elsewhere in the reel is wider than the
pointer and would hide a contact, so the zoom never relies on it. Clips are
still chosen from attempts whose line visibly crosses a wall somewhere, so the
failure reads at a glance before the zoom confirms it.

The look is synthwave: a dusk gradient, a neon perspective floor, a sliced
sun on the title cards. The maze images themselves are the exact pictures the
models saw and are never restyled.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mazerunner.analysis.board import UPPER, build_board, paired  # noqa: E402
from mazerunner.geometry import densify_polyline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "mazerunner-promo.mp4"
OUT = ROOT / "results" / "promo-frames"
W = H = 1080
FPS = 30

# Synthwave palette: shared with the site and the figures.
NIGHT = (13, 2, 33)        # top of the sky
DUSK = (43, 8, 79)         # toward the horizon
PINK = (255, 42, 109)      # failure, the thing to notice
CYAN = (5, 217, 232)       # the subject, success
SUN_TOP = (249, 200, 14)
SUN_BOTTOM = (255, 41, 117)
ORANGE = (255, 108, 17)
VIOLET = (140, 30, 255)
WHITE = (245, 240, 255)
LAVENDER = (185, 162, 230)
DIM = (130, 108, 170)
FLOOR = (255, 41, 117)

FUTURA = "/System/Library/Fonts/Supplemental/Futura.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"

META = json.loads((ROOT / "configs/board.v2.json").read_text())
NAMES = {leg: v["name"].upper() for leg, v in META["legs"].items()}
MODEL = {leg: v["model"].upper() for leg, v in META["legs"].items()}

# What reads as an unmistakable wall crossing at the reel's zoom: the drawn
# centreline runs at least WALL_DEPTH px inside a wall for MIN_CROSSING px of
# path, reaching CLEAR_DEPTH px deep somewhere along it.
WALL_DEPTH = 5.0
CLEAR_DEPTH = 9.0
MIN_CROSSING = 30


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


F_HUGE = font(FUTURA, 150, 4)       # Condensed ExtraBold
F_HEAD = font(FUTURA, 58, 4)
F_BOLD = font(FUTURA, 36, 2)
F_MED = font(FUTURA, 26, 0)
F_SMALL = font(FUTURA, 22, 0)
F_STAMP = font(FUTURA, 110, 4)
F_DATA = font(MONO, 28)
F_DATA_S = font(MONO, 22)
F_MARK = font(FUTURA, 22, 2)


def text(d, xy, s, f, fill, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def fade(color, a):
    return tuple(int(c * a) for c in color)


def lerp(a, b, t):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


# ------------------------------------------------------------ backgrounds

def _sky(horizon):
    """Vertical dusk gradient down to the horizon, dark floor below it."""
    img = Image.new("RGB", (W, H), NIGHT)
    px = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = min(1.0, y / max(1, horizon))
        px[y, :] = lerp(NIGHT, DUSK, t ** 1.6) if y < horizon else (16, 3, 38)
    return Image.fromarray(px)


def _floor(img, horizon, phase=0.0, alpha=1.0):
    """Neon perspective grid below the horizon; `phase` scrolls it toward the
    viewer so title cards feel like driving into the sunset."""
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    vx = W / 2
    for k in range(-14, 15):
        x_far = vx + k * 26
        x_near = vx + k * 260
        d.line([(x_far, horizon), (x_near, H)], fill=fade(FLOOR, 0.55 * alpha), width=2)
    for j in range(14):
        z = (j + phase) / 14
        y = horizon + (H - horizon) * (z ** 2.2)
        d.line([(0, y), (W, y)], fill=fade(FLOOR, (0.25 + 0.6 * z) * alpha), width=2)
    d.line([(0, horizon), (W, horizon)], fill=fade(PINK, alpha), width=3)
    glowed = layer.filter(ImageFilter.GaussianBlur(4))
    return ImageChops.add(ImageChops.add(img, glowed), layer)


def _sun(img, cx, cy, r):
    """The sliced synthwave sun: a yellow-to-pink disc cut by horizontal gaps
    that widen toward its base."""
    disc = Image.new("RGB", (2 * r, 2 * r), (0, 0, 0))
    mask = Image.new("L", (2 * r, 2 * r), 0)
    dd = ImageDraw.Draw(disc)
    for y in range(2 * r):
        dd.line([(0, y), (2 * r, y)], fill=lerp(SUN_TOP, SUN_BOTTOM, y / (2 * r)))
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
    for k in range(7):
        y = int(r * 1.05 + k * r * 0.14)
        h = int(2 + k * 2.6)
        md.rectangle([0, y, 2 * r, y + h], fill=0)
    halo = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(halo).ellipse([cx - r - 40, cy - r - 40, cx + r + 40, cy + r + 40], fill=(120, 20, 80))
    img = ImageChops.add(img, halo.filter(ImageFilter.GaussianBlur(60)))
    img.paste(disc, (cx - r, cy - r), mask)
    return img


BASE = None


def base_frame():
    """The clip and data-scene backdrop: dusk sky, low neon floor."""
    global BASE
    if BASE is None:
        BASE = _floor(_sky(960), 960, 0.0, 0.7)
    img = BASE.copy()
    return img, ImageDraw.Draw(img)


def neon(img, draw_fn, color, radius=10, strength=1.0):
    """Draw something twice: blurred underneath as glow, sharp on top."""
    layer = Image.new("RGB", img.size, (0, 0, 0))
    draw_fn(ImageDraw.Draw(layer), color)
    glow_layer = layer.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        glow_layer = Image.eval(glow_layer, lambda v: int(v * strength))
    img = ImageChops.add(img, glow_layer)
    draw_fn(ImageDraw.Draw(img), color)
    return img


def chrome_text(img, xy, s, f, anchor="mm", top=CYAN, bottom=PINK):
    """Text filled with a vertical gradient, with a soft neon glow."""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).text(xy, s, font=f, fill=255, anchor=anchor)
    box = mask.getbbox()
    if not box:
        return img
    grad = Image.new("RGB", img.size, (0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for y in range(box[1], box[3] + 1):
        t = (y - box[1]) / max(1, box[3] - box[1])
        gd.line([(box[0], y), (box[2], y)], fill=lerp(top, WHITE, t * 2) if t < 0.5 else lerp(WHITE, bottom, (t - 0.5) * 2))
    glow_src = Image.new("RGB", img.size, (0, 0, 0))
    glow_src.paste(Image.new("RGB", img.size, PINK), (0, 0), mask)
    img = ImageChops.add(img, Image.eval(glow_src.filter(ImageFilter.GaussianBlur(14)), lambda v: int(v * 0.8)))
    img.paste(grad, (0, 0), mask)
    return img


def wordmark(d, y=H - 46):
    text(d, (46, y), "MAZE", F_MARK, CYAN)
    x = 46 + d.textlength("MAZE", font=F_MARK)
    text(d, (x, y), "RUNNER", F_MARK, PINK)


# ------------------------------------------------------------------ title

def title_frames(n=165):
    """Opening card: drive toward the sliced sun, then the wordmark lands."""
    frames = []
    sky = _sky(700)
    lit = _sun(sky.copy(), W // 2, 600, 200)
    for i in range(n):
        t = i / n
        bg = _floor(lit.copy(), 700, (t * 3) % 1.0)
        # Reveal fast, then hold every line on screen for about three seconds.
        a = min(1.0, t * 5)
        img = Image.blend(bg, chrome_text(bg.copy(), (W / 2, 190), "MAZERUNNER", F_HUGE), a)
        d = ImageDraw.Draw(img)
        if t > 0.2:
            b = min(1.0, (t - 0.2) * 8)
            text(d, (W / 2, 302), "CAN AN AI DRAW ITS WAY OUT OF A MAZE?", F_BOLD, fade(WHITE, b), anchor="mm")
        if t > 0.3:
            b = min(1.0, (t - 0.3) * 8)
            text(d, (W / 2, 350), "ONE PICTURE  ·  ONE STROKE  ·  TOUCH A WALL AND IT'S OVER", F_SMALL,
                 fade(CYAN, b), anchor="mm")
        frames.append(img)
    return frames


# ------------------------------------------------------------------ clips

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


def stamp(img, label, colr, y, pop=1.0):
    f = font(FUTURA, int(110 * pop), 4)
    return neon(img, lambda d, c: d.text((W / 2, y), label, font=f, fill=c, anchor="ma"), colr, 18)


def contact_stage(src, task, mask, points_norm, size):
    """The maze at true scale for the zoom: the path at the pointer's real
    width as a translucent tube, and every wall pixel it touched in pink.
    Built at the task's own resolution, then scaled with nearest-neighbour so
    the magnified pixels are the real ones."""
    w, h = task["width"], task["height"]
    r = task["pointer_radius_px"]
    pts = [(float(p["x"]) * (w - 1), float(p["y"]) * (h - 1)) for p in points_norm]
    tube_img = Image.new("L", (w, h), 0)
    td = ImageDraw.Draw(tube_img)
    td.line(pts, fill=255, width=2 * r + 1, joint="curve")
    for x, y in pts:
        td.ellipse([x - r, y - r, x + r, y + r], fill=255)
    tube = np.asarray(tube_img) > 0
    hit = tube & ~mask
    rgb = np.asarray(src, dtype=np.float32).copy()
    rgb[tube] = rgb[tube] * 0.35 + np.array(CYAN, dtype=np.float32) * 0.65
    # A soft halo around the touched pixels, so a one-pixel graze still reads
    # at video resolution; the touched pixels themselves stay solid pink.
    halo = np.clip(gaussian_filter(hit.astype(np.float32), 2.5) * 5, 0, 1)[..., None] * 0.75
    rgb = rgb * (1 - halo) + np.array(PINK, dtype=np.float32) * halo
    rgb[hit] = np.array(PINK, dtype=np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(size, Image.NEAREST)


def clip_frames(rec, tasks, n_draw=40, n_hold=16, n_in=6, n_zoom=30, n_linger=54):
    """One attempt replayed: draw the path, stamp the verdict, and on a
    failure zoom into the first wall contact and linger there.

    The zoom crossfades from the drawn stage to the true-scale contact view,
    both rendered at the stage's resolution, so the magnified pixels stay
    pinned to the ones the path was drawn against.
    """
    provider, task_id, trial, success, points_norm, crossing, caption, where = rec
    task, mask = tasks[task_id]
    src = Image.open(Path(task["_dir"]) / task["image_file"]).convert("RGB")
    maze, scale = fit(src, 880, 690)
    truth = contact_stage(src, task, mask, points_norm, maze.size) if not success else None
    ox = (W - maze.width) // 2
    oy = 196

    pts = [(p["x"] * (task["width"] - 1) * scale + ox, p["y"] * (task["height"] - 1) * scale + oy)
           for p in points_norm]
    path_len = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    n_draw = max(n_draw, min(96, round(1.6 * math.sqrt(path_len))))
    col = (crossing["x_px"] * scale + ox, crossing["y_px"] * scale + oy) if crossing else None

    frames = []
    zoom_phase = (n_zoom + n_linger) if (not success and col) else 0
    total = n_in + n_draw + n_hold + zoom_phase
    accent = CYAN if success else PINK
    box = [ox - 3, oy - 3, ox + maze.width + 2, oy + maze.height + 2]
    framed = neon(base_frame()[0], lambda dd, c: dd.rectangle(box, outline=c, width=3), VIOLET, 12)
    framed_zoom = neon(base_frame()[0], lambda dd, c: dd.rectangle(box, outline=c, width=4), PINK, 14)
    for i in range(total):
        img = framed.copy()
        stage = maze.copy()
        sd = ImageDraw.Draw(stage)
        img.paste(maze, (ox, oy))
        d = ImageDraw.Draw(img)

        def hud(d, right_label="ATTEMPT", right_value=f"#{trial + 1}", value_color=CYAN):
            text(d, (46, 64), MODEL.get(provider, provider.upper()), F_BOLD, WHITE)
            text(d, (46, 112), where.upper(), F_SMALL, LAVENDER)
            text(d, (46, 146), caption.upper(), F_SMALL, accent if i >= n_in + n_draw - 1 else DIM)
            text(d, (W - 46, 70), right_label, F_SMALL, DIM, anchor="ra")
            text(d, (W - 46, 100), right_value, F_BOLD, value_color, anchor="ra")

        hud(d)
        if i >= n_in:
            frac = min(1.0, (i - n_in) / max(1, n_draw - 1))
            drawn = walk(pts, frac)
            if len(drawn) >= 2:
                for target, off in ((d, 0), (sd, 1)):
                    line = drawn if off == 0 else [(x - ox, y - oy) for x, y in drawn]
                    target.line(line, fill=(255, 255, 255), width=9, joint="curve")
                    target.line(line, fill=CYAN if frac < 1 or success else PINK, width=5, joint="curve")
                hx, hy = drawn[-1]
                d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=WHITE)
            sx, sy = pts[0]
            for target, (x0, y0) in ((d, (sx, sy)), (sd, (sx - ox, sy - oy))):
                target.ellipse([x0 - 11, y0 - 11, x0 + 11, y0 + 11], outline=WHITE, width=3)

        done = i >= n_in + n_draw - 1
        if done and not success and col:
            cx, cy = col
            d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], outline=PINK, width=6)
            d.line([(cx - 17, cy - 17), (cx + 17, cy + 17)], fill=PINK, width=7)
            d.line([(cx - 17, cy + 17), (cx + 17, cy - 17)], fill=PINK, width=7)

        zoom_start = n_in + n_draw + n_hold
        if zoom_phase and i >= zoom_start:
            step = i - zoom_start
            z = min(1.0, step / max(1, n_zoom - 1))
            ease = z * z * (3 - 2 * z)
            lingering = step >= n_zoom
            # Keep the crop at the frame's aspect ratio so x and y magnify by
            # the same factor; clamp the centre, not the box, to stay inside.
            half_w, half_h = maze.width / 2, maze.height / 2
            tile_w = 80.0
            tile_h = tile_w * maze.height / maze.width
            cw = half_w + (tile_w - half_w) * ease
            ch = half_h + (tile_h - half_h) * ease
            tx, ty = col[0] - ox, col[1] - oy
            cx = min(max(half_w + (tx - half_w) * ease, cw), maze.width - cw)
            cy = min(max(half_h + (ty - half_h) * ease, ch), maze.height - ch)
            crop_box = tuple(int(v) for v in (cx - cw, cy - ch, cx + cw, cy + ch))
            drawn_crop = stage.crop(crop_box).resize((maze.width, maze.height), Image.LANCZOS)
            true_crop = truth.crop(crop_box).resize((maze.width, maze.height), Image.NEAREST)
            blend = min(1.0, step / 12)
            crop = Image.blend(drawn_crop, true_crop, blend)
            mag = maze.width / max(1, crop_box[2] - crop_box[0])
            img = framed_zoom.copy()
            img.paste(crop, (ox, oy))
            d = ImageDraw.Draw(img)
            hud(d, "FIRST WALL CONTACT", f"{mag:.0f}×", PINK)
            if blend >= 1:
                d.rectangle([46, 140, 660, 172], fill=NIGHT)
                text(d, (46, 146), "CYAN: THE POINTER'S REAL WIDTH   ·   PINK: WALL IT TOUCHED", F_SMALL, WHITE)
            zx = ox + (col[0] - ox - crop_box[0]) * mag
            zy = oy + (col[1] - oy - crop_box[1]) * mag
            if not lingering:
                r = 26 * min(2.2, mag)
                d.ellipse([zx - r, zy - r, zx + r, zy + r], outline=PINK, width=6)
            else:
                # An open ring points at the crossing without covering it.
                r = 50 * min(2.2, mag)
                for a0, a1 in ((200, 250), (290, 340), (20, 70), (110, 160)):
                    d.arc([zx - r, zy - r, zx + r, zy + r], a0, a1, fill=PINK, width=6)
            img = stamp(img, "FAIL", PINK, oy + maze.height + 18)
            wordmark(ImageDraw.Draw(img))
            frames.append(img)
            continue

        if done:
            k = (i - (n_in + n_draw - 1)) / max(1, n_hold)
            img = stamp(img, "WIN" if success else "FAIL", CYAN if success else PINK,
                        oy + maze.height + 18, 1.0 + 0.14 * max(0.0, 1 - k * 6))
        wordmark(ImageDraw.Draw(img))
        frames.append(img)
    return frames


# ----------------------------------------------------------- data scenes

def heading(img, title, sub):
    img = chrome_text(img, (W / 2, 120), title, F_HEAD, "mm", CYAN, PINK)
    text(ImageDraw.Draw(img), (W / 2, 186), sub, F_SMALL, LAVENDER, anchor="mm")
    return img


def leaderboard_frames(rank, n=176):
    """The standings, rows sliding in from the top."""
    frames = []
    best = rank[0][1]
    for i in range(n):
        t = i / n
        img, _ = base_frame()
        img = heading(img, "LEADERBOARD", "192 MAZES  ·  4 TRIES EACH")
        d = ImageDraw.Draw(img)
        text(d, (520, 232), "PASS@1 · FIRST TRY", F_SMALL, DIM)
        text(d, (1000, 232), "PASS@4", F_SMALL, DIM, anchor="ra")
        for r, (name, p1, pk) in enumerate(rank):
            appear = 0.05 + r * 0.055
            if t < appear:
                continue
            a = min(1.0, (t - appear) * 9)
            y = 262 + r * 72
            lead = r == 0
            bar_x = 520
            bar_w = int(300 * (p1 / 100) * a)
            colr = CYAN if lead else lerp(VIOLET, PINK, r / max(1, len(rank) - 1))
            d.rectangle([bar_x, y + 8, bar_x + max(3, bar_w), y + 40], fill=fade(colr, 0.9))
            text(d, (80, y + 2), f"{r + 1}", F_BOLD, fade(SUN_TOP if lead else DIM, a))
            text(d, (126, y + 6), name, F_MED, fade(WHITE if lead else LAVENDER, a))
            text(d, (bar_x + max(3, bar_w) + 16, y + 6), f"{p1:.0f}%", F_DATA, fade(WHITE if lead else LAVENDER, a))
            text(d, (1000, y + 6), f"{pk:.0f}%", F_DATA, fade(CYAN if lead else DIM, a), anchor="ra")
        if t > 0.72:
            a = min(1.0, (t - 0.72) * 5)
            img = neon(img, lambda dd, c: dd.text((W / 2, 880), f"THE BEST MODEL STILL FAILS {100 - best:.0f}% OF THE TIME",
                                                   font=F_BOLD, fill=c, anchor="mm"), fade(SUN_TOP, a), 10)
        wordmark(ImageDraw.Draw(img))
        frames.append(img)
    return frames


def warp_frames(straight, warped, n=186):
    """Punchline 1: the same maze, bent, gets harder the bigger it is."""
    labels = ["SIZE 1", "SIZE 2", "SIZE 3", "SIZE 4"]
    x0, x1, y0, y1 = 190, 900, 720, 320
    frames = []
    for i in range(n):
        t = i / n
        img, _ = base_frame()
        img = heading(img, "BEND THE SAME MAZE", "GPT-6 ASTRA  ·  EACH BENT MAZE AGAINST THE STRAIGHT ONE IT CAME FROM")
        d = ImageDraw.Draw(img)
        d.line([(x0, y0), (x1, y0)], fill=DIM, width=2)
        d.line([(x0, y0), (x0, y1 - 30)], fill=DIM, width=2)
        for k, lab in enumerate(labels):
            text(d, (x0 + (x1 - x0) * k / 3, y0 + 20), lab, F_SMALL, LAVENDER, anchor="ma")
        for pct in (0, 50, 100):
            y = y0 - (y0 - y1) * pct / 100
            text(d, (x0 - 18, y), f"{pct}%", F_DATA_S, DIM, anchor="rm")
        grow = min(1.0, max(0.0, (t - 0.06) * 1.8))
        for values, colr, w in ((straight, LAVENDER, 5), (warped, CYAN, 7)):
            p = [(x0 + (x1 - x0) * k / 3, y0 - (y0 - y1) * v / 100) for k, v in enumerate(values)]
            shown = walk(p, grow) if grow < 1 else p

            def draw(dd, c, shown=shown, w=w):
                if len(shown) >= 2:
                    dd.line(shown, fill=c, width=w, joint="curve")
                for px, py in shown:
                    dd.ellipse([px - 8, py - 8, px + 8, py + 8], fill=c)
            img = neon(img, draw, colr, 8, 0.8)
        d = ImageDraw.Draw(img)
        if grow >= 1:
            sy = y0 - (y0 - y1) * straight[-1] / 100
            wy = y0 - (y0 - y1) * warped[-1] / 100
            text(d, (x1 + 22, sy), f"{straight[-1]:.0f}%", F_DATA, LAVENDER, anchor="lm")
            text(d, (x1 + 22, wy), f"{warped[-1]:.0f}%", F_DATA, CYAN, anchor="lm")
            if t > 0.62:
                d.line([(x1 - 30, sy + 14), (x1 - 30, wy - 14)], fill=SUN_TOP, width=4)
        if t > 0.5:
            a = min(1.0, (t - 0.5) * 5)
            for r, (name, colr) in enumerate((("STRAIGHT", LAVENDER), ("BENT", CYAN))):
                y = 792 + r * 38
                d.rectangle([190, y + 10, 226, y + 16], fill=fade(colr, a))
                text(d, (242, y), name, F_SMALL, fade(colr, a))
        if t > 0.66:
            a = min(1.0, (t - 0.66) * 6)
            img = neon(img, lambda dd, c: dd.text((W / 2, 900), f"−{straight[-1] - warped[-1]:.0f} POINTS ON THE BIGGEST MAZES",
                                                   font=F_BOLD, fill=c, anchor="mm"), fade(SUN_TOP, a), 10)
        wordmark(ImageDraw.Draw(img))
        frames.append(img)
    return frames


def effort_frames(effort, n=186):
    """Punchline 2: thinking longer helps GPT and Claude, not Gemini."""
    frames = []
    x0, x1 = 430, 960
    for i in range(n):
        t = i / n
        img, _ = base_frame()
        img = heading(img, "THINK LONGER?", "SAME BIG MAZES  ·  NORMAL EFFORT  VS  EXTRA EFFORT")
        d = ImageDraw.Draw(img)
        for r, e in enumerate(effort):
            appear = 0.08 + r * 0.07
            if t < appear:
                continue
            a = min(1.0, (t - appear) * 6)
            y = 280 + r * 104
            text(d, (100, y), e["model"].upper(), F_MED, fade(WHITE, a))
            text(d, (100, y + 34), f"{e['less'].upper()}  ›  {e['more'].upper()}", F_SMALL, fade(DIM, a))
            lo = x0 + (x1 - x0) * e["less_score"] / 100
            hi = x0 + (x1 - x0) * (e["less_score"] + (e["more_score"] - e["less_score"]) * a) / 100
            d.line([(x0, y + 28), (x1, y + 28)], fill=(52, 22, 92), width=2)
            sig = e["ci_lo"] > 0
            d.line([(lo, y + 28), (hi, y + 28)], fill=SUN_TOP if sig else DIM, width=6)
            d.ellipse([lo - 9, y + 19, lo + 9, y + 37], fill=LAVENDER)
            d.ellipse([hi - 11, y + 17, hi + 11, y + 39], fill=CYAN)
            if a >= 1:
                text(d, (max(lo, hi) + 26, y + 28), f"{e['delta']:+.0f}", F_DATA, WHITE if sig else DIM, anchor="lm")
        if t > 0.66:
            a = min(1.0, (t - 0.66) * 6)
            img = neon(img, lambda dd, c: dd.text((W / 2, 850), "GPT AND CLAUDE GAIN 8–14 POINTS",
                                                   font=F_BOLD, fill=c, anchor="mm"), fade(CYAN, a), 10)
            img = neon(img, lambda dd, c: dd.text((W / 2, 900), "GEMINI'S THINKING ADDS NOTHING",
                                                   font=F_BOLD, fill=c, anchor="mm"), fade(PINK, a), 10)
        wordmark(ImageDraw.Draw(img))
        frames.append(img)
    return frames


def end_frames(n=72):
    frames = []
    lit = _sun(_sky(700), W // 2, 600, 200)
    for i in range(n):
        img = _floor(lit.copy(), 700, (i / n * 2) % 1.0)
        img = chrome_text(img, (W / 2, 190), "MAZERUNNER", F_HUGE)
        d = ImageDraw.Draw(img)
        text(d, (W / 2, 302), "CAN AN AI DRAW ITS WAY OUT OF A MAZE?", F_BOLD, WHITE, anchor="mm")
        text(d, (W / 2, 350), "EVERY PATH IN THIS CLIP IS A REAL MODEL SUBMISSION", F_SMALL, CYAN, anchor="mm")
        frames.append(img)
    return frames


# ------------------------------------------------------------- selection

def crossings(task, mask, depth_map, points_norm):
    """(the clearest wall crossing or None, its length in px).

    A crossing is a stretch of the drawn centreline that runs at least
    WALL_DEPTH px inside a wall; the clearest is the longest one, and it only
    counts if it is long enough to read on screen and goes CLEAR_DEPTH deep
    somewhere. The zoom lands in its middle."""
    w, h = task["width"], task["height"]
    pts = [(float(p["x"]) * (w - 1), float(p["y"]) * (h - 1)) for p in points_norm]
    dense = densify_polyline(pts, 1.0)
    best, run, peak = (0, None, 0.0), [], 0.0
    for i, (x, y) in enumerate(dense + [(-1.0, -1.0)]):
        xi, yi = int(round(x)), int(round(y))
        inside = 0 <= xi < w and 0 <= yi < h
        dpt = float(depth_map[yi, xi]) if inside else 0.0
        if dpt >= WALL_DEPTH:
            run.append(i)
            peak = max(peak, dpt)
            continue
        if run and len(run) > best[0] and peak >= CLEAR_DEPTH:
            best = (len(run), run[len(run) // 2], peak)
        run, peak = [], 0.0
    length, mid, _peak = best
    if length < MIN_CROSSING:
        return None, length
    x, y = dense[mid]
    return {"x_px": x, "y_px": y, "at": mid / max(1, len(dense) - 1)}, length


def main():
    """Render every frame of the reel to results/promo-frames, then encode
    mazerunner-promo.mp4 with ffmpeg."""
    attempts_path = ROOT / META["attempts"]
    rows = []
    with attempts_path.open() as handle:
        for line in handle:
            r = json.loads(line)
            r.pop("raw_response", None)
            rows.append(r)
    index = {json.loads(l)["task_id"]: json.loads(l)
             for l in (ROOT / META["dataset"] / "index.jsonl").read_text().splitlines() if l.strip()}
    verdicts = {}
    for line in (ROOT / "results/analysis/failure-modes.jsonl").read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            verdicts[(v["provider"], v["maze"], v["trial"])] = v

    task_cache = {}

    def task_of(tid):
        if tid not in task_cache:
            d = ROOT / index[tid]["dir"]
            t = json.loads((d / "task.json").read_text())
            t["_dir"] = str(d)
            mask = np.asarray(Image.open(d / t["mask_file"]).convert("L")) > 127
            task_cache[tid] = (t, mask, distance_transform_edt(~mask))
        return task_cache[tid]

    def where(meta):
        size = f"size {meta['level'][1]}"
        return f"{meta['family']} maze · {size}" + (" · bent" if meta["warped"] else "")

    CAPTION = {
        "figure_ground_inversion": "read the walls as the corridors",
        "graph_abstraction": "jumped between imagined junctions",
        "corridor_departure": "cut straight through a wall",
        "almost": "nearly there — then through a wall",
    }
    main_legs = [k for k, v in META["legs"].items() if not v.get("upper_only")]

    def in_frame(pts):
        try:
            return bool(pts) and all(-0.02 <= float(p["x"]) <= 1.02 and -0.02 <= float(p["y"]) <= 1.02
                                     for p in pts)
        except (TypeError, ValueError, KeyError):
            return False

    by_mode = defaultdict(list)
    wins_by_task = defaultdict(list)
    fails_by_task = defaultdict(list)
    for r in rows:
        if r.get("error") or not r.get("submission") or r["provider"] not in main_legs:
            continue
        ev = r.get("evaluation") or {}
        meta = index.get(r["maze"])
        pts = r["submission"].get("points") or []
        if not meta or len(pts) < 6 or not in_frame(pts):
            continue
        key = (r["provider"], r["maze"], r["trial"])
        if ev.get("success"):
            rec = (r["provider"], r["maze"], r["trial"], True, pts, None)
            wins_by_task[(r["provider"], r["maze"])].append(rec)
            if meta["level"] == "L4" and len(pts) >= 12:
                m = meta.get("measures") or {}
                by_mode["hard_win"].append((m.get("turns", 0) + 2 * m.get("route_branches", 0), rec))
            continue
        v = verdicts.get(key)
        mode = v["primary"] if v else None
        progress = (r.get("derived") or {}).get("route_progress", 0) or 0
        wanted = mode in CAPTION or progress > 0.75 or (meta["warped"] and meta["level"] in ("L3", "L4"))
        if not wanted:
            continue
        task, mask, depth = task_of(r["maze"])
        first, deepest = crossings(task, mask, depth, pts)  # deepest = crossing length
        if first is None:
            continue  # fails the scorer, but would look correct on screen
        contact = ev.get("first_collision")
        if not contact:
            continue
        rec = (r["provider"], r["maze"], r["trial"], False, pts, contact)
        fails_by_task[(r["provider"], r["maze"])].append((deepest, rec))
        if mode in CAPTION:
            by_mode[mode].append((deepest, rec))
        died_at = contact.get("segment_index", 0) / max(1, len(pts) - 1)
        if progress > 0.75 and died_at > 0.6:
            by_mode["almost"].append((progress + died_at, rec))

    used, picked = set(), []

    def label(rec, caption):
        return rec + (caption, where(index[rec[1]]))

    def take(mode, n=1, providers=None, caption=None):
        out = []
        for _s, rec in sorted(by_mode.get(mode, []), key=lambda x: -x[0]):
            if rec[1] in used or (providers and rec[0] not in providers):
                continue
            used.add(rec[1])
            out.append(label(rec, caption or CAPTION.get(mode, "")))
            if len(out) >= n:
                break
        return out

    def take_warp_pair(provider):
        """The leader solving a big straight maze, then visibly failing its
        bent twin."""
        best = None
        for (prov, maze), fails in fails_by_task.items():
            meta = index[maze]
            if prov != provider or not meta["warped"] or meta["level"] not in ("L3", "L4"):
                continue
            twin = meta["twin"]
            wins = wins_by_task.get((prov, twin))
            if not wins or maze in used or twin in used:
                continue
            deepest, rec = max(fails, key=lambda x: x[0])
            score = deepest + (5 if meta["level"] == "L4" else 0)
            if best is None or score > best[0]:
                best = (score, wins[0], rec)
        if not best:
            return []
        _s, win, fail = best
        used.update({win[1], fail[1]})
        return [label(win, "solved it"), label(fail, "same maze, bent — failed")]

    picked += take("hard_win", 1, {"gpt6-astra"}, "biggest size — solved it")
    picked += take_warp_pair("gpt6-astra")
    picked += take("figure_ground_inversion", 1)
    picked += take("graph_abstraction", 1)
    picked += take("hard_win", 1, {"argon", "gpt61-terra"}, "biggest size — solved it")
    picked += take("corridor_departure", 1, {"opus55"})
    picked += take("almost", 2)
    print(f"selected {len(picked)} clips: {sum(1 for p in picked if p[3])} wins, "
          f"{sum(1 for p in picked if not p[3])} fails")
    for p in picked:
        print(f"  {p[0]:12s} {p[1]:24s} trial {p[2]}  {'WIN ' if p[3] else 'FAIL'}  {p[6]}")
    tasks = {p[1]: task_of(p[1])[:2] for p in picked}

    upper_only = frozenset(k for k, v in META["legs"].items() if v.get("upper_only"))
    board, trials = build_board(attempts_path, upper_only)
    rank = sorted(((MODEL[k], 100 * board[k]["headline"]["score"], 100 * board[k]["pass_at_k"]["score"])
                   for k in main_legs), key=lambda x: -x[1])
    warp = board["gpt6-astra"]["warp"]
    effort = []
    for pair in META["effort_pairs"]:
        hi, lo = pair["more"], pair["less"]
        delta = paired(trials[hi], trials[lo], UPPER)
        effort.append({"model": META["legs"][hi]["model"], "more": META["legs"][hi]["effort"],
                       "less": META["legs"][lo]["effort"],
                       "less_score": 100 * board[lo]["upper"]["score"],
                       "more_score": 100 * board[hi]["upper"]["score"],
                       "delta": 100 * delta["score"], "ci_lo": delta["lo"]})
    effort.sort(key=lambda e: -e["delta"])

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    frames = title_frames()
    marks = {}
    for rec in picked:
        marks[f"{rec[0]}/{rec[1]}/{rec[2]}"] = len(frames)
        frames += clip_frames(rec, tasks)
    data_start = len(frames)
    frames += leaderboard_frames(rank)
    frames += warp_frames([100 * warp[lv]["straight"] for lv in ("L1", "L2", "L3", "L4")],
                          [100 * warp[lv]["warped"] for lv in ("L1", "L2", "L3", "L4")])
    frames += effort_frames(effort)
    frames += end_frames()
    for i, f in enumerate(frames):
        f.save(OUT / f"f{i:05d}.png")
    (OUT / "clips.json").write_text(json.dumps(marks, indent=2))
    print(f"rendered {len(frames)} frames ({len(frames)/FPS:.1f}s)")
    # The soundtrack is rendered to the reel's exact length and cued to it:
    # the race starts on the first clip, the key lifts on the first data
    # scene and the final hit rings over the end card.
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_music import write as write_music
    music = write_music(len(frames) / FPS, OUT / "music.wav",
                        go=min(marks.values()) / FPS, lift=data_start / FPS)
    # Frames render at 1080 for the zoom crops; the site shows the reel at
    # most 720 wide, and 720 keeps the file under the repo's 5MB blob limit.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(OUT / "f%05d.png"), "-i", str(music),
                    "-vf", "scale=720:720:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-c:a", "aac", "-b:a", "128k", "-shortest",
                    "-movflags", "+faststart", str(VIDEO)], check=True)
    print(f"encoded {VIDEO.name} ({VIDEO.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
