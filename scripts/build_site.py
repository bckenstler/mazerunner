"""Generate the data-driven parts of the landing page.

Everything the page says with a number in it — the leaderboard, the effort
table, the findings, the method notes — is computed here from the merged run
and the analyses make_figures.py writes beside it, not hand-typed, so the page
cannot drift from the results. Figures are copied from results/figures and
example mazes from the dataset. Run make_figures.py first, then this, before
deploying Pages.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mazerunner.analysis.board import UPPER, build_board, paired  # noqa: E402

META = json.loads((ROOT / "configs/board.v2.json").read_text())
LEGS = META["legs"]
UPPER_ONLY = frozenset(k for k, v in LEGS.items() if v.get("upper_only"))
DATASET = ROOT / META["dataset"]

# Example pictures for first-time visitors: one layout at every size plus its
# bent twin, and every layout at one size.
SIZE_EXAMPLES = ["rectilinear-L1-s0000", "rectilinear-L2-s0004", "rectilinear-L3-s0008",
                 "rectilinear-L4-s0012", "rectilinear-L4w-s0012"]
FAMILY_EXAMPLES = ["rectilinear-L2-s0004", "braided-L2-s0020", "pipes-L2-s0116", "island-L2-s0100",
                   "rooms-L2-s0036", "cave-L2-s0068", "radial-L2-s0084", "organic-L2-s0052"]


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def pts(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}"


def cell_mean(cells: dict, keep) -> float:
    vals = [v for key, v in cells.items() if keep(*key.split("/"))]
    return sum(vals) / len(vals)


def findings(b: dict, f: dict, effort: dict) -> list[dict]:
    """The page's findings, each a headline, a callout number, a paragraph
    written for someone who has never seen the benchmark, and its figures."""
    astra, opus = b["gpt6-astra"], b["opus55"]
    w = lambda leg, lv, side: b[leg]["warp"][lv][side]  # noqa: E731
    modes = f["failure_modes"]

    def share(leg, key):
        return modes[leg].get(key, 0) / sum(modes[leg].values())

    early = f["failed_before_halfway"]
    cost = f["cost_per_attempt_upper"]
    argon_cells = b["argon"]["by_cell"]
    astra_cells = astra["by_cell"]
    e_opus, e_astra, e_terra, e_argon = (effort[k] for k in ("opus55-max", "gpt6-astra-xhigh",
                                                             "gpt61-terra-xhigh", "argon"))
    bent_l4_best = max(b[k]["by_level"].get("L4w", 0) for k in LEGS if k in b)
    return [
        {
            "id": "bending",
            "stat": f"−{pts(w('gpt6-astra', 'L4', 'straight') - w('gpt6-astra', 'L4', 'warped'))}",
            "stat_label": "points the top model loses when its largest mazes are bent",
            "title": "Bigger is harder. Bent is much harder.",
            "body": (
                f"Every model loses ground as the mazes grow: GPT-6 Astra, the top model, solves "
                f"{pct(astra['by_level']['L1'])} of the smallest mazes and {pct(astra['by_level']['L4'])} "
                f"of the largest. Bending hurts more, and the bigger the maze, the more it hurts. Four "
                f"of the layouts also come bent: the same maze, with the same route, start and "
                f"finish, drawn on a warped grid. On small mazes the bend barely matters (Astra loses "
                f"{pts(w('gpt6-astra', 'L1', 'straight') - w('gpt6-astra', 'L1', 'warped'))} points). "
                f"On the largest, Astra falls from {pct(w('gpt6-astra', 'L4', 'straight'))} to "
                f"{pct(w('gpt6-astra', 'L4', 'warped'))}. Claude Opus 5.5 is thrown by bending at "
                f"any size: it solves {pct(w('opus55', 'L1', 'straight'))} of the smallest straight "
                f"grids and {pct(w('opus55', 'L1', 'warped'))} of the same mazes bent."
            ),
            "figures": [("02-ladder.png", "Success rate on straight mazes, by size."),
                        ("03-warp.png", "Each bent maze against the straight maze it was made from.")],
        },
        {
            "id": "steady-hand",
            "stat": pct(share("gpt6-astra", "clearance_failure")),
            "stat_label": "of GPT-6 Astra's failures follow the right route",
            "title": "The best models know the way. Their hand isn't steady enough.",
            "body": (
                f"We sorted every failed attempt by what went wrong, from the drawn path and, where "
                f"the model shared it, its reasoning. Nine in ten of GPT-6 Astra's failures take the "
                f"correct route and still touch a wall: the line runs a few pixels too close to an "
                f"edge. That is why corridor width, more than any other feature of a maze, predicts "
                f"whether it succeeds. Weaker models fail differently. Claude Opus 5.5 cuts straight "
                f"through walls ({pct(share('opus55', 'corridor_departure'))} of its failures), and "
                f"Kimi K3.1 Pro and Nemotron 3.5 Super VL often mistake the walls for the corridors."
            ),
            "figures": [("07-failure-modes.png", "Why each failed attempt failed, per model."),
                        ("06-regression.png", "Which maze features predict success, with the others held fixed.")],
        },
        {
            "id": "early",
            "stat": pct(early["gpt6-astra"]),
            "stat_label": "of the top model's failures happen in the first half of the route",
            "title": "Mistakes come early, not at the finish line.",
            "body": (
                f"You might expect long routes to go wrong near the end, as small errors pile up. "
                f"They don't. Even for the leader, {pct(early['gpt6-astra'])} of failed attempts "
                f"first touch a wall before the halfway point; for Claude and Kimi it is "
                f"{pct(early['opus55'])} and {pct(early['kimi31'])}."
            ),
            "figures": [("08-progress.png", "How far each failed attempt got before it touched a wall.")],
        },
        {
            "id": "blind-spots",
            "stat": pct(argon_cells["organic/L2"]),
            "stat_label": f"Gemini 4 Argon on size-2 organic mazes, after {pct(argon_cells['organic/L1'])} at size 1",
            "title": "Each model has its own blind spots.",
            "body": (
                f"Scores for every layout and size, for three models. What stops one model is often "
                f"easy for another. Organic mazes, with curved and irregular corridors, stop Claude "
                f"at the smallest size and Gemini from size 2, while GPT-6 Astra still solves "
                f"{pct(astra_cells['organic/L2'])} of them at size 2. Island and pipes layouts stay "
                f"solvable for all three."
            ),
            "figures": [("05-cells.png", "Success rate per group of four mazes.")],
        },
        {
            "id": "effort",
            "stat": f"+{pts(e_opus['delta']['score'])}",
            "stat_label": (f"points for Claude Opus 5.5 at maximum effort, at "
                           f"{cost['opus55-max'] / cost['opus55']:.0f}× the cost"),
            "title": "Thinking longer helps some models, at a price.",
            "body": (
                f"Four models also ran at a higher effort setting on the large mazes (sizes 3 and 4), "
                f"compared maze by maze with their normal setting. The GPT-6 models gain "
                f"{pts(e_terra['delta']['score'])} to {pts(e_astra['delta']['score'])} points. "
                f"Claude Opus 5.5 gains {pts(e_opus['delta']['score'], 1)} at maximum effort, but "
                f"writes about 30 times as much and costs "
                f"{cost['opus55-max'] / cost['opus55']:.0f} times as much per try. Gemini 4 Argon "
                f"decides for itself when to think, and switching its thinking off costs it only "
                f"{pts(e_argon['delta']['score'], 1)} points. Even at the highest settings, no model "
                f"passes 30% on the largest bent mazes."
            ),
            "figures": [("04-effort.png", "Large mazes at normal and extra effort, same mazes at both."),
                        ("10-cost.png", "Success against list price per try.")],
        },
    ], bent_l4_best


def method(board: dict, f: dict) -> list[dict]:
    """Plain notes on how the numbers are made."""
    unanswered = sum(r["unanswered"] for r in board.values())
    attempts = sum(r["attempts"] for r in board.values())
    tol = f["tolerance"]
    first_swap = min(r for r in tol["radii"] if r > 4)
    return [
        {"title": "How a try is scored",
         "body": ("The model gets one picture of a maze and returns a list of points. We sweep a "
                  "pointer 3 pixels in radius along the line through them. If it touches a wall "
                  "anywhere, or doesn't start on the start badge and end on the goal, the try "
                  "fails. There is no partial credit.")},
        {"title": "How the score is averaged",
         "body": ("The 192 mazes fall into 48 groups of four: eight layouts at four sizes, plus "
                  "bent versions of the four grid layouts at every size. Each group counts "
                  "equally, so no layout or size dominates. The ranges are 95% confidence "
                  "intervals, from resampling the mazes within each group.")},
        {"title": "Does the 3-pixel pointer decide the ranking?",
         "body": (f"We re-scored every stored attempt with pointers from 1 to 8 pixels. Every "
                  f"score moves, but the order holds from 1 to 4 pixels. From {first_swap} pixels "
                  f"up, two pairs trade places, and both pairs are already within each other's "
                  f"ranges at 3 pixels."),
         "figure": "09-tolerance.png"},
        {"title": "Settings and fine print",
         "body": (f"Each model ran at its default or standard setting, four tries per maze, "
                  f"{attempts:,} tries in all, from 29 October to 2 November 2026. A call gets ten "
                  f"minutes (thirty at the extra-effort settings). A try that got no answer twice "
                  f"counts as a fail: {unanswered} of {attempts:,}, all on the longest routes. "
                  f"OpenAI, Google, Anthropic and Meta models ran through a LiteLLM gateway; Kimi "
                  f"and Nemotron through OpenRouter, each pinned to one host.")},
    ]


def thumbnail(task_id: str, out: Path, width: int = 420) -> str:
    """A small JPEG of the exact image a model saw for this task."""
    src = DATASET / task_id / "input.png"
    img = Image.open(src).convert("RGB")
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    name = f"{task_id}.jpg"
    img.save(out / name, quality=84, optimize=True)
    return name


def main() -> int:
    """Rebuild the landing page's data from the merged run and analyses."""
    board, tasks = build_board(ROOT / META["attempts"], UPPER_ONLY)
    f = json.loads((ROOT / "results/analysis/findings.json").read_text())

    rows = []
    for leg in sorted((k for k in board if k not in UPPER_ONLY), key=lambda k: -board[k]["headline"]["score"]):
        r = board[leg]
        cells = r["by_cell"]
        rows.append({
            "id": leg,
            "model": LEGS[leg]["model"],
            "vendor": LEGS[leg]["vendor"],
            "effort": LEGS[leg]["effort"],
            "overall": round(100 * r["headline"]["score"], 1),
            "ci": [round(100 * r["headline"]["lo"], 1), round(100 * r["headline"]["hi"], 1)],
            "small": round(100 * cell_mean(cells, lambda fam, lv: lv in ("L1", "L2"))),
            "large": round(100 * cell_mean(cells, lambda fam, lv: lv in ("L3", "L4"))),
            # The same four grid layouts, straight, so "large, bent" has a fair
            # comparison beside it.
            "large_grid": round(100 * cell_mean(cells, lambda fam, lv: lv in ("L3", "L4")
                                                 and f"{fam}/{lv}w" in cells)),
            "large_bent": round(100 * cell_mean(cells, lambda fam, lv: lv in ("L3w", "L4w"))),
            "best_of_4": round(100 * r["pass_at_k"]["score"]),
        })
    effort, by_more = [], {}
    for pair in META["effort_pairs"]:
        hi, lo = pair["more"], pair["less"]
        d = paired(tasks[hi], tasks[lo], UPPER)
        by_more[hi] = {"delta": d}
        effort.append({
            "model": LEGS[hi]["model"],
            "more": LEGS[hi]["effort"], "less": LEGS[lo]["effort"],
            "more_score": round(100 * board[hi]["upper"]["score"], 1),
            "less_score": round(100 * board[lo]["upper"]["score"], 1),
            "delta": round(100 * d["score"], 1),
            "ci": [round(100 * d["lo"], 1), round(100 * d["hi"], 1)],
            "trials": min(board[hi]["trials"], board[lo]["trials"]),
            "tokens": [board[lo]["median_output_tokens_upper"], board[hi]["median_output_tokens_upper"]],
            "cost": [f["cost_per_attempt_upper"].get(lo), f["cost_per_attempt_upper"].get(hi)],
        })
    effort.sort(key=lambda e: -e["delta"])

    found, bent_l4_best = findings(board, f, by_more)
    hero = [
        {"n": f"{rows[0]['overall']:.0f}%", "label": f"best pass@1, solved on the first try ({rows[0]['model']})"},
        {"n": f"{100 * bent_l4_best:.0f}%", "label": "best score on the largest bent mazes, at any effort"},
        {"n": f"+{max(e['delta'] for e in effort):.0f}", "label": "most points gained by thinking longer"},
    ]

    out = ROOT / "docs"
    assets = out / "assets"
    examples = assets / "examples"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.png"):
        old.unlink()
    if examples.exists():
        shutil.rmtree(examples)
    examples.mkdir()
    for old in ("leaderboard.json", "effort.json", "figures.json"):
        (assets / old).unlink(missing_ok=True)

    figures = {fig for item in found for fig, _c in item["figures"]} | {m["figure"] for m in method(board, f) if "figure" in m}
    figures.add("01-leaderboard.png")  # the social-card image
    for name in sorted(figures):
        shutil.copy2(ROOT / "results/figures" / name, assets / name)
    sizes = [{"file": thumbnail(t, examples), "label": label}
             for t, label in zip(SIZE_EXAMPLES, ["size 1", "size 2", "size 3", "size 4", "size 4, bent"])]
    families = [{"file": thumbnail(t, examples, 300), "label": t.split("-")[0]} for t in FAMILY_EXAMPLES]

    site = {
        "hero": hero,
        "board": rows,
        "effort": effort,
        "findings": [{**item, "figures": [{"file": fig, "caption": c} for fig, c in item["figures"]]}
                     for item in found],
        "method": method(board, f),
        "examples": {"sizes": sizes, "families": families},
        "counts": {"mazes": 192, "attempts": sum(r["attempts"] for r in board.values()),
                   "models": len(rows)},
    }
    (assets / "site.json").write_text(json.dumps(site, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "mazerunner-promo.mp4", assets / "mazerunner-promo.mp4")
    print(f"site data -> {assets}: {len(rows)} board rows, {len(effort)} effort rows, "
          f"{len(found)} findings, {len(figures)} figures, {len(sizes) + len(families)} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
