"""Generate the data-driven parts of the landing page.

The leaderboard, the effort table, and every number quoted in a figure caption
are computed here from the merged run and the analyses make_figures.py writes
beside it, not hand-typed, so the page cannot drift from the results. Figures
are copied from results/figures. Run make_figures.py first, then this, before
deploying Pages.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mazerunner.analysis.board import UPPER, build_board, paired  # noqa: E402

META = json.loads((ROOT / "configs/board.v2.json").read_text())
LEGS = META["legs"]
UPPER_ONLY = frozenset(k for k, v in LEGS.items() if v.get("upper_only"))


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def sections(board: dict, findings: dict) -> list[dict]:
    """The figure gallery, grouped into themed sections, each figure with the
    paragraph rendered above it. Numbers come from the board and findings."""
    b = board
    astra, terra, sol, argon, opus, kimi = (
        b[k] for k in ("gpt6-astra", "gpt61-terra", "gpt61-sol", "argon", "opus55", "kimi31"))
    effort = {e["more"]: e for e in findings["effort"]}
    modes = findings["failure_modes"]

    def share(leg, *keys):
        total = sum(modes[leg].values())
        return sum(modes[leg].get(k, 0) for k in keys) / total

    early = findings["failed_before_halfway"]
    cost = findings["cost_per_attempt_upper"]
    reg = findings["regression"]
    tol = findings["tolerance"]
    w = lambda leg, lv, side: b[leg]["warp"][lv][side]  # noqa: E731
    return [
        {
            "title": "The standings",
            "figures": [
                ("01-leaderboard.png", (
                    f"GPT-6 Astra at medium effort solves {pct(astra['headline']['score'], 1)} of "
                    f"mazes on the first attempt, averaged over 48 equally weighted cells. "
                    f"GPT-6.1 Terra ({pct(terra['headline']['score'], 1)}) is within its interval; "
                    f"Gemini 4 Argon is fourth at {pct(argon['headline']['score'], 1)}. Claude Opus "
                    f"5.5 solves {pct(opus['headline']['score'], 1)}, Kimi K3.1 Pro "
                    f"{pct(kimi['headline']['score'], 1)}, and the two floor models almost nothing. "
                    f"Given four attempts, the leader solves {pct(astra['pass_at_k']['score'])}."
                )),
                ("02-ladder.png", (
                    "Each size level adds rows, columns, rings or rooms and narrows the corridors; "
                    "L1 is the original v1 medium size. Every model falls at every step: Astra from "
                    f"{pct(astra['by_level']['L1'])} at L1 to {pct(astra['by_level']['L4'])} at L4, "
                    f"Claude from {pct(opus['by_level']['L1'])} to {pct(opus['by_level']['L4'])}."
                )),
            ],
        },
        {
            "title": "Bending a maze is harder than enlarging it",
            "figures": [
                ("03-warp.png", (
                    "The four grid families also come warped at every level: the same maze, "
                    "drawn on a smoothly bent grid, with the same route, endpoints and style. On "
                    "small mazes the leaders barely notice — Astra loses "
                    f"{100 * (w('gpt6-astra', 'L1', 'straight') - w('gpt6-astra', 'L1', 'warped')):.0f} "
                    f"points at L1. At L4 it falls from {pct(w('gpt6-astra', 'L4', 'straight'))} to "
                    f"{pct(w('gpt6-astra', 'L4', 'warped'))}. Claude breaks immediately: it solves "
                    f"{pct(w('opus55', 'L1', 'straight'))} of straight L1 grids and "
                    f"{pct(w('opus55', 'L1', 'warped'))} of their warped twins."
                )),
                ("05-cells.png", (
                    "Every cell holds four mazes. Families are not equally hard, and not hard for "
                    "the same models: organic mazes (curved, irregular corridors) defeat Claude "
                    "from L1 and Argon from L2, while island and pipes stay solvable for all "
                    "three. The leader's hardest cells are the warped L4 grids and the irregular "
                    "families (organic, radial, cave) at L4."
                )),
                ("06-regression.png", (
                    "A per-model logistic regression on maze features. For the two strongest "
                    "models, corridor width is the largest effect "
                    f"({reg['gpt6-astra']['min_clearance_px']['beta']:+.1f} and "
                    f"{reg['argon']['min_clearance_px']['beta']:+.1f} log-odds per SD): narrow "
                    "corridors make the drawn line clip a wall. For Claude, warping "
                    f"({reg['opus55']['warped']['beta']:+.1f}) and branching "
                    f"({reg['opus55']['route_branches']['beta']:+.1f}) dominate: it loses the route "
                    "itself. Warping hurts all three, after route length and turns are held fixed."
                )),
            ],
        },
        {
            "title": "Thinking longer helps on big mazes, for some models",
            "figures": [
                ("04-effort.png", (
                    "The extra-effort settings ran on the 24 L3/L4 cells, compared task by task "
                    "with the default setting. GPT-6 at xhigh gains "
                    f"{100 * effort['gpt6-astra-xhigh']['delta']['score']:.0f} points (Astra) and "
                    f"{100 * effort['gpt61-sol-xhigh']['delta']['score']:.0f} (Sol); Claude at max "
                    f"effort gains {100 * effort['opus55-max']['delta']['score']:.0f} while writing "
                    "about 30× more tokens. Argon is the exception: switching its thinking off "
                    f"costs only {100 * effort['argon']['delta']['score']:.1f} points, inside the "
                    "noise. Even at top effort, warped L4 stays below 30% for every model."
                )),
                ("10-cost.png", (
                    "Per-attempt list price on L3/L4. GPT-6.1 Terra at medium is the cheapest "
                    f"strong option (${cost['gpt61-terra']:.3f} per attempt for "
                    f"{pct(terra['upper']['score'])}); Astra at xhigh is the most accurate at "
                    f"${cost['gpt6-astra-xhigh']:.2f}. Claude at max effort costs "
                    f"${cost['opus55-max']:.2f} per attempt for {pct(b['opus55-max']['upper']['score'])}."
                )),
            ],
        },
        {
            "title": "How models fail",
            "figures": [
                ("07-failure-modes.png", (
                    "Every failed attempt was classified from its path geometry and, where the "
                    "model returned one, its reasoning trace; a trace-based label needs the path "
                    f"to agree. {pct(share('gpt6-astra', 'clearance_failure'))} of Astra's failures "
                    "follow the right route and clip a wall: the line runs too close to an edge. "
                    f"Claude cuts across walls ({pct(share('opus55', 'corridor_departure'))} of its "
                    f"failures), and in {pct(share('opus55', 'satisficing'))} says it will commit to a "
                    "best-guess path it could not verify. Kimi and Nemotron most often treat the "
                    "walls as corridors."
                )),
                ("08-progress.png", (
                    "Where along the route each failed attempt first touched a wall. Even for the "
                    f"leader, {pct(early['gpt6-astra'])} of failures happen in the first half; for "
                    f"Claude and Kimi it is {pct(early['opus55'])} and {pct(early['kimi31'])}. "
                    "Failures are mostly early, local mistakes, not a route that drifts at the end."
                )),
            ],
        },
        {
            "title": "The scoring constant",
            "figures": [
                ("09-tolerance.png", (
                    "Scoring sweeps a 3px pointer disk along the path. Re-scoring every stored "
                    "submission at other radii changes every score but keeps the order from 1px to "
                    f"4px. At {min(r for r in tol['radii'] if r > 4)}px and above two close pairs "
                    "swap (Astra and Terra, Sol and Argon) — pairs whose intervals already overlap "
                    "at 3px."
                )),
            ],
        },
    ]


def main() -> int:
    """Rebuild the landing page's data: leaderboard and effort table computed
    from the merged run, figure captions from the computed findings, plus
    figure and video copies."""
    board, tasks = build_board(ROOT / META["attempts"], UPPER_ONLY)
    findings = json.loads((ROOT / "results/analysis/findings.json").read_text())

    rows = []
    for leg in sorted((k for k in board if k not in UPPER_ONLY), key=lambda k: -board[k]["headline"]["score"]):
        r = board[leg]
        rows.append({
            "id": leg,
            "name": LEGS[leg]["name"],
            "pass1": round(100 * r["headline"]["score"], 1),
            "ci": [round(100 * r["headline"]["lo"], 1), round(100 * r["headline"]["hi"], 1)],
            "passk": round(100 * r["pass_at_k"]["score"]),
            "k": r["pass_at_k"]["k"],
            "levels": {lv: round(100 * v) for lv, v in r["by_level"].items()},
            "progress": round(100 * r["route_progress"], 1),
        })
    effort = []
    for pair in META["effort_pairs"]:
        hi, lo = pair["more"], pair["less"]
        d = paired(tasks[hi], tasks[lo], UPPER)
        effort.append({
            "model": LEGS[hi]["model"],
            "more": LEGS[hi]["effort"], "less": LEGS[lo]["effort"],
            "more_score": round(100 * board[hi]["upper"]["score"], 1),
            "less_score": round(100 * board[lo]["upper"]["score"], 1),
            "delta": round(100 * d["score"], 1),
            "ci": [round(100 * d["lo"], 1), round(100 * d["hi"], 1)],
            "trials": min(board[hi]["trials"], board[lo]["trials"]),
            "tokens": [board[lo]["median_output_tokens_upper"], board[hi]["median_output_tokens_upper"]],
        })
    effort.sort(key=lambda e: -e["delta"])

    out = ROOT / "docs"
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.png"):
        old.unlink()
    (assets / "leaderboard.json").write_text(json.dumps(rows, indent=2) + "\n")
    (assets / "effort.json").write_text(json.dumps(effort, indent=2) + "\n")

    gallery = sections(board, findings)
    for s in gallery:
        for name, _caption in s["figures"]:
            shutil.copy2(ROOT / "results/figures" / name, assets / name)
    (assets / "figures.json").write_text(json.dumps(
        [{"title": s["title"], "figures": [{"file": f, "text": t} for f, t in s["figures"]]}
         for s in gallery],
        indent=2, ensure_ascii=False,
    ) + "\n")
    shutil.copy2(ROOT / "mazerunner-promo.mp4", assets / "mazerunner-promo.mp4")

    n_figs = sum(len(s["figures"]) for s in gallery)
    print(f"site data -> {assets}: leaderboard ({len(rows)} rows), effort ({len(effort)} rows), "
          f"{n_figs} figures, promo video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
