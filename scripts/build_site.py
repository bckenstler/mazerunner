"""Generate the data-driven parts of the landing page.

The leaderboard on the site is computed from the merged run, not hand-typed,
so the page cannot drift from the results. Figures are copied from
results/figures. Run before deploying Pages.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mazerunner.analysis.load import load_attempts, task_means, by_task  # noqa: E402
from mazerunner.analysis.stats import bootstrap_ci, pass_at_k  # noqa: E402

NAMES = {
    "gpt-xhigh": "GPT-5.6 Sol · xhigh", "openai": "GPT-5.6 Sol · medium",
    "gemini": "Gemini 3.6 Flash · medium", "kimi": "Kimi K3 · high",
    "anthropic": "Claude Opus 5 · high", "muse-spark": "Muse Spark 1.1 · medium",
    "inkling": "Inkling · default",
}

# The figure gallery, grouped into themed sections. Each figure carries its
# own paragraph, rendered above it, so prose and chart interleave; the full
# treatment with caveats stays in STUDY.md.
SECTIONS = [
    {
        "title": "The standings",
        "figures": [
            ("01-leaderboard.png", (
                "The spread runs the whole scale: the best configuration draws "
                "its way out of 61% of mazes on the first try, the weakest out "
                "of none in 800 attempts. Eight tries lift everyone — the ghost "
                "bars — without reordering anyone."
            )),
            ("02-tiers.png", (
                "Difficulty is measured from each maze's geometry — route "
                "length, turns, branches — and every model's pass rate falls "
                "cleanly across the tiers. Hard mazes are hard for everyone, "
                "just not equally."
            )),
            ("04-tolerance.png", (
                "The ordering is not fragile: re-scored with a pointer anywhere "
                "from 1px to 8px wide, the ranking never inverts."
            )),
        ],
    },
    {
        "title": "Thinking longer is not a portable knob",
        "figures": [
            ("03-effort.png", (
                "Every model ran its full reasoning-effort ladder, and the same "
                "knob produces three different machines: GPT converts test-time "
                "compute into accuracy (+37 points from low to xhigh, still "
                "unsaturated), Gemini steps up once and then returns an "
                "identical score across a 6× spread in reasoning tokens, and "
                "Claude and Kimi are flat everywhere. Whatever limits the flat "
                "models, it is not a deliberation budget."
            )),
        ],
    },
    {
        "title": "What makes a maze hard depends on who is solving it",
        "figures": [
            ("12-regression.png", (
                "A regression of pass/fail on maze geometry says turns and "
                "branches are what sink most models — but not the best one. For "
                "GPT at xhigh, turns and branches stop mattering entirely and "
                "the only predictor left is corridor width: it has solved "
                "routing and is now limited by how precisely it can draw."
            )),
            ("13-response.png", (
                "The raw pass rates tell the same story without a model on "
                "top: GPT climbs as corridors widen while Gemini collapses as "
                "turns accumulate."
            )),
        ],
    },
    {
        "title": "Every model fails in its own way",
        "figures": [
            ("05-failure-modes.png", (
                "All 4,140 failed attempts were classified from the reasoning "
                "trace plus the drawn path — a verdict needs both to agree. The "
                "signatures are distinct enough to identify a model from its "
                "failures alone: GPT runs out of corridor width, Gemini and "
                "Claude drive through walls they described correctly, Kimi "
                "announces its path is approximate and submits it anyway, and "
                "Inkling reports positions from a coarse, quantized sketch of "
                "the scene."
            )),
        ],
    },
    {
        "title": "Perception, isolated",
        "figures": [
            ("06-resolution.png", (
                "Re-sending identical mazes at 0.5× and 2× resolution doubles "
                "Kimi (20% → 42%) while its effort ladder was flat — extra "
                "pixels bought what extra thinking could not. Its constraint is "
                "perceptual acuity, not planning."
            )),
            ("09-fingerprints.png", (
                "The drawn coordinates themselves carry fingerprints. Models "
                "that measure the image write messy decimals and start on the "
                "badge; Inkling snaps 68% of its coordinates to an exact 0.01 "
                "grid and misses the badge by 25px at the median — estimating "
                "the scene rather than measuring it, which a continuous drag "
                "punishes and a grid world would forgive."
            )),
        ],
    },
    {
        "title": "Showing a model its own mistake makes it worse",
        "figures": [
            ("07-feedback.png", (
                "After a failed attempt, each model saw its own path drawn on "
                "the maze with the exact collision marked, then tried again. No "
                "model beats a blind retry of the same task; for mid-tier "
                "models the feedback costs up to 24 points. Closed-loop visual "
                "correction — look at your stroke, fix your stroke — is a "
                "capability none of them has yet."
            )),
        ],
    },
    {
        "title": "The image is the interface",
        "figures": [
            ("10-dimensions.png", (
                "Disclosing the canvas dimensions in the prompt — sparing the "
                "model from inferring the aspect ratio before it can place a "
                "coordinate — helps some models almost exactly as much as it "
                "hurts others, so the dimension-free prompt stays."
            )),
            ("08-variance.png", (
                "Re-rendering identical mazes in five visual styles moves the "
                "average less than a point while swinging individual mazes by "
                "24 — style is a per-maze difficulty axis that vanishes in "
                "aggregate."
            )),
        ],
    },
]

FIGURES = [fig for s in SECTIONS for fig in s["figures"]]


def main() -> int:
    """Rebuild the landing page's data: leaderboard computed from the merged
    run (never hand-typed, so the page cannot drift from the results), plus
    figure and video copies."""
    rows = load_attempts(
        [ROOT / "results/main/merged/attempts.jsonl"], ROOT / "datasets/v1/dev"
    )
    means = task_means(rows)
    succ = by_task(rows)
    board = []
    for provider, tasks in means.items():
        values = list(tasks.values())
        mean = sum(values) / len(values)
        lo, hi = bootstrap_ci(values)
        full = [a for a in succ[provider].values() if len(a) >= 8]
        p8 = sum(pass_at_k(len(a), sum(a), 8) for a in full) / len(full)
        board.append({
            "id": provider,
            "name": NAMES.get(provider, provider),
            "pass1": round(100 * mean, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
            "pass8": round(100 * p8),
        })
    board.sort(key=lambda r: -r["pass1"])

    out = ROOT / "docs"
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "leaderboard.json").write_text(json.dumps(board, indent=2))

    for name, _caption in FIGURES:
        shutil.copy2(ROOT / "results/figures" / name, out / "assets" / name)
    (out / "assets" / "figures.json").write_text(
        json.dumps(
            [
                {
                    "title": s["title"],
                    "figures": [{"file": f, "text": t} for f, t in s["figures"]],
                }
                for s in SECTIONS
            ],
            indent=2,
        )
    )
    shutil.copy2(ROOT / "mazerunner-promo.mp4", out / "assets" / "mazerunner-promo.mp4")

    print(f"site data -> {out/'assets'}: leaderboard ({len(board)} rows), "
          f"{len(FIGURES)} figures, promo video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
