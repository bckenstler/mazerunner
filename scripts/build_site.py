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

# The figure gallery, grouped into themed sections. Each section carries a
# short narrative so the page reads as an argument, not a chart dump; the full
# treatment with caveats stays in STUDY.md.
SECTIONS = [
    {
        "title": "The ranking is real",
        "intro": (
            "Before reading anything into the ordering, it has to survive its own "
            "assumptions. It does, twice over: every model's pass rate falls "
            "monotonically across measured difficulty tiers, so the tiers capture "
            "something models actually feel — and re-scoring all 5,600 stored "
            "submissions at pointer radii from 1px to 8px never inverts a single "
            "adjacent pair. The leaderboard is not an artifact of the 3px scoring "
            "constant."
        ),
        "figures": [
            ("01-leaderboard.png", "The full picture: pass@1 with task-clustered CIs, pass@8 as the ghost bar."),
            ("02-tiers.png", "Difficulty is monotone for every model — the tiers measure something real."),
            ("04-tolerance.png", "Re-scored at pointer radii 1–8px: the ranking never inverts."),
        ],
    },
    {
        "title": "Thinking longer is not a portable knob",
        "intro": (
            "Every model was run across its full reasoning-effort ladder. The same "
            "knob produces three different machines: GPT converts test-time compute "
            "into accuracy (+37 points from low to xhigh, still unsaturated), "
            "Gemini steps up once and then returns an identical score across a 6× "
            "spread in reasoning tokens, and Claude and Kimi are flat everywhere. "
            "Whatever is limiting the flat models, it is not a deliberation budget."
        ),
        "figures": [
            ("03-effort.png", "Only GPT converts test-time compute into accuracy. Gemini returns 32% three times."),
        ],
    },
    {
        "title": "What makes a maze hard depends on who is solving it",
        "intro": (
            "A logistic regression of pass/fail on task geometry says turns and "
            "branches are what sink most models — but not the best one. For GPT at "
            "xhigh, turns and branches stop mattering entirely and the only "
            "significant predictor left is corridor width: it has solved routing "
            "and is now bound by drawing precision. The raw response curves tell "
            "the same story without any model on top."
        ),
        "figures": [
            ("12-regression.png", "Turns and branches make mazes hard — except for GPT, which is bound only by corridor width."),
            ("13-response.png", "The same story as raw pass rates: GPT climbs with corridor width; Gemini collapses with turns."),
        ],
    },
    {
        "title": "Every model fails in its own way",
        "intro": (
            "All 4,140 failed attempts were classified from the reasoning trace "
            "plus the submitted geometry — a verdict needs both to agree. The "
            "signatures are distinct enough to identify a model from its failures "
            "alone: GPT runs out of corridor width, Gemini and Claude drive "
            "through walls they described correctly, Kimi announces its path is "
            "approximate and submits it anyway, and Inkling reports positions from "
            "a coarse, quantized sketch of the scene."
        ),
        "figures": [
            ("05-failure-modes.png", "4,140 failures classified from trace + geometry: every model fails its own way."),
        ],
    },
    {
        "title": "Perception, isolated",
        "intro": (
            "Two probes separate seeing from planning. Re-sending identical mazes "
            "at 0.5× and 2× resolution doubles Kimi (20% → 42%) while its effort "
            "ladder was flat — extra pixels bought what extra thinking could not, "
            "so its constraint is perceptual acuity. And the submitted coordinates "
            "carry fingerprints: models that measure the image write messy "
            "decimals and land on the start badge, while Inkling snaps 68% of its "
            "coordinates to an exact 0.01 grid and misses the badge by 25px at "
            "the median — estimation, not measurement, fatal at this task's "
            "tolerances."
        ),
        "figures": [
            ("06-resolution.png", "Kimi doubles on pixels alone — its effort ladder was flat."),
            ("09-fingerprints.png", "Measuring vs. writing round numbers: Inkling snaps to a 0.01 grid and misses the badge by 25px."),
        ],
    },
    {
        "title": "Showing a model its own mistake makes it worse",
        "intro": (
            "After a failed attempt, each model was shown its own path drawn on "
            "the maze with the exact collision marked, and asked to try again. No "
            "model beats a blind retry of the same task; for mid-tier models the "
            "feedback costs up to 24 points. Retrying from feedback is worse than "
            "resampling — closed-loop visual correction is a capability none of "
            "them has yet."
        ),
        "figures": [
            ("07-feedback.png", "Seeing its own mistake beats a blind retry for no model."),
        ],
    },
    {
        "title": "The image is the interface",
        "intro": (
            "Two ablations probe how the picture itself is read. Disclosing the "
            "canvas dimensions in the prompt — sparing the model from inferring "
            "the aspect ratio — helps some models almost exactly as much as it "
            "hurts others, so the dimension-free prompt stays. And re-rendering "
            "identical mazes in five visual styles moves the average less than a "
            "point while swinging individual mazes by 24 — style is a per-maze "
            "difficulty axis that vanishes in aggregate."
        ),
        "figures": [
            ("10-dimensions.png", "Disclosing the canvas size helps some models exactly as much as it hurts others."),
            ("08-variance.png", "Style is a per-maze difficulty axis: tiny main effect, 20× larger interaction."),
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
                    "intro": s["intro"],
                    "figures": [{"file": f, "caption": c} for f, c in s["figures"]],
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
