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
                "GPT-5.6 Sol at xhigh effort solves 60.6% of mazes on the "
                "first attempt. Four of the seven models are below 21%, and "
                "Inkling solved none of its 800 attempts. Giving each model "
                "eight attempts raises every score but does not change the "
                "order."
            )),
            ("02-tiers.png", (
                "Each maze's difficulty tier is computed from its geometry: "
                "route length, turn count, branch count. Pass rates drop from "
                "easy to hard for every model."
            )),
            ("04-tolerance.png", (
                "The pointer is the disk swept along the submitted path, and "
                "its size sets the scoring tolerance: a larger pointer needs "
                "more clearance, so corridors are effectively tighter; a "
                "smaller one is more forgiving of a stroke that hugs a wall. "
                "All 5,600 submissions were re-scored with pointer sizes from "
                "1px to 8px. The ranking is the same at every tolerance."
            )),
        ],
    },
    {
        "title": "Thinking longer helps one model",
        "figures": [
            ("03-effort.png", (
                "The four strongest models were run at every reasoning-effort "
                "setting they support, on a 25-maze subset. GPT gains 37 "
                "points from low to xhigh and has not plateaued. Gemini gains "
                "about 12 points from any thinking at all, then returns the "
                "same score while using 6× more reasoning tokens. Claude "
                "scores the same at every setting; Kimi moves only at its "
                "maximum setting, at nine minutes per maze. Muse Spark and "
                "Inkling were checked at their top setting only and stayed at "
                "their floor."
            )),
        ],
    },
    {
        "title": "What makes a maze hard depends on the model",
        "figures": [
            ("12-regression.png", (
                "A regression of pass/fail on maze geometry: for most models, "
                "turns and branches are the strongest predictors of failure. "
                "For GPT at xhigh neither matters — the only significant "
                "predictor is corridor width. Its remaining errors are drawing "
                "errors, not routing errors."
            )),
            ("13-response.png", (
                "The raw pass rates show the same pattern: GPT's rate rises "
                "with corridor width, and Gemini's falls as turns increase."
            )),
        ],
    },
    {
        "title": "Each model fails differently",
        "figures": [
            ("05-failure-modes.png", (
                "Each of the 4,140 failed attempts was classified using the "
                "model's reasoning trace and the geometry of its path "
                "together. GPT drifts out of narrow corridors. Gemini and "
                "Claude cross walls they had described correctly. Kimi notes "
                "in its reasoning that its path is approximate and submits it "
                "anyway. Inkling reports positions that do not match the "
                "image."
            )),
        ],
    },
    {
        "title": "Isolating perception",
        "figures": [
            ("06-resolution.png", (
                "A 25-maze subset was re-sent at half and double resolution. "
                "At 2×, Kimi goes from 20% to 42% — the same model whose scores "
                "never moved on the effort ladder. Its bottleneck is "
                "resolution, not reasoning."
            )),
            ("09-fingerprints.png", (
                "The submitted coordinates show how each model reads the "
                "image. Models that measure write irregular decimals and "
                "start on the badge. Inkling places 68% of its points on an "
                "exact 0.01 grid and misses the start badge by a median of "
                "25px: it is estimating positions rather than reading them "
                "off the image."
            )),
        ],
    },
    {
        "title": "Showing a model its own mistake makes it worse",
        "figures": [
            ("07-feedback.png", (
                "On a 50-maze subset, each model that failed an attempt was "
                "shown its own path drawn on the maze with the collision "
                "marked, and asked to correct it. Every model scores lower after seeing its "
                "mistake than on a fresh blind attempt — up to 24 points "
                "lower."
            )),
        ],
    },
    {
        "title": "Style and canvas size",
        "figures": [
            ("10-dimensions.png", (
                "Telling the model the image's pixel dimensions changes "
                "scores in both directions: GPT at medium gains 8 points and "
                "Gemini loses 8. The default prompt leaves dimensions out."
            )),
            ("08-variance.png", (
                "Twenty mazes were each re-rendered in five visual styles. "
                "Average scores barely move, but individual mazes swing by up "
                "to 24 points — style changes which mazes are hard without "
                "changing how hard the set is."
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
