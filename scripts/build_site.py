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
    "gpt-xhigh": "GPT-5.6 Sol · xhigh", "openai": "GPT-5.6 Sol",
    "gemini": "Gemini 3.6 Flash", "kimi": "Kimi K3",
    "anthropic": "Claude Opus 5", "muse-spark": "Muse Spark 1.1", "inkling": "Inkling",
}

FIGURES = [
    ("01-leaderboard.png", "The full picture: pass@1 with task-clustered CIs, pass@8 as the ghost bar."),
    ("02-tiers.png", "Difficulty is monotone for every model — the tiers measure something real."),
    ("03-effort.png", "Only GPT converts test-time compute into accuracy. Gemini returns 32% three times."),
    ("04-tolerance.png", "Re-scored at pointer radii 1–8px: the ranking never inverts."),
    ("05-failure-modes.png", "4,140 failures classified from trace + geometry: every model fails its own way."),
    ("06-resolution.png", "Kimi doubles on pixels alone — its effort ladder was flat."),
    ("07-feedback.png", "Seeing its own mistake beats a blind retry for no model."),
    ("10-dimensions.png", "Disclosing the canvas size helps some models exactly as much as it hurts others."),
    ("11-blind.png", "Take the image away and every model drops to zero — the scores come from seeing."),
    ("08-variance.png", "Style is a per-maze difficulty axis: tiny main effect, 20× larger interaction."),
    ("09-fingerprints.png", "Measuring vs. writing round numbers: Inkling snaps to a 0.01 grid and misses the badge by 25px."),
]


def main() -> int:
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
        json.dumps([{"file": f, "caption": c} for f, c in FIGURES], indent=2)
    )
    shutil.copy2(ROOT / "mazerunner-promo.mp4", out / "assets" / "mazerunner-promo.mp4")

    print(f"site data -> {out/'assets'}: leaderboard ({len(board)} rows), "
          f"{len(FIGURES)} figures, promo video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
