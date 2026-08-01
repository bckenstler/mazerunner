"""Threshold-free robustness: does the ranking survive a different tolerance?

Every score depends on one arbitrary constant — the 3px pointer disk. If the
leaderboard reorders when that becomes 2px or 5px, the ranking is an artifact
of the constant rather than a property of the models. Re-scoring stored
submissions costs no API calls, so there is no excuse for not checking.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..evaluator import evaluate

DEFAULT_RADII = (1, 2, 3, 5, 8)


def _load_task_and_mask(task_dir: Path, cache: dict) -> tuple[dict, np.ndarray]:
    key = str(task_dir)
    if key not in cache:
        task = json.loads((task_dir / "task.json").read_text())
        mask = np.asarray(Image.open(task_dir / task["mask_file"]).convert("L")) > 127
        cache[key] = (task, mask)
    return cache[key]


def rescore(
    rows: list[dict],
    radii: tuple[int, ...] = DEFAULT_RADII,
    *,
    index: dict[str, dict] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """radius -> provider -> task -> pass rate, from stored submissions."""
    cache: dict = {}
    out: dict[int, dict[str, dict[str, list]]] = {
        r: defaultdict(lambda: defaultdict(list)) for r in radii
    }

    for row in rows:
        if row.get("error") or row.get("submission") is None:
            continue
        task_dir = row.get("task_dir") or (index or {}).get(row["maze"], {}).get("dir")
        if not task_dir:
            continue
        task, mask = _load_task_and_mask(Path(task_dir), cache)
        width, height = task["width"], task["height"]
        start = (task["start"]["x"] * (width - 1), task["start"]["y"] * (height - 1))
        goal = (task["goal"]["x"] * (width - 1), task["goal"]["y"] * (height - 1))
        reference = task["reference"].get(
            "optimal_length_px_geometric", task["reference"]["optimal_length_px"]
        )
        for radius in radii:
            ev = evaluate(
                row["submission"],
                mask,
                width=width,
                height=height,
                start_px=start,
                goal_px=goal,
                start_radius_px=task["start_radius_px"],
                goal_radius_px=task["goal_radius_px"],
                pointer_radius_px=radius,
                reference_length_px=reference,
                compute_clearance=False,  # a full distance transform per call
            )
            out[radius][row["provider"]][row["maze"]].append(ev.success)

    return {
        radius: {
            provider: {task: sum(v) / len(v) for task, v in tasks.items()}
            for provider, tasks in providers.items()
        }
        for radius, providers in out.items()
    }


def rank_stability(curves: dict[int, dict[str, dict[str, float]]]) -> dict:
    """Does the ordering hold across tolerances?"""
    rankings = {}
    rates = {}
    for radius, providers in sorted(curves.items()):
        means = {p: sum(t.values()) / len(t) for p, t in providers.items() if t}
        rates[radius] = means
        rankings[radius] = [p for p, _ in sorted(means.items(), key=lambda kv: -kv[1])]

    baseline = rankings.get(3) or next(iter(rankings.values()), [])
    inversions = {
        radius: [p for i, p in enumerate(order) if baseline and baseline.index(p) != i]
        for radius, order in rankings.items()
    }
    return {
        "rates": rates,
        "rankings": rankings,
        "stable": all(order == baseline for order in rankings.values()),
        "inversions": {r: v for r, v in inversions.items() if v},
    }
