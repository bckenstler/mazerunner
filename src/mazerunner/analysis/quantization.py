"""Coordinate fingerprints: how a model actually places points.

Two signatures separate a model that measures the image from one that emits
plausible-looking geometry. A measurer's coordinates are irregular, because
they come from features it located. A confabulator's snap to round numbers —
0.05 and 0.1 grids — because they come from a mental sketch rather than pixels.

Localization error on the start badge is the same question asked directly: the
badge is drawn at a known place, so distance from it is pure perception with no
routing involved.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

GRIDS = (0.5, 0.25, 0.1, 0.05, 0.01)


def _points(row: dict) -> list[tuple[float, float]]:
    submission = row.get("submission")
    if not isinstance(submission, dict):
        return []
    out = []
    for point in submission.get("points") or []:
        if isinstance(point, dict):
            try:
                out.append((float(point["x"]), float(point["y"])))
            except (TypeError, ValueError, KeyError):
                continue
    return out


def decimal_places(rows: list[dict]) -> dict[str, Counter]:
    """provider -> histogram of decimal places used in submitted coordinates."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for x, y in _points(row):
            for value in (x, y):
                text = f"{value:.10f}".rstrip("0")
                places = len(text.split(".")[1]) if "." in text else 0
                out[row["provider"]][min(places, 6)] += 1
    return out


def grid_snap(rows: list[dict], tolerance: float = 1e-6) -> dict[str, dict[float, float]]:
    """provider -> grid -> fraction of coordinates landing exactly on it."""
    totals: Counter = Counter()
    hits: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for x, y in _points(row):
            for value in (x, y):
                totals[row["provider"]] += 1
                for grid in GRIDS:
                    if abs(value / grid - round(value / grid)) < tolerance:
                        hits[row["provider"]][grid] += 1
    return {
        provider: {grid: hits[provider][grid] / totals[provider] for grid in GRIDS}
        for provider in totals
    }


def localization_error(rows: list[dict], tasks: dict[str, dict]) -> dict[str, list[float]]:
    """provider -> distance in px from each submission's first point to the start badge.

    Pure perception: the badge is a fixed, high-contrast marker, so error here
    is unconfounded by planning or route quality.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        points = _points(row)
        task = tasks.get(row.get("maze"))
        if not points or task is None:
            continue
        width, height = task["width"], task["height"]
        sx = task["start"]["x"] * (width - 1)
        sy = task["start"]["y"] * (height - 1)
        px, py = points[0][0] * (width - 1), points[0][1] * (height - 1)
        out[row["provider"]].append(math.hypot(px - sx, py - sy))
    return out


def percentiles(values: list[float], points=(50, 75, 90, 95)) -> dict[int, float]:
    if not values:
        return {p: float("nan") for p in points}
    ordered = sorted(values)
    return {
        p: ordered[min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))]
        for p in points
    }
