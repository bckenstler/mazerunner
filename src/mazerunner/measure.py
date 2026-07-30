"""Task difficulty measurement.

Tiers are measured from certified artifacts, never declared by the sampler:
the geodesic length (normalized by canvas diagonal), the geodesic's turn
count, and the branchiness of the retained route. The sampler *aims* for a
tier by choosing difficulty params, but the recorded tier is what the built
task actually measures.
"""

from __future__ import annotations

import math

from .world import World, WorldValidation, adjacency

TURN_ANGLE_DEG = 30.0
# Bands over the combined score, calibrated on the smoke set (diameter-length
# routes measure 1.5–5.7); reviewed via `dataset stats`.
TIER_BANDS = {"easy": 1.6, "medium": 3.0}  # score < easy → easy, < medium → medium, else hard
BRANCH_CAP = 15  # raster worlds have hundreds of micro-junctions; cap their weight


def turn_count(points: list[tuple[float, float]], angle_deg: float = TURN_ANGLE_DEG) -> int:
    turns = 0
    for a, b, c in zip(points, points[1:], points[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_t = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.degrees(math.acos(cos_t)) >= angle_deg:
            turns += 1
    return turns


def route_branches(world: World) -> int:
    """Junction nodes (degree ≥ 3) along the retained route — decision points."""
    adj = adjacency(world)
    return sum(1 for n in world.path_nodes if len(adj[n]) >= 3)


def measure_task(world: World, validation: WorldValidation) -> dict:
    diagonal = math.hypot(world.width, world.height)
    norm_length = validation.geodesic_length_px / diagonal
    turns = turn_count(validation.geodesic_points_px)
    branches = route_branches(world)
    score = norm_length + turns / 12 + min(branches, BRANCH_CAP) / 10
    if score < TIER_BANDS["easy"]:
        tier = "easy"
    elif score < TIER_BANDS["medium"]:
        tier = "medium"
    else:
        tier = "hard"
    return {
        "geodesic_length_px": round(validation.geodesic_length_px, 2),
        "normalized_length": round(norm_length, 4),
        "turns": turns,
        "route_branches": branches,
        "difficulty_score": round(score, 4),
        "tier": tier,
    }
