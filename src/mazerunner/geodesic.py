"""Mask-certified geometric optimum.

The certified shortest legal route through the *hidden mask itself*, not the
generator's graph: erode the mask by the pointer disk (legal centers), run
8-connected Dijkstra start→goal over that region, then string-pull the grid
path with the real swept-disk collision check so grid metrication error is
removed. The result is the efficiency denominator — a submitted path can
never be legally shorter than it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from .evaluator import check_path_collision, disk_offsets
from .geometry import Point, polyline_length

SQRT2 = math.sqrt(2.0)


def legal_region(mask: np.ndarray, pointer_radius: int) -> np.ndarray:
    """Pixels where the pointer disk fits entirely inside the mask."""
    offsets = disk_offsets(pointer_radius)
    span = 2 * pointer_radius + 1
    structure = np.zeros((span, span), dtype=bool)
    structure[offsets[:, 0] + pointer_radius, offsets[:, 1] + pointer_radius] = True
    # border_value=0 so disks poking off-canvas are illegal, matching the scorer
    return ndimage.binary_erosion(mask, structure=structure, border_value=0)


def _grid_shortest_path(
    legal: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """8-connected Dijkstra over the legal region; returns (x, y) pixel chain."""
    h, w = legal.shape
    idx = -np.ones(legal.shape, dtype=np.int64)
    ys, xs = np.nonzero(legal)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    if n == 0 or idx[start[1], start[0]] < 0 or idx[goal[1], goal[0]] < 0:
        return None

    rows, cols, weights = [], [], []
    for dy, dx, weight in (
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, SQRT2),
        (1, -1, SQRT2),
    ):
        src_y = ys
        src_x = xs
        dst_y = src_y + dy
        dst_x = src_x + dx
        ok = (dst_y >= 0) & (dst_y < h) & (dst_x >= 0) & (dst_x < w)
        ok[ok] &= legal[dst_y[ok], dst_x[ok]]
        rows.append(idx[src_y[ok], src_x[ok]])
        cols.append(idx[dst_y[ok], dst_x[ok]])
        weights.append(np.full(int(ok.sum()), weight))
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    source = int(idx[start[1], start[0]])
    target = int(idx[goal[1], goal[0]])
    dist, predecessors = sparse_dijkstra(
        graph, directed=False, indices=source, return_predecessors=True
    )
    if not np.isfinite(dist[target]):
        return None
    chain = [target]
    while chain[-1] != source:
        chain.append(int(predecessors[chain[-1]]))
    chain.reverse()
    return [(int(xs[i]), int(ys[i])) for i in chain]


def _string_pull(
    points: list[Point], mask: np.ndarray, pointer_radius: int
) -> list[Point]:
    """Greedy shortcutting: replace runs of grid steps with the longest
    straight segments that pass the real swept-disk check. Removes the
    8-connected metrication overestimate."""
    pulled: list[Point] = [points[0]]
    i = 0
    while i < len(points) - 1:
        lo, hi = i + 1, len(points) - 1
        best = i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            free, _ = check_path_collision(mask, [points[i], points[mid]], pointer_radius)
            if free:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        pulled.append(points[best])
        i = best
    return pulled


def geodesic_optimum(
    mask: np.ndarray,
    start_px: Point,
    goal_px: Point,
    pointer_radius: int,
    margin: int = 1,
) -> tuple[float, list[Point]]:
    """Certified shortest legal route. Raises if none exists (fail closed).

    Computed with `margin` extra pixels of clearance so the stored polyline
    survives coordinate rounding and replays through the scorer at the true
    pointer radius. The tiny resulting overestimate of the optimum is covered
    by the evaluator's canary buffer.
    """
    effective = pointer_radius + margin
    legal = legal_region(mask, effective)
    start = (int(round(start_px[0])), int(round(start_px[1])))
    goal = (int(round(goal_px[0])), int(round(goal_px[1])))
    chain = _grid_shortest_path(legal, start, goal)
    if chain is None:
        raise ValueError("no legal route exists through the eroded mask")
    pulled = _string_pull([(float(x), float(y)) for x, y in chain], mask, effective)
    # String-pulled segments were verified with the swept disk at radius
    # pointer+margin; the polyline is a valid submission with slack to spare.
    return polyline_length(pulled), pulled
