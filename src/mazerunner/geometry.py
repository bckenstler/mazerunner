"""Continuous geometry helpers shared by generators, renderer, and evaluator."""

from __future__ import annotations

import math


Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polyline_length(points: list[Point]) -> float:
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def cumulative_lengths(points: list[Point]) -> list[float]:
    """Arclength at each vertex, starting at 0."""
    out = [0.0]
    for i in range(len(points) - 1):
        out.append(out[-1] + dist(points[i], points[i + 1]))
    return out


def project_onto_polyline(p: Point, points: list[Point]) -> tuple[float, float, int]:
    """Nearest point on a polyline. Returns (arclength, perpendicular distance, segment).

    A route that doubles back can pass near the same spot twice, so the same
    query point projects onto two arclengths equally well. Ties break toward the
    *smallest* arclength: progress should never be credited for reaching a place
    the route only visits later.
    """
    if len(points) < 2:
        return 0.0, dist(p, points[0]) if points else 0.0, 0
    cumulative = cumulative_lengths(points)
    best = (0.0, float("inf"), 0)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg_sq = dx * dx + dy * dy
        if seg_sq == 0.0:
            t = 0.0
        else:
            t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_sq
            t = max(0.0, min(1.0, t))
        foot = (a[0] + dx * t, a[1] + dy * t)
        perpendicular = dist(p, foot)
        # 1px slack so near-identical passes resolve to the earlier one.
        if perpendicular < best[1] - 1.0:
            best = (cumulative[i] + t * math.sqrt(seg_sq), perpendicular, i)
    return best


def densify_polyline(points: list[Point], spacing: float) -> list[Point]:
    """Resample a polyline so consecutive samples are at most `spacing` apart.

    Original vertices are always retained so bends are never cut.
    """
    if len(points) < 2:
        return list(points)
    out: list[Point] = [points[0]]
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        seg = dist(a, b)
        n = max(1, math.ceil(seg / spacing))
        for k in range(1, n + 1):
            out.append(lerp(a, b, k / n))
    return out


def quad_bezier(p0: Point, c: Point, p1: Point, n: int = 24) -> list[Point]:
    """Densified quadratic Bezier from p0 to p1 with control point c."""
    pts: list[Point] = []
    for k in range(n + 1):
        t = k / n
        u = 1.0 - t
        x = u * u * p0[0] + 2 * u * t * c[0] + t * t * p1[0]
        y = u * u * p0[1] + 2 * u * t * c[1] + t * t * p1[1]
        pts.append((x, y))
    return pts


def arc_points(
    center: Point, radius: float, a0: float, a1: float, max_step: float = 0.06
) -> list[Point]:
    """Circular arc from angle a0 to a1 (radians), densified by angle step."""
    sweep = a1 - a0
    n = max(2, math.ceil(abs(sweep) / max_step))
    pts: list[Point] = []
    for k in range(n + 1):
        a = a0 + sweep * (k / n)
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    return pts


def simplify_polyline(points: list[Point], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker: drop every vertex within `epsilon` px of the
    chord that replaces it, keeping both endpoints."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = points[a], points[b]
        chord = math.hypot(x2 - x1, y2 - y1) or 1e-9
        worst, index = -1.0, None
        for i in range(a + 1, b):
            x0, y0 = points[i]
            d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / chord
            if d > worst:
                worst, index = d, i
        if index is not None and worst > epsilon:
            keep[index] = True
            stack += [(a, index), (index, b)]
    return [p for p, k in zip(points, keep) if k]
