"""Shared generator machinery: carving, endpoint selection, route retention."""

from __future__ import annotations

import math
import random

from .. import solver as solver_mod
from ..geometry import densify_polyline
from ..world import World, adjacency

WARP_WAVES = 4
WARP_SAMPLE_SPACING = 4.0


def dfs_spanning_tree(
    nodes: list[int], neighbors: dict[int, list[int]], rng: random.Random
) -> set[tuple[int, int]]:
    """Randomized DFS spanning tree over a candidate adjacency."""
    start = rng.choice(nodes)
    visited = {start}
    stack = [start]
    edges: set[tuple[int, int]] = set()
    while stack:
        node = stack[-1]
        unvisited = [n for n in neighbors[node] if n not in visited]
        if not unvisited:
            stack.pop()
            continue
        nxt = rng.choice(unvisited)
        visited.add(nxt)
        edges.add((min(node, nxt), max(node, nxt)))
        stack.append(nxt)
    if len(visited) != len(nodes):
        raise ValueError("candidate adjacency is not connected")
    return edges


def add_loops(
    tree: set[tuple[int, int]],
    candidates: list[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Braid a tree by re-adding a sample of removed candidate edges."""
    closed = [e for e in candidates if (min(e), max(e)) not in tree]
    picked = rng.sample(closed, min(count, len(closed)))
    return tree | {(min(a, b), max(a, b)) for a, b in picked}


def pick_far_endpoints(
    world: World, rng: random.Random, quantile: float = 1.0
) -> tuple[int, int]:
    """Endpoint pair by pixel distance.

    quantile=1.0 is the double-sweep diameter heuristic (hardest); lower
    quantiles pick a goal at that quantile of the distance distribution from
    the start — the primary difficulty lever for route length. Floored at the
    40th percentile so tasks never become trivial.
    """
    adj = adjacency(world)
    a = solver_mod.farthest_node(adj, rng.choice(sorted(world.nodes)))
    dist, _prev = solver_mod.dijkstra(adj, a)
    ranked = sorted(dist, key=lambda n: dist[n])
    quantile = max(0.4, min(1.0, quantile))
    b = ranked[round(quantile * (len(ranked) - 1))]
    return a, b


def mst_edges(nodes: dict[int, tuple[float, float]]) -> set[tuple[int, int]]:
    """Prim's MST over Euclidean distances."""
    from ..geometry import dist

    ids = sorted(nodes)
    in_tree = {ids[0]}
    edges: set[tuple[int, int]] = set()
    while len(in_tree) < len(ids):
        best = None
        for a in in_tree:
            for b in ids:
                if b in in_tree:
                    continue
                d = dist(nodes[a], nodes[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        in_tree.add(b)
        edges.add((min(a, b), max(a, b)))
    return edges


def scatter_points(
    rng: random.Random,
    count: int,
    width: float,
    height: float,
    margin: float,
    min_dist: float,
    max_tries: int = 20000,
) -> list[tuple[float, float]]:
    """Seeded rejection sampling with a minimum pairwise distance."""
    from ..geometry import dist

    points: list[tuple[float, float]] = []
    tries = 0
    while len(points) < count and tries < max_tries:
        tries += 1
        p = (rng.uniform(margin, width - margin), rng.uniform(margin, height - margin))
        if all(dist(p, q) >= min_dist for q in points):
            points.append(p)
    if len(points) < count:
        raise ValueError(f"could only place {len(points)}/{count} points")
    return points


def delaunay_candidates(
    points: list[tuple[float, float]], max_length: float
) -> list[tuple[int, int]]:
    """Delaunay edges shorter than max_length, as candidate adjacency."""
    import numpy as np
    from scipy.spatial import Delaunay

    from ..geometry import dist

    tri = Delaunay(np.array(points))
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            if dist(points[a], points[b]) <= max_length:
                edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def retain_route(world: World) -> None:
    """Solve with pixel-weighted Dijkstra and retain the route on the world."""
    adj = adjacency(world)
    path, _length = solver_mod.dijkstra_path(adj, world.start_node, world.goal_node)
    if path is None:
        raise ValueError(f"{world.id}: goal unreachable from start")
    world.path_nodes = path


def warp_world(world: World, seed: int, amplitude: float, wavelength: float) -> None:
    """Bend a maze through a smooth random displacement field, in place.

    Topology is untouched; only geometry moves. Straight corridors become
    curves, so a grid maze keeps its structure but loses the axis-aligned
    regularity that makes it easy to read. The field is a sum of a few plane
    waves with wavelengths well above corridor spacing, so neighboring
    corridors bend together; where shear still pushes two corridors too
    close, validation rejects the world and the sampler resamples.

    Edges are densified before displacement so a two-point segment becomes a
    curve rather than a moved straight line. The field draws from its own
    stream derived from `seed`, so a warped build never shifts the random
    sequence the unwarped generator consumes.

    Generators call this after picking endpoints and before retaining the
    route, so a warped maze and its unwarped twin (same seed) share topology,
    start, and goal; only the drawn geometry differs. The route is solved on
    the warped lengths, so the stored reference is still a true shortest path.
    """
    rng = random.Random(f"warp:{seed}")
    waves = []
    for _ in range(WARP_WAVES):
        k = 2 * math.pi / (wavelength * rng.uniform(1.0, 1.5))
        theta = rng.uniform(0, 2 * math.pi)
        push = rng.uniform(0, 2 * math.pi)
        waves.append((k * math.cos(theta), k * math.sin(theta), rng.uniform(0, 2 * math.pi),
                      math.cos(push), math.sin(push)))
    gain = amplitude / math.sqrt(WARP_WAVES)

    def displace(p: tuple[float, float]) -> tuple[float, float]:
        x, y = p
        dx = dy = 0.0
        for kx, ky, phase, ux, uy in waves:
            s = math.sin(kx * x + ky * y + phase)
            dx += ux * s
            dy += uy * s
        return x + gain * dx, y + gain * dy

    world.nodes = {n: displace(p) for n, p in world.nodes.items()}
    for edge in world.edges:
        edge.geometry = [displace(p) for p in densify_polyline(edge.geometry, WARP_SAMPLE_SPACING)]
