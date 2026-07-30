"""Organic maze: scattered nodes, Delaunay candidates, curved Bezier corridors."""

from __future__ import annotations

import math
import random

from ..geometry import quad_bezier
from ..world import Edge, World
from .common import (
    add_loops,
    delaunay_candidates,
    dfs_spanning_tree,
    pick_far_endpoints,
    retain_route,
    scatter_points,
)

WIDTH, HEIGHT = 800, 740
NODES = 30
CORRIDOR = 22.0


def curved(a: tuple[float, float], b: tuple[float, float], rng: random.Random, max_bow: float = 30.0):
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    # Perpendicular control-point offset gives each corridor a gentle bow,
    # capped so dense node layouts don't bow corridors into each other.
    off = rng.uniform(0.4 * max_bow, max_bow) * rng.choice([-1, 1])
    c = (mx - dy / length * off, my + dx / length * off)
    return quad_bezier(a, c, b, n=24)


def build(seed: int = 5, overrides: dict | None = None) -> World:
    # The sampler picks (nodes, min_dist, corridor) as a coherent tuple —
    # dense layouts need narrower corridors to satisfy edge separation.
    o = {"nodes": NODES, "corridor": CORRIDOR, "loops": 1, "min_dist": 92, **(overrides or {})}
    rng = random.Random(seed)
    min_dist = o["min_dist"]
    points = scatter_points(rng, o["nodes"], WIDTH, HEIGHT, margin=85, min_dist=min_dist)
    nodes = dict(enumerate(points))
    candidates = delaunay_candidates(points, max_length=210)
    neighbors: dict[int, list[int]] = {n: [] for n in nodes}
    for a, b in candidates:
        neighbors[a].append(b)
        neighbors[b].append(a)
    tree = dfs_spanning_tree(sorted(nodes), neighbors, rng)
    carved = add_loops(tree, candidates, o["loops"], rng)
    max_bow = min(30.0, min_dist * 0.22)
    edges = [
        Edge(a, b, curved(nodes[a], nodes[b], rng, max_bow), o["corridor"])
        for a, b in sorted(carved)
    ]
    world = World(
        id="organic",
        type="Organic",
        style="Enchanted forest",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Curved graph embedding",
        solver="BFS",
        nodes=nodes,
        edges=edges,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
