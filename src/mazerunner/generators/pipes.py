"""Pipes: sparse orthogonal network with loops on a coarse grid."""

from __future__ import annotations

import random

from ..world import Edge, World
from .common import add_loops, dfs_spanning_tree, pick_far_endpoints, retain_route

WIDTH, HEIGHT = 800, 740
COLS, ROWS = 6, 5
MARGIN = 95
CORRIDOR = 18.0
LOOPS = 4


def build(seed: int = 3, overrides: dict | None = None) -> World:
    o = {"cols": COLS, "rows": ROWS, "corridor": CORRIDOR, "loops": LOOPS, **(overrides or {})}
    cols, rows = o["cols"], o["rows"]
    rng = random.Random(seed)
    dx = (WIDTH - 2 * MARGIN) / (cols - 1)
    dy = (HEIGHT - 2 * MARGIN) / (rows - 1)
    nodes = {
        r * cols + c: (MARGIN + c * dx, MARGIN + r * dy)
        for r in range(rows)
        for c in range(cols)
    }
    candidates = []
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                candidates.append((n, n + 1))
            if r + 1 < rows:
                candidates.append((n, n + cols))
    neighbors: dict[int, list[int]] = {n: [] for n in nodes}
    for a, b in candidates:
        neighbors[a].append(b)
        neighbors[b].append(a)
    tree = dfs_spanning_tree(sorted(nodes), neighbors, rng)
    carved = add_loops(tree, candidates, o["loops"], rng)
    edges = [Edge(a, b, [nodes[a], nodes[b]], o["corridor"]) for a, b in sorted(carved)]
    world = World(
        id="pipes",
        type="Pipes",
        style="Neon factory",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Orthogonal network",
        solver="Dijkstra",
        nodes=nodes,
        edges=edges,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
