"""Rectilinear maze: randomized DFS spanning tree on a rectangular grid."""

from __future__ import annotations

import random

from ..world import Edge, World
from .common import dfs_spanning_tree, pick_far_endpoints, retain_route

COLS, ROWS = 8, 7
WIDTH, HEIGHT = 800, 740
MARGIN = 70
CORRIDOR = 26.0


def grid_layout(cols: int = COLS, rows: int = ROWS):
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
    return nodes, candidates, neighbors


def build(seed: int = 11, overrides: dict | None = None) -> World:
    """Randomized DFS spanning tree on a rectangular grid.

    Honors cols/rows/corridor. The plain grid maze: exactly one route between
    any two cells, so a model that reasons about connectivity correctly cannot
    be defeated by ambiguity — only by drawing.
    """
    o = {"cols": COLS, "rows": ROWS, "corridor": CORRIDOR, **(overrides or {})}
    rng = random.Random(seed)
    nodes, candidates, neighbors = grid_layout(o["cols"], o["rows"])
    tree = dfs_spanning_tree(sorted(nodes), neighbors, rng)
    edges = [Edge(a, b, [nodes[a], nodes[b]], o["corridor"]) for a, b in sorted(tree)]
    world = World(
        id="rectilinear",
        type="Rectilinear",
        style="Notebook ink",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Randomized DFS",
        solver="BFS",
        nodes=nodes,
        edges=edges,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
