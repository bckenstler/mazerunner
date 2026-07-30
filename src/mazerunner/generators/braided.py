"""Braided maze: DFS spanning tree on a grid plus carved loops."""

from __future__ import annotations

import random

from ..world import Edge, World
from .common import add_loops, dfs_spanning_tree, pick_far_endpoints, retain_route
from .rectilinear import CORRIDOR, HEIGHT, WIDTH, grid_layout

LOOPS = 12


def build(seed: int = 23, overrides: dict | None = None) -> World:
    o = {"cols": 8, "rows": 7, "corridor": CORRIDOR, "loops": LOOPS, **(overrides or {})}
    rng = random.Random(seed)
    nodes, candidates, neighbors = grid_layout(o["cols"], o["rows"])
    tree = dfs_spanning_tree(sorted(nodes), neighbors, rng)
    carved = add_loops(tree, candidates, o["loops"], rng)
    edges = [Edge(a, b, [nodes[a], nodes[b]], o["corridor"]) for a, b in sorted(carved)]
    world = World(
        id="braided",
        type="Braided",
        style="Dungeon stone",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="DFS plus loop carving",
        solver="BFS",
        nodes=nodes,
        edges=edges,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
