"""Island path: scattered islands joined by an MST of routes plus loops."""

from __future__ import annotations

import random

from ..world import Edge, World
from .common import (
    delaunay_candidates,
    mst_edges,
    pick_far_endpoints,
    retain_route,
    scatter_points,
)

WIDTH, HEIGHT = 800, 740
ISLANDS = 16
CORRIDOR = 16.0


def build(seed: int = 29, overrides: dict | None = None) -> World:
    o = {"islands": ISLANDS, "corridor": CORRIDOR, "extra_edges": 2, "min_dist": 128, **(overrides or {})}
    rng = random.Random(seed)
    points = scatter_points(rng, o["islands"], WIDTH, HEIGHT, margin=95, min_dist=o["min_dist"])
    nodes = dict(enumerate(points))
    pads = {n: rng.uniform(27, 36) for n in nodes}
    candidates = delaunay_candidates(points, max_length=280)
    candidate_set = set(candidates)

    chosen = mst_edges(nodes)
    if not chosen <= candidate_set:
        # Keep routes on Delaunay neighbors so bridges stay short and local.
        chosen &= candidate_set
    extras = [e for e in candidates if e not in chosen]
    rng.shuffle(extras)
    chosen |= set(extras[: o["extra_edges"]])

    edges = [Edge(a, b, [nodes[a], nodes[b]], o["corridor"]) for a, b in sorted(chosen)]
    world = World(
        id="island",
        type="Island path",
        style="Watercolor archipelago",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Spatial island graph",
        solver="Dijkstra",
        nodes=nodes,
        edges=edges,
        node_pad_radius=pads,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
