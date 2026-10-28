"""Rooms and corridors: jittered room grid joined by an MST plus one loop."""

from __future__ import annotations

import random

from ..geometry import polyline_length
from ..world import Edge, World
from .common import mst_edges, pick_far_endpoints, retain_route

WIDTH, HEIGHT = 800, 740
COLS, ROWS = 4, 3
MARGIN = 110
CORRIDOR = 18.0


def build(seed: int = 7, overrides: dict | None = None) -> World:
    """Jittered room grid joined by a minimum spanning tree plus one loop.

    Honors corridor/extra_edges/cols/rows/room_scale. Open rooms punctuate
    narrow corridors, so the pointer has slack in some places and almost none
    in others — clearance varies within a single route. A denser room grid
    needs smaller rooms to keep walls between them, which is what room_scale
    shrinks (room size and placement jitter together).
    """
    o = {"corridor": CORRIDOR, "extra_edges": 1, "cols": COLS, "rows": ROWS, "room_scale": 1.0,
         **(overrides or {})}
    cols, rows, s = o["cols"], o["rows"], o["room_scale"]
    rng = random.Random(seed)
    dx = (WIDTH - 2 * MARGIN) / (cols - 1)
    dy = (HEIGHT - 2 * MARGIN) / (rows - 1)
    nodes: dict[int, tuple[float, float]] = {}
    rects: dict[int, tuple[float, float, float, float]] = {}
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            x = MARGIN + c * dx + rng.uniform(-24 * s, 24 * s)
            y = MARGIN + r * dy + rng.uniform(-24 * s, 24 * s)
            half_w = rng.uniform(38 * s, 55 * s)
            half_h = rng.uniform(34 * s, 50 * s)
            nodes[n] = (x, y)
            rects[n] = (x - half_w, y - half_h, x + half_w, y + half_h)

    chosen = mst_edges(nodes)
    # One extra loop edge between near-but-unconnected rooms.
    extra = sorted(
        (
            (polyline_length([nodes[a], nodes[b]]), a, b)
            for a in nodes
            for b in nodes
            if a < b and (a, b) not in chosen
        )
    )
    for _d, a, b in extra[: o["extra_edges"]]:
        chosen.add((a, b))

    def hvh(a: int, b: int) -> list[tuple[float, float]]:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        mx = (x1 + x2) / 2
        return [(x1, y1), (mx, y1), (mx, y2), (x2, y2)]

    edges = [Edge(a, b, hvh(a, b), o["corridor"]) for a, b in sorted(chosen)]
    world = World(
        id="rooms",
        type="Rooms and corridors",
        style="Architect's plan",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Room graph plus MST",
        solver="Dijkstra",
        nodes=nodes,
        edges=edges,
        node_rects=rects,
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
