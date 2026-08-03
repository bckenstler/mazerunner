"""Radial maze: polar cells carved by DFS; arcs within rings, spokes between."""

from __future__ import annotations

import math
import random

from ..geometry import arc_points
from ..world import Edge, World
from .common import add_loops, dfs_spanning_tree, pick_far_endpoints, retain_route

WIDTH, HEIGHT = 800, 740
CENTER = (400.0, 370.0)
RING_RADII = [80.0, 140.0, 200.0, 260.0, 320.0]
RING_SECTORS = [8, 10, 12, 16, 18]
CORRIDOR = 24.0
CENTER_PAD = 30.0


def build(seed: int = 17, overrides: dict | None = None) -> World:
    """Polar cells carved by DFS — arcs within rings, spokes between them.

    Honors rings/corridor/loops. Concentric geometry yields few but long
    routes, and every arc is a continuous curve, so route length here is
    dominated by arcs rather than turn count.
    """
    o = {"rings": len(RING_RADII), "corridor": CORRIDOR, "loops": 1, **(overrides or {})}
    ring_radii = RING_RADII[: o["rings"]]
    ring_sectors = RING_SECTORS[: o["rings"]]
    rng = random.Random(seed)
    nodes: dict[int, tuple[float, float]] = {0: CENTER}
    angle: dict[int, float] = {}
    ring_of: dict[int, int] = {0: -1}
    rings: list[list[int]] = []
    for i, (radius, sectors) in enumerate(zip(ring_radii, ring_sectors)):
        ring = []
        rotation = rng.uniform(0, 2 * math.pi)
        for j in range(sectors):
            n = len(nodes)
            theta = rotation + 2 * math.pi * j / sectors
            nodes[n] = (
                CENTER[0] + radius * math.cos(theta),
                CENTER[1] + radius * math.sin(theta),
            )
            angle[n] = theta
            ring_of[n] = i
            ring.append(n)
        rings.append(ring)

    geometries: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def register(a: int, b: int, geometry) -> tuple[int, int]:
        key = (min(a, b), max(a, b))
        geometries[key] = geometry if a < b else geometry[::-1]
        return key

    candidates: list[tuple[int, int]] = []
    # Arcs between angular neighbors within each ring.
    for i, ring in enumerate(rings):
        for k, n in enumerate(ring):
            m = ring[(k + 1) % len(ring)]
            a0, a1 = angle[n], angle[m]
            if a1 < a0:
                a1 += 2 * math.pi
            candidates.append(register(n, m, arc_points(CENTER, ring_radii[i], a0, a1)))
    # Spokes: center to the innermost ring, then nearest-angle links outward.
    for n in rings[0]:
        candidates.append(register(0, n, [CENTER, nodes[n]]))
    for i in range(len(rings) - 1):
        for n in rings[i]:
            m = min(
                rings[i + 1],
                key=lambda cand: abs(
                    math.remainder(angle[cand] - angle[n], 2 * math.pi)
                ),
            )
            candidates.append(register(n, m, [nodes[n], nodes[m]]))

    candidates = sorted(set(candidates))
    neighbors: dict[int, list[int]] = {n: [] for n in nodes}
    for a, b in candidates:
        neighbors[a].append(b)
        neighbors[b].append(a)
    tree = dfs_spanning_tree(sorted(nodes), neighbors, rng)
    carved = add_loops(tree, candidates, o["loops"], rng)
    edges = [Edge(a, b, geometries[(a, b)], o["corridor"]) for a, b in sorted(carved)]

    world = World(
        id="radial",
        type="Radial",
        style="Astronomer's chart",
        state_representation="GRAPH",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Polar-cell DFS",
        solver="BFS",
        nodes=nodes,
        edges=edges,
        node_pad_radius={0: CENTER_PAD},
        start_node=0,
        goal_node=0,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
