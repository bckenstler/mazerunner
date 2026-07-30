"""Cave: seeded cellular automata, largest open component, 4-neighbor grid."""

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from ..world import Edge, World
from .common import pick_far_endpoints, retain_route

WIDTH, HEIGHT = 800, 740
CELL = 20
COLS, ROWS = WIDTH // CELL, HEIGHT // CELL
OPEN_PROBABILITY = 0.58
SMOOTHING_STEPS = 5
MIN_COMPONENT_CELLS = 250


def cellular_field(rng: random.Random, open_probability: float = OPEN_PROBABILITY) -> np.ndarray:
    field = np.array(
        [[rng.random() < open_probability for _ in range(COLS)] for _ in range(ROWS)]
    )
    field[0, :] = field[-1, :] = False
    field[:, 0] = field[:, -1] = False
    for _ in range(SMOOTHING_STEPS):
        open_neighbors = ndimage.convolve(
            field.astype(int), np.ones((3, 3), dtype=int), mode="constant"
        ) - field.astype(int)
        field = open_neighbors >= 5
        field[0, :] = field[-1, :] = False
        field[:, 0] = field[:, -1] = False
    return field


def build(seed: int = 45, overrides: dict | None = None) -> World:
    o = {"open_probability": OPEN_PROBABILITY, **(overrides or {})}
    rng = random.Random(seed)
    field = cellular_field(rng, o["open_probability"])
    labels, count = ndimage.label(field, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count == 0:
        raise ValueError("cave: no open cells")
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    if sizes[largest - 1] < MIN_COMPONENT_CELLS:
        raise ValueError(f"cave: largest component {int(sizes[largest - 1])} cells is too small")
    open_cells = labels == largest

    nodes: dict[int, tuple[float, float]] = {}
    cell_ids: dict[tuple[int, int], int] = {}
    rects = []
    for r in range(ROWS):
        for c in range(COLS):
            if open_cells[r, c]:
                n = len(nodes)
                cell_ids[(r, c)] = n
                nodes[n] = (c * CELL + CELL / 2, r * CELL + CELL / 2)
                rects.append(("rect", c * CELL, r * CELL, (c + 1) * CELL - 1, (r + 1) * CELL - 1))

    edges = []
    for (r, c), n in cell_ids.items():
        for rr, cc in ((r, c + 1), (r + 1, c)):
            m = cell_ids.get((rr, cc))
            if m is not None:
                edges.append(Edge(n, m, [nodes[n], nodes[m]], 10.0))

    world = World(
        id="cave",
        type="Cave",
        style="Bioluminescent cavern",
        state_representation="RASTER",
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
        generator="Cellular automata",
        solver="Grid BFS",
        nodes=nodes,
        edges=edges,
        extra_open=rects,
        start_node=0,
        goal_node=0,
        check_edge_separation=False,
    )
    world.start_node, world.goal_node = pick_far_endpoints(world, rng, o.get("endpoint_quantile", 1.0))
    retain_route(world)
    return world
