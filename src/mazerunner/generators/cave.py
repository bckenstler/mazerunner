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


def cellular_field(
    rng: random.Random,
    open_probability: float = OPEN_PROBABILITY,
    cols: int = COLS,
    rows: int = ROWS,
) -> np.ndarray:
    field = np.array(
        [[rng.random() < open_probability for _ in range(cols)] for _ in range(rows)]
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
    """Cellular-automata cave: its largest open component, on a 4-neighbor grid.

    Honors open_probability/cell. The only RASTER family — adjacency comes from cell
    geometry rather than a designed graph, so there is no tidy node structure to
    reason over and corridor edges are irregular. `check_edge_separation` is off
    for exactly this reason: here touching corridors *are* the connectivity.

    A smaller `cell` gives the automaton more cells on the same canvas, so the
    cave grows finer, more winding passages; the minimum component size scales
    with cell area so the same fraction of the canvas must stay connected.

    `seamless` makes neighboring cell rects share their boundary pixel. The v1
    rects stop one pixel short, which is invisible at scale 1 but opens
    one-pixel seams once augmentation scales the cave up to fit the canvas;
    the renderer stencils the mask, so the seams show as dashed lines that
    block the pointer. Off by default so v1 tasks rebuild byte-identically.
    """
    o = {"open_probability": OPEN_PROBABILITY, "cell": CELL, "seamless": False, **(overrides or {})}
    cell = o["cell"]
    inset = 0 if o["seamless"] else 1
    cols, rows = WIDTH // cell, HEIGHT // cell
    min_cells = MIN_COMPONENT_CELLS * (CELL / cell) ** 2
    rng = random.Random(seed)
    field = cellular_field(rng, o["open_probability"], cols, rows)
    labels, count = ndimage.label(field, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count == 0:
        raise ValueError("cave: no open cells")
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    if sizes[largest - 1] < min_cells:
        raise ValueError(f"cave: largest component {int(sizes[largest - 1])} cells is too small")
    open_cells = labels == largest

    nodes: dict[int, tuple[float, float]] = {}
    cell_ids: dict[tuple[int, int], int] = {}
    rects = []
    for r in range(rows):
        for c in range(cols):
            if open_cells[r, c]:
                n = len(nodes)
                cell_ids[(r, c)] = n
                nodes[n] = (c * cell + cell / 2, r * cell + cell / 2)
                rects.append(("rect", c * cell, r * cell, (c + 1) * cell - inset, (r + 1) * cell - inset))

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
