"""Start/goal placement with 4 endpoint types."""

from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from mazerunner.common.types import Cell
from mazerunner.generator.maze_graph import bfs_distances

ENDPOINT_TYPES = ["edge-edge", "interior-interior", "edge-interior", "interior-edge"]


def is_edge_cell(cell: Cell, rows: int, cols: int) -> bool:
    """Check if a cell is on the grid border."""
    r, c = cell
    return r == 0 or r == rows - 1 or c == 0 or c == cols - 1


def is_interior_cell(cell: Cell, rows: int, cols: int) -> bool:
    """Check if a cell is not on the grid border."""
    return not is_edge_cell(cell, rows, cols)


def get_dead_ends(passages: Set[FrozenSet[Cell]], rows: int, cols: int) -> List[Cell]:
    """Return all cells with exactly 1 passage neighbor (dead ends)."""
    degree: Dict[Cell, int] = {}
    for edge in passages:
        for cell in edge:
            degree[cell] = degree.get(cell, 0) + 1
    return [cell for cell, d in degree.items() if d == 1]


def _cell_degree(cell: Cell, passages: Set[FrozenSet[Cell]]) -> int:
    """Count how many passages connect to a cell."""
    count = 0
    for edge in passages:
        if cell in edge:
            count += 1
    return count


def place_endpoints(
    passages: Set[FrozenSet[Cell]],
    rows: int,
    cols: int,
    endpoint_type: str,
    min_solution_length: int,
    rng: np.random.Generator,
) -> Tuple[Cell, Cell]:
    """Place start and goal according to endpoint type and min distance."""
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    edge_cells = [c for c in all_cells if is_edge_cell(c, rows, cols)]
    dead_ends = get_dead_ends(passages, rows, cols)
    interior_dead_ends = [c for c in dead_ends if is_interior_cell(c, rows, cols)]

    # Determine start and goal pools
    start_type, goal_type = endpoint_type.split("-")

    if start_type == "edge":
        start_pool = edge_cells
    else:
        start_pool = interior_dead_ends

    if not start_pool:
        # Fallback: use edge cells
        start_pool = edge_cells

    # Pick start randomly
    start_idx = rng.integers(len(start_pool))
    start = start_pool[start_idx]

    # Compute distances from start
    distances = bfs_distances(passages, start, rows, cols)

    # Build goal pool
    if goal_type == "edge":
        goal_pool = edge_cells
        # For edge-edge, goal must be on a different border than start
        if start_type == "edge":
            start_borders = _get_borders(start, rows, cols)
            goal_pool = [
                c for c in goal_pool
                if c != start and not _get_borders(c, rows, cols).issubset(start_borders)
            ]
    else:
        goal_pool = interior_dead_ends
        goal_pool = [c for c in goal_pool if c != start]

    if not goal_pool:
        # Fallback: use all reachable cells except start
        goal_pool = [c for c in all_cells if c != start and c in distances]

    # Filter by min distance
    qualifying = [c for c in goal_pool if c in distances and distances[c] >= min_solution_length]

    if qualifying:
        # Pick randomly from qualifying
        goal_idx = rng.integers(len(qualifying))
        goal = qualifying[goal_idx]
    else:
        # Fallback: pick the farthest qualifying cell
        goal_pool_with_dist = [(c, distances.get(c, 0)) for c in goal_pool if c in distances]
        if goal_pool_with_dist:
            goal = max(goal_pool_with_dist, key=lambda x: x[1])[0]
        else:
            # Last resort
            goal = (rows - 1, cols - 1) if start != (rows - 1, cols - 1) else (0, 0)

    return start, goal


def _get_borders(cell: Cell, rows: int, cols: int) -> set:
    """Return the set of borders a cell is on."""
    r, c = cell
    borders = set()
    if r == 0:
        borders.add("top")
    if r == rows - 1:
        borders.add("bottom")
    if c == 0:
        borders.add("left")
    if c == cols - 1:
        borders.add("right")
    return borders
