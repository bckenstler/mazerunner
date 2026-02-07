"""Start/goal placement logic for structural maze entrances."""

from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from mazerunner.common.types import Cell, RenderConfig


PLACEMENT_STYLES = ["edge_to_edge", "edge_to_center", "edge_to_dead_end"]


def choose_placement_style(rng: np.random.Generator) -> str:
    """Uniform random pick from the 3 placement styles."""
    idx = int(rng.integers(0, len(PLACEMENT_STYLES)))
    return PLACEMENT_STYLES[idx]


def _edge_cells(rows: int, cols: int) -> Dict[str, List[Cell]]:
    """Return dict mapping edge name to list of cells on that edge."""
    edges: Dict[str, List[Cell]] = {
        "top": [(0, c) for c in range(cols)],
        "bottom": [(rows - 1, c) for c in range(cols)],
        "left": [(r, 0) for r in range(rows)],
        "right": [(r, cols - 1) for r in range(rows)],
    }
    return edges


def _find_dead_ends(
    rows: int, cols: int, passages: Set[FrozenSet[Cell]]
) -> List[Cell]:
    """Find cells with exactly 1 passage (dead ends)."""
    degree: Dict[Cell, int] = {}
    for r in range(rows):
        for c in range(cols):
            degree[(r, c)] = 0
    for passage in passages:
        for cell in passage:
            degree[cell] = degree.get(cell, 0) + 1
    return [cell for cell, d in degree.items() if d == 1]


def _bfs_distances(
    rows: int,
    cols: int,
    passages: Set[FrozenSet[Cell]],
    start: Cell,
) -> Dict[Cell, int]:
    """BFS from start, return dict mapping each cell to its distance."""
    dist: Dict[Cell, int] = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols:
                if neighbor not in dist and frozenset({current, neighbor}) in passages:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
    return dist


def _is_interior(cell: Cell, rows: int, cols: int) -> bool:
    """Return True if cell is not on any edge of the grid."""
    r, c = cell
    return 0 < r < rows - 1 and 0 < c < cols - 1


def choose_start_goal_placed(
    rows: int,
    cols: int,
    passages: Set[FrozenSet[Cell]],
    min_distance: int,
    rng: np.random.Generator,
) -> Tuple[Cell, Cell, str, str, str]:
    """Pick start/goal with structural placement.

    Returns (start, goal, style, start_edge, goal_edge).
    """
    style = choose_placement_style(rng)
    edges = _edge_cells(rows, cols)
    edge_names = ["top", "bottom", "left", "right"]

    # Pick random start edge and cell
    start_edge_idx = int(rng.integers(0, len(edge_names)))
    start_edge = edge_names[start_edge_idx]
    start_cells = edges[start_edge]
    start_idx = int(rng.integers(0, len(start_cells)))
    start = start_cells[start_idx]

    distances = _bfs_distances(rows, cols, passages, start)

    if style == "edge_to_edge":
        # Goal on a different edge, with BFS distance >= min_distance
        other_edges = [e for e in edge_names if e != start_edge]
        goal_candidates = []
        for e in other_edges:
            for cell in edges[e]:
                if cell in distances and distances[cell] >= min_distance:
                    goal_candidates.append((cell, e))
        if not goal_candidates:
            # Fallback: farthest cell on any other edge
            best_cell = None
            best_edge = ""
            best_dist = -1
            for e in other_edges:
                for cell in edges[e]:
                    if cell in distances and distances[cell] > best_dist:
                        best_dist = distances[cell]
                        best_cell = cell
                        best_edge = e
            goal_candidates = [(best_cell, best_edge)]
        goal_idx = int(rng.integers(0, len(goal_candidates)))
        goal, goal_edge = goal_candidates[goal_idx]

    elif style == "edge_to_center":
        # Goal is an interior dead-end near maze center
        center_r = rows / 2.0
        center_c = cols / 2.0
        radius = max(rows, cols) // 4
        dead_ends = _find_dead_ends(rows, cols, passages)
        interior_dead_ends = [
            c for c in dead_ends
            if c != start and _is_interior(c, rows, cols)
        ]
        # Prefer dead ends near center with sufficient distance
        center_candidates = [
            c for c in interior_dead_ends
            if abs(c[0] - center_r) <= radius and abs(c[1] - center_c) <= radius
            and distances.get(c, 0) >= min_distance
        ]
        if not center_candidates:
            # Relax: any interior dead end with sufficient distance
            center_candidates = [
                c for c in interior_dead_ends
                if distances.get(c, 0) >= min_distance
            ]
        if not center_candidates:
            # Relax further: any interior dead end, sorted by distance from center
            center_candidates = interior_dead_ends
        if not center_candidates:
            # Final fallback: farthest interior cell
            interior_cells = [
                c for c in distances.keys()
                if c != start and _is_interior(c, rows, cols)
            ]
            if interior_cells:
                interior_cells.sort(key=lambda c: distances.get(c, 0), reverse=True)
                center_candidates = [interior_cells[0]]
            else:
                # Extremely small grid — just use farthest cell
                center_candidates = [max(distances, key=lambda c: distances[c])]
        goal_idx = int(rng.integers(0, len(center_candidates)))
        goal = center_candidates[goal_idx]
        goal_edge = ""  # interior goal

    else:  # edge_to_dead_end
        # Goal is the farthest interior dead-end cell from start
        dead_ends = _find_dead_ends(rows, cols, passages)
        interior_dead_ends = [
            c for c in dead_ends
            if c != start and _is_interior(c, rows, cols)
        ]
        if interior_dead_ends:
            interior_dead_ends.sort(key=lambda c: distances.get(c, 0), reverse=True)
            goal = interior_dead_ends[0]
        else:
            # Fallback: farthest interior cell
            interior_cells = [
                c for c in distances.keys()
                if c != start and _is_interior(c, rows, cols)
            ]
            if interior_cells:
                interior_cells.sort(key=lambda c: distances.get(c, 0), reverse=True)
                goal = interior_cells[0]
            else:
                goal = max(distances, key=lambda c: distances[c])
        goal_edge = ""  # always interior for this style

    return start, goal, style, start_edge, goal_edge


def opening_pixel_rect(
    cell: Cell,
    edge: str,
    config: RenderConfig,
    rows: int,
    cols: int,
) -> Tuple[int, int, int, int]:
    """Return (y_min, y_max, x_min, x_max) for the corridor-width gap in the outer wall.

    The gap is carved through the wall on the specified edge at the given cell.
    """
    cell_size = config.corridor_width + config.wall_thickness
    maze_origin_x = config.chrome_width_left + config.wall_thickness
    maze_origin_y = config.chrome_height_top + config.wall_thickness

    r, c = cell
    cw = config.corridor_width
    wt = config.wall_thickness

    if edge == "top":
        # Gap in the top wall, above cell (r, c)
        x_min = maze_origin_x + c * cell_size
        x_max = x_min + cw
        y_min = config.chrome_height_top
        y_max = y_min + wt
    elif edge == "bottom":
        # Gap in the bottom wall, below cell (r, c)
        x_min = maze_origin_x + c * cell_size
        x_max = x_min + cw
        y_min = maze_origin_y + r * cell_size + cw
        y_max = y_min + wt
    elif edge == "left":
        # Gap in the left wall, left of cell (r, c)
        x_min = config.chrome_width_left
        x_max = x_min + wt
        y_min = maze_origin_y + r * cell_size
        y_max = y_min + cw
    elif edge == "right":
        # Gap in the right wall, right of cell (r, c)
        x_min = maze_origin_x + c * cell_size + cw
        x_max = x_min + wt
        y_min = maze_origin_y + r * cell_size
        y_max = y_min + cw
    else:
        raise ValueError(f"Invalid edge: {edge}")

    return (y_min, y_max, x_min, x_max)


def opening_center(
    cell: Cell,
    edge: str,
    config: RenderConfig,
    rows: int,
    cols: int,
) -> Tuple[float, float]:
    """Return (x, y) pixel midpoint of the opening."""
    y_min, y_max, x_min, x_max = opening_pixel_rect(cell, edge, config, rows, cols)
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
