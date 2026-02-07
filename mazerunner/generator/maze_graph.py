"""Core maze generation and solving algorithms."""

from collections import deque
from typing import Dict, List, Set, Tuple, FrozenSet

import numpy as np

from mazerunner.common.types import Cell, MazeGrid


def _get_neighbors(row: int, col: int, rows: int, cols: int) -> List[Cell]:
    """Return valid orthogonal neighbors within grid bounds."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def generate_maze_dfs(rows: int, cols: int, rng: np.random.Generator) -> Set[FrozenSet[Cell]]:
    """Generate a perfect maze using randomized DFS with explicit stack."""
    visited = set()
    passages = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    while stack:
        current = stack[-1]
        neighbors = _get_neighbors(current[0], current[1], rows, cols)
        unvisited = [n for n in neighbors if n not in visited]

        if unvisited:
            # Pick a random unvisited neighbor
            idx = int(rng.integers(0, len(unvisited)))
            neighbor = unvisited[idx]
            passages.add(frozenset({current, neighbor}))
            visited.add(neighbor)
            stack.append(neighbor)
        else:
            stack.pop()

    return passages


def solve_bfs(
    rows: int,
    cols: int,
    passages: Set[FrozenSet[Cell]],
    start: Cell,
    goal: Cell,
) -> List[Cell]:
    """BFS from start to goal. Returns ordered cell list including start and goal."""
    parent: Dict[Cell, Cell] = {start: start}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in _get_neighbors(current[0], current[1], rows, cols):
            if neighbor not in parent and frozenset({current, neighbor}) in passages:
                parent[neighbor] = current
                queue.append(neighbor)

    # Reconstruct path
    path = []
    cell = goal
    while cell != start:
        path.append(cell)
        cell = parent[cell]
    path.append(start)
    path.reverse()
    return path


def bfs_distances(
    rows: int,
    cols: int,
    passages: Set[FrozenSet[Cell]],
    start: Cell,
) -> Dict[Cell, int]:
    """BFS from start, return dict mapping each cell to its distance from start."""
    dist: Dict[Cell, int] = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in _get_neighbors(current[0], current[1], rows, cols):
            if neighbor not in dist and frozenset({current, neighbor}) in passages:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)

    return dist


def choose_start_goal(
    rows: int,
    cols: int,
    passages: Set[FrozenSet[Cell]],
    min_distance: int,
    rng: np.random.Generator,
) -> Tuple[Cell, Cell]:
    """Pick random start cell, then pick goal at least min_distance away."""
    # Pick random start
    start_row = int(rng.integers(0, rows))
    start_col = int(rng.integers(0, cols))
    start = (start_row, start_col)

    distances = bfs_distances(rows, cols, passages, start)
    candidates = [cell for cell, d in distances.items() if d >= min_distance]

    if not candidates:
        # Use the farthest cell
        farthest = max(distances, key=lambda c: distances[c])
        candidates = [farthest]

    idx = int(rng.integers(0, len(candidates)))
    goal = candidates[idx]
    return start, goal


def build_maze(
    rows: int,
    cols: int,
    min_solution_length: int,
    rng: np.random.Generator,
) -> MazeGrid:
    """Generate maze, choose start/goal, solve, return MazeGrid."""
    passages = generate_maze_dfs(rows, cols, rng)
    start, goal = choose_start_goal(rows, cols, passages, min_solution_length, rng)
    solution_path = solve_bfs(rows, cols, passages, start, goal)
    return MazeGrid(
        rows=rows,
        cols=cols,
        passages=passages,
        start=start,
        goal=goal,
        solution_path=solution_path,
    )
