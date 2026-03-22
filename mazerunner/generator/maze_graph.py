"""Randomized DFS maze generation and BFS solving."""

from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from mazerunner.common.types import Cell


def _neighbors(cell: Cell, rows: int, cols: int) -> List[Cell]:
    """Return valid grid neighbors (up, down, left, right)."""
    r, c = cell
    result = []
    if r > 0:
        result.append((r - 1, c))
    if r < rows - 1:
        result.append((r + 1, c))
    if c > 0:
        result.append((r, c - 1))
    if c < cols - 1:
        result.append((r, c + 1))
    return result


def generate_maze(rows: int, cols: int, rng: np.random.Generator) -> Set[FrozenSet[Cell]]:
    """Generate a perfect maze using iterative randomized DFS.

    Uses an iterative (not recursive) approach to avoid stack overflow on large grids.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        rng: numpy random Generator for deterministic shuffling.

    Returns:
        A set of passages (edges) as frozensets of two adjacent cells.
        Invariant: ``len(passages) == rows * cols - 1`` (spanning tree).
    """
    visited = set()
    passages: Set[FrozenSet[Cell]] = set()
    start = (0, 0)
    visited.add(start)
    stack = [start]

    while stack:
        current = stack[-1]
        unvisited = [n for n in _neighbors(current, rows, cols) if n not in visited]

        if not unvisited:
            stack.pop()
            continue

        # Shuffle using the provided RNG for determinism
        indices = rng.permutation(len(unvisited))
        next_cell = unvisited[indices[0]]

        passages.add(frozenset((current, next_cell)))
        visited.add(next_cell)
        stack.append(next_cell)

    return passages


def _build_adjacency(passages: Set[FrozenSet[Cell]]) -> Dict[Cell, List[Cell]]:
    """Build adjacency list from passage set."""
    adj: Dict[Cell, List[Cell]] = {}
    for edge in passages:
        a, b = tuple(edge)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def solve_bfs(
    passages: Set[FrozenSet[Cell]], start: Cell, goal: Cell, rows: int, cols: int
) -> List[Cell]:
    """Find shortest path from start to goal using BFS.

    Args:
        passages: Set of passage edges (frozensets of two cells).
        start: Starting cell.
        goal: Goal cell.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.

    Returns:
        The shortest path as a list of cells from start to goal inclusive,
        or an empty list if no path exists.
    """
    adj = _build_adjacency(passages)
    queue = deque([start])
    parent: Dict[Cell, Cell | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in adj.get(current, []):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    if goal not in parent:
        return []

    path = []
    cell: Cell | None = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path


def bfs_distances(
    passages: Set[FrozenSet[Cell]], start: Cell, rows: int, cols: int
) -> Dict[Cell, int]:
    """Compute BFS distances from start to all reachable cells.

    Args:
        passages: Set of passage edges (frozensets of two cells).
        start: Starting cell for distance computation.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.

    Returns:
        Dict mapping each reachable cell to its BFS distance from start.
    """
    adj = _build_adjacency(passages)
    distances: Dict[Cell, int] = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in adj.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances
