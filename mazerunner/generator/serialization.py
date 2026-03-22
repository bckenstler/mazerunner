"""MazeGrid -> canonical JSON serialization."""

import json
from typing import Any, Dict

from mazerunner.common.types import MazeGrid, MazeInstance
from mazerunner.generator.difficulty import compute_difficulty_score


def _cell_to_str(cell: tuple) -> str:
    """Convert a cell tuple to a string key."""
    return f"{cell[0]},{cell[1]}"


def _str_to_cell(s: str) -> tuple:
    """Convert a string key back to a cell tuple."""
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]))


def maze_grid_to_adjacency(grid: MazeGrid) -> Dict[str, list]:
    """Convert passages set to a sorted adjacency dict.

    Keys are ``"row,col"`` strings for all cells in the grid. Neighbor lists
    and keys are sorted lexicographically for deterministic JSON output.

    Args:
        grid: The maze grid to convert.

    Returns:
        Sorted adjacency dict mapping cell keys to sorted neighbor key lists.
    """
    adj: Dict[str, list] = {}

    # Initialize all cells
    for r in range(grid.rows):
        for c in range(grid.cols):
            key = _cell_to_str((r, c))
            adj[key] = []

    # Populate from passages
    for edge in grid.passages:
        cells = list(edge)
        a, b = cells[0], cells[1]
        adj[_cell_to_str(a)].append(_cell_to_str(b))
        adj[_cell_to_str(b)].append(_cell_to_str(a))

    # Sort adjacency lists for determinism
    for key in adj:
        adj[key] = sorted(adj[key])

    # Sort keys
    sorted_adj = dict(sorted(adj.items(), key=lambda x: (_str_to_cell(x[0])[0], _str_to_cell(x[0])[1])))
    return sorted_adj


def compute_branching_factor(adjacency: Dict[str, list]) -> float:
    """Compute average branching factor (average number of passages per cell).

    Args:
        adjacency: Adjacency dict mapping cell keys to neighbor key lists.

    Returns:
        Average degree across all cells, or 0.0 if adjacency is empty.
    """
    if not adjacency:
        return 0.0
    total = sum(len(neighbors) for neighbors in adjacency.values())
    return total / len(adjacency)


def maze_grid_to_instance(
    grid: MazeGrid, maze_id: str, color_schema: Dict[str, str] | None = None
) -> MazeInstance:
    """Convert a MazeGrid to a serializable MazeInstance.

    The returned instance has ``metadata["tier"]`` set to 0; the caller is
    responsible for setting the correct tier value.

    Args:
        grid: The in-memory maze grid.
        maze_id: Unique identifier for this maze (e.g. "maze_000042").
        color_schema: Optional color schema dict to embed in metadata.

    Returns:
        A MazeInstance with computed adjacency, difficulty score, and metadata.
    """
    adjacency = maze_grid_to_adjacency(grid)
    path_length = len(grid.solution_path)
    difficulty_score = compute_difficulty_score(path_length, grid.rows, grid.cols)
    branching_factor = compute_branching_factor(adjacency)

    metadata = {
        "tier": 0,  # Set by caller
        "endpoint_type": grid.endpoint_type,
        "difficulty_score": difficulty_score,
        "path_length": path_length,
        "branching_factor": round(branching_factor, 4),
        "grid_rows": grid.rows,
        "grid_cols": grid.cols,
        "color_schema": color_schema or {},
    }

    return MazeInstance(
        id=maze_id,
        grid_rows=grid.rows,
        grid_cols=grid.cols,
        start=grid.start,
        goal=grid.goal,
        adjacency=adjacency,
        shortest_path_cells=grid.solution_path,
        metadata=metadata,
    )


def instance_to_dict(instance: MazeInstance) -> Dict[str, Any]:
    """Convert a MazeInstance to a JSON-serializable dict.

    Cell coordinates are converted to ``"row,col"`` strings.

    Args:
        instance: The maze instance to convert.

    Returns:
        A dict suitable for ``json.dumps()``.
    """
    return {
        "id": instance.id,
        "grid_rows": instance.grid_rows,
        "grid_cols": instance.grid_cols,
        "start": _cell_to_str(instance.start),
        "goal": _cell_to_str(instance.goal),
        "adjacency": instance.adjacency,
        "shortest_path_cells": [_cell_to_str(c) for c in instance.shortest_path_cells],
        "metadata": instance.metadata,
    }


def instance_to_json(instance: MazeInstance, indent: int = 2) -> str:
    """Serialize a MazeInstance to a JSON string.

    Args:
        instance: The maze instance to serialize.
        indent: JSON indentation level.

    Returns:
        A formatted JSON string.
    """
    return json.dumps(instance_to_dict(instance), indent=indent)
