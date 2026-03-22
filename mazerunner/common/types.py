"""Core data types for the MazeRunner benchmark."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

Cell = Tuple[int, int]
"""A maze cell coordinate as (row, col), 0-indexed."""


@dataclass
class DifficultyConfig:
    """Configuration for a single maze's difficulty parameters, sampled from a tier.

    Attributes:
        tier: Difficulty tier (1=easy, 2=medium, 3=hard).
        grid_rows: Number of rows in the maze grid.
        grid_cols: Number of columns in the maze grid.
        min_solution_length: Minimum required BFS solution path length.
    """

    tier: int
    grid_rows: int
    grid_cols: int
    min_solution_length: int


@dataclass
class MazeGrid:
    """In-memory representation of a generated maze before serialization.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        passages: Set of edges as frozensets of two Cell tuples. For a perfect maze,
            ``len(passages) == rows * cols - 1`` (spanning tree invariant).
        start: Starting cell coordinate.
        goal: Goal cell coordinate.
        solution_path: BFS shortest path from start to goal as a list of cells.
        endpoint_type: One of "edge-edge", "interior-interior", "edge-interior",
            or "interior-edge".
    """

    rows: int
    cols: int
    passages: Set[FrozenSet[Cell]]
    start: Cell
    goal: Cell
    solution_path: List[Cell]
    endpoint_type: str


@dataclass
class MazeInstance:
    """Serializable maze instance ready for JSON output.

    Attributes:
        id: Unique maze identifier (e.g. "maze_000042").
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
        start: Starting cell coordinate.
        goal: Goal cell coordinate.
        adjacency: Sorted adjacency dict with ``"row,col"`` string keys mapping to
            sorted lists of neighbor keys.
        shortest_path_cells: BFS shortest path as a list of cell coordinates.
        metadata: Dict containing tier, endpoint_type, difficulty_score, path_length,
            branching_factor, and color_schema.
    """

    id: str
    grid_rows: int
    grid_cols: int
    start: Cell
    goal: Cell
    adjacency: Dict[str, List[str]]
    shortest_path_cells: List[Cell]
    metadata: Dict = field(default_factory=dict)
