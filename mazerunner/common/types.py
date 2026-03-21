"""Core data types for the MazeRunner benchmark."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

Cell = Tuple[int, int]  # (row, col)


@dataclass
class DifficultyConfig:
    tier: int
    grid_rows: int
    grid_cols: int
    min_solution_length: int


@dataclass
class MazeGrid:
    rows: int
    cols: int
    passages: Set[FrozenSet[Cell]]
    start: Cell
    goal: Cell
    solution_path: List[Cell]
    endpoint_type: str


@dataclass
class MazeInstance:
    id: str
    grid_rows: int
    grid_cols: int
    start: Cell
    goal: Cell
    adjacency: Dict[str, List[str]]
    shortest_path_cells: List[Cell]
    metadata: Dict = field(default_factory=dict)
