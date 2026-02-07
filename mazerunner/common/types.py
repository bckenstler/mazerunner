"""Shared data structures for MazeRunner benchmark."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

Point = Tuple[float, float]
Polyline = List[Point]
Cell = Tuple[int, int]


@dataclass
class MazeGrid:
    rows: int
    cols: int
    passages: Set[FrozenSet[Cell]]
    start: Cell
    goal: Cell
    solution_path: List[Cell]
    placement_style: str = "edge_to_edge"
    start_edge: str = "top"
    goal_edge: str = ""


@dataclass
class DifficultyConfig:
    tier: int
    grid_rows: int
    grid_cols: int
    corridor_width: int
    wall_thickness: int
    min_solution_length: int


@dataclass
class RenderConfig:
    image_width: int
    image_height: int
    corridor_width: int
    wall_thickness: int
    chrome_height_top: int
    chrome_width_left: int
    theme_name: str


@dataclass
class EvalResult:
    maze_id: str
    success: Dict[str, bool]
    valid_frac: Dict[str, float]
    min_clearance: float
    goal_distance: float
    path_length: float
    length_regret: float
    mono_score: float
    start_ok: bool
    goal_ok: bool
