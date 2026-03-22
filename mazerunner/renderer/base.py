"""Shared utilities and config dataclasses for maze rendering."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


DEFAULT_COLOR_SCHEMA = {
    "name": "classic",
    "wall": "#1a1a2e",
    "corridor": "#e8e8e8",
    "start": "#22c55e",
    "goal": "#ef4444",
    "solution_path": "#3b82f6",
    "background": "#f5f5f5",
}


def parse_cell(s: str) -> Tuple[int, int]:
    """Convert 'r,c' string to (row, col) tuple."""
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' hex string to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def load_instance(path: str) -> Dict[str, Any]:
    """Load a maze instance from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def has_wall(adjacency: Dict[str, List[str]], cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> bool:
    """Return True if there is no passage between two adjacent cells (i.e., a wall exists)."""
    key_a = f"{cell_a[0]},{cell_a[1]}"
    key_b = f"{cell_b[0]},{cell_b[1]}"
    return key_b not in adjacency.get(key_a, [])


def get_color_schema(instance: Dict[str, Any]) -> Dict[str, str]:
    """Extract color schema from instance, falling back to classic if missing."""
    schema = instance.get("metadata", {}).get("color_schema", {})
    if not schema or "wall" not in schema:
        return DEFAULT_COLOR_SCHEMA.copy()
    return schema


def is_edge_cell(r: int, c: int, rows: int, cols: int) -> bool:
    """Return True if the cell is on the border of the grid."""
    return r == 0 or r == rows - 1 or c == 0 or c == cols - 1


def get_border_sides(r: int, c: int, rows: int, cols: int) -> List[str]:
    """Return list of border sides for a cell, e.g. ['top'], ['left', 'top']."""
    sides = []
    if r == 0:
        sides.append("top")
    if r == rows - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == cols - 1:
        sides.append("right")
    return sides


_CLOCKWISE_ORDER = ["top", "right", "bottom", "left"]


def get_opening_side(r: int, c: int, rows: int, cols: int) -> str | None:
    """Return the single border side to open for an edge cell entrance/exit.

    For non-corner edge cells, returns the one border side.
    For corner cells, returns the most clockwise side (top → right → bottom → left).
    For interior cells, returns None.
    """
    sides = get_border_sides(r, c, rows, cols)
    if not sides:
        return None
    if len(sides) == 1:
        return sides[0]
    return max(sides, key=lambda s: _CLOCKWISE_ORDER.index(s))


@dataclass
class DragRenderConfig:
    wall_thickness: int = 4
    corridor_width: int = 20
    margin: int = 0
    marker_radius_frac: float = 0.4
    antialias: bool = True

    @property
    def cell_size(self) -> int:
        return self.corridor_width + self.wall_thickness


@dataclass
class GridRenderConfig:
    cell_size: int = 30
    wall_thickness: int = 4
    margin: int = 0
    marker_radius_frac: float = 0.35
    antialias: bool = True
    gridline_thickness: int = 1
