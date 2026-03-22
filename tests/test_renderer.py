"""Tests for the renderer module."""

import pytest

from mazerunner.renderer.base import (
    DragRenderConfig,
    GridRenderConfig,
    get_border_sides,
    get_opening_side,
    has_wall,
    hex_to_rgb,
    is_edge_cell,
    parse_cell,
)
from mazerunner.renderer.text_grid import render_text_grid
from mazerunner.renderer.vision_drag import (
    cell_to_pixel_center,
    cell_to_pixel_rect,
    render_vision_drag,
)
from mazerunner.renderer.vision_grid import render_vision_grid


def _make_simple_instance(rows: int, cols: int, passages=None, start=None, goal=None) -> dict:
    """Build a maze instance dict for testing.

    If passages is None, connects every cell to its right and down neighbor
    (all walls removed). Otherwise, uses the provided passage list of
    ((r1,c1),(r2,c2)) tuples.
    """
    adjacency = {}
    for r in range(rows):
        for c in range(cols):
            adjacency[f"{r},{c}"] = []

    if passages is None:
        # Fully connected grid
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    adjacency[f"{r},{c}"].append(f"{r},{c + 1}")
                    adjacency[f"{r},{c + 1}"].append(f"{r},{c}")
                if r + 1 < rows:
                    adjacency[f"{r},{c}"].append(f"{r + 1},{c}")
                    adjacency[f"{r + 1},{c}"].append(f"{r},{c}")
    else:
        for (r1, c1), (r2, c2) in passages:
            adjacency[f"{r1},{c1}"].append(f"{r2},{c2}")
            adjacency[f"{r2},{c2}"].append(f"{r1},{c1}")

    # Sort adjacency lists
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (rows - 1, cols - 1)

    return {
        "id": "test_maze",
        "grid_rows": rows,
        "grid_cols": cols,
        "start": f"{start[0]},{start[1]}",
        "goal": f"{goal[0]},{goal[1]}",
        "adjacency": adjacency,
        "shortest_path_cells": [f"{start[0]},{start[1]}", f"{goal[0]},{goal[1]}"],
        "metadata": {
            "color_schema": {
                "name": "classic",
                "wall": "#1a1a2e",
                "corridor": "#e8e8e8",
                "start": "#22c55e",
                "goal": "#ef4444",
                "solution_path": "#3b82f6",
                "background": "#f5f5f5",
            }
        },
    }


# ─── base utilities ───────────────────────────────────────────────


class TestParseCell:
    def test_basic(self):
        assert parse_cell("3,7") == (3, 7)

    def test_zero(self):
        assert parse_cell("0,0") == (0, 0)

    def test_large(self):
        assert parse_cell("27,39") == (27, 39)


class TestHexToRgb:
    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)

    def test_red(self):
        assert hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_mixed_case(self):
        assert hex_to_rgb("#aaBBcc") == (170, 187, 204)

    def test_without_hash(self):
        # Should still work if someone passes without hash
        assert hex_to_rgb("1a1a2e") == (26, 26, 46)


class TestHasWall:
    def test_passage_exists(self):
        adj = {"0,0": ["0,1"], "0,1": ["0,0"]}
        assert has_wall(adj, (0, 0), (0, 1)) is False

    def test_wall_exists(self):
        adj = {"0,0": [], "0,1": []}
        assert has_wall(adj, (0, 0), (0, 1)) is True

    def test_missing_key(self):
        adj = {}
        assert has_wall(adj, (0, 0), (0, 1)) is True


class TestIsEdgeCell:
    def test_corner(self):
        assert is_edge_cell(0, 0, 5, 5) is True

    def test_top_edge(self):
        assert is_edge_cell(0, 2, 5, 5) is True

    def test_interior(self):
        assert is_edge_cell(2, 2, 5, 5) is False

    def test_bottom_right(self):
        assert is_edge_cell(4, 4, 5, 5) is True

    def test_left_edge(self):
        assert is_edge_cell(3, 0, 5, 5) is True


class TestGetBorderSides:
    def test_top_left_corner(self):
        sides = get_border_sides(0, 0, 5, 5)
        assert sorted(sides) == ["left", "top"]

    def test_bottom_right_corner(self):
        sides = get_border_sides(4, 4, 5, 5)
        assert sorted(sides) == ["bottom", "right"]

    def test_top_edge_only(self):
        assert get_border_sides(0, 2, 5, 5) == ["top"]

    def test_interior(self):
        assert get_border_sides(2, 2, 5, 5) == []


class TestGetOpeningSide:
    def test_top_left_corner(self):
        # top and left → left (most clockwise)
        assert get_opening_side(0, 0, 5, 5) == "left"

    def test_top_right_corner(self):
        # top and right → right (most clockwise)
        assert get_opening_side(0, 4, 5, 5) == "right"

    def test_bottom_right_corner(self):
        # bottom and right → bottom (most clockwise)
        assert get_opening_side(4, 4, 5, 5) == "bottom"

    def test_bottom_left_corner(self):
        # bottom and left → left (most clockwise)
        assert get_opening_side(4, 0, 5, 5) == "left"

    def test_non_corner_edge(self):
        assert get_opening_side(0, 2, 5, 5) == "top"

    def test_interior(self):
        assert get_opening_side(2, 2, 5, 5) is None


# ─── text_grid ────────────────────────────────────────────────────


class TestTextGrid:
    def test_2x2_fully_connected(self):
        instance = _make_simple_instance(2, 2)
        text = render_text_grid(instance)
        lines = text.split("\n")
        # 2*2+1 = 5 lines
        assert len(lines) == 5
        # 4*2+1 = 9 chars per line
        for line in lines:
            assert len(line) == 9

    def test_markers_present(self):
        instance = _make_simple_instance(2, 2)
        text = render_text_grid(instance)
        assert " S " in text
        assert " G " in text

    @pytest.mark.parametrize("rows,cols", [(3, 3), (5, 7), (2, 4)])
    def test_dimensions(self, rows, cols):
        instance = _make_simple_instance(rows, cols)
        text = render_text_grid(instance)
        lines = text.split("\n")
        assert len(lines) == 2 * rows + 1
        for line in lines:
            assert len(line) == 4 * cols + 1

    def test_all_walls(self):
        """A maze with no passages should have walls everywhere.
        Start (0,0) corner: opening on left (most clockwise of top/left).
        Goal (1,1) corner: opening on bottom (most clockwise of bottom/right)."""
        instance = _make_simple_instance(2, 2, passages=[])
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Top border: no opening (start corner opens on left, not top)
        assert lines[0] == "+---+---+"
        assert lines[2] == "+---+---+"
        # Bottom border: opening below goal (1,1) — bottom is most clockwise
        assert lines[4] == "+---+   +"
        # Row 0: left border open for start (0,0)
        assert lines[1] == "  S |   |"
        # Row 1: right border NOT open for goal (bottom chosen instead)
        assert lines[3] == "|   | G |"

    def test_all_open(self):
        """A fully connected maze should have no interior walls."""
        instance = _make_simple_instance(2, 2)
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Top border: no opening (start at corner opens left)
        assert lines[0] == "+---+---+"
        # No horizontal walls between rows
        assert lines[2] == "+   +   +"
        # Row 0: left border open for start
        assert lines[1] == "  S     |"
        # Row 1: right border NOT open (goal opens bottom)
        assert lines[3] == "|     G |"
        # Bottom border: opening below goal (1,1)
        assert lines[4] == "+---+   +"

    def test_border_opening_top_edge(self):
        """Start on top edge should have opening in top border."""
        instance = _make_simple_instance(3, 3, start=(0, 1), goal=(2, 2))
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Top wall row: opening above (0,1) — col 1 wall segment at indices 5-7
        assert lines[0][5:8] == "   "

    def test_border_opening_left_edge(self):
        """Start on left edge should have opening on left border."""
        instance = _make_simple_instance(3, 3, start=(1, 0), goal=(2, 2))
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Cell row for r=1: left border should be space
        assert lines[3][0] == " "

    def test_border_opening_bottom_edge(self):
        """Goal on bottom edge should have opening in bottom border."""
        instance = _make_simple_instance(3, 3, start=(0, 0), goal=(2, 1))
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Bottom wall row: opening below (2,1) — col 1 wall segment at indices 5-7
        assert lines[6][5:8] == "   "

    def test_border_opening_right_edge(self):
        """Goal on right edge should have opening on right border."""
        instance = _make_simple_instance(3, 3, start=(0, 0), goal=(1, 2))
        text = render_text_grid(instance)
        lines = text.split("\n")
        # Cell row for r=1: right border should be space
        assert lines[3][-1] == " "

    def test_no_opening_interior_endpoints(self):
        """Interior endpoints should not create border openings."""
        instance = _make_simple_instance(3, 3, start=(1, 1), goal=(1, 1))
        text = render_text_grid(instance)
        lines = text.split("\n")
        # All borders should be intact
        assert lines[0][0] == "+"
        assert lines[0][-1] == "+"
        # Left/right borders should be |
        assert lines[1][0] == "|"
        assert lines[1][-1] == "|"

    def test_real_instance_from_generator(self):
        """Render a maze from the actual generator and check basic structure."""
        from mazerunner.generator.maze_graph import generate_maze, solve_bfs
        from mazerunner.generator.seed_utils import make_rng
        from mazerunner.generator.serialization import (
            instance_to_dict,
            maze_grid_to_instance,
        )
        from mazerunner.common.types import MazeGrid

        rng = make_rng(42)
        rows, cols = 5, 7
        passages = generate_maze(rows, cols, rng)
        start, goal = (0, 0), (rows - 1, cols - 1)
        solution = solve_bfs(passages, start, goal, rows, cols)
        grid = MazeGrid(rows=rows, cols=cols, passages=passages,
                        start=start, goal=goal, solution_path=solution,
                        endpoint_type="edge-edge")
        inst = maze_grid_to_instance(grid, "test")
        d = instance_to_dict(inst)

        text = render_text_grid(d)
        lines = text.split("\n")
        assert len(lines) == 2 * rows + 1
        assert " S " in text
        assert " G " in text


# ─── vision_drag ──────────────────────────────────────────────────


class TestVisionDrag:
    def test_image_dimensions(self):
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 4)
        img = render_vision_drag(instance, config)
        expected_w = 0 + 4 + 4 * (20 + 4)
        expected_h = 0 + 4 + 3 * (20 + 4)
        assert img.size == (expected_w, expected_h)

    def test_image_dimensions_with_margin(self):
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=10, antialias=False)
        instance = _make_simple_instance(3, 4)
        img = render_vision_drag(instance, config)
        expected_w = 20 + 4 + 4 * 24
        expected_h = 20 + 4 + 3 * 24
        assert img.size == (expected_w, expected_h)

    def test_rgb_mode(self):
        config = DragRenderConfig(antialias=False)
        instance = _make_simple_instance(2, 2)
        img = render_vision_drag(instance, config)
        assert img.mode == "RGB"

    def test_corridor_pixel_color(self):
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_drag(instance, config)
        # Cell (1,1) center should be corridor color (no marker there)
        cx, cy = cell_to_pixel_center(1, 1, config)
        pixel = img.getpixel((int(cx), int(cy)))
        corridor_rgb = (232, 232, 232)  # #e8e8e8
        assert pixel == corridor_rgb

    def test_wall_pixel_color(self):
        """Interior wall junction should be wall color."""
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3, passages=[])
        img = render_vision_drag(instance, config)
        # Pick a wall junction point (between cells, not on border where opening may exist)
        # Wall between (0,0) and (0,1) at x=wt+cw=24, y=1
        wt, cw = 4, 20
        pixel = img.getpixel((wt + cw, 1))
        wall_rgb = (26, 26, 46)  # #1a1a2e
        assert pixel == wall_rgb

    def test_start_goal_letter_markers(self):
        """Start and goal centers should have non-corridor, non-wall colored pixels (letter ink)."""
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_drag(instance, config)
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        # Start at (0,0) — center pixel should be letter color (not corridor or wall)
        cx, cy = cell_to_pixel_center(0, 0, config)
        pixel = img.getpixel((int(cx), int(cy)))
        assert pixel != corridor_rgb
        assert pixel != wall_rgb
        # Goal at (2,2)
        cx, cy = cell_to_pixel_center(2, 2, config)
        pixel = img.getpixel((int(cx), int(cy)))
        assert pixel != corridor_rgb
        assert pixel != wall_rgb

    def test_border_opening_start_corner(self):
        """Corner start (0,0) should open only on left (most clockwise of top/left)."""
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_drag(instance, config)
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        wt, cw = 4, 20
        # Left border opening — pixel at (0, wt + cw//2)
        pixel = img.getpixel((0, wt + cw // 2))
        assert pixel == corridor_rgb
        # Top border should NOT be open — pixel at (wt + cw//2, 0) should be wall
        pixel = img.getpixel((wt + cw // 2, 0))
        assert pixel == wall_rgb

    def test_border_opening_goal_corner(self):
        """Corner goal (2,2) should open only on bottom (most clockwise of bottom/right)."""
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_drag(instance, config)
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        w, h = img.size
        wt, cw, cs = 4, 20, 24
        # Bottom: pixel at (wt + 2*cs + cw//2, h-1) should be corridor
        pixel = img.getpixel((wt + 2 * cs + cw // 2, h - 1))
        assert pixel == corridor_rgb
        # Right: pixel at (w-1, wt + 2*cs + cw//2) should NOT be open
        pixel = img.getpixel((w - 1, wt + 2 * cs + cw // 2))
        assert pixel == wall_rgb

    def test_border_opening_non_corner_edge(self):
        """Non-corner edge cell should still get its single opening."""
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3, start=(0, 1), goal=(2, 1))
        img = render_vision_drag(instance, config)
        corridor_rgb = (232, 232, 232)
        wt, cw, cs = 4, 20, 24
        # Start (0,1) on top edge — opening at top
        pixel = img.getpixel((wt + 1 * cs + cw // 2, 0))
        assert pixel == corridor_rgb
        # Goal (2,1) on bottom edge — opening at bottom
        h = img.size[1]
        pixel = img.getpixel((wt + 1 * cs + cw // 2, h - 1))
        assert pixel == corridor_rgb

    def test_cell_to_pixel_center_formula(self):
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0)
        cx, cy = cell_to_pixel_center(0, 0, config)
        assert cx == 4 + 20 / 2
        assert cy == 4 + 20 / 2

    def test_cell_to_pixel_rect(self):
        config = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0)
        x0, y0, x1, y1 = cell_to_pixel_rect(1, 2, config)
        cs = 24  # 20 + 4
        assert x0 == 4 + 2 * cs
        assert y0 == 4 + 1 * cs
        assert x1 == x0 + 19
        assert y1 == y0 + 19

    def test_antialias_same_final_size(self):
        instance = _make_simple_instance(3, 4)
        config_aa = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=True)
        config_no = DragRenderConfig(wall_thickness=4, corridor_width=20, margin=0, antialias=False)
        img_aa = render_vision_drag(instance, config_aa)
        img_no = render_vision_drag(instance, config_no)
        assert img_aa.size == img_no.size


# ─── vision_grid ──────────────────────────────────────────────────


class TestVisionGrid:
    def test_image_dimensions(self):
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 4)
        img = render_vision_grid(instance, config)
        expected_w = 4 * 5 + 30 * 4  # wt*(cols+1) + cs*cols
        expected_h = 4 * 4 + 30 * 3
        assert img.size == (expected_w, expected_h)

    def test_cell_color(self):
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        # Cell (1,1) center — should be corridor color
        cs, wt = 30, 4
        x = wt + 1 * (cs + wt) + cs // 2
        y = wt + 1 * (cs + wt) + cs // 2
        pixel = img.getpixel((x, y))
        assert pixel == (232, 232, 232)

    def test_start_cell_has_letter(self):
        """Start cell should be corridor-colored with a letter marker drawn on it."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        cs, wt = 30, 4
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        # Center of start cell (0,0) should have letter pixel (not corridor, not wall)
        x = wt + cs // 2
        y = wt + cs // 2
        pixel = img.getpixel((x, y))
        assert pixel != corridor_rgb
        assert pixel != wall_rgb

    def test_goal_cell_has_letter(self):
        """Goal cell should be corridor-colored with a letter marker drawn on it."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        cs, wt = 30, 4
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        # Center of goal cell (2,2) should have letter pixel
        x = wt + 2 * (cs + wt) + cs // 2
        y = wt + 2 * (cs + wt) + cs // 2
        pixel = img.getpixel((x, y))
        assert pixel != corridor_rgb
        assert pixel != wall_rgb

    def test_start_cell_background_is_corridor(self):
        """Start cell background (off-center) should be corridor color, not start fill."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        corridor_rgb = (232, 232, 232)
        # Sample a corner of start cell (0,0) — should be corridor, not start color
        x = wt = 4
        y = wt
        pixel = img.getpixel((x, y))
        assert pixel == corridor_rgb

    def test_wall_between_unconnected_cells(self):
        """Wall pixel between cells with no passage."""
        passages = [((0, 0), (0, 1))]  # Only one passage
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(2, 2, passages=passages)
        img = render_vision_grid(instance, config)
        # Wall between (0,0) and (1,0) — vertical, between rows
        cs, wt = 30, 4
        x = wt + cs // 2  # middle of col 0
        y = wt + cs  # start of wall gap area between row 0 and row 1
        pixel = img.getpixel((x, y))
        assert pixel == (26, 26, 46)  # wall color

    def test_passage_between_connected_cells(self):
        """Passage pixel between connected cells should be corridor color (off gridline center)."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(2, 2)  # fully connected
        img = render_vision_grid(instance, config)
        # Passage between (0,0) and (0,1) — horizontal
        # Sample slightly off-center to avoid the gridline
        cs, wt = 30, 4
        x = wt + cs  # wall gap area between col 0 and col 1
        y = wt + cs // 2  # middle of row 0
        pixel = img.getpixel((x, y))
        assert pixel == (232, 232, 232)  # corridor color

    def test_gridline_in_passage(self):
        """A thin gridline (wall color) should appear centered in passage gaps."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False, gridline_thickness=1)
        instance = _make_simple_instance(2, 2)  # fully connected
        img = render_vision_grid(instance, config)
        cs, wt = 30, 4
        wall_rgb = (26, 26, 46)
        # Horizontal passage between (0,0) and (0,1): vertical gridline at center of wall gap
        gap_x0 = wt + cs  # start of wall gap
        mid_x = gap_x0 + wt // 2  # center of the 4px wall gap
        y = wt + cs // 2
        pixel = img.getpixel((mid_x, y))
        assert pixel == wall_rgb

    def test_gridline_does_not_fill_passage(self):
        """Gridline should be thin — pixels adjacent to gridline center should be corridor."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False, gridline_thickness=1)
        instance = _make_simple_instance(2, 2)
        img = render_vision_grid(instance, config)
        cs, wt = 30, 4
        corridor_rgb = (232, 232, 232)
        # Pixel to the left of gridline center should be corridor
        gap_x0 = wt + cs
        mid_x = gap_x0 + wt // 2
        y = wt + cs // 2
        pixel = img.getpixel((mid_x - 1, y))
        assert pixel == corridor_rgb

    def test_border_opening_start_corner(self):
        """Corner start (0,0) opens only on left (most clockwise of top/left)."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        cs, wt = 30, 4
        # Left border opening — pixel at (0, wt + cs//2)
        pixel = img.getpixel((0, wt + cs // 2))
        assert pixel == corridor_rgb
        # Top border should NOT be open
        pixel = img.getpixel((wt + cs // 2, 0))
        assert pixel == wall_rgb

    def test_border_opening_goal_corner(self):
        """Corner goal (2,2) opens only on bottom (most clockwise of bottom/right)."""
        config = GridRenderConfig(cell_size=30, wall_thickness=4, margin=0, antialias=False)
        instance = _make_simple_instance(3, 3)
        img = render_vision_grid(instance, config)
        corridor_rgb = (232, 232, 232)
        wall_rgb = (26, 26, 46)
        cs, wt = 30, 4
        w, h = img.size
        cell_x = wt + 2 * (cs + wt) + cs // 2
        cell_y = wt + 2 * (cs + wt) + cs // 2
        # Bottom opening
        pixel = img.getpixel((cell_x, h - 1))
        assert pixel == corridor_rgb
        # Right should NOT be open
        pixel = img.getpixel((w - 1, cell_y))
        assert pixel == wall_rgb

    def test_fallback_color_schema(self):
        """Instance without color_schema in metadata should still render."""
        instance = _make_simple_instance(2, 2)
        del instance["metadata"]["color_schema"]
        config = GridRenderConfig(antialias=False)
        img = render_vision_grid(instance, config)
        assert img.mode == "RGB"
