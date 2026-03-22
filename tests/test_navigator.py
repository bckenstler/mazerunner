"""Tests for the navigator module."""

import pytest
from PIL import Image

from mazerunner.navigator import DragNavigator, GridNavigator, InteractionResult
from mazerunner.renderer.base import DragRenderConfig, GridRenderConfig
from mazerunner.renderer.vision_drag import cell_to_pixel_center, cell_to_pixel_rect


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


# ─── GridNavigator ───────────────────────────────────────────────


class TestGridNavigator:
    def test_initial_position_is_start(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        assert nav.position == (0, 0)

    def test_initial_not_finished(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        assert nav.finished is False

    def test_single_valid_step(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        result = nav.interact("R")
        assert result.valid is True
        assert result.steps_applied == 1
        assert nav.position == (0, 1)

    def test_multi_step_sequence(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        result = nav.interact("RRDD")
        assert result.valid is True
        assert result.steps_applied == 4
        assert nav.position == (2, 2)

    def test_reaches_goal_sets_finished(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        nav.interact("RRDD")
        assert nav.finished is True

    def test_wall_hit_rejects_entire_action(self):
        # No passages — every move hits a wall
        instance = _make_simple_instance(3, 3, passages=[])
        nav = GridNavigator(instance)
        result = nav.interact("R")
        assert result.valid is False
        assert result.steps_applied == 0
        assert nav.position == (0, 0)

    def test_out_of_bounds_rejects(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        result = nav.interact("U")
        assert result.valid is False
        assert result.steps_applied == 0
        assert nav.position == (0, 0)

    def test_later_invalid_step_rejects_all(self):
        # RRR: third R goes out of bounds (col 3 in a 3-col grid)
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        result = nav.interact("RRR")
        assert result.valid is False
        assert result.steps_applied == 0
        assert nav.position == (0, 0)

    def test_invalid_action_no_movement(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        nav.interact("R")  # valid, now at (0,1)
        result = nav.interact("RRR")  # goes out of bounds
        assert result.valid is False
        assert nav.position == (0, 1)  # unchanged

    def test_history_records_all(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        nav.interact("R")
        nav.interact("D")
        nav.interact("L")
        assert len(nav.history) == 3

    def test_history_includes_invalid(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance)
        nav.interact("U")  # invalid (out of bounds from start)
        assert len(nav.history) == 1
        assert nav.history[0].result.valid is False

    def test_render_text_shows_x(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance, render_mode="text_grid")
        nav.interact("R")
        text = nav.render()
        assert " X " in text

    def test_render_x_overlays_start(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance, render_mode="text_grid")
        text = nav.render()
        # X should appear at start, S should not
        lines = text.split("\n")
        start_line = lines[1]  # row 0 cell line
        assert " X " in start_line
        assert " S " not in start_line

    def test_render_s_reappears(self):
        instance = _make_simple_instance(3, 3)
        nav = GridNavigator(instance, render_mode="text_grid")
        nav.interact("R")
        text = nav.render()
        assert " S " in text  # S reappears at (0,0)
        assert " X " in text  # X at (0,1)

    def test_render_vision_grid_returns_image(self):
        instance = _make_simple_instance(3, 3)
        config = GridRenderConfig(antialias=False)
        nav = GridNavigator(instance, render_mode="vision_grid", config=config)
        img = nav.render()
        assert isinstance(img, Image.Image)


# ─── DragNavigator ───────────────────────────────────────────────


class TestDragNavigator:
    @pytest.fixture
    def config(self):
        return DragRenderConfig(
            wall_thickness=4, corridor_width=20, margin=0, antialias=False
        )

    @pytest.fixture
    def instance(self):
        return _make_simple_instance(3, 3)

    def test_initial_position_is_start_center(self, instance, config):
        nav = DragNavigator(instance, config=config)
        expected = cell_to_pixel_center(0, 0, config)
        assert nav.position == expected

    def test_initial_not_finished(self, instance, config):
        nav = DragNavigator(instance, config=config)
        assert nav.finished is False

    def test_first_drag_must_start_in_start_cell(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        # Drag within start cell corridor
        result = nav.interact([[cx, cy], [cx + 2, cy]])
        assert result.valid is True

    def test_first_drag_outside_start_rejected(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(1, 1, config)
        result = nav.interact([[cx, cy], [cx + 2, cy]])
        assert result.valid is False
        # Position unchanged
        expected = cell_to_pixel_center(0, 0, config)
        assert nav.position == expected

    def test_subsequent_drag_must_continue(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        nav.interact([[cx, cy], [cx + 5, cy]])
        # Second drag starts at previous endpoint
        result = nav.interact([[cx + 5, cy], [cx + 8, cy]])
        assert result.valid is True

    def test_discontinuous_drag_rejected(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        nav.interact([[cx, cy], [cx + 5, cy]])
        # Second drag starts far from previous endpoint
        result = nav.interact([[cx + 50, cy], [cx + 55, cy]])
        assert result.valid is False

    def test_valid_corridor_path(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        # Move right within the corridor
        result = nav.interact([[cx, cy], [cx + 5, cy]])
        assert result.valid is True
        assert nav.position == (cx + 5, cy)

    def test_wall_crossing_rejects_entire_drag(self, config):
        # Maze with no passages — wall between every cell
        instance = _make_simple_instance(3, 3, passages=[])
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        # Try to move far right (into wall)
        cs = config.cell_size
        result = nav.interact([[cx, cy], [cx + cs, cy]])
        assert result.valid is False
        assert nav.position == (cx, cy)

    def test_reaching_goal_sets_finished(self, config):
        # Small 1x2 maze for easy navigation
        instance = _make_simple_instance(1, 2, start=(0, 0), goal=(0, 1))
        nav = DragNavigator(instance, config=config)
        start_cx, start_cy = cell_to_pixel_center(0, 0, config)
        goal_cx, goal_cy = cell_to_pixel_center(0, 1, config)
        result = nav.interact([[start_cx, start_cy], [goal_cx, goal_cy]])
        assert result.valid is True
        assert nav.finished is True

    def test_path_accumulates(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        nav.interact([[cx, cy], [cx + 3, cy]])
        nav.interact([[cx + 3, cy], [cx + 6, cy]])
        # Path should have: initial + 2 from first drag + 2 from second
        assert len(nav._path) == 3  # initial + 1 from each drag (start points deduped)

    def test_history_records_all(self, instance, config):
        nav = DragNavigator(instance, config=config)
        cx, cy = cell_to_pixel_center(0, 0, config)
        nav.interact([[cx, cy], [cx + 3, cy]])
        nav.interact([[cx + 3, cy], [cx + 6, cy]])
        assert len(nav.history) == 2

    def test_render_returns_image(self, instance, config):
        nav = DragNavigator(instance, config=config)
        img = nav.render()
        assert isinstance(img, Image.Image)

    def test_collision_mask_wall_impassable(self, config):
        instance = _make_simple_instance(3, 3, passages=[])
        nav = DragNavigator(instance, config=config)
        mask = nav.mask
        # A wall pixel: between cell (0,0) and (0,1) — at x=wt+cw, y inside corridor
        wt, cw = config.wall_thickness, config.corridor_width
        x = wt + cw  # first wall pixel after cell (0,0) corridor
        y = wt + cw // 2  # middle of corridor row
        assert mask[y, x] is False or mask[y, x] == False

    def test_collision_mask_corridor_passable(self, instance, config):
        nav = DragNavigator(instance, config=config)
        mask = nav.mask
        # Cell (1,1) center should be passable
        cx, cy = cell_to_pixel_center(1, 1, config)
        assert mask[int(cy), int(cx)] == True
