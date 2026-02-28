"""Tests for placement module."""

import numpy as np
import pytest

from mazerunner.common.types import RenderConfig
from mazerunner.generator.maze_graph import generate_maze_dfs
from mazerunner.generator.placement import (
    PLACEMENT_STYLES,
    _edge_cells,
    _find_dead_ends,
    _is_interior,
    choose_placement_style,
    choose_start_goal_placed,
    opening_center,
    opening_pixel_rect,
)
from mazerunner.generator.seed_utils import make_rng


class TestChoosePlacementStyle:
    def test_returns_valid_style(self):
        rng = make_rng(42)
        for _ in range(50):
            style = choose_placement_style(rng)
            assert style in PLACEMENT_STYLES

    def test_all_styles_reachable(self):
        """Over many draws, all 3 styles should appear."""
        styles_seen = set()
        rng = make_rng(0)
        for _ in range(100):
            styles_seen.add(choose_placement_style(rng))
        assert styles_seen == set(PLACEMENT_STYLES)


class TestEdgeCells:
    @pytest.mark.parametrize("rows,cols", [(5, 5), (3, 7), (10, 10)])
    def test_edge_cell_counts(self, rows, cols):
        edges = _edge_cells(rows, cols)
        assert len(edges["top"]) == cols
        assert len(edges["bottom"]) == cols
        assert len(edges["left"]) == rows
        assert len(edges["right"]) == rows

    def test_top_cells_are_row_zero(self):
        edges = _edge_cells(5, 7)
        for r, c in edges["top"]:
            assert r == 0

    def test_bottom_cells_are_last_row(self):
        edges = _edge_cells(5, 7)
        for r, c in edges["bottom"]:
            assert r == 4

    def test_left_cells_are_col_zero(self):
        edges = _edge_cells(5, 7)
        for r, c in edges["left"]:
            assert c == 0

    def test_right_cells_are_last_col(self):
        edges = _edge_cells(5, 7)
        for r, c in edges["right"]:
            assert c == 6


class TestFindDeadEnds:
    def test_dead_ends_have_one_passage(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        dead_ends = _find_dead_ends(rows, cols, passages)
        for cell in dead_ends:
            count = sum(1 for p in passages if cell in p)
            assert count == 1

    def test_dead_ends_exist_in_perfect_maze(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        dead_ends = _find_dead_ends(rows, cols, passages)
        assert len(dead_ends) > 0


class TestChooseStartGoalPlaced:
    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (8, 12)])
    def test_returns_valid_tuple(self, rows, cols):
        rng = make_rng(42)
        passages = generate_maze_dfs(rows, cols, rng)
        start, goal, style, start_edge, goal_edge = choose_start_goal_placed(
            rows, cols, passages, 3, rng
        )
        assert 0 <= start[0] < rows and 0 <= start[1] < cols
        assert 0 <= goal[0] < rows and 0 <= goal[1] < cols
        assert start != goal
        assert style in PLACEMENT_STYLES
        assert start_edge in ("top", "bottom", "left", "right")
        assert goal_edge in ("top", "bottom", "left", "right", "")

    def test_edge_to_edge_different_edges(self):
        """For edge_to_edge, start and goal should be on different edges."""
        rng = make_rng(42)
        rows, cols = 10, 10
        # Run many times to find an edge_to_edge case
        for seed in range(200):
            rng = make_rng(seed)
            passages = generate_maze_dfs(rows, cols, rng)
            start, goal, style, start_edge, goal_edge = choose_start_goal_placed(
                rows, cols, passages, 3, rng
            )
            if style == "edge_to_edge":
                assert start_edge != goal_edge
                assert goal_edge != ""
                return
        pytest.skip("No edge_to_edge style found in 200 tries")

    def test_edge_to_center_goal_is_interior_dead_end(self):
        """For edge_to_center, goal should be an interior dead-end cell."""
        rows, cols = 10, 10
        for seed in range(200):
            rng = make_rng(seed)
            passages = generate_maze_dfs(rows, cols, rng)
            start, goal, style, start_edge, goal_edge = choose_start_goal_placed(
                rows, cols, passages, 3, rng
            )
            if style == "edge_to_center":
                assert goal_edge == ""
                assert _is_interior(goal, rows, cols), f"goal {goal} is on perimeter"
                return
        pytest.skip("No edge_to_center style found in 200 tries")

    def test_edge_to_dead_end_goal_is_interior(self):
        """For edge_to_dead_end, goal should be interior."""
        rows, cols = 10, 10
        for seed in range(200):
            rng = make_rng(seed)
            passages = generate_maze_dfs(rows, cols, rng)
            start, goal, style, start_edge, goal_edge = choose_start_goal_placed(
                rows, cols, passages, 3, rng
            )
            if style == "edge_to_dead_end":
                assert goal_edge == ""
                assert _is_interior(goal, rows, cols), f"goal {goal} is on perimeter"
                return
        pytest.skip("No edge_to_dead_end style found in 200 tries")

    def test_start_is_on_start_edge(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        start, goal, style, start_edge, goal_edge = choose_start_goal_placed(
            rows, cols, passages, 3, rng
        )
        r, c = start
        if start_edge == "top":
            assert r == 0
        elif start_edge == "bottom":
            assert r == rows - 1
        elif start_edge == "left":
            assert c == 0
        elif start_edge == "right":
            assert c == cols - 1


class TestOpeningPixelRect:
    @pytest.fixture
    def config(self):
        return RenderConfig(
            image_width=100,
            image_height=100,
            corridor_width=10,
            wall_thickness=2,
            chrome_height_top=0,
            chrome_width_left=0,
            theme_name="light_classic",
        )

    def test_top_opening(self, config):
        y_min, y_max, x_min, x_max = opening_pixel_rect((0, 3), "top", config, 8, 8)
        assert y_min == 0  # starts at chrome top
        assert y_max == 2  # wall thickness
        assert x_max - x_min == 10  # corridor width

    def test_bottom_opening(self, config):
        y_min, y_max, x_min, x_max = opening_pixel_rect((7, 3), "bottom", config, 8, 8)
        cell_size = 12  # cw + wt
        expected_y_min = 2 + 7 * cell_size + 10  # origin + row*cell_size + cw
        assert y_min == expected_y_min
        assert y_max - y_min == 2  # wall thickness

    def test_left_opening(self, config):
        y_min, y_max, x_min, x_max = opening_pixel_rect((3, 0), "left", config, 8, 8)
        assert x_min == 0
        assert x_max == 2
        assert y_max - y_min == 10  # corridor width

    def test_right_opening(self, config):
        y_min, y_max, x_min, x_max = opening_pixel_rect((3, 7), "right", config, 8, 8)
        cell_size = 12
        expected_x_min = 2 + 7 * cell_size + 10
        assert x_min == expected_x_min
        assert x_max - x_min == 2

    def test_invalid_edge_raises(self, config):
        with pytest.raises(ValueError):
            opening_pixel_rect((0, 0), "invalid", config, 8, 8)


class TestOpeningCenter:
    def test_center_is_midpoint(self):
        config = RenderConfig(
            image_width=100,
            image_height=100,
            corridor_width=10,
            wall_thickness=2,
            chrome_height_top=0,
            chrome_width_left=0,
            theme_name="light_classic",
        )
        y_min, y_max, x_min, x_max = opening_pixel_rect((0, 3), "top", config, 8, 8)
        cx, cy = opening_center((0, 3), "top", config, 8, 8)
        assert cx == (x_min + x_max) / 2.0
        assert cy == (y_min + y_max) / 2.0
