"""End-to-end tests: generate maze, evaluate GT solution, evaluate bad path."""

import math

import numpy as np
import pytest

from mazerunner.common.rle import encode_rle
from mazerunner.common.types import RenderConfig
from mazerunner.evaluator.evaluate import evaluate_single
from mazerunner.generator.difficulty import sample_difficulty_params
from mazerunner.generator.masks import (
    carve_outer_openings,
    compute_cell_center,
    compute_image_size,
    generate_free_space_mask,
    generate_region_mask,
    generate_wall_mask,
    solution_cells_to_polyline,
)
from mazerunner.generator.maze_graph import build_maze
from mazerunner.generator.placement import opening_center
from mazerunner.generator.seed_utils import derive_seed, make_rng


def _build_gt_data():
    """Build a full GT data dict for tier-1 maze with seed=42."""
    seed = derive_seed(42, 0)
    rng = make_rng(seed)
    diff = sample_difficulty_params(1, rng)

    chrome_height_top = 0
    chrome_width_left = 0
    image_width, image_height, render_config = compute_image_size(
        diff, chrome_height_top, chrome_width_left
    )
    render_config = RenderConfig(
        image_width=image_width,
        image_height=image_height,
        corridor_width=render_config.corridor_width,
        wall_thickness=render_config.wall_thickness,
        chrome_height_top=chrome_height_top,
        chrome_width_left=chrome_width_left,
        theme_name="light_classic",
    )

    maze = build_maze(diff.grid_rows, diff.grid_cols, diff.min_solution_length, rng)
    wall_mask = generate_wall_mask(maze, render_config)
    carve_outer_openings(wall_mask, maze, render_config)
    free_mask = generate_free_space_mask(wall_mask)

    if maze.start_edge:
        start_center = opening_center(
            maze.start, maze.start_edge, render_config, maze.rows, maze.cols
        )
    else:
        start_center = compute_cell_center(
            maze.start[0], maze.start[1], render_config, maze.rows, maze.cols
        )
    if maze.goal_edge:
        goal_center = opening_center(
            maze.goal, maze.goal_edge, render_config, maze.rows, maze.cols
        )
    else:
        goal_center = compute_cell_center(
            maze.goal[0], maze.goal[1], render_config, maze.rows, maze.cols
        )

    start_mask = generate_region_mask(
        start_center, render_config.corridor_width * 0.4, (image_height, image_width)
    )
    goal_mask = generate_region_mask(
        goal_center, render_config.corridor_width * 0.4, (image_height, image_width)
    )

    solution_polyline = solution_cells_to_polyline(
        maze.solution_path, render_config, maze.rows, maze.cols,
        start_edge=maze.start_edge, goal_edge=maze.goal_edge,
    )

    solution_length = 0.0
    for i in range(1, len(solution_polyline)):
        x0, y0 = solution_polyline[i - 1]
        x1, y1 = solution_polyline[i]
        solution_length += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    gt_data = {
        "id": "test_000",
        "image_size": {"w": image_width, "h": image_height},
        "regions": {
            "free_space_mask_rle": encode_rle(free_mask),
            "wall_mask_rle": encode_rle(wall_mask),
            "start_mask_rle": encode_rle(start_mask),
            "goal_mask_rle": encode_rle(goal_mask),
        },
        "gt": {
            "solution_polyline": [[float(x), float(y)] for x, y in solution_polyline],
            "solution_length": solution_length,
        },
        "render_config": {
            "corridor_width": render_config.corridor_width,
            "wall_thickness": render_config.wall_thickness,
            "chrome_height_top": render_config.chrome_height_top,
            "chrome_width_left": render_config.chrome_width_left,
            "theme_name": render_config.theme_name,
        },
    }
    return gt_data, solution_polyline, start_center, goal_center


class TestEndToEndGTSolution:
    def test_gt_polyline_evaluates_as_success(self):
        gt_data, solution_polyline, start_center, goal_center = _build_gt_data()
        prediction = {
            "encoding": "polyline",
            "data": {"points": gt_data["gt"]["solution_polyline"]},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok is True
        assert result.goal_ok is True
        assert result.success["0"] is True
        assert result.valid_frac["0"] == pytest.approx(1.0, abs=0.01)

    def test_gt_polyline_mono_score_near_one(self):
        gt_data, solution_polyline, _, _ = _build_gt_data()
        prediction = {
            "encoding": "polyline",
            "data": {"points": gt_data["gt"]["solution_polyline"]},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.mono_score >= 0.99

    def test_gt_polyline_length_regret_near_zero(self):
        gt_data, solution_polyline, _, _ = _build_gt_data()
        prediction = {
            "encoding": "polyline",
            "data": {"points": gt_data["gt"]["solution_polyline"]},
        }
        result = evaluate_single(prediction, gt_data)
        assert abs(result.length_regret) < 0.05


class TestEndToEndBadPath:
    def test_straight_line_fails_at_r1(self):
        gt_data, solution_polyline, start_center, goal_center = _build_gt_data()
        # Straight line from start center to goal center (cuts through walls).
        # success@0 uses radius=0 (clearance >= 0, always true), so we check
        # success@1 which requires clearance >= 1 (free space only).
        straight_line = [
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ]
        prediction = {
            "encoding": "polyline",
            "data": {"points": straight_line},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.success["1"] is False

    def test_straight_line_endpoints_ok(self):
        gt_data, solution_polyline, start_center, goal_center = _build_gt_data()
        straight_line = [
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ]
        prediction = {
            "encoding": "polyline",
            "data": {"points": straight_line},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok is True
        assert result.goal_ok is True

    def test_straight_line_valid_frac_below_one_at_r1(self):
        gt_data, solution_polyline, start_center, goal_center = _build_gt_data()
        # At radius=1, wall pixels (clearance=0) fail, so valid_frac@1 < 1.0
        straight_line = [
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ]
        prediction = {
            "encoding": "polyline",
            "data": {"points": straight_line},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.valid_frac["1"] < 1.0
