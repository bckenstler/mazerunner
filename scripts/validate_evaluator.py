#!/usr/bin/env python
"""Validate the MazeRunner evaluator with synthetic test cases."""

import math
import sys

import numpy as np

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


def build_gt_data():
    """Generate a tier-1 maze with fixed seed and build full GT data dict."""
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
        "id": "validate_000",
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
    return gt_data, maze, solution_polyline, start_center, goal_center, render_config


def run_test(name, fn):
    """Run a test function, print PASS/FAIL."""
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"        {e}")
        return False


def main():
    print("=" * 60)
    print("MazeRunner Evaluator Validation")
    print("=" * 60)

    gt_data, maze, solution_polyline, start_center, goal_center, render_config = build_gt_data()
    gt_points = gt_data["gt"]["solution_polyline"]
    results = []

    # ---- Test 1: GT solution polyline ----
    def test_gt_solution():
        prediction = {
            "encoding": "polyline",
            "data": {"points": gt_points},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok, f"start_ok={result.start_ok}"
        assert result.goal_ok, f"goal_ok={result.goal_ok}"
        assert result.valid_frac["0"] >= 0.99, f"valid_frac@0={result.valid_frac['0']}"
        assert result.success["0"], f"success@0={result.success['0']}"

    results.append(run_test("GT solution polyline", test_gt_solution))

    # ---- Test 2: GT + 2px noise ----
    def test_gt_with_noise():
        rng = np.random.default_rng(123)
        noisy_points = []
        for pt in gt_points:
            noisy_points.append([
                pt[0] + rng.uniform(-1.0, 1.0),
                pt[1] + rng.uniform(-1.0, 1.0),
            ])
        prediction = {
            "encoding": "polyline",
            "data": {"points": noisy_points},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok, f"start_ok={result.start_ok}"
        assert result.goal_ok, f"goal_ok={result.goal_ok}"
        assert result.success["0"], f"success@0={result.success['0']}"

    results.append(run_test("GT + 2px noise", test_gt_with_noise))

    # ---- Test 3: Reversed path ----
    def test_reversed_path():
        reversed_points = list(reversed(gt_points))
        prediction = {
            "encoding": "polyline",
            "data": {"points": reversed_points},
        }
        result = evaluate_single(prediction, gt_data)
        # Endpoints are swapped: start of pred is at goal region, end at start region
        assert result.start_ok is False, f"start_ok={result.start_ok} (expected False)"
        assert result.goal_ok is False, f"goal_ok={result.goal_ok} (expected False)"
        # The path itself is still valid (same corridor)
        assert result.valid_frac["0"] >= 0.95, f"valid_frac@0={result.valid_frac['0']}"

    results.append(run_test("Reversed path", test_reversed_path))

    # ---- Test 4: Straight line start -> goal ----
    def test_straight_line():
        straight_points = [
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ]
        prediction = {
            "encoding": "polyline",
            "data": {"points": straight_points},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok, f"start_ok={result.start_ok}"
        assert result.goal_ok, f"goal_ok={result.goal_ok}"
        # Use radius=1 to check wall penetration (radius=0 means clearance>=0, always true)
        assert result.valid_frac["1"] < 1.0, f"valid_frac@1={result.valid_frac['1']}"
        assert result.success["1"] is False, f"success@1={result.success['1']}"

    results.append(run_test("Straight line start->goal", test_straight_line))

    # ---- Test 5: First half only ----
    def test_first_half():
        half_idx = len(gt_points) // 2
        half_points = gt_points[:half_idx]
        if len(half_points) < 2:
            half_points = gt_points[:2]
        prediction = {
            "encoding": "polyline",
            "data": {"points": half_points},
        }
        result = evaluate_single(prediction, gt_data)
        assert result.start_ok, f"start_ok={result.start_ok}"
        assert result.goal_ok is False, f"goal_ok={result.goal_ok} (expected False)"
        assert result.valid_frac["0"] >= 0.99, f"valid_frac@0={result.valid_frac['0']}"

    results.append(run_test("First half only", test_first_half))

    # ---- Test 6: Delta-encoded GT ----
    def test_delta_encoded():
        deltas = []
        for i in range(1, len(gt_points)):
            deltas.append([
                gt_points[i][0] - gt_points[i - 1][0],
                gt_points[i][1] - gt_points[i - 1][1],
            ])
        prediction = {
            "encoding": "delta",
            "data": {
                "start": gt_points[0],
                "deltas": deltas,
            },
        }
        result = evaluate_single(prediction, gt_data)
        assert result.success["0"], f"success@0={result.success['0']}"

    results.append(run_test("Delta-encoded GT", test_delta_encoded))

    # ---- Test 7: Cell-route encoded GT ----
    def test_cell_route_encoded():
        cells = [[int(cell[0]), int(cell[1])] for cell in maze.solution_path]
        prediction = {
            "encoding": "cell_route",
            "data": {"cells": cells},
        }
        result = evaluate_single(prediction, gt_data)
        # Cell-route only covers cell centers, not opening centers, so
        # endpoints may not land in start/goal regions. Check valid_frac instead.
        assert result.valid_frac["0"] >= 0.99, f"valid_frac@0={result.valid_frac['0']}"

    results.append(run_test("Cell-route encoded GT", test_cell_route_encoded))

    # ---- Test 8: Empty path ----
    def test_empty_path():
        prediction = {
            "encoding": "polyline",
            "data": {"points": []},
        }
        raised = False
        try:
            evaluate_single(prediction, gt_data)
        except (ValueError, Exception):
            raised = True
        assert raised, "Expected ValueError for empty path"

    results.append(run_test("Empty path raises error", test_empty_path))

    # ---- Summary ----
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)

    if passed < total:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
