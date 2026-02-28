#!/usr/bin/env python
"""Validate the MazeRunner agentic evaluator with synthetic scenarios."""

import math
import sys

from mazerunner.common.rle import encode_rle
from mazerunner.common.types import RenderConfig
from mazerunner.evaluator.session import MazeSession, SegmentStatus
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
        maze.solution_path,
        render_config,
        maze.rows,
        maze.cols,
        start_edge=maze.start_edge,
        goal_edge=maze.goal_edge,
    )

    solution_length = 0.0
    for i in range(1, len(solution_polyline)):
        x0, y0 = solution_polyline[i - 1]
        x1, y1 = solution_polyline[i]
        solution_length += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    gt_data = {
        "id": "validate_agentic_000",
        "image_size": {"w": image_width, "h": image_height},
        "regions": {
            "free_space_mask_rle": encode_rle(free_mask),
            "wall_mask_rle": encode_rle(wall_mask),
            "start_mask_rle": encode_rle(start_mask),
            "goal_mask_rle": encode_rle(goal_mask),
        },
        "gt": {
            "solution_polyline": [
                [float(x), float(y)] for x, y in solution_polyline
            ],
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
    print("MazeRunner Agentic Evaluator Validation")
    print("=" * 60)

    gt_data, solution_polyline, start_center, goal_center = build_gt_data()
    gt_points = gt_data["gt"]["solution_polyline"]
    results = []

    # ---- Scenario 1: GT path as single segment ----
    def test_gt_single_segment():
        session = MazeSession(gt_data)
        r = session.submit_segment(gt_points)
        assert r.status == SegmentStatus.ACCEPTED, (
            f"Expected ACCEPTED, got {r.status}"
        )
        result = session.finish()
        assert result.eval_result is not None
        assert result.eval_result.success["0"] is True, (
            f"success@0={result.eval_result.success['0']}"
        )
        assert result.finish_reason == "completed"

    results.append(run_test("GT path as single segment", test_gt_single_segment))

    # ---- Scenario 2: GT path in 3 segments (overlapping at boundaries) ----
    def test_gt_three_segments():
        session = MazeSession(gt_data)
        n = len(gt_points)
        third = n // 3
        # Overlap at boundaries for contiguity
        r1 = session.submit_segment(gt_points[:third])
        assert r1.status == SegmentStatus.ACCEPTED, f"Seg 1: {r1.status}"
        r2 = session.submit_segment(gt_points[third - 1 : 2 * third])
        assert r2.status == SegmentStatus.ACCEPTED, f"Seg 2: {r2.status}"
        r3 = session.submit_segment(gt_points[2 * third - 1 :])
        assert r3.status == SegmentStatus.ACCEPTED, f"Seg 3: {r3.status}"

        result = session.finish()
        assert result.eval_result is not None
        assert result.eval_result.success["0"] is True
        assert result.stats.segments_accepted == 3

    results.append(run_test("GT path in 3 segments", test_gt_three_segments))

    # ---- Scenario 3: Straight-line wall crossing ----
    def test_wall_crossing():
        session = MazeSession(gt_data)
        r = session.submit_segment([
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ])
        assert r.status == SegmentStatus.REJECTED_WALL, (
            f"Expected REJECTED_WALL, got {r.status}"
        )
        assert r.violation_point is not None

    results.append(
        run_test("Straight-line wall crossing rejected", test_wall_crossing)
    )

    # ---- Scenario 4: Contiguity gap ----
    def test_contiguity_gap():
        session = MazeSession(gt_data)
        mid = len(gt_points) // 2
        r1 = session.submit_segment(gt_points[:mid])
        assert r1.status == SegmentStatus.ACCEPTED
        # Jump to a far-away point
        r2 = session.submit_segment([[0.0, 0.0], [1.0, 1.0]])
        assert r2.status == SegmentStatus.REJECTED_CONTIGUITY

    results.append(run_test("Contiguity gap rejected", test_contiguity_gap))

    # ---- Scenario 5: Correction after rejection ----
    def test_correction_after_rejection():
        session = MazeSession(gt_data)
        mid = len(gt_points) // 2
        r1 = session.submit_segment(gt_points[:mid])
        assert r1.status == SegmentStatus.ACCEPTED

        # Submit bad segment
        r2 = session.submit_segment([[0.0, 0.0]])
        assert r2.status != SegmentStatus.ACCEPTED

        # Re-submit correct continuation (overlapping at boundary)
        r3 = session.submit_segment(gt_points[mid - 1 :])
        assert r3.status == SegmentStatus.ACCEPTED

        result = session.finish()
        assert result.eval_result is not None
        assert result.stats.segments_accepted == 2
        assert result.stats.segments_rejected == 1

    results.append(
        run_test("Correction after rejection", test_correction_after_rejection)
    )

    # ---- Scenario 6: Empty finish ----
    def test_empty_finish():
        session = MazeSession(gt_data)
        result = session.finish()
        assert result.eval_result is None
        assert result.finish_reason == "empty_path"
        assert len(result.accepted_path) == 0

    results.append(run_test("Empty finish", test_empty_finish))

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
