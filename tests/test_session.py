"""Tests for agentic MazeSession."""

import math

import pytest

from mazerunner.common.rle import encode_rle
from mazerunner.common.types import RenderConfig
from mazerunner.evaluator.evaluate import evaluate_single
from mazerunner.evaluator.session import (
    MazeSession,
    SegmentStatus,
)
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


@pytest.fixture
def gt_data_fixture():
    """Build GT data for a tier-1 maze, same pattern as test_end_to_end.py."""
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
        "id": "test_session_000",
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


class TestSubmitSegment:
    def test_empty_rejection(self, gt_data_fixture):
        gt_data, _, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        result = session.submit_segment([])
        assert result.status == SegmentStatus.REJECTED_EMPTY

    def test_wall_rejection(self, gt_data_fixture):
        gt_data, _, start_center, goal_center = gt_data_fixture
        session = MazeSession(gt_data)
        # Straight line from start to goal will hit walls
        result = session.submit_segment([
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ])
        assert result.status == SegmentStatus.REJECTED_WALL

    def test_gt_path_accepted(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        points = [[p[0], p[1]] for p in solution_polyline]
        result = session.submit_segment(points)
        assert result.status == SegmentStatus.ACCEPTED

    def test_contiguity_check(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        # Submit first segment (first third of GT path)
        third = len(solution_polyline) // 3
        first_part = [[p[0], p[1]] for p in solution_polyline[:third]]
        result1 = session.submit_segment(first_part)
        assert result1.status == SegmentStatus.ACCEPTED
        # Submit second segment starting far away
        result2 = session.submit_segment([[0.0, 0.0], [1.0, 1.0]])
        assert result2.status == SegmentStatus.REJECTED_CONTIGUITY

    def test_start_region_check(self, gt_data_fixture):
        gt_data, _, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        # Point (0, 0) is a wall corner, not in start region
        result = session.submit_segment([[0.0, 0.0], [1.0, 0.0]])
        assert result.status in (
            SegmentStatus.REJECTED_NOT_IN_START,
            SegmentStatus.REJECTED_WALL,
        )

    def test_rejection_does_not_modify_path(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        # Submit first third of GT path
        third = len(solution_polyline) // 3
        first = [[p[0], p[1]] for p in solution_polyline[:third]]
        result1 = session.submit_segment(first)
        assert result1.status == SegmentStatus.ACCEPTED
        points_after_accept = result1.num_points_so_far
        # Reject a bad segment (far away, contiguity failure)
        result2 = session.submit_segment([[0.0, 0.0]])
        assert result2.status != SegmentStatus.ACCEPTED
        assert result2.num_points_so_far == points_after_accept

    def test_stats_tracking(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        # Accept one segment
        points = [[p[0], p[1]] for p in solution_polyline]
        session.submit_segment(points)
        # Reject empty
        session.submit_segment([])
        assert session._stats.total_segments_submitted == 2
        assert session._stats.segments_accepted == 1
        assert session._stats.segments_rejected == 1

    def test_violation_point_reported(self, gt_data_fixture):
        gt_data, _, start_center, goal_center = gt_data_fixture
        session = MazeSession(gt_data)
        result = session.submit_segment([
            [start_center[0], start_center[1]],
            [goal_center[0], goal_center[1]],
        ])
        if result.status == SegmentStatus.REJECTED_WALL:
            assert result.violation_point is not None


class TestFinish:
    def test_gt_path_success(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        points = [[p[0], p[1]] for p in solution_polyline]
        session.submit_segment(points)
        result = session.finish()
        assert result.eval_result is not None
        assert result.eval_result.success["0"] is True
        assert result.finish_reason == "completed"

    def test_empty_path_finish(self, gt_data_fixture):
        gt_data, _, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        result = session.finish()
        assert result.eval_result is None
        assert result.finish_reason == "empty_path"

    def test_partial_path_goal_not_ok(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        mid = len(solution_polyline) // 2
        points = [[p[0], p[1]] for p in solution_polyline[:mid]]
        session.submit_segment(points)
        result = session.finish()
        assert result.eval_result is not None
        assert result.eval_result.goal_ok is False

    def test_finish_twice_raises(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        points = [[p[0], p[1]] for p in solution_polyline]
        session.submit_segment(points)
        session.finish()
        with pytest.raises(RuntimeError):
            session.finish()

    def test_eval_result_matches_evaluate_single(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        points = [[p[0], p[1]] for p in solution_polyline]
        session.submit_segment(points)
        session_result = session.finish()
        # Compare with direct evaluate_single
        prediction = {
            "encoding": "polyline",
            "data": {"points": [[p[0], p[1]] for p in solution_polyline]},
        }
        direct_result = evaluate_single(prediction, gt_data)
        assert session_result.eval_result.success == direct_result.success
        assert session_result.eval_result.start_ok == direct_result.start_ok
        assert session_result.eval_result.goal_ok == direct_result.goal_ok


class TestMultiSegment:
    def test_gt_path_three_segments(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        n = len(solution_polyline)
        third = n // 3
        # Overlap at boundaries so contiguity check passes
        seg1 = [[p[0], p[1]] for p in solution_polyline[:third]]
        seg2 = [[p[0], p[1]] for p in solution_polyline[third - 1 : 2 * third]]
        seg3 = [[p[0], p[1]] for p in solution_polyline[2 * third - 1 :]]

        r1 = session.submit_segment(seg1)
        assert r1.status == SegmentStatus.ACCEPTED
        r2 = session.submit_segment(seg2)
        assert r2.status == SegmentStatus.ACCEPTED
        r3 = session.submit_segment(seg3)
        assert r3.status == SegmentStatus.ACCEPTED

        result = session.finish()
        assert result.eval_result is not None
        assert result.eval_result.success["0"] is True

    def test_reject_then_correct(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)
        n = len(solution_polyline)
        mid = n // 2

        # Submit first half
        seg1 = [[p[0], p[1]] for p in solution_polyline[:mid]]
        r1 = session.submit_segment(seg1)
        assert r1.status == SegmentStatus.ACCEPTED

        # Submit bad segment (far away — contiguity failure)
        r2 = session.submit_segment([[0.0, 0.0], [1.0, 1.0]])
        assert r2.status != SegmentStatus.ACCEPTED

        # Submit correct continuation (overlap at boundary)
        seg3 = [[p[0], p[1]] for p in solution_polyline[mid - 1 :]]
        r3 = session.submit_segment(seg3)
        assert r3.status == SegmentStatus.ACCEPTED

    def test_many_small_segments(self, gt_data_fixture):
        gt_data, solution_polyline, _, _ = gt_data_fixture
        session = MazeSession(gt_data)

        # Submit first pair of points
        first_seg = [
            [solution_polyline[0][0], solution_polyline[0][1]],
            [solution_polyline[1][0], solution_polyline[1][1]],
        ]
        r = session.submit_segment(first_seg)
        assert r.status == SegmentStatus.ACCEPTED

        # Submit remaining points as pairs overlapping at boundary
        for i in range(2, len(solution_polyline)):
            pts = [
                [solution_polyline[i - 1][0], solution_polyline[i - 1][1]],
                [solution_polyline[i][0], solution_polyline[i][1]],
            ]
            result = session.submit_segment(pts)
            assert result.status == SegmentStatus.ACCEPTED, (
                f"Segment {i} rejected: {result.reason}"
            )

        result = session.finish()
        assert result.eval_result is not None
