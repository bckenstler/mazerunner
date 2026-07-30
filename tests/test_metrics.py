"""Route progress and failure taxonomy."""

from __future__ import annotations

import math

import pytest

from mazerunner.geometry import cumulative_lengths, project_onto_polyline
from mazerunner.metrics import derive, failure_category, route_progress


def _task(points, width=101, height=101):
    """Task whose reference path is `points`, given in normalized coords."""
    return {
        "width": width,
        "height": height,
        "reference": {"optimal_path": [{"x": x, "y": y} for x, y in points]},
    }


STRAIGHT = _task([(0.0, 0.0), (1.0, 0.0)])  # 100px horizontal run


def _eval(**kwargs):
    base = {
        "success": False,
        "schema_valid": True,
        "starts_correctly": True,
        "ends_correctly": False,
        "collision_free": True,
        "first_collision": None,
    }
    base.update(kwargs)
    return base


# ---------- projection primitive ----------

def test_cumulative_lengths():
    assert cumulative_lengths([(0, 0), (3, 0), (3, 4)]) == [0.0, 3.0, 7.0]


def test_projection_on_straight_line():
    arclength, perpendicular, segment = project_onto_polyline((5.0, 2.0), [(0, 0), (10, 0)])
    assert arclength == pytest.approx(5.0)
    assert perpendicular == pytest.approx(2.0)
    assert segment == 0


def test_projection_clamps_past_the_end():
    arclength, _, _ = project_onto_polyline((50.0, 0.0), [(0, 0), (10, 0)])
    assert arclength == pytest.approx(10.0)


def test_projection_clamps_before_the_start():
    arclength, perpendicular, _ = project_onto_polyline((-7.0, 0.0), [(0, 0), (10, 0)])
    assert arclength == pytest.approx(0.0)
    assert perpendicular == pytest.approx(7.0)


def test_projection_prefers_earlier_pass_on_a_doubling_route():
    """A hairpin passes the same spot twice; progress takes the earlier one."""
    route = [(0, 0), (10, 0), (10, 2), (0, 2)]
    arclength, _, _ = project_onto_polyline((5.0, 1.0), route)
    assert arclength == pytest.approx(5.0)


def test_projection_still_finds_a_clearly_closer_later_segment():
    route = [(0, 0), (10, 0), (10, 50), (0, 50)]
    arclength, perpendicular, _ = project_onto_polyline((5.0, 50.0), route)
    assert perpendicular == pytest.approx(0.0)
    assert arclength > 50.0


# ---------- route progress ----------

def test_pass_is_full_credit():
    assert route_progress(STRAIGHT, _eval(success=True)) == 1.0


def test_no_tool_call_is_zero():
    assert route_progress(STRAIGHT, None) == 0.0


def test_invalid_schema_is_zero():
    assert route_progress(STRAIGHT, _eval(schema_valid=False)) == 0.0


def test_wrong_start_is_zero():
    """No partial credit for a route never entered."""
    assert route_progress(STRAIGHT, _eval(starts_correctly=False)) == 0.0


def test_midroute_collision_scores_halfway():
    evaluation = _eval(
        collision_free=False, first_collision={"x_px": 50.0, "y_px": 0.0, "segment_index": 1}
    )
    assert route_progress(STRAIGHT, evaluation) == pytest.approx(0.5, abs=0.02)


def test_early_collision_scores_low():
    evaluation = _eval(
        collision_free=False, first_collision={"x_px": 10.0, "y_px": 0.0, "segment_index": 0}
    )
    assert route_progress(STRAIGHT, evaluation) == pytest.approx(0.1, abs=0.02)


def test_stopped_short_uses_the_final_submitted_point():
    """The frozen plan defined only the collision case; this is the amendment."""
    submission = {"points": [{"x": 0.0, "y": 0.0}, {"x": 0.7, "y": 0.0}]}
    progress = route_progress(STRAIGHT, _eval(), submission)
    assert progress == pytest.approx(0.7, abs=0.02)


def test_stopped_short_without_a_submission_is_zero():
    assert route_progress(STRAIGHT, _eval(), None) == 0.0


def test_progress_is_bounded():
    """A path wandering past the goal cannot score above 1.0."""
    submission = {"points": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0}]}
    assert route_progress(STRAIGHT, _eval(), submission) == 1.0


def test_progress_on_an_L_shaped_route():
    task = _task([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])  # 100px right, then 100px down
    evaluation = _eval(
        collision_free=False, first_collision={"x_px": 100.0, "y_px": 50.0, "segment_index": 1}
    )
    assert route_progress(task, evaluation) == pytest.approx(0.75, abs=0.02)


def test_degenerate_reference_does_not_divide_by_zero():
    task = _task([(0.5, 0.5), (0.5, 0.5)])
    assert route_progress(task, _eval()) == 0.0


# ---------- failure taxonomy ----------

@pytest.mark.parametrize(
    "evaluation,expected",
    [
        (None, "no_tool_call"),
        ({"success": True}, "pass"),
        ({"success": False, "schema_valid": False}, "schema_invalid"),
        ({"success": False, "schema_valid": True, "starts_correctly": False}, "wrong_start"),
        (
            {
                "success": False,
                "schema_valid": True,
                "starts_correctly": True,
                "collision_free": False,
            },
            "collision",
        ),
        (
            {
                "success": False,
                "schema_valid": True,
                "starts_correctly": True,
                "collision_free": True,
                "ends_correctly": False,
            },
            "stopped_short",
        ),
        (
            {
                "success": False,
                "schema_valid": True,
                "starts_correctly": True,
                "collision_free": True,
                "ends_correctly": True,
            },
            "other",
        ),
    ],
)
def test_failure_categories(evaluation, expected):
    assert failure_category(evaluation) == expected


def test_derive_returns_both_metrics():
    row = {"evaluation": _eval(success=True), "submission": None}
    out = derive(STRAIGHT, row)
    assert out == {"route_progress": 1.0, "failure_category": "pass"}


def test_derive_is_idempotent_on_stored_rows():
    """Back-filling an already-derived row must not change it."""
    row = {
        "evaluation": _eval(
            collision_free=False, first_collision={"x_px": 50.0, "y_px": 0.0, "segment_index": 1}
        ),
        "submission": {"points": [{"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}]},
    }
    first = derive(STRAIGHT, row)
    second = derive(STRAIGHT, {**row, "derived": first})
    assert first == second
