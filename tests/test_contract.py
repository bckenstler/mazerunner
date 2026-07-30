import math

from mazerunner.contract import MAX_POINTS, validate_submission


def ok_points(n):
    return {"points": [{"x": 0.1 + 0.001 * i, "y": 0.5} for i in range(n)]}


def test_valid_submission():
    points, error = validate_submission(ok_points(3))
    assert error is None
    assert len(points) == 3
    assert points[0] == (0.1, 0.5)


def test_rejects_non_object():
    assert validate_submission(None)[1] is not None
    assert validate_submission([1, 2])[1] is not None


def test_rejects_missing_points():
    assert validate_submission({})[1] == "missing required 'points' array"


def test_rejects_too_few_points():
    points, error = validate_submission(ok_points(1))
    assert points is None
    assert "too few" in error


def test_rejects_too_many_points():
    points, error = validate_submission(ok_points(MAX_POINTS + 1))
    assert points is None
    assert "too many" in error


def test_rejects_nan_and_inf():
    for bad in (math.nan, math.inf, -math.inf):
        args = {"points": [{"x": bad, "y": 0.5}, {"x": 0.2, "y": 0.5}]}
        points, error = validate_submission(args)
        assert points is None
        assert "finite" in error


def test_rejects_out_of_range():
    args = {"points": [{"x": -0.01, "y": 0.5}, {"x": 0.2, "y": 0.5}]}
    assert validate_submission(args)[0] is None
    args = {"points": [{"x": 0.5, "y": 1.01}, {"x": 0.2, "y": 0.5}]}
    assert validate_submission(args)[0] is None


def test_rejects_missing_coordinate():
    args = {"points": [{"x": 0.1}, {"x": 0.2, "y": 0.5}]}
    points, error = validate_submission(args)
    assert points is None
    assert "missing 'y'" in error


def test_rejects_non_numeric_coordinate():
    args = {"points": [{"x": "0.1", "y": 0.5}, {"x": 0.2, "y": 0.5}]}
    assert validate_submission(args)[0] is None
    args = {"points": [{"x": True, "y": 0.5}, {"x": 0.2, "y": 0.5}]}
    assert validate_submission(args)[0] is None
