"""Tests for canonicalize module."""

import math

import pytest

from mazerunner.evaluator.canonicalize import (
    _parse_delta,
    _parse_polyline,
    canonicalize,
    clamp_points,
    densify,
)


class TestParsePolyline:
    def test_basic_extraction(self):
        data = {"points": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
        result = _parse_polyline(data)
        assert result == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

    def test_single_point(self):
        data = {"points": [[10, 20]]}
        result = _parse_polyline(data)
        assert result == [(10.0, 20.0)]

    def test_missing_points_key_raises(self):
        with pytest.raises(ValueError, match="missing 'points'"):
            _parse_polyline({"wrong_key": []})

    def test_empty_points_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_polyline({"points": []})

    def test_invalid_point_format_raises(self):
        with pytest.raises(ValueError, match="must be"):
            _parse_polyline({"points": [[1, 2, 3]]})

    def test_integer_points_converted_to_float(self):
        data = {"points": [[1, 2], [3, 4]]}
        result = _parse_polyline(data)
        assert all(isinstance(p[0], float) and isinstance(p[1], float) for p in result)


class TestParseDelta:
    def test_cumulative_deltas(self):
        data = {"start": [10.0, 20.0], "deltas": [[5.0, 0.0], [0.0, 3.0], [-2.0, 1.0]]}
        result = _parse_delta(data)
        assert len(result) == 4
        assert result[0] == (10.0, 20.0)
        assert result[1] == (15.0, 20.0)
        assert result[2] == (15.0, 23.0)
        assert result[3] == (13.0, 24.0)

    def test_no_deltas(self):
        data = {"start": [5.0, 5.0], "deltas": []}
        result = _parse_delta(data)
        assert result == [(5.0, 5.0)]

    def test_missing_start_raises(self):
        with pytest.raises(ValueError, match="missing 'start'"):
            _parse_delta({"deltas": []})

    def test_missing_deltas_raises(self):
        with pytest.raises(ValueError, match="missing 'deltas'"):
            _parse_delta({"start": [0, 0]})


class TestClampPoints:
    def test_points_inside_bounds_unchanged(self):
        polyline = [(5.0, 5.0), (50.0, 50.0)]
        result = clamp_points(polyline, 100, 100)
        assert result == [(5.0, 5.0), (50.0, 50.0)]

    def test_points_outside_bounds_clamped(self):
        polyline = [(-10.0, -5.0), (200.0, 300.0)]
        result = clamp_points(polyline, 100, 100)
        assert result[0] == (0.0, 0.0)
        assert result[1] == (99.0, 99.0)  # W-1, H-1

    def test_boundary_points(self):
        polyline = [(0.0, 0.0), (99.0, 99.0)]
        result = clamp_points(polyline, 100, 100)
        assert result == [(0.0, 0.0), (99.0, 99.0)]

    def test_mixed_in_out(self):
        polyline = [(-1.0, 50.0), (50.0, -1.0), (50.0, 50.0), (150.0, 50.0)]
        result = clamp_points(polyline, 100, 100)
        assert result[0] == (0.0, 50.0)
        assert result[1] == (50.0, 0.0)
        assert result[2] == (50.0, 50.0)
        assert result[3] == (99.0, 50.0)


class TestDensify:
    def test_no_gap_exceeds_max(self):
        polyline = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        result = densify(polyline, max_gap=2.0)
        for i in range(1, len(result)):
            dx = result[i][0] - result[i - 1][0]
            dy = result[i][1] - result[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy)
            assert dist <= 2.0 + 1e-9, f"Gap {dist} at index {i}"

    def test_original_points_preserved(self):
        polyline = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        result = densify(polyline, max_gap=2.0)
        # Original points must appear in the result
        for p in polyline:
            assert p in result, f"Original point {p} not in densified result"

    def test_already_dense_unchanged(self):
        polyline = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        result = densify(polyline, max_gap=2.0)
        assert result == polyline

    def test_single_point(self):
        polyline = [(5.0, 5.0)]
        result = densify(polyline, max_gap=2.0)
        assert result == [(5.0, 5.0)]

    def test_densify_diagonal(self):
        polyline = [(0.0, 0.0), (10.0, 10.0)]
        result = densify(polyline, max_gap=1.0)
        # Diagonal distance ~14.14, so should add many intermediate points
        assert len(result) > 10
        for i in range(1, len(result)):
            dx = result[i][0] - result[i - 1][0]
            dy = result[i][1] - result[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy)
            assert dist <= 1.0 + 1e-9


class TestCanonicalize:
    def test_polyline_encoding_full_pipeline(self):
        prediction = {
            "encoding": "polyline",
            "data": {"points": [[0, 0], [50, 0], [50, 50]]},
        }
        image_size = {"w": 100, "h": 100}
        result = canonicalize(prediction, image_size)
        assert len(result) > 3  # densified
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (50.0, 50.0)

    def test_delta_encoding_round_trip(self):
        # Create a path as polyline, convert to delta, canonicalize both
        points = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]
        deltas = []
        for i in range(1, len(points)):
            deltas.append([
                points[i][0] - points[i - 1][0],
                points[i][1] - points[i - 1][1],
            ])

        pred_polyline = {
            "encoding": "polyline",
            "data": {"points": [[p[0], p[1]] for p in points]},
        }
        pred_delta = {
            "encoding": "delta",
            "data": {"start": [10.0, 10.0], "deltas": deltas},
        }
        image_size = {"w": 100, "h": 100}
        result_poly = canonicalize(pred_polyline, image_size)
        result_delta = canonicalize(pred_delta, image_size)
        assert len(result_poly) == len(result_delta)
        for p1, p2 in zip(result_poly, result_delta):
            assert abs(p1[0] - p2[0]) < 1e-9
            assert abs(p1[1] - p2[1]) < 1e-9

    def test_unknown_encoding_raises(self):
        prediction = {"encoding": "unknown", "data": {}}
        with pytest.raises(ValueError, match="Unknown encoding"):
            canonicalize(prediction, {"w": 100, "h": 100})

    def test_missing_encoding_raises(self):
        with pytest.raises(ValueError, match="must have 'encoding'"):
            canonicalize({"data": {}}, {"w": 100, "h": 100})

    def test_missing_data_raises(self):
        with pytest.raises(ValueError, match="must have 'encoding' and 'data'"):
            canonicalize({"encoding": "polyline"}, {"w": 100, "h": 100})
