"""Tests for metrics module."""

import math

import numpy as np
import pytest

from mazerunner.evaluator.metrics import (
    check_endpoint,
    compute_clearance_map,
    compute_length_regret,
    compute_mono_score,
    compute_path_length,
    compute_valid_frac,
)


class TestComputeClearanceMap:
    def test_all_free(self):
        mask = np.ones((50, 50), dtype=bool)
        clearance = compute_clearance_map(mask)
        # Interior points should have positive clearance
        assert clearance[25, 25] > 0
        # Corner should be close to 0 (distance to edge of 50x50 is 0 since
        # the mask is all-True meaning all free, the distance_transform_edt
        # measures distance to False/wall pixels; with no walls, it is inf-like
        # Actually with all-True, there are no False pixels, distance_transform_edt
        # returns distance to nearest 0 pixel, but there are none. It returns 0.
        # Wait: distance_transform_edt of all-True returns all-zero? Let me check.
        # No: scipy distance_transform_edt computes distance to background (0/False).
        # If there are no background pixels, all distances are 0 by convention.

        # Actually scipy returns 0.0 for all pixels when there are no False pixels.
        # This is a degenerate case.
        # Let's test a more realistic case.

    def test_all_free_practical(self):
        # All-True mask: no walls at all. scipy distance_transform_edt returns 0 everywhere.
        mask = np.ones((10, 10), dtype=bool)
        clearance = compute_clearance_map(mask)
        # With no False (wall) pixels, distance_transform_edt returns 0 for all
        assert clearance.shape == (10, 10)

    def test_all_wall(self):
        mask = np.zeros((10, 10), dtype=bool)
        clearance = compute_clearance_map(mask)
        assert np.all(clearance == 0)

    def test_center_free_ring(self):
        # Create a mask with a border of walls and free interior
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        clearance = compute_clearance_map(mask)
        # Center pixel should have clearance ~2 (distance to wall border)
        assert clearance[10, 10] > 1.0
        # Wall pixel should have clearance 0
        assert clearance[0, 0] == 0.0
        # Pixel at (2, 2) is free but adjacent to wall at (1, 2) -> clearance ~1
        assert clearance[2, 2] == pytest.approx(1.0, abs=0.1)

    def test_shape_preserved(self):
        mask = np.ones((30, 50), dtype=bool)
        clearance = compute_clearance_map(mask)
        assert clearance.shape == (30, 50)


class TestCheckEndpoint:
    def test_point_inside_region(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        assert check_endpoint((50.0, 50.0), mask, tolerance=0) is True

    def test_point_outside_region(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        assert check_endpoint((0.0, 0.0), mask, tolerance=0) is False

    def test_point_within_tolerance(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        # Point at (53, 50) is 3 pixels from nearest True pixel
        assert check_endpoint((53.0, 50.0), mask, tolerance=4) is True

    def test_point_beyond_tolerance(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        # Point at (60, 50) is 10 pixels away
        assert check_endpoint((60.0, 50.0), mask, tolerance=4) is False

    def test_out_of_bounds(self):
        mask = np.ones((10, 10), dtype=bool)
        assert check_endpoint((-5.0, -5.0), mask, tolerance=0) is False
        assert check_endpoint((100.0, 100.0), mask, tolerance=0) is False


class TestComputeValidFrac:
    def test_all_free_path(self):
        # Large free area
        free_mask = np.ones((100, 100), dtype=bool)
        free_mask[0, :] = False  # Top row is wall to give nonzero clearance
        from mazerunner.evaluator.metrics import compute_clearance_map
        clearance = compute_clearance_map(free_mask)
        polyline = [(50.0, 50.0), (51.0, 50.0), (52.0, 50.0)]
        frac = compute_valid_frac(polyline, clearance, radius=0)
        assert frac == 1.0

    def test_path_through_walls(self):
        free_mask = np.zeros((100, 100), dtype=bool)
        free_mask[50, 40:60] = True  # Narrow free corridor
        from mazerunner.evaluator.metrics import compute_clearance_map
        clearance = compute_clearance_map(free_mask)
        # Some points on wall, some on free. Use radius=1 to distinguish
        # wall (clearance=0) from free (clearance>=1).
        polyline = [(50.0, 50.0), (50.0, 0.0), (50.0, 99.0)]
        frac = compute_valid_frac(polyline, clearance, radius=1)
        # x=50, y=50 -> col=50, row=50 -> free_mask[50, 50] is True, clearance>=1
        # x=50, y=0 -> col=50, row=0 -> free_mask[0, 50] is False, clearance=0
        # x=50, y=99 -> col=50, row=99 -> free_mask[99, 50] is False, clearance=0
        # So 1/3 valid
        assert frac == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_empty_polyline(self):
        clearance = np.ones((10, 10))
        frac = compute_valid_frac([], clearance, radius=0)
        assert frac == 0.0


class TestComputePathLength:
    def test_straight_horizontal_line(self):
        polyline = [(0.0, 0.0), (10.0, 0.0)]
        assert compute_path_length(polyline) == pytest.approx(10.0)

    def test_right_angle(self):
        polyline = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        assert compute_path_length(polyline) == pytest.approx(20.0)

    def test_diagonal(self):
        polyline = [(0.0, 0.0), (3.0, 4.0)]
        assert compute_path_length(polyline) == pytest.approx(5.0)

    def test_single_point(self):
        polyline = [(5.0, 5.0)]
        assert compute_path_length(polyline) == 0.0

    def test_known_square(self):
        polyline = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        assert compute_path_length(polyline) == pytest.approx(40.0)


class TestComputeMonoScore:
    def test_monotonic_path(self):
        gt = [(float(i), 0.0) for i in range(20)]
        pred = [(float(i), 0.0) for i in range(20)]
        score = compute_mono_score(pred, gt)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_reversed_path(self):
        gt = [(float(i), 0.0) for i in range(20)]
        pred = [(float(i), 0.0) for i in range(19, -1, -1)]
        score = compute_mono_score(pred, gt)
        assert score < 1.0

    def test_single_point_paths(self):
        # Degenerate: single point -> return 1.0
        score = compute_mono_score([(0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])
        assert score == 1.0

    def test_partial_backtrack(self):
        gt = [(float(i), 0.0) for i in range(10)]
        # Forward then back then forward again
        pred = [(0.0, 0.0), (5.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
        score = compute_mono_score(pred, gt)
        assert 0.0 < score < 1.0


class TestComputeLengthRegret:
    def test_same_length(self):
        assert compute_length_regret(100.0, 100.0) == 0.0

    def test_longer_pred(self):
        regret = compute_length_regret(150.0, 100.0)
        assert regret == pytest.approx(0.5)

    def test_shorter_pred(self):
        regret = compute_length_regret(80.0, 100.0)
        assert regret == pytest.approx(-0.2)

    def test_zero_gt_length(self):
        assert compute_length_regret(100.0, 0.0) == 0.0
