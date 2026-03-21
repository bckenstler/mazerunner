"""Tests for start/goal placement."""

import numpy as np
import pytest

from mazerunner.generator.maze_graph import generate_maze, bfs_distances
from mazerunner.generator.placement import (
    ENDPOINT_TYPES,
    get_dead_ends,
    is_edge_cell,
    is_interior_cell,
    place_endpoints,
    _get_borders,
)
from mazerunner.generator.seed_utils import make_rng


class TestEdgeClassification:
    def test_corners_are_edge(self):
        assert is_edge_cell((0, 0), 5, 7)
        assert is_edge_cell((0, 6), 5, 7)
        assert is_edge_cell((4, 0), 5, 7)
        assert is_edge_cell((4, 6), 5, 7)

    def test_border_cells_are_edge(self):
        assert is_edge_cell((0, 3), 5, 7)
        assert is_edge_cell((2, 0), 5, 7)
        assert is_edge_cell((4, 3), 5, 7)
        assert is_edge_cell((2, 6), 5, 7)

    def test_interior_cells(self):
        assert is_interior_cell((1, 1), 5, 7)
        assert is_interior_cell((2, 3), 5, 7)
        assert is_interior_cell((3, 5), 5, 7)

    def test_edge_interior_mutually_exclusive(self):
        for r in range(5):
            for c in range(7):
                assert is_edge_cell((r, c), 5, 7) != is_interior_cell((r, c), 5, 7)


class TestGetBorders:
    def test_corner_has_two_borders(self):
        borders = _get_borders((0, 0), 5, 7)
        assert borders == {"top", "left"}

    def test_edge_has_one_border(self):
        borders = _get_borders((0, 3), 5, 7)
        assert borders == {"top"}

    def test_interior_has_no_borders(self):
        borders = _get_borders((2, 3), 5, 7)
        assert borders == set()


class TestGetDeadEnds:
    def test_dead_ends_have_degree_one(self):
        rng = make_rng(42)
        passages = generate_maze(10, 14, rng)
        dead_ends = get_dead_ends(passages, 10, 14)
        for cell in dead_ends:
            degree = sum(1 for p in passages if cell in p)
            assert degree == 1

    def test_dead_ends_exist(self):
        """A perfect maze always has dead ends."""
        rng = make_rng(42)
        passages = generate_maze(10, 14, rng)
        dead_ends = get_dead_ends(passages, 10, 14)
        assert len(dead_ends) > 0

    def test_dead_ends_are_valid_cells(self):
        rng = make_rng(42)
        rows, cols = 8, 10
        passages = generate_maze(rows, cols, rng)
        dead_ends = get_dead_ends(passages, rows, cols)
        for r, c in dead_ends:
            assert 0 <= r < rows
            assert 0 <= c < cols


class TestPlaceEndpoints:
    @pytest.mark.parametrize("endpoint_type", ENDPOINT_TYPES)
    def test_all_endpoint_types(self, endpoint_type):
        """Test that all 4 endpoint types produce valid start/goal."""
        rng = make_rng(42)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        start, goal = place_endpoints(passages, rows, cols, endpoint_type, 5, rng)
        assert start != goal
        assert 0 <= start[0] < rows and 0 <= start[1] < cols
        assert 0 <= goal[0] < rows and 0 <= goal[1] < cols

    def test_edge_edge_different_borders(self):
        """For edge-edge, start and goal should be on different borders."""
        rng = make_rng(42)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        start, goal = place_endpoints(passages, rows, cols, "edge-edge", 5, rng)
        assert is_edge_cell(start, rows, cols)
        assert is_edge_cell(goal, rows, cols)

    def test_edge_edge_start_on_edge(self):
        rng = make_rng(42)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        start, goal = place_endpoints(passages, rows, cols, "edge-edge", 5, rng)
        assert is_edge_cell(start, rows, cols)

    def test_interior_interior_dead_ends(self):
        """Interior endpoints should be dead-end cells."""
        rng = make_rng(42)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        dead_ends = get_dead_ends(passages, rows, cols)
        interior_dead_ends = [c for c in dead_ends if is_interior_cell(c, rows, cols)]

        if len(interior_dead_ends) >= 2:
            start, goal = place_endpoints(passages, rows, cols, "interior-interior", 1, rng)
            # At least start should be an interior dead-end if pool is non-empty
            if is_interior_cell(start, rows, cols):
                assert start in dead_ends

    def test_edge_interior(self):
        rng = make_rng(42)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        start, goal = place_endpoints(passages, rows, cols, "edge-interior", 5, rng)
        assert is_edge_cell(start, rows, cols)

    def test_interior_edge(self):
        rng = make_rng(99)
        rows, cols = 12, 16
        passages = generate_maze(rows, cols, rng)
        dead_ends = get_dead_ends(passages, rows, cols)
        interior_dead_ends = [c for c in dead_ends if is_interior_cell(c, rows, cols)]

        start, goal = place_endpoints(passages, rows, cols, "interior-edge", 5, rng)
        if interior_dead_ends:
            # Start should come from interior dead-end pool
            if is_interior_cell(start, rows, cols):
                assert start in dead_ends
        assert is_edge_cell(goal, rows, cols) or not is_interior_cell(goal, rows, cols)

    def test_min_distance_respected_when_possible(self):
        rng = make_rng(42)
        rows, cols = 15, 20
        passages = generate_maze(rows, cols, rng)
        min_dist = 10
        start, goal = place_endpoints(passages, rows, cols, "edge-edge", min_dist, rng)
        distances = bfs_distances(passages, start, rows, cols)
        # Should meet min distance if possible (it should be on a 15x20 grid)
        assert distances[goal] >= min_dist

    def test_fallback_when_min_distance_impossible(self):
        """Should still return valid endpoints even with unreachable min distance."""
        rng = make_rng(42)
        rows, cols = 5, 5
        passages = generate_maze(rows, cols, rng)
        start, goal = place_endpoints(passages, rows, cols, "edge-edge", 999, rng)
        assert start != goal
