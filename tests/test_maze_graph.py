"""Tests for maze generation and BFS solving."""

import numpy as np
import pytest

from mazerunner.generator.maze_graph import generate_maze, solve_bfs, bfs_distances
from mazerunner.generator.seed_utils import make_rng


GRID_SIZES = [(5, 7), (8, 10), (10, 14), (15, 20), (5, 5)]


class TestGenerateMaze:
    @pytest.mark.parametrize("rows,cols", GRID_SIZES)
    def test_passage_count_invariant(self, rows, cols):
        """Perfect maze has exactly rows*cols - 1 passages."""
        rng = make_rng(42)
        passages = generate_maze(rows, cols, rng)
        assert len(passages) == rows * cols - 1

    @pytest.mark.parametrize("rows,cols", GRID_SIZES)
    def test_all_cells_reachable(self, rows, cols):
        """Every cell should be reachable from (0,0)."""
        rng = make_rng(42)
        passages = generate_maze(rows, cols, rng)
        distances = bfs_distances(passages, (0, 0), rows, cols)
        assert len(distances) == rows * cols

    def test_passages_are_frozensets(self):
        rng = make_rng(42)
        passages = generate_maze(5, 7, rng)
        for p in passages:
            assert isinstance(p, frozenset)
            assert len(p) == 2

    def test_passages_connect_adjacent_cells(self):
        """Each passage should connect cells that differ by 1 in exactly one dimension."""
        rng = make_rng(42)
        passages = generate_maze(5, 7, rng)
        for p in passages:
            cells = list(p)
            a, b = cells[0], cells[1]
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

    def test_deterministic(self):
        """Same seed produces same maze."""
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        p1 = generate_maze(10, 10, rng1)
        p2 = generate_maze(10, 10, rng2)
        assert p1 == p2

    def test_different_seeds_different_mazes(self):
        rng1 = make_rng(42)
        rng2 = make_rng(99)
        p1 = generate_maze(10, 10, rng1)
        p2 = generate_maze(10, 10, rng2)
        assert p1 != p2

    def test_small_maze(self):
        """1x1 maze has no passages."""
        rng = make_rng(42)
        passages = generate_maze(1, 1, rng)
        assert len(passages) == 0

    def test_2x2_maze(self):
        rng = make_rng(42)
        passages = generate_maze(2, 2, rng)
        assert len(passages) == 3


class TestSolveBfs:
    @pytest.mark.parametrize("rows,cols", GRID_SIZES)
    def test_solution_exists(self, rows, cols):
        """Any two cells in a perfect maze should be connected."""
        rng = make_rng(42)
        passages = generate_maze(rows, cols, rng)
        path = solve_bfs(passages, (0, 0), (rows - 1, cols - 1), rows, cols)
        assert len(path) >= 2
        assert path[0] == (0, 0)
        assert path[-1] == (rows - 1, cols - 1)

    def test_path_is_valid(self):
        """Each consecutive pair in the path should be connected by a passage."""
        rng = make_rng(42)
        rows, cols = 10, 14
        passages = generate_maze(rows, cols, rng)
        path = solve_bfs(passages, (0, 0), (rows - 1, cols - 1), rows, cols)
        for i in range(len(path) - 1):
            edge = frozenset((path[i], path[i + 1]))
            assert edge in passages

    def test_path_no_duplicates(self):
        rng = make_rng(42)
        rows, cols = 10, 14
        passages = generate_maze(rows, cols, rng)
        path = solve_bfs(passages, (0, 0), (rows - 1, cols - 1), rows, cols)
        assert len(path) == len(set(path))

    def test_same_start_goal(self):
        rng = make_rng(42)
        passages = generate_maze(5, 5, rng)
        path = solve_bfs(passages, (0, 0), (0, 0), 5, 5)
        assert path == [(0, 0)]


class TestBfsDistances:
    def test_start_distance_zero(self):
        rng = make_rng(42)
        passages = generate_maze(5, 7, rng)
        distances = bfs_distances(passages, (0, 0), 5, 7)
        assert distances[(0, 0)] == 0

    def test_all_distances_non_negative(self):
        rng = make_rng(42)
        passages = generate_maze(5, 7, rng)
        distances = bfs_distances(passages, (0, 0), 5, 7)
        assert all(d >= 0 for d in distances.values())

    def test_distances_match_path_length(self):
        rng = make_rng(42)
        rows, cols = 8, 10
        passages = generate_maze(rows, cols, rng)
        distances = bfs_distances(passages, (0, 0), rows, cols)
        goal = (rows - 1, cols - 1)
        path = solve_bfs(passages, (0, 0), goal, rows, cols)
        assert distances[goal] == len(path) - 1
