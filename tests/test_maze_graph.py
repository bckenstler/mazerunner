"""Tests for maze generation and solving."""

from collections import deque

import numpy as np
import pytest

from mazerunner.generator.maze_graph import (
    build_maze,
    choose_start_goal,
    generate_maze_dfs,
    solve_bfs,
    bfs_distances,
    _get_neighbors,
)
from mazerunner.generator.seed_utils import make_rng


class TestGenerateMazeDFS:
    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20), (3, 7)])
    def test_perfect_maze_passage_count(self, rows, cols):
        rng = make_rng(42)
        passages = generate_maze_dfs(rows, cols, rng)
        # A perfect maze (spanning tree) has exactly rows*cols - 1 edges
        assert len(passages) == rows * cols - 1

    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20)])
    def test_all_cells_reachable(self, rows, cols):
        rng = make_rng(99)
        passages = generate_maze_dfs(rows, cols, rng)

        # BFS from (0,0) should reach all cells
        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))

        while queue:
            current = queue.popleft()
            for neighbor in _get_neighbors(current[0], current[1], rows, cols):
                if neighbor not in visited and frozenset({current, neighbor}) in passages:
                    visited.add(neighbor)
                    queue.append(neighbor)

        assert len(visited) == rows * cols

    def test_passages_connect_adjacent_cells(self):
        rng = make_rng(42)
        rows, cols = 8, 8
        passages = generate_maze_dfs(rows, cols, rng)

        for passage in passages:
            cells = list(passage)
            assert len(cells) == 2
            r1, c1 = cells[0]
            r2, c2 = cells[1]
            # Must be orthogonal neighbors
            assert abs(r1 - r2) + abs(c1 - c2) == 1
            # Must be within bounds
            assert 0 <= r1 < rows and 0 <= c1 < cols
            assert 0 <= r2 < rows and 0 <= c2 < cols


class TestSolveBFS:
    def test_path_starts_at_start_ends_at_goal(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        path = solve_bfs(rows, cols, passages, start, goal)
        assert path[0] == start
        assert path[-1] == goal

    def test_path_follows_passages(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        start = (0, 0)
        goal = (9, 9)
        path = solve_bfs(rows, cols, passages, start, goal)

        for i in range(len(path) - 1):
            edge = frozenset({path[i], path[i + 1]})
            assert edge in passages, f"Step {i}: {path[i]} -> {path[i+1]} not in passages"

    def test_path_has_no_cycles(self):
        rng = make_rng(42)
        rows, cols = 8, 8
        passages = generate_maze_dfs(rows, cols, rng)
        start = (0, 0)
        goal = (7, 7)
        path = solve_bfs(rows, cols, passages, start, goal)
        # No repeated cells
        assert len(path) == len(set(path))

    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20)])
    def test_solve_various_sizes(self, rows, cols):
        rng = make_rng(77)
        passages = generate_maze_dfs(rows, cols, rng)
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        path = solve_bfs(rows, cols, passages, start, goal)
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal


class TestChooseStartGoal:
    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20)])
    def test_within_grid_bounds(self, rows, cols):
        rng = make_rng(42)
        passages = generate_maze_dfs(rows, cols, rng)
        start, goal = choose_start_goal(rows, cols, passages, 3, rng)
        assert 0 <= start[0] < rows and 0 <= start[1] < cols
        assert 0 <= goal[0] < rows and 0 <= goal[1] < cols

    def test_sufficient_distance(self):
        rng = make_rng(42)
        rows, cols = 15, 15
        passages = generate_maze_dfs(rows, cols, rng)
        min_dist = 10
        rng2 = make_rng(42)
        start, goal = choose_start_goal(rows, cols, passages, min_dist, rng2)
        distances = bfs_distances(rows, cols, passages, start)
        assert distances[goal] >= min_dist or distances[goal] == max(distances.values())

    def test_start_and_goal_are_different(self):
        rng = make_rng(42)
        rows, cols = 10, 10
        passages = generate_maze_dfs(rows, cols, rng)
        start, goal = choose_start_goal(rows, cols, passages, 5, rng)
        assert start != goal


class TestBuildMaze:
    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20)])
    def test_returns_valid_maze_grid(self, rows, cols):
        rng = make_rng(42)
        maze = build_maze(rows, cols, 3, rng)
        assert maze.rows == rows
        assert maze.cols == cols
        assert len(maze.passages) == rows * cols - 1
        assert maze.start != maze.goal
        assert maze.solution_path[0] == maze.start
        assert maze.solution_path[-1] == maze.goal

    def test_solution_path_follows_passages(self):
        rng = make_rng(42)
        maze = build_maze(10, 10, 5, rng)
        for i in range(len(maze.solution_path) - 1):
            edge = frozenset({maze.solution_path[i], maze.solution_path[i + 1]})
            assert edge in maze.passages
