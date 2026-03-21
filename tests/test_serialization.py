"""Tests for serialization."""

import json

import pytest

from mazerunner.common.types import MazeGrid
from mazerunner.generator.maze_graph import generate_maze, solve_bfs
from mazerunner.generator.seed_utils import make_rng
from mazerunner.generator.serialization import (
    compute_branching_factor,
    instance_to_dict,
    instance_to_json,
    maze_grid_to_adjacency,
    maze_grid_to_instance,
)


def _make_test_grid(rows=8, cols=10, seed=42):
    rng = make_rng(seed)
    passages = generate_maze(rows, cols, rng)
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    solution = solve_bfs(passages, start, goal, rows, cols)
    return MazeGrid(
        rows=rows, cols=cols, passages=passages,
        start=start, goal=goal, solution_path=solution,
        endpoint_type="edge-edge",
    )


class TestMazeGridToAdjacency:
    def test_symmetric(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        for cell, neighbors in adj.items():
            for n in neighbors:
                assert cell in adj[n], f"{cell} -> {n} but not {n} -> {cell}"

    def test_all_cells_present(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        assert len(adj) == grid.rows * grid.cols

    def test_sorted_adjacency_lists(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        for neighbors in adj.values():
            assert neighbors == sorted(neighbors)

    def test_sorted_keys(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        keys = list(adj.keys())
        parsed = [(int(k.split(",")[0]), int(k.split(",")[1])) for k in keys]
        assert parsed == sorted(parsed)

    def test_edge_count_matches_passages(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        total_edges = sum(len(n) for n in adj.values()) // 2
        assert total_edges == len(grid.passages)


class TestComputeBranchingFactor:
    def test_perfect_maze_branching(self):
        grid = _make_test_grid()
        adj = maze_grid_to_adjacency(grid)
        bf = compute_branching_factor(adj)
        # Perfect maze: edges = rows*cols - 1, avg degree = 2*(rows*cols-1)/(rows*cols)
        expected = 2 * (grid.rows * grid.cols - 1) / (grid.rows * grid.cols)
        assert abs(bf - expected) < 0.01

    def test_empty_adjacency(self):
        assert compute_branching_factor({}) == 0.0


class TestMazeGridToInstance:
    def test_instance_fields(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        assert instance.id == "test_001"
        assert instance.grid_rows == grid.rows
        assert instance.grid_cols == grid.cols
        assert instance.start == grid.start
        assert instance.goal == grid.goal
        assert len(instance.shortest_path_cells) == len(grid.solution_path)

    def test_metadata_keys(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        meta = instance.metadata
        assert "endpoint_type" in meta
        assert "difficulty_score" in meta
        assert "path_length" in meta
        assert "branching_factor" in meta

    def test_difficulty_score_in_range(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        assert 1 <= instance.metadata["difficulty_score"] <= 9


class TestInstanceToJson:
    def test_valid_json(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        json_str = instance_to_json(instance)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_json_schema(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        data = json.loads(instance_to_json(instance))
        assert "id" in data
        assert "grid_rows" in data
        assert "grid_cols" in data
        assert "start" in data
        assert "goal" in data
        assert "adjacency" in data
        assert "shortest_path_cells" in data
        assert "metadata" in data

    def test_start_goal_format(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        data = json.loads(instance_to_json(instance))
        assert "," in data["start"]
        assert "," in data["goal"]

    def test_roundtrip_dict(self):
        grid = _make_test_grid()
        instance = maze_grid_to_instance(grid, "test_001")
        d = instance_to_dict(instance)
        assert d["id"] == "test_001"
        assert d["grid_rows"] == grid.rows
        assert isinstance(d["adjacency"], dict)
        assert isinstance(d["shortest_path_cells"], list)
