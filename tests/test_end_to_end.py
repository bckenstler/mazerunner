"""End-to-end tests for the maze generation pipeline."""

import json
import os
import tempfile

import pytest

from mazerunner.common.types import MazeGrid
from mazerunner.generate import generate_dataset
from mazerunner.generator.difficulty import sample_difficulty_params
from mazerunner.generator.maze_graph import generate_maze, solve_bfs
from mazerunner.generator.placement import ENDPOINT_TYPES, place_endpoints
from mazerunner.generator.seed_utils import derive_seed, make_rng
from mazerunner.generator.serialization import maze_grid_to_instance, instance_to_json


class TestFullPipeline:
    def test_deterministic_pipeline(self):
        """Same seed produces identical maze instances."""
        results = []
        for _ in range(2):
            seed = derive_seed(42, 0)
            rng = make_rng(seed)
            config = sample_difficulty_params(1, rng)
            passages = generate_maze(config.grid_rows, config.grid_cols, rng)
            start, goal = place_endpoints(
                passages, config.grid_rows, config.grid_cols, "edge-edge", config.min_solution_length, rng
            )
            solution = solve_bfs(passages, start, goal, config.grid_rows, config.grid_cols)
            grid = MazeGrid(
                rows=config.grid_rows, cols=config.grid_cols, passages=passages,
                start=start, goal=goal, solution_path=solution, endpoint_type="edge-edge",
            )
            instance = maze_grid_to_instance(grid, "test")
            results.append(instance_to_json(instance))
        assert results[0] == results[1]

    def test_solution_valid(self):
        """Generated solution path is valid."""
        seed = derive_seed(42, 5)
        rng = make_rng(seed)
        config = sample_difficulty_params(2, rng)
        passages = generate_maze(config.grid_rows, config.grid_cols, rng)
        endpoint_type = "edge-edge"
        start, goal = place_endpoints(
            passages, config.grid_rows, config.grid_cols,
            endpoint_type, config.min_solution_length, rng,
        )
        solution = solve_bfs(passages, start, goal, config.grid_rows, config.grid_cols)

        assert solution[0] == start
        assert solution[-1] == goal
        for i in range(len(solution) - 1):
            assert frozenset((solution[i], solution[i + 1])) in passages

    @pytest.mark.parametrize("tier", [1, 2, 3])
    def test_all_tiers_solvable(self, tier):
        """Mazes from all tiers are solvable."""
        rng = make_rng(42)
        config = sample_difficulty_params(tier, rng)
        passages = generate_maze(config.grid_rows, config.grid_cols, rng)
        start, goal = place_endpoints(
            passages, config.grid_rows, config.grid_cols,
            "edge-edge", config.min_solution_length, rng,
        )
        solution = solve_bfs(passages, start, goal, config.grid_rows, config.grid_cols)
        assert len(solution) >= 2

    @pytest.mark.parametrize("endpoint_type", ENDPOINT_TYPES)
    def test_all_endpoint_types_solvable(self, endpoint_type):
        rng = make_rng(42)
        config = sample_difficulty_params(2, rng)
        passages = generate_maze(config.grid_rows, config.grid_cols, rng)
        start, goal = place_endpoints(
            passages, config.grid_rows, config.grid_cols,
            endpoint_type, config.min_solution_length, rng,
        )
        solution = solve_bfs(passages, start, goal, config.grid_rows, config.grid_cols)
        assert len(solution) >= 2


class TestFileIO:
    def test_generate_dataset(self):
        """Full dataset generation writes correct files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir, num_mazes=10, master_seed=42, tier_distribution=[3, 4, 3])
            instances_dir = os.path.join(tmpdir, "instances")
            assert os.path.isdir(instances_dir)

            files = sorted(os.listdir(instances_dir))
            assert len(files) == 10

            # Verify each file
            tiers_seen = set()
            for fname in files:
                assert fname.endswith(".json")
                with open(os.path.join(instances_dir, fname)) as f:
                    data = json.load(f)
                assert "id" in data
                assert "adjacency" in data
                assert "shortest_path_cells" in data
                assert "start" in data
                assert "goal" in data
                assert "metadata" in data
                assert "color_schema" in data["metadata"]
                assert "name" in data["metadata"]["color_schema"]
                tiers_seen.add(data["metadata"]["tier"])

            assert tiers_seen == {1, 2, 3}

    def test_dataset_determinism(self):
        """Same params produce identical datasets."""
        jsons = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                generate_dataset(tmpdir, num_mazes=5, master_seed=42, tier_distribution=[1, 2, 2])
                instances_dir = os.path.join(tmpdir, "instances")
                files = sorted(os.listdir(instances_dir))
                run_jsons = []
                for fname in files:
                    with open(os.path.join(instances_dir, fname)) as f:
                        run_jsons.append(f.read())
                jsons.append(run_jsons)
        assert jsons[0] == jsons[1]

    def test_file_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir, num_mazes=3, master_seed=42, tier_distribution=[1, 1, 1])
            instances_dir = os.path.join(tmpdir, "instances")
            files = sorted(os.listdir(instances_dir))
            assert files == ["maze_000000.json", "maze_000001.json", "maze_000002.json"]
