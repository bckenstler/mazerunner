"""Tests for seed determinism and reproducibility."""

import numpy as np
import pytest

from mazerunner.generator.seed_utils import derive_seed, make_rng
from mazerunner.generator.maze_graph import build_maze


class TestDeriveSeed:
    def test_same_inputs_same_output(self):
        s1 = derive_seed(42, 0)
        s2 = derive_seed(42, 0)
        assert s1 == s2

    def test_different_index_different_output(self):
        s1 = derive_seed(42, 0)
        s2 = derive_seed(42, 1)
        assert s1 != s2

    def test_different_master_seed_different_output(self):
        s1 = derive_seed(42, 0)
        s2 = derive_seed(99, 0)
        assert s1 != s2

    def test_output_is_positive_integer(self):
        for master in [0, 1, 42, 999999]:
            for idx in [0, 1, 100]:
                s = derive_seed(master, idx)
                assert isinstance(s, int)
                assert s >= 0

    def test_derives_distinct_seeds_across_range(self):
        seeds = [derive_seed(42, i) for i in range(100)]
        # All should be unique (extremely unlikely collision with SHA-256)
        assert len(set(seeds)) == 100


class TestMakeRNG:
    def test_same_seed_same_sequence(self):
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 == vals2

    def test_different_seed_different_sequence(self):
        rng1 = make_rng(42)
        rng2 = make_rng(99)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 != vals2


class TestMazeReproducibility:
    def test_same_seed_identical_maze(self):
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        maze1 = build_maze(10, 10, 5, rng1)
        maze2 = build_maze(10, 10, 5, rng2)
        assert maze1.rows == maze2.rows
        assert maze1.cols == maze2.cols
        assert maze1.passages == maze2.passages
        assert maze1.start == maze2.start
        assert maze1.goal == maze2.goal
        assert maze1.solution_path == maze2.solution_path

    def test_different_seed_different_maze(self):
        rng1 = make_rng(42)
        rng2 = make_rng(99)
        maze1 = build_maze(10, 10, 5, rng1)
        maze2 = build_maze(10, 10, 5, rng2)
        # Very unlikely to be identical
        assert maze1.passages != maze2.passages or maze1.start != maze2.start

    def test_full_pipeline_determinism(self):
        """Ensure the full derive_seed -> make_rng -> build_maze pipeline is deterministic."""
        from mazerunner.generator.difficulty import sample_difficulty_params

        for run in range(2):
            seed = derive_seed(42, 7)
            rng = make_rng(seed)
            diff = sample_difficulty_params(1, rng)
            maze = build_maze(diff.grid_rows, diff.grid_cols, diff.min_solution_length, rng)
            if run == 0:
                first_maze = maze
            else:
                assert maze.passages == first_maze.passages
                assert maze.start == first_maze.start
                assert maze.goal == first_maze.goal
                assert maze.solution_path == first_maze.solution_path
