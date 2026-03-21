"""Tests for difficulty tier configs and sampling."""

import pytest

from mazerunner.generator.difficulty import (
    TIER_PARAMS,
    compute_difficulty_score,
    sample_difficulty_params,
)
from mazerunner.generator.seed_utils import make_rng


class TestSampleDifficultyParams:
    @pytest.mark.parametrize("tier", [1, 2, 3])
    def test_values_within_range(self, tier):
        """Sampled params should be within tier bounds."""
        params = TIER_PARAMS[tier]
        for seed in range(20):
            rng = make_rng(seed)
            config = sample_difficulty_params(tier, rng)
            assert config.tier == tier
            assert params["row_range"][0] <= config.grid_rows <= params["row_range"][1]
            assert params["col_range"][0] <= config.grid_cols <= params["col_range"][1]
            assert config.min_solution_length == params["min_solution_length"]

    def test_invalid_tier(self):
        rng = make_rng(42)
        with pytest.raises(ValueError):
            sample_difficulty_params(0, rng)
        with pytest.raises(ValueError):
            sample_difficulty_params(4, rng)

    def test_deterministic(self):
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        c1 = sample_difficulty_params(2, rng1)
        c2 = sample_difficulty_params(2, rng2)
        assert c1 == c2

    def test_tier1_easy(self):
        rng = make_rng(42)
        config = sample_difficulty_params(1, rng)
        assert 5 <= config.grid_rows <= 8
        assert 7 <= config.grid_cols <= 12
        assert config.min_solution_length == 8

    def test_tier2_medium(self):
        rng = make_rng(42)
        config = sample_difficulty_params(2, rng)
        assert 10 <= config.grid_rows <= 16
        assert 14 <= config.grid_cols <= 22
        assert config.min_solution_length == 20

    def test_tier3_hard(self):
        rng = make_rng(42)
        config = sample_difficulty_params(3, rng)
        assert 18 <= config.grid_rows <= 28
        assert 25 <= config.grid_cols <= 40
        assert config.min_solution_length == 40


class TestComputeDifficultyScore:
    def test_minimum_score(self):
        assert compute_difficulty_score(1, 10, 10) == 1

    def test_maximum_score(self):
        assert compute_difficulty_score(100, 10, 10) == 9

    def test_score_range(self):
        for path_len in range(1, 200):
            score = compute_difficulty_score(path_len, 10, 10)
            assert 1 <= score <= 9

    def test_longer_path_higher_score(self):
        s1 = compute_difficulty_score(5, 10, 10)
        s2 = compute_difficulty_score(50, 10, 10)
        assert s2 >= s1

    def test_larger_grid_lower_score(self):
        """Same path length on larger grid = lower score."""
        s1 = compute_difficulty_score(20, 5, 5)
        s2 = compute_difficulty_score(20, 20, 20)
        assert s1 >= s2
