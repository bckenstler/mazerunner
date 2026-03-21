"""Tests for seed derivation and RNG factory."""

import numpy as np
import pytest

from mazerunner.generator.seed_utils import derive_seed, make_rng


class TestDeriveSeed:
    def test_deterministic(self):
        """Same input always produces same output."""
        assert derive_seed(42, 0) == derive_seed(42, 0)
        assert derive_seed(42, 99) == derive_seed(42, 99)

    def test_different_indices_different_seeds(self):
        """Different indices produce different seeds."""
        seeds = [derive_seed(42, i) for i in range(100)]
        assert len(set(seeds)) == 100

    def test_different_master_seeds(self):
        """Different master seeds produce different results."""
        assert derive_seed(42, 0) != derive_seed(43, 0)

    def test_returns_int(self):
        seed = derive_seed(42, 0)
        assert isinstance(seed, int)

    def test_fits_in_4_bytes(self):
        """Seed should be <= 2^32 - 1."""
        for i in range(50):
            seed = derive_seed(42, i)
            assert 0 <= seed <= 0xFFFFFFFF

    def test_large_master_seed(self):
        seed = derive_seed(999999999, 0)
        assert isinstance(seed, int)
        assert 0 <= seed <= 0xFFFFFFFF

    def test_zero_inputs(self):
        seed = derive_seed(0, 0)
        assert isinstance(seed, int)


class TestMakeRng:
    def test_returns_generator(self):
        rng = make_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_deterministic(self):
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        assert rng1.random() == rng2.random()

    def test_different_seeds_different_output(self):
        rng1 = make_rng(42)
        rng2 = make_rng(43)
        # Very unlikely to be equal
        assert rng1.random() != rng2.random()
