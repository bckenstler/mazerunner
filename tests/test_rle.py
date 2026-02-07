"""Tests for RLE encode/decode."""

import numpy as np
import pytest

from mazerunner.common.rle import decode_rle, encode_rle


class TestRLERoundTrip:
    def test_random_mask_round_trip(self):
        rng = np.random.default_rng(123)
        mask = rng.random((20, 30)) > 0.5
        encoded = encode_rle(mask)
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_all_false_mask(self):
        mask = np.zeros((10, 15), dtype=bool)
        encoded = encode_rle(mask)
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)
        # All false: should be single count of 150
        assert encoded["counts"] == [150]

    def test_all_true_mask(self):
        mask = np.ones((8, 12), dtype=bool)
        encoded = encode_rle(mask)
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)
        # All true: starts with 0 zeros, then 96 ones
        assert encoded["counts"] == [0, 96]

    def test_single_row_mask(self):
        mask = np.array([[True, False, True, True, False]], dtype=bool)
        encoded = encode_rle(mask)
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_known_small_mask(self):
        mask = np.array([
            [False, False, True],
            [True, True, False],
        ], dtype=bool)
        encoded = encode_rle(mask)
        # Flat: F F T T T F -> counts: [2, 3, 1]
        assert encoded["counts"] == [2, 3, 1]
        assert encoded["shape"] == [2, 3]
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_shape_preservation(self):
        for shape in [(1, 1), (1, 100), (50, 1), (33, 47)]:
            mask = np.random.default_rng(0).random(shape) > 0.5
            encoded = encode_rle(mask)
            assert encoded["shape"] == list(shape)
            decoded = decode_rle(encoded)
            assert decoded.shape == shape

    def test_alternating_pattern(self):
        # Checkerboard-like pattern in 1D
        mask = np.array([[True, False, True, False, True, False]], dtype=bool)
        encoded = encode_rle(mask)
        # Flat: T F T F T F -> counts: [0, 1, 1, 1, 1, 1, 1]
        assert encoded["counts"] == [0, 1, 1, 1, 1, 1, 1]
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_large_mask_round_trip(self):
        rng = np.random.default_rng(999)
        mask = rng.random((200, 300)) > 0.3
        encoded = encode_rle(mask)
        decoded = decode_rle(encoded)
        np.testing.assert_array_equal(decoded, mask)
