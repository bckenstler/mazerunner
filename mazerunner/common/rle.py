"""Run-length encoding for binary masks."""

import numpy as np


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a (H, W) bool array into run-length encoding.

    Returns {"counts": [...], "shape": [H, W]} where counts alternates
    [num_zeros, num_ones, num_zeros, num_ones, ...], always starting
    with a zero-count (can be 0 if mask starts with True).
    """
    h, w = mask.shape
    flat = mask.flatten().astype(bool)

    if len(flat) == 0:
        return {"counts": [0], "shape": [h, w]}

    counts = []
    current_val = False  # always start counting zeros
    current_count = 0

    for val in flat:
        if val == current_val:
            current_count += 1
        else:
            counts.append(current_count)
            current_val = val
            current_count = 1

    counts.append(current_count)
    return {"counts": counts, "shape": [h, w]}


def decode_rle(rle_dict: dict) -> np.ndarray:
    """Decode RLE dict back to (H, W) bool array."""
    counts = rle_dict["counts"]
    h, w = rle_dict["shape"]

    flat = []
    current_val = False  # starts with zero-count
    for count in counts:
        flat.extend([current_val] * count)
        current_val = not current_val

    arr = np.array(flat, dtype=bool).reshape(h, w)
    return arr
