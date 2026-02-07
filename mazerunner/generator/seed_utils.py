"""Deterministic seed derivation utilities."""

import hashlib

import numpy as np


def derive_seed(master_seed: int, index: int) -> int:
    """Derive a deterministic seed from master_seed and index using SHA-256."""
    data = f"{master_seed}:{index}".encode()
    digest = hashlib.sha256(data).digest()
    seed = int.from_bytes(digest[:4], byteorder="big")
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy random Generator from a seed."""
    return np.random.default_rng(seed)
