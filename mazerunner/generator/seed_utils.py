"""SHA-256 seed derivation and numpy RNG factory."""

import hashlib

import numpy as np


def derive_seed(master_seed: int, index: int) -> int:
    """Derive a deterministic seed from a master seed and index using SHA-256."""
    data = f"{master_seed}:{index}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big")


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy random Generator from a seed."""
    return np.random.default_rng(seed)
