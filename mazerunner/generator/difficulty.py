"""Difficulty tier parameter configs and sampling."""

import numpy as np

from mazerunner.common.types import DifficultyConfig

# Tier definitions: (row_min, row_max, col_min, col_max, min_solution_length)
TIER_PARAMS = {
    1: {"row_range": (5, 8), "col_range": (7, 12), "min_solution_length": 8},
    2: {"row_range": (10, 16), "col_range": (14, 22), "min_solution_length": 20},
    3: {"row_range": (18, 28), "col_range": (25, 40), "min_solution_length": 40},
}


def sample_difficulty_params(tier: int, rng: np.random.Generator) -> DifficultyConfig:
    """Sample grid dimensions for a given difficulty tier."""
    if tier not in TIER_PARAMS:
        raise ValueError(f"Invalid tier: {tier}. Must be 1, 2, or 3.")

    params = TIER_PARAMS[tier]
    row_min, row_max = params["row_range"]
    col_min, col_max = params["col_range"]

    grid_rows = int(rng.integers(row_min, row_max + 1))
    grid_cols = int(rng.integers(col_min, col_max + 1))

    return DifficultyConfig(
        tier=tier,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        min_solution_length=params["min_solution_length"],
    )


def compute_difficulty_score(path_length: int, rows: int, cols: int) -> int:
    """Compute difficulty score (1-9) from path length and grid size."""
    return min(9, 1 + int(8 * path_length / (rows * cols)))
