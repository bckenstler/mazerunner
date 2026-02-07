"""Difficulty tier configurations and sampling."""

import numpy as np

from mazerunner.common.types import DifficultyConfig

TIER_CONFIGS = {
    1: {"rows": (5, 8), "cols": (7, 12), "corridor": (28, 40), "wall": (3, 6), "min_solution": 8},
    2: {"rows": (10, 16), "cols": (14, 22), "corridor": (16, 26), "wall": (3, 5), "min_solution": 20},
    3: {"rows": (18, 28), "cols": (25, 40), "corridor": (8, 16), "wall": (2, 4), "min_solution": 40},
}


def sample_difficulty_params(tier: int, rng: np.random.Generator) -> DifficultyConfig:
    """Sample difficulty parameters uniformly from tier ranges."""
    cfg = TIER_CONFIGS[tier]
    rows = int(rng.integers(cfg["rows"][0], cfg["rows"][1] + 1))
    cols = int(rng.integers(cfg["cols"][0], cfg["cols"][1] + 1))
    corridor_width = int(rng.integers(cfg["corridor"][0], cfg["corridor"][1] + 1))
    wall_thickness = int(rng.integers(cfg["wall"][0], cfg["wall"][1] + 1))
    return DifficultyConfig(
        tier=tier,
        grid_rows=rows,
        grid_cols=cols,
        corridor_width=corridor_width,
        wall_thickness=wall_thickness,
        min_solution_length=cfg["min_solution"],
    )
