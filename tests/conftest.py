"""Shared fixtures for MazeRunner tests."""

import numpy as np
import pytest

from mazerunner.common.types import DifficultyConfig, RenderConfig
from mazerunner.generator.maze_graph import build_maze
from mazerunner.generator.masks import (
    compute_cell_center,
    compute_image_size,
    generate_free_space_mask,
    generate_wall_mask,
    solution_cells_to_polyline,
)
from mazerunner.generator.difficulty import sample_difficulty_params
from mazerunner.generator.seed_utils import derive_seed, make_rng


@pytest.fixture
def rng():
    """Deterministic RNG for tests."""
    return make_rng(42)


@pytest.fixture
def small_maze(rng):
    """A small 5x7 maze for testing."""
    return build_maze(5, 7, 4, rng)


@pytest.fixture
def tier1_maze_data():
    """Generate a full tier-1 maze with all metadata, matching generate.py pipeline."""
    seed = derive_seed(42, 0)
    rng = make_rng(seed)
    diff = sample_difficulty_params(1, rng)

    chrome_height_top = 38
    chrome_width_left = 0
    image_width, image_height, render_config = compute_image_size(
        diff, chrome_height_top, chrome_width_left
    )
    render_config = RenderConfig(
        image_width=image_width,
        image_height=image_height,
        corridor_width=render_config.corridor_width,
        wall_thickness=render_config.wall_thickness,
        chrome_height_top=chrome_height_top,
        chrome_width_left=chrome_width_left,
        theme_name="light_classic",
    )

    maze = build_maze(diff.grid_rows, diff.grid_cols, diff.min_solution_length, rng)
    wall_mask = generate_wall_mask(maze, render_config)
    free_mask = generate_free_space_mask(wall_mask)

    start_center = compute_cell_center(
        maze.start[0], maze.start[1], render_config, maze.rows, maze.cols
    )
    goal_center = compute_cell_center(
        maze.goal[0], maze.goal[1], render_config, maze.rows, maze.cols
    )

    solution_polyline = solution_cells_to_polyline(
        maze.solution_path, render_config, maze.rows, maze.cols
    )

    return {
        "maze": maze,
        "render_config": render_config,
        "wall_mask": wall_mask,
        "free_mask": free_mask,
        "start_center": start_center,
        "goal_center": goal_center,
        "solution_polyline": solution_polyline,
        "image_width": image_width,
        "image_height": image_height,
        "diff": diff,
    }
