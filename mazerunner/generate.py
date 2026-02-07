"""CLI module for maze dataset generation."""

import argparse
import json
import math
import os

import numpy as np

from mazerunner.common.rle import encode_rle
from mazerunner.common.types import RenderConfig
from mazerunner.generator.difficulty import sample_difficulty_params
from mazerunner.generator.masks import (
    carve_outer_openings,
    compute_cell_center,
    compute_image_size,
    generate_free_space_mask,
    generate_region_mask,
    generate_wall_mask,
    solution_cells_to_polyline,
)
from mazerunner.generator.maze_graph import build_maze
from mazerunner.generator.placement import opening_center
from mazerunner.generator.renderer import MazeRenderer
from mazerunner.generator.seed_utils import derive_seed, make_rng
from mazerunner.generator.themes import pick_theme


def generate_dataset(output_dir: str, num_mazes: int, master_seed: int, tier_distribution: list):
    """Generate a maze dataset with images and ground truth."""
    assert sum(tier_distribution) == num_mazes, (
        f"Tier distribution {tier_distribution} must sum to {num_mazes}"
    )

    images_dir = os.path.join(output_dir, "images")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    renderer = MazeRenderer()

    # Build tier assignment: first N are tier 1, next M tier 2, rest tier 3
    tier_assignment = []
    for tier_idx, count in enumerate(tier_distribution):
        tier_assignment.extend([tier_idx + 1] * count)

    for index in range(num_mazes):
        tier = tier_assignment[index]
        seed = derive_seed(master_seed, index)
        rng = make_rng(seed)

        diff = sample_difficulty_params(tier, rng)

        chrome_height_top = 0
        chrome_width_left = 0

        image_width, image_height, render_config = compute_image_size(
            diff, chrome_height_top, chrome_width_left
        )

        theme_name = pick_theme(rng)
        render_config = RenderConfig(
            image_width=image_width,
            image_height=image_height,
            corridor_width=render_config.corridor_width,
            wall_thickness=render_config.wall_thickness,
            chrome_height_top=chrome_height_top,
            chrome_width_left=chrome_width_left,
            theme_name=theme_name,
        )

        maze = build_maze(diff.grid_rows, diff.grid_cols, diff.min_solution_length, rng)

        wall_mask = generate_wall_mask(maze, render_config)
        carve_outer_openings(wall_mask, maze, render_config)
        free_mask = generate_free_space_mask(wall_mask)

        if maze.start_edge:
            start_center = opening_center(
                maze.start, maze.start_edge, render_config, maze.rows, maze.cols
            )
        else:
            start_center = compute_cell_center(
                maze.start[0], maze.start[1], render_config, maze.rows, maze.cols
            )
        if maze.goal_edge:
            goal_center = opening_center(
                maze.goal, maze.goal_edge, render_config, maze.rows, maze.cols
            )
        else:
            goal_center = compute_cell_center(
                maze.goal[0], maze.goal[1], render_config, maze.rows, maze.cols
            )

        start_mask = generate_region_mask(
            start_center, render_config.corridor_width * 0.4, (image_height, image_width)
        )
        goal_mask = generate_region_mask(
            goal_center, render_config.corridor_width * 0.4, (image_height, image_width)
        )

        solution_polyline = solution_cells_to_polyline(
            maze.solution_path, render_config, maze.rows, maze.cols,
            start_edge=maze.start_edge, goal_edge=maze.goal_edge,
        )

        # Compute solution length
        solution_length = 0.0
        for i in range(1, len(solution_polyline)):
            x0, y0 = solution_polyline[i - 1]
            x1, y1 = solution_polyline[i]
            solution_length += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Render image
        img = renderer.render(maze, render_config, antialias=True)

        # Save image
        maze_id = f"{index:06d}"
        img_path = os.path.join(images_dir, f"{maze_id}.png")
        img.save(img_path)

        # Build GT JSON
        gt = {
            "id": maze_id,
            "seed": seed,
            "image_size": {"w": image_width, "h": image_height},
            "maze_family": "orthogonal",
            "difficulty": {
                "tier": diff.tier,
                "corridor_width_px": diff.corridor_width,
                "grid_rows": diff.grid_rows,
                "grid_cols": diff.grid_cols,
                "wall_thickness_px": diff.wall_thickness,
            },
            "regions": {
                "start_mask_rle": encode_rle(start_mask),
                "goal_mask_rle": encode_rle(goal_mask),
                "free_space_mask_rle": encode_rle(free_mask),
                "wall_mask_rle": encode_rle(wall_mask),
            },
            "gt": {
                "solution_polyline": [[float(x), float(y)] for x, y in solution_polyline],
                "solution_length": solution_length,
            },
            "render_config": {
                "corridor_width": render_config.corridor_width,
                "wall_thickness": render_config.wall_thickness,
                "chrome_height_top": render_config.chrome_height_top,
                "chrome_width_left": render_config.chrome_width_left,
                "theme_name": render_config.theme_name,
            },
            "placement": {
                "style": maze.placement_style,
                "start_edge": maze.start_edge,
                "goal_edge": maze.goal_edge,
            },
        }

        gt_path = os.path.join(gt_dir, f"{maze_id}.json")
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)

        print(f"Generated maze {maze_id} (tier {tier}, theme {theme_name}, "
              f"{diff.grid_rows}x{diff.grid_cols}, solution_len={len(maze.solution_path)})")

    print(f"\nDataset generation complete: {num_mazes} mazes in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate MazeRunner benchmark dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-mazes", type=int, default=50, help="Number of mazes to generate")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--tier-distribution",
        type=str,
        default="15,20,15",
        help="Comma-separated count per tier (must sum to num-mazes)",
    )
    args = parser.parse_args()

    tier_dist = [int(x) for x in args.tier_distribution.split(",")]
    assert len(tier_dist) == 3, "Tier distribution must have exactly 3 values"

    generate_dataset(args.output_dir, args.num_mazes, args.master_seed, tier_dist)


if __name__ == "__main__":
    main()
