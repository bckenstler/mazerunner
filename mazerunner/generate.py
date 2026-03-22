"""CLI entry point for maze dataset generation."""

import argparse
import json
import os
import sys

from mazerunner.common.types import MazeGrid
from mazerunner.generator.seed_utils import derive_seed, make_rng
from mazerunner.generator.maze_graph import generate_maze, solve_bfs
from mazerunner.generator.placement import ENDPOINT_TYPES, place_endpoints
from mazerunner.generator.difficulty import sample_difficulty_params
from mazerunner.generator.color_schemas import sample_color_schema
from mazerunner.generator.serialization import maze_grid_to_instance, instance_to_json


def generate_dataset(output_dir: str, num_mazes: int, master_seed: int, tier_distribution: list):
    """Generate a dataset of maze instances and write them as JSON files.

    Orchestrates the full pipeline: seed derivation, difficulty sampling,
    maze generation, endpoint placement, BFS solving, and serialization.
    Output is written to ``{output_dir}/instances/maze_{i:06d}.json``.

    Args:
        output_dir: Root output directory.
        num_mazes: Total number of mazes to generate.
        master_seed: Master seed for deterministic generation.
        tier_distribution: List of 3 counts specifying how many mazes per tier
            (e.g. [300, 400, 300]).
    """
    instances_dir = os.path.join(output_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    # Build tier assignments from distribution
    tier_assignments = []
    for tier_idx, count in enumerate(tier_distribution, start=1):
        tier_assignments.extend([tier_idx] * count)

    if len(tier_assignments) != num_mazes:
        print(
            f"Error: tier distribution sum ({len(tier_assignments)}) != num_mazes ({num_mazes})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use master RNG to shuffle tier assignments
    master_rng = make_rng(master_seed)
    tier_assignments_arr = master_rng.permutation(tier_assignments)

    for i in range(num_mazes):
        seed = derive_seed(master_seed, i)
        rng = make_rng(seed)

        tier = int(tier_assignments_arr[i])
        config = sample_difficulty_params(tier, rng)

        # Generate maze
        passages = generate_maze(config.grid_rows, config.grid_cols, rng)

        # Pick endpoint type
        endpoint_type = ENDPOINT_TYPES[rng.integers(len(ENDPOINT_TYPES))]

        # Place start/goal
        start, goal = place_endpoints(
            passages, config.grid_rows, config.grid_cols,
            endpoint_type, config.min_solution_length, rng,
        )

        # Solve
        solution = solve_bfs(passages, start, goal, config.grid_rows, config.grid_cols)

        # Build grid
        grid = MazeGrid(
            rows=config.grid_rows,
            cols=config.grid_cols,
            passages=passages,
            start=start,
            goal=goal,
            solution_path=solution,
            endpoint_type=endpoint_type,
        )

        # Sample color schema
        color_schema = sample_color_schema(rng)

        # Serialize
        maze_id = f"maze_{i:06d}"
        instance = maze_grid_to_instance(grid, maze_id, color_schema=color_schema)
        instance.metadata["tier"] = tier

        json_str = instance_to_json(instance)
        filepath = os.path.join(instances_dir, f"{maze_id}.json")
        with open(filepath, "w") as f:
            f.write(json_str)

    print(f"Generated {num_mazes} mazes in {instances_dir}")


def main():
    """Parse CLI arguments and run dataset generation."""
    parser = argparse.ArgumentParser(description="Generate maze benchmark dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-mazes", type=int, required=True, help="Number of mazes to generate")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--tier-distribution",
        type=str,
        required=True,
        help="Comma-separated counts per tier (e.g., 300,400,300)",
    )

    args = parser.parse_args()
    tier_distribution = [int(x) for x in args.tier_distribution.split(",")]

    if len(tier_distribution) != 3:
        print("Error: tier-distribution must have exactly 3 values", file=sys.stderr)
        sys.exit(1)

    generate_dataset(args.output_dir, args.num_mazes, args.master_seed, tier_distribution)


if __name__ == "__main__":
    main()
