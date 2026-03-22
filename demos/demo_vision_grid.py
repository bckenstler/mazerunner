#!/usr/bin/env python3
"""Interactive vision grid navigator demo.

Generates a small maze and lets you navigate with U/D/L/R keys.
Each move renders a new image and saves it to disk (opens automatically on macOS).
Type a sequence of moves (e.g. "RRDD") and press Enter.
Type "q" to quit.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mazerunner.common.types import MazeGrid
from mazerunner.generator.maze_graph import generate_maze, solve_bfs
from mazerunner.generator.seed_utils import make_rng
from mazerunner.generator.serialization import instance_to_dict, maze_grid_to_instance
from mazerunner.navigator import GridNavigator
from mazerunner.renderer.base import GridRenderConfig


def make_instance(rows: int = 5, cols: int = 7, seed: int = 42) -> dict:
    rng = make_rng(seed)
    passages = generate_maze(rows, cols, rng)
    start, goal = (0, 0), (rows - 1, cols - 1)
    solution = solve_bfs(passages, start, goal, rows, cols)
    grid = MazeGrid(
        rows=rows, cols=cols, passages=passages,
        start=start, goal=goal, solution_path=solution,
        endpoint_type="edge-edge",
    )
    return instance_to_dict(maze_grid_to_instance(grid, "demo"))


def show_image(img, tmpdir: str, step: int):
    path = os.path.join(tmpdir, f"step_{step:03d}.png")
    img.save(path)
    # Open with default viewer on macOS; adjust for other platforms
    if sys.platform == "darwin":
        subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Image saved to: {path}")


def main():
    instance = make_instance()
    config = GridRenderConfig(cell_size=40, wall_thickness=4, margin=10)
    nav = GridNavigator(instance, render_mode="vision_grid", config=config)
    solution_len = len(instance["shortest_path_cells"])

    tmpdir = tempfile.mkdtemp(prefix="mazerunner_vision_grid_")
    step = 0

    print("=== Vision Grid Navigator Demo ===")
    print(f"Goal: navigate from S to G (shortest path: {solution_len} cells)")
    print(f"Images saved to: {tmpdir}")
    print("Enter moves as U/D/L/R (e.g. 'RRDD'), 'q' to quit\n")

    img = nav.render()
    show_image(img, tmpdir, step)
    print()

    move_count = 0
    while not nav.finished:
        action = input("Move> ").strip().upper()
        if action == "Q":
            print("Bye!")
            return

        if not action:
            continue

        result = nav.interact(action)
        step += 1

        if result.valid:
            move_count += result.steps_applied
            img = nav.render()
            show_image(img, tmpdir, step)
            print(f"  Moved {result.steps_applied} step(s). Total steps: {move_count}")
            print(f"  Position: row={nav.position[0]}, col={nav.position[1]}")
        else:
            print(f"  Invalid move '{action}' — hit a wall or out of bounds. Try again.")

        if nav.finished:
            print(f"\n  You reached the goal in {move_count} steps!")
            print(f"  (Shortest path was {solution_len} cells / {solution_len - 1} steps)")


if __name__ == "__main__":
    main()
