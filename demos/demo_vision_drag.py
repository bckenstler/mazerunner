#!/usr/bin/env python3
"""Automated vision drag navigator demo.

Generates a small maze, solves it, then replays the solution as a series of
drag interactions (pixel-coordinate paths along the BFS solution).
Saves each frame as a PNG so you can see breadcrumbs accumulate.
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
from mazerunner.navigator import DragNavigator
from mazerunner.renderer.base import DragRenderConfig, parse_cell
from mazerunner.renderer.vision_drag import cell_to_pixel_center


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


def save_and_show(img, tmpdir: str, step: int):
    path = os.path.join(tmpdir, f"frame_{step:03d}.png")
    img.save(path)
    print(f"  Saved: {path}")
    return path


def main():
    instance = make_instance()
    config = DragRenderConfig(wall_thickness=4, corridor_width=24, margin=8, antialias=True)
    nav = DragNavigator(instance, config=config)

    # Build the solution path as pixel centers
    solution_cells = [parse_cell(s) for s in instance["shortest_path_cells"]]
    pixel_centers = [cell_to_pixel_center(r, c, config, config.margin) for r, c in solution_cells]

    tmpdir = tempfile.mkdtemp(prefix="mazerunner_vision_drag_")

    print("=== Vision Drag Navigator Demo ===")
    print(f"Maze: {instance['grid_rows']}x{instance['grid_cols']}")
    print(f"Solution length: {len(solution_cells)} cells")
    print(f"Frames saved to: {tmpdir}\n")

    # Save initial state
    img = nav.render()
    last_path = save_and_show(img, tmpdir, 0)

    # Replay solution one segment at a time (drag from cell center to next cell center)
    for i in range(len(pixel_centers) - 1):
        p1 = pixel_centers[i]
        p2 = pixel_centers[i + 1]

        # For antialias mode, drag coords are in render-space (pre-downscale).
        # The DragNavigator uses the non-AA mask internally, so we pass non-scaled coords.
        result = nav.interact([[p1[0], p1[1]], [p2[0], p2[1]]])

        if not result.valid:
            print(f"  Segment {i} rejected! {p1} -> {p2}")
            continue

        img = nav.render()
        last_path = save_and_show(img, tmpdir, i + 1)

        cell = solution_cells[i + 1]
        status = " (GOAL!)" if nav.finished else ""
        print(f"    -> cell ({cell[0]},{cell[1]}){status}")

    print(f"\nDone! {len(pixel_centers)} frames in {tmpdir}")

    # Open final frame
    if sys.platform == "darwin":
        subprocess.Popen(["open", tmpdir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", tmpdir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
