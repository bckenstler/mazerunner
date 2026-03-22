#!/usr/bin/env python3
"""Interactive text grid navigator demo.

Generates a small maze and lets you navigate with U/D/L/R keys.
Type a sequence of moves (e.g. "RRDD") and press Enter.
Type "q" to quit.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mazerunner.common.types import MazeGrid
from mazerunner.generator.maze_graph import generate_maze, solve_bfs
from mazerunner.generator.seed_utils import make_rng
from mazerunner.generator.serialization import instance_to_dict, maze_grid_to_instance
from mazerunner.navigator import GridNavigator


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


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    instance = make_instance()
    nav = GridNavigator(instance, render_mode="text_grid")
    solution_len = len(instance["shortest_path_cells"])

    clear_screen()
    print("=== Text Grid Navigator Demo ===")
    print(f"Goal: navigate from S to G (shortest path: {solution_len} cells)")
    print("Enter moves as U/D/L/R (e.g. 'RRDD'), 'q' to quit\n")
    print(nav.render())
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
        clear_screen()
        print("=== Text Grid Navigator Demo ===")
        print(f"Goal: navigate from S to G (shortest path: {solution_len} cells)\n")
        print(nav.render())
        print()

        if result.valid:
            move_count += result.steps_applied
            print(f"  Moved {result.steps_applied} step(s). Total steps: {move_count}")
            print(f"  Position: row={nav.position[0]}, col={nav.position[1]}")
        else:
            print(f"  Invalid move '{action}' — hit a wall or out of bounds. Try again.")

        if nav.finished:
            print(f"\n  You reached the goal in {move_count} steps!")
            print(f"  (Shortest path was {solution_len} cells / {solution_len - 1} steps)")


if __name__ == "__main__":
    main()
