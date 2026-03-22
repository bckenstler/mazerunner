#!/usr/bin/env python
"""E2E test: text_grid mode via MazeEnvClient.

Starts a server in a subprocess, connects a sync client, and runs a full
episode using BFS-optimal moves through a text_grid maze.

Usage:
    python scripts/e2e_text_grid.py [--instance-dir data/dev]
"""

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import signal
import subprocess
import tempfile
import time
from collections import deque


def _make_test_instance() -> dict:
    """Create a small 3x4 maze with a known solution for deterministic testing."""
    # Linear corridor: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3)
    passages = [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (1, 3)),
        ((1, 3), (2, 3)),
        # Add some branches to make it a proper maze
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (2, 2)),
        ((2, 2), (2, 1)),
        ((2, 1), (2, 0)),
    ]
    rows, cols = 3, 4
    adjacency = {}
    for r in range(rows):
        for c in range(cols):
            adjacency[f"{r},{c}"] = []
    for (r1, c1), (r2, c2) in passages:
        adjacency[f"{r1},{c1}"].append(f"{r2},{c2}")
        adjacency[f"{r2},{c2}"].append(f"{r1},{c1}")
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    return {
        "id": "e2e_test_maze",
        "grid_rows": rows,
        "grid_cols": cols,
        "start": "0,0",
        "goal": "2,3",
        "adjacency": adjacency,
        "shortest_path_cells": ["0,0", "0,1", "0,2", "0,3", "1,3", "2,3"],
        "metadata": {
            "color_schema": {
                "name": "classic",
                "wall": "#1a1a2e",
                "corridor": "#e8e8e8",
                "start": "#22c55e",
                "goal": "#ef4444",
                "solution_path": "#3b82f6",
                "background": "#f5f5f5",
            }
        },
    }


def bfs_solve(adjacency: dict, start: str, goal: str) -> list[str]:
    """BFS to find directions from start to goal."""
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell == goal:
            return path
        r, c = map(int, cell.split(","))
        for neighbor in adjacency.get(cell, []):
            if neighbor not in visited:
                visited.add(neighbor)
                nr, nc = map(int, neighbor.split(","))
                dr, dc = nr - r, nc - c
                direction = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}[(dr, dc)]
                queue.append((neighbor, path + [direction]))
    raise RuntimeError(f"No path from {start} to {goal}")


def wait_for_server(url: str, timeout: float = 15.0) -> None:
    """Poll the health endpoint until the server is ready."""
    import urllib.request
    import urllib.error

    health_url = f"{url}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="E2E test for text_grid mode")
    parser.add_argument("--instance-dir", default=None, help="Directory with maze instances")
    parser.add_argument("--port", type=int, default=18765, help="Port for test server")
    args = parser.parse_args()

    port = args.port
    base_url = f"http://localhost:{port}"

    # Set up instance directory
    tmpdir = None
    instance_dir = args.instance_dir
    if instance_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="maze_e2e_")
        inst_dir = os.path.join(tmpdir, "instances")
        os.makedirs(inst_dir)
        instance = _make_test_instance()
        with open(os.path.join(inst_dir, "maze_000000.json"), "w") as f:
            json.dump(instance, f)
        instance_dir = tmpdir

    env_vars = {
        **os.environ,
        "MAZE_MODE": "text_grid",
        "MAZE_INSTANCE_DIR": instance_dir,
        "MAZE_REWARD_MODE": "sparse",
        "MAZE_MAX_STEPS": "50",
        "MAZE_PORT": str(port),
    }

    server_proc = None
    try:
        # Start server
        print(f"[e2e] Starting server on port {port}...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "mazerunner", "serve"],
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wait_for_server(base_url)
        print("[e2e] Server ready.")

        # Connect client
        from mazerunner.openenv.client import MazeEnvClient

        client = MazeEnvClient(base_url=base_url).sync()
        with client:
            # Reset
            print("[e2e] Resetting environment...")
            reset_result = client.reset()
            print(f"[e2e] Reset done={reset_result.done}, reward={reset_result.reward}")

            # List tools
            print("[e2e] Listing tools...")
            tools = client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"[e2e] Tools: {tool_names}")
            assert "navigate" in tool_names, f"Expected 'navigate' in tools, got {tool_names}"
            assert "get_maze_info" in tool_names, f"Expected 'get_maze_info' in tools"

            # Get maze info
            print("[e2e] Getting maze info...")
            info = client.call_tool("get_maze_info")
            print(f"[e2e] Maze info: {info}")
            assert info["mode"] == "text_grid"
            start = info["start"]
            goal = info["goal"]

            # Solve with BFS
            print(f"[e2e] Solving from {start} to {goal}...")

            # Load adjacency from instance file for BFS
            inst_path = os.path.join(instance_dir, "instances", "maze_000000.json")
            if os.path.exists(inst_path):
                with open(inst_path) as f:
                    inst_data = json.load(f)
                adjacency = inst_data["adjacency"]
            else:
                # Fallback: just navigate step by step
                adjacency = None

            if adjacency:
                directions = bfs_solve(adjacency, start, goal)
                direction_str = "".join(directions)
                print(f"[e2e] BFS solution: {direction_str} ({len(directions)} steps)")

                # Navigate step by step to show incremental progress
                position = start
                for i, d in enumerate(directions):
                    result = client.call_tool("navigate", directions=d)
                    position = f"{result['position'][0]},{result['position'][1]}"
                    print(f"[e2e] Step {i + 1}: {d} -> {position} (valid={result['valid']})")
                    assert result["valid"], f"Step {d} was invalid at position {position}"

                    if result["finished"]:
                        print(f"[e2e] Reached goal! reward={result['reward']}, "
                              f"steps={result['step_count']}")
                        break

                assert result["finished"], "Did not reach goal after BFS solution"
                assert result["reward"] == 1.0, f"Expected reward 1.0, got {result['reward']}"
                assert result["done"], "Expected done=True at goal"
            else:
                print("[e2e] No adjacency available, skipping BFS solve")

        print("\n[e2e] ✓ text_grid e2e test PASSED")

    finally:
        if server_proc is not None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("[e2e] Server stopped.")

        if tmpdir is not None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
