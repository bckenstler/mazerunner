#!/usr/bin/env python
"""E2E test: verify max_steps cutoff terminates episodes correctly.

Starts a server with max_steps=3, navigates without reaching the goal,
and verifies that done=True triggers at the step limit.

Usage:
    python scripts/e2e_max_steps.py
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import signal
import subprocess
import tempfile
import time


def _make_long_maze() -> dict:
    """1x6 linear maze — goal is 5 steps away, but max_steps will be 3."""
    cols = 6
    adjacency = {}
    for c in range(cols):
        adjacency[f"0,{c}"] = []
    for c in range(cols - 1):
        adjacency[f"0,{c}"].append(f"0,{c + 1}")
        adjacency[f"0,{c + 1}"].append(f"0,{c}")

    return {
        "id": "max_steps_maze",
        "grid_rows": 1,
        "grid_cols": cols,
        "start": "0,0",
        "goal": "0,5",
        "adjacency": adjacency,
        "shortest_path_cells": [f"0,{c}" for c in range(cols)],
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


def wait_for_server(url: str, timeout: float = 15.0) -> None:
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
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def main():
    port = 18775
    base_url = f"http://localhost:{port}"
    max_steps = 3

    tmpdir = tempfile.mkdtemp(prefix="maze_e2e_maxstep_")
    inst_dir = os.path.join(tmpdir, "instances")
    os.makedirs(inst_dir)
    with open(os.path.join(inst_dir, "maze_000000.json"), "w") as f:
        json.dump(_make_long_maze(), f)

    env_vars = {
        **os.environ,
        "MAZE_MODE": "text_grid",
        "MAZE_INSTANCE_DIR": tmpdir,
        "MAZE_REWARD_MODE": "sparse",
        "MAZE_MAX_STEPS": str(max_steps),
        "MAZE_PORT": str(port),
    }

    server_proc = None
    try:
        print(f"[e2e] Starting server with max_steps={max_steps}...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "mazerunner", "serve"],
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wait_for_server(base_url)
        print("[e2e] Server ready.")

        from mazerunner.openenv.client import MazeEnvClient

        client = MazeEnvClient(base_url=base_url).sync()
        with client:
            client.reset()

            results = []
            for i in range(max_steps + 1):  # try one extra step past the limit
                result = client.call_tool("navigate", directions="R")
                results.append(result)
                print(f"[e2e] Step {i + 1}: pos={result['position']}, "
                      f"done={result['done']}, reward={result['reward']}")
                if result["done"]:
                    break

            # Should have hit done at step 3
            assert len(results) == max_steps, \
                f"Expected {max_steps} steps before done, got {len(results)}"
            assert results[-1]["done"], "Last step should have done=True"
            assert not results[-1]["finished"], "Should NOT have reached goal"
            assert results[-1]["reward"] == -1.0, \
                f"Sparse timeout should give -1.0, got {results[-1]['reward']}"
            assert results[-1]["step_count"] == max_steps

        print(f"\n[e2e] ✓ max_steps cutoff e2e test PASSED")

    finally:
        if server_proc is not None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("[e2e] Server stopped.")

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
