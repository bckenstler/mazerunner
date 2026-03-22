#!/usr/bin/env python
"""E2E test: verify all three reward modes produce expected signals.

Runs three server instances (one per reward mode) in sequence, navigates the
same maze, and checks that rewards match expectations.

Usage:
    python scripts/e2e_reward_modes.py
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


def _make_test_instance() -> dict:
    """3-cell linear maze: (0,0) -> (0,1) -> (0,2)."""
    adjacency = {
        "0,0": ["0,1"],
        "0,1": ["0,0", "0,2"],
        "0,2": ["0,1"],
    }
    return {
        "id": "reward_test_maze",
        "grid_rows": 1,
        "grid_cols": 3,
        "start": "0,0",
        "goal": "0,2",
        "adjacency": adjacency,
        "shortest_path_cells": ["0,0", "0,1", "0,2"],
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


def run_episode(base_url: str) -> list[dict]:
    """Navigate R, R and collect step results."""
    from mazerunner.openenv.client import MazeEnvClient

    client = MazeEnvClient(base_url=base_url).sync()
    results = []
    with client:
        client.reset()
        for d in ["R", "R"]:
            result = client.call_tool("navigate", directions=d)
            results.append(result)
    return results


def test_reward_mode(reward_mode: str, tmpdir: str, port: int) -> None:
    base_url = f"http://localhost:{port}"

    env_vars = {
        **os.environ,
        "MAZE_MODE": "text_grid",
        "MAZE_INSTANCE_DIR": tmpdir,
        "MAZE_REWARD_MODE": reward_mode,
        "MAZE_MAX_STEPS": "50",
        "MAZE_PORT": str(port),
    }

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "mazerunner", "serve"],
        env=env_vars,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        wait_for_server(base_url)
        results = run_episode(base_url)

        step1, step2 = results[0], results[1]

        if reward_mode == "sparse":
            assert step1["reward"] == 0.0, \
                f"sparse: mid-path reward should be 0, got {step1['reward']}"
            assert step2["reward"] == 1.0, \
                f"sparse: goal reward should be 1.0, got {step2['reward']}"
            print(f"[e2e]   sparse: step1={step1['reward']}, step2={step2['reward']} ✓")

        elif reward_mode == "shaped":
            assert step1["reward"] > 0, \
                f"shaped: moving closer should give positive reward, got {step1['reward']}"
            assert step2["reward"] > 0, \
                f"shaped: reaching goal should give positive reward, got {step2['reward']}"
            print(f"[e2e]   shaped: step1={step1['reward']:.4f}, "
                  f"step2={step2['reward']:.4f} ✓")

        elif reward_mode == "efficiency":
            assert step1["reward"] == 0.0, \
                f"efficiency: mid-path reward should be 0, got {step1['reward']}"
            assert step2["reward"] == 1.0, \
                f"efficiency: optimal (2 steps / 2 optimal) should be 1.0, got {step2['reward']}"
            print(f"[e2e]   efficiency: step1={step1['reward']}, "
                  f"step2={step2['reward']} ✓")

    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()


def main():
    tmpdir = tempfile.mkdtemp(prefix="maze_e2e_reward_")
    inst_dir = os.path.join(tmpdir, "instances")
    os.makedirs(inst_dir)
    instance = _make_test_instance()
    with open(os.path.join(inst_dir, "maze_000000.json"), "w") as f:
        json.dump(instance, f)

    try:
        base_port = 18770

        for i, mode in enumerate(["sparse", "shaped", "efficiency"]):
            print(f"[e2e] Testing reward_mode={mode}...")
            test_reward_mode(mode, tmpdir, base_port + i)

        print("\n[e2e] ✓ All reward mode e2e tests PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
