#!/usr/bin/env python
"""E2E test: vision_drag mode via MazeEnvClient.

Starts a server, connects a sync client, drags through a maze using pixel
coordinates, and verifies rendered PNGs.

Usage:
    python scripts/e2e_vision_drag.py
"""

import base64
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
    """Small fully-connected 3x3 maze for drag testing."""
    rows, cols = 3, 3
    adjacency = {}
    for r in range(rows):
        for c in range(cols):
            adjacency[f"{r},{c}"] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                adjacency[f"{r},{c}"].append(f"{r},{c + 1}")
                adjacency[f"{r},{c + 1}"].append(f"{r},{c}")
            if r + 1 < rows:
                adjacency[f"{r},{c}"].append(f"{r + 1},{c}")
                adjacency[f"{r + 1},{c}"].append(f"{r},{c}")
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    return {
        "id": "e2e_drag_maze",
        "grid_rows": rows,
        "grid_cols": cols,
        "start": "0,0",
        "goal": "2,2",
        "adjacency": adjacency,
        "shortest_path_cells": ["0,0", "0,1", "0,2", "1,2", "2,2"],
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
    port = 18767
    base_url = f"http://localhost:{port}"

    tmpdir = tempfile.mkdtemp(prefix="maze_e2e_drag_")
    inst_dir = os.path.join(tmpdir, "instances")
    os.makedirs(inst_dir)
    instance = _make_test_instance()
    with open(os.path.join(inst_dir, "maze_000000.json"), "w") as f:
        json.dump(instance, f)

    env_vars = {
        **os.environ,
        "MAZE_MODE": "vision_drag",
        "MAZE_INSTANCE_DIR": tmpdir,
        "MAZE_REWARD_MODE": "sparse",
        "MAZE_MAX_STEPS": "50",
        "MAZE_PORT": str(port),
    }

    server_proc = None
    try:
        print(f"[e2e] Starting vision_drag server on port {port}...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "mazerunner", "serve"],
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wait_for_server(base_url)
        print("[e2e] Server ready.")

        from mazerunner.openenv.client import MazeEnvClient
        from mazerunner.renderer.base import DragRenderConfig
        from mazerunner.renderer.vision_drag import cell_to_pixel_center

        client = MazeEnvClient(base_url=base_url).sync()
        with client:
            # Reset
            print("[e2e] Resetting environment...")
            reset_result = client.reset()
            assert not reset_result.done

            # List tools — should have drag, not navigate
            tools = client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"[e2e] Tools: {tool_names}")
            assert "drag" in tool_names, f"Expected 'drag' tool, got {tool_names}"
            assert "navigate" not in tool_names

            # Get maze info
            info = client.call_tool("get_maze_info")
            assert info["mode"] == "vision_drag"
            print(f"[e2e] Maze: {info['grid_rows']}x{info['grid_cols']}, "
                  f"start={info['start']}, goal={info['goal']}")

            # Compute pixel path through the maze: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
            config = DragRenderConfig()
            margin = config.margin

            waypoints = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
            pixel_centers = [
                cell_to_pixel_center(r, c, config, margin)
                for r, c in waypoints
            ]

            # Drag in segments, showing incremental progress
            current_pos = pixel_centers[0]
            for i in range(len(pixel_centers) - 1):
                p1 = pixel_centers[i]
                p2 = pixel_centers[i + 1]
                points = [[p1[0], p1[1]], [p2[0], p2[1]]]

                print(f"[e2e] Drag {i + 1}: ({p1[0]:.0f},{p1[1]:.0f}) -> "
                      f"({p2[0]:.0f},{p2[1]:.0f})")
                result = client.call_tool("drag", points=points)

                # Verify PNG
                img_bytes = base64.b64decode(result["rendered"])
                assert img_bytes[:4] == b"\x89PNG", f"Drag {i + 1}: not a valid PNG"

                print(f"[e2e]   valid={result['valid']}, pos={result['position']}, "
                      f"PNG={len(img_bytes)} bytes")

                if not result["valid"]:
                    print(f"[e2e]   WARNING: drag was invalid, continuing...")
                    # For subsequent drags, start from current position
                    continue

                current_pos = (result["position"][0], result["position"][1])

                if result["finished"]:
                    print(f"[e2e] Reached goal! reward={result['reward']}, "
                          f"steps={result['step_count']}")
                    break

            if result.get("finished"):
                assert result["done"]
                assert result["reward"] == 1.0
                print("\n[e2e] ✓ vision_drag e2e test PASSED")
            else:
                print("\n[e2e] ✗ Did not reach goal (drag path may have been invalid)")
                sys.exit(1)

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
