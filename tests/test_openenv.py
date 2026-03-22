"""Tests for the OpenEnv integration module."""

import base64
import json
import os
import tempfile

import pytest
from PIL import Image

from openenv.core.env_server import CallToolAction, ListToolsAction

from mazerunner.navigator.base import InteractionResult
from mazerunner.navigator.grid_navigator import GridNavigator
from mazerunner.openenv.models import MazeObservation
from mazerunner.openenv.reward import compute_reward
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _make_simple_instance(rows: int = 3, cols: int = 3, passages=None, start=None, goal=None) -> dict:
    """Build a maze instance dict for testing."""
    adjacency = {}
    for r in range(rows):
        for c in range(cols):
            adjacency[f"{r},{c}"] = []

    if passages is None:
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    adjacency[f"{r},{c}"].append(f"{r},{c + 1}")
                    adjacency[f"{r},{c + 1}"].append(f"{r},{c}")
                if r + 1 < rows:
                    adjacency[f"{r},{c}"].append(f"{r + 1},{c}")
                    adjacency[f"{r + 1},{c}"].append(f"{r},{c}")
    else:
        for (r1, c1), (r2, c2) in passages:
            adjacency[f"{r1},{c1}"].append(f"{r2},{c2}")
            adjacency[f"{r2},{c2}"].append(f"{r1},{c1}")

    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (rows - 1, cols - 1)

    return {
        "id": "test_maze",
        "grid_rows": rows,
        "grid_cols": cols,
        "start": f"{start[0]},{start[1]}",
        "goal": f"{goal[0]},{goal[1]}",
        "adjacency": adjacency,
        "shortest_path_cells": [f"{start[0]},{start[1]}", f"{goal[0]},{goal[1]}"],
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


def _make_linear_instance() -> dict:
    """3x1 linear maze: (0,0) -> (0,1) -> (0,2). Shortest path = 2 steps."""
    return _make_simple_instance(
        rows=1,
        cols=3,
        passages=[((0, 0), (0, 1)), ((0, 1), (0, 2))],
        start=(0, 0),
        goal=(0, 2),
    )


# ─── Reward Functions ────────────────────────────────────────────


class TestRewardFunctions:
    def test_sparse_at_goal(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        nav.interact("RR")  # reach goal
        result = nav.history[-1].result
        reward = compute_reward(nav, instance, result, (0, 0), 1, "sparse", "text_grid")
        assert reward == 1.0

    def test_sparse_not_at_goal(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        result = nav.interact("R")
        reward = compute_reward(nav, instance, result, (0, 0), 1, "sparse", "text_grid")
        assert reward == 0.0

    def test_sparse_max_steps_penalty(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        result = nav.interact("R")
        reward = compute_reward(
            nav, instance, result, (0, 0), 5, "sparse", "text_grid", max_steps=5
        )
        assert reward == -1.0

    def test_shaped_positive(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        result = nav.interact("R")  # moved closer to goal
        reward = compute_reward(nav, instance, result, (0, 0), 1, "shaped", "text_grid")
        assert reward > 0.0

    def test_shaped_invalid_move(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        result = nav.interact("U")  # invalid (wall / out of bounds)
        reward = compute_reward(nav, instance, result, (0, 0), 1, "shaped", "text_grid")
        assert reward == 0.0

    def test_efficiency_at_goal_optimal(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        nav.interact("RR")
        result = nav.history[-1].result
        # Reached in 1 interaction, optimal is 2 steps
        reward = compute_reward(
            nav, instance, result, (0, 0), 1, "efficiency", "text_grid"
        )
        assert reward == 1.0  # min(1.0, 2/1) capped at 1.0

    def test_efficiency_at_goal_suboptimal(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        nav.interact("R")
        nav.interact("R")
        result = nav.history[-1].result
        # Reached in 2 interactions, optimal is 2 steps
        reward = compute_reward(
            nav, instance, result, (0, 1), 2, "efficiency", "text_grid"
        )
        assert reward == 1.0  # min(1.0, 2/2)

    def test_efficiency_not_at_goal(self):
        instance = _make_linear_instance()
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
        nav = GridNavigator(instance)
        result = nav.interact("R")
        reward = compute_reward(
            nav, instance, result, (0, 0), 1, "efficiency", "text_grid"
        )
        assert reward == 0.0


# ─── MazeEnvironment (Grid Mode) ────────────────────────────────


class TestMazeEnvironmentGrid:
    def test_reset_returns_observation(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance)
        obs = env.reset()
        assert obs.done is False
        assert obs.reward == 0.0
        meta = obs.metadata
        assert meta["mode"] == "text_grid"
        assert meta["position"] == [0, 0]
        assert isinstance(meta["rendered"], str)

    def test_navigate_valid_move(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance)
        env.reset()
        obs = env.step(CallToolAction(tool_name="navigate", arguments={"directions": "R"}))
        result = obs.result
        assert result is not None
        # Extract structured content
        data = result.structured_content
        assert data["valid"] is True
        assert data["position"] == [0, 1]
        assert data["step_count"] == 1

    def test_navigate_invalid_move_rejected(self):
        instance = _make_simple_instance(
            rows=3,
            cols=3,
            passages=[((0, 0), (0, 1)), ((0, 1), (0, 2)),
                      ((0, 2), (1, 2)), ((1, 2), (2, 2))],
        )
        env = MazeEnvironment(mode="text_grid", instance=instance)
        env.reset()
        # Try to move down from (0,0) — no passage there
        obs = env.step(CallToolAction(tool_name="navigate", arguments={"directions": "D"}))
        data = obs.result.structured_content
        assert data["valid"] is False
        assert data["position"] == [0, 0]

    def test_navigate_to_goal(self):
        instance = _make_simple_instance(rows=1, cols=3)
        env = MazeEnvironment(mode="text_grid", instance=instance, reward_mode="sparse")
        env.reset()
        obs = env.step(CallToolAction(tool_name="navigate", arguments={"directions": "RR"}))
        data = obs.result.structured_content
        assert data["finished"] is True
        assert data["done"] is True
        assert data["reward"] == 1.0

    def test_max_steps_cutoff(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance, max_steps=2)
        env.reset()
        env.step(CallToolAction(tool_name="navigate", arguments={"directions": "R"}))
        obs = env.step(CallToolAction(tool_name="navigate", arguments={"directions": "R"}))
        data = obs.result.structured_content
        assert data["done"] is True
        assert data["step_count"] == 2


# ─── MazeEnvironment (Drag Mode) ─────────────────────────────────


class TestMazeEnvironmentDrag:
    def test_reset_returns_observation(self):
        instance = _make_simple_instance(rows=3, cols=3)
        env = MazeEnvironment(mode="vision_drag", instance=instance)
        obs = env.reset()
        assert obs.done is False
        meta = obs.metadata
        assert meta["mode"] == "vision_drag"
        # Position should be pixel coordinates (floats)
        assert len(meta["position"]) == 2

    def test_drag_tool_exposed(self):
        instance = _make_simple_instance(rows=3, cols=3)
        env = MazeEnvironment(mode="vision_drag", instance=instance)
        env.reset()
        obs = env.step(ListToolsAction())
        tool_names = [t.name for t in obs.tools]
        assert "drag" in tool_names
        assert "navigate" not in tool_names

    def test_get_maze_info(self):
        instance = _make_simple_instance(rows=3, cols=3)
        env = MazeEnvironment(mode="vision_drag", instance=instance)
        env.reset()
        obs = env.step(CallToolAction(tool_name="get_maze_info", arguments={}))
        data = obs.result.structured_content
        assert data["grid_rows"] == 3
        assert data["grid_cols"] == 3
        assert data["mode"] == "vision_drag"


# ─── Tool Discovery ──────────────────────────────────────────────


class TestMazeEnvironmentTools:
    def test_grid_mode_exposes_navigate(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance)
        env.reset()
        obs = env.step(ListToolsAction())
        tool_names = [t.name for t in obs.tools]
        assert "navigate" in tool_names
        assert "get_maze_info" in tool_names
        assert "drag" not in tool_names

    def test_drag_mode_exposes_drag(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="vision_drag", instance=instance)
        env.reset()
        obs = env.step(ListToolsAction())
        tool_names = [t.name for t in obs.tools]
        assert "drag" in tool_names
        assert "navigate" not in tool_names
        assert "get_maze_info" in tool_names


# ─── Maze Loading ────────────────────────────────────────────────


class TestMazeLoading:
    def test_single_instance_reuses(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance)
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1.metadata["maze_id"] == obs2.metadata["maze_id"]

    def test_directory_mode_loads_mazes(self):
        instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = os.path.join(tmpdir, "instances")
            os.makedirs(inst_dir)
            for i in range(3):
                inst = _make_simple_instance()
                inst["id"] = f"maze_{i}"
                path = os.path.join(inst_dir, f"maze_{i:06d}.json")
                with open(path, "w") as f:
                    json.dump(inst, f)
                instances.append(inst)

            env = MazeEnvironment(mode="text_grid", instance_dir=tmpdir, seed=42)
            ids = []
            for _ in range(3):
                obs = env.reset()
                ids.append(obs.metadata["maze_id"])
            # All 3 mazes should have been loaded (possibly in shuffled order)
            assert len(ids) == 3

    def test_directory_mode_deterministic_with_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = os.path.join(tmpdir, "instances")
            os.makedirs(inst_dir)
            for i in range(5):
                inst = _make_simple_instance()
                inst["id"] = f"maze_{i}"
                path = os.path.join(inst_dir, f"maze_{i:06d}.json")
                with open(path, "w") as f:
                    json.dump(inst, f)

            env1 = MazeEnvironment(mode="text_grid", instance_dir=tmpdir, seed=123)
            env2 = MazeEnvironment(mode="text_grid", instance_dir=tmpdir, seed=123)
            ids1 = [env1.reset().metadata["maze_id"] for _ in range(5)]
            ids2 = [env2.reset().metadata["maze_id"] for _ in range(5)]
            assert ids1 == ids2


# ─── Image Encoding ──────────────────────────────────────────────


class TestImageEncoding:
    def test_vision_grid_returns_valid_base64_png(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="vision_grid", instance=instance)
        obs = env.reset()
        rendered = obs.metadata["rendered"]
        # Should be valid base64
        img_bytes = base64.b64decode(rendered)
        # Should be valid PNG (starts with PNG signature)
        assert img_bytes[:4] == b"\x89PNG"

    def test_vision_drag_returns_valid_base64_png(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="vision_drag", instance=instance)
        obs = env.reset()
        rendered = obs.metadata["rendered"]
        img_bytes = base64.b64decode(rendered)
        assert img_bytes[:4] == b"\x89PNG"

    def test_text_grid_returns_text(self):
        instance = _make_simple_instance()
        env = MazeEnvironment(mode="text_grid", instance=instance)
        obs = env.reset()
        rendered = obs.metadata["rendered"]
        # Text grid should contain wall characters and not be base64
        assert isinstance(rendered, str)
        # Should contain the X marker
        assert "X" in rendered


# ─── MazeObservation Model ───────────────────────────────────────


class TestMazeObservation:
    def test_model_roundtrip(self):
        obs = MazeObservation(
            rendered="test",
            mode="text_grid",
            position=[0.0, 0.0],
            valid=True,
            finished=False,
            steps_applied=0,
            reward=0.0,
            maze_id="test",
            step_count=0,
            done=False,
        )
        data = obs.model_dump()
        obs2 = MazeObservation(**data)
        assert obs == obs2
