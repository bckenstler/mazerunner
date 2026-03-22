"""Tests for the eval module: metrics, IO, harness, and protocol."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass

import pytest

from mazerunner.eval.harness import run_eval
from mazerunner.eval.io import load_eval_result, save_eval_result
from mazerunner.eval.metrics import compute_metrics
from mazerunner.eval.protocol import EpisodeRecord, EpisodeRunner, EvalResult, StepRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _make_simple_instance(rows=3, cols=3, passages=None, start=None, goal=None):
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


# ─── Metrics ─────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_empty_records(self):
        metrics = compute_metrics([])
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_steps"] == 0.0

    def test_all_success(self):
        records = [
            EpisodeRecord(
                maze_id=f"maze_{i}",
                success=True,
                steps=5,
                reward=1.0,
                trajectory=[
                    StepRecord(action={"directions": "R"}, tool_name="navigate", reward=0.0, valid=True)
                    for _ in range(5)
                ],
            )
            for i in range(3)
        ]
        metrics = compute_metrics(records)
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_steps"] == 5.0
        assert metrics["avg_steps_successful"] == 5.0
        assert metrics["avg_reward"] == 1.0
        assert metrics["invalid_action_rate"] == 0.0

    def test_mixed_results(self):
        records = [
            EpisodeRecord(
                maze_id="maze_0",
                success=True,
                steps=4,
                reward=1.0,
                trajectory=[
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True)
                    for _ in range(4)
                ],
            ),
            EpisodeRecord(
                maze_id="maze_1",
                success=False,
                steps=10,
                reward=-1.0,
                trajectory=[
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True)
                    for _ in range(10)
                ],
            ),
        ]
        metrics = compute_metrics(records)
        assert metrics["success_rate"] == 0.5
        assert metrics["avg_steps"] == 7.0
        assert metrics["avg_steps_successful"] == 4.0
        assert metrics["avg_reward"] == 0.0

    def test_invalid_action_rate(self):
        records = [
            EpisodeRecord(
                maze_id="maze_0",
                success=True,
                steps=4,
                reward=1.0,
                trajectory=[
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True),
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=False),
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True),
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True),
                ],
            ),
        ]
        metrics = compute_metrics(records)
        assert metrics["invalid_action_rate"] == 0.25

    def test_efficiency_with_instances(self):
        records = [
            EpisodeRecord(
                maze_id="maze_0",
                success=True,
                steps=8,
                reward=1.0,
                trajectory=[
                    StepRecord(action={}, tool_name="navigate", reward=0.0, valid=True)
                    for _ in range(8)
                ],
            ),
        ]
        instances = {
            "maze_0": {
                "shortest_path_cells": ["0,0", "0,1", "0,2", "1,2", "2,2"],  # 4 steps optimal
            }
        }
        metrics = compute_metrics(records, instances)
        assert metrics["efficiency"] == 0.5  # 4/8

    def test_efficiency_without_instances(self):
        records = [
            EpisodeRecord(maze_id="m", success=True, steps=5, reward=1.0, trajectory=[]),
        ]
        metrics = compute_metrics(records)
        assert metrics["efficiency"] == 0.0


# ─── IO ──────────────────────────────────────────────────────────


class TestEvalIO:
    def test_save_and_load_roundtrip(self):
        result = EvalResult(
            run_id="test123",
            mode="text_grid",
            model="gpt-4o",
            num_episodes=2,
            records=[
                EpisodeRecord(
                    maze_id="maze_0",
                    success=True,
                    steps=3,
                    reward=1.0,
                    trajectory=[
                        StepRecord(
                            action={"directions": "RRD"},
                            tool_name="navigate",
                            reward=1.0,
                            valid=True,
                            raw_result={"valid": True, "position": [2, 2], "finished": True},
                        )
                    ],
                    mode="text_grid",
                    initial_observation={"rendered": "maze_ascii", "position": [0, 0]},
                ),
                EpisodeRecord(
                    maze_id="maze_1",
                    success=False,
                    steps=10,
                    reward=-1.0,
                    trajectory=[],
                    mode="text_grid",
                    initial_observation={"rendered": "maze2", "position": [0, 0]},
                ),
            ],
            metrics={"success_rate": 0.5, "avg_steps": 6.5},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_eval_result(result, path)
            loaded = load_eval_result(path)

            assert loaded.run_id == "test123"
            assert loaded.mode == "text_grid"
            assert loaded.model == "gpt-4o"
            assert loaded.num_episodes == 2
            assert len(loaded.records) == 2
            assert loaded.records[0].success is True
            assert loaded.records[0].trajectory[0].tool_name == "navigate"
            assert loaded.records[0].trajectory[0].raw_result["finished"] is True
            assert loaded.records[0].mode == "text_grid"
            assert loaded.records[0].initial_observation["position"] == [0, 0]
            assert loaded.metrics["success_rate"] == 0.5
        finally:
            os.unlink(path)

    def test_save_creates_valid_json(self):
        result = EvalResult(
            run_id="x",
            mode="text_grid",
            model="m",
            num_episodes=0,
            records=[],
            metrics={},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_eval_result(result, path)
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == "x"
            assert isinstance(data["records"], list)
        finally:
            os.unlink(path)


# ─── Protocol ────────────────────────────────────────────────────


class TestEpisodeRunnerProtocol:
    def test_protocol_is_runtime_checkable(self):
        class MyRunner:
            def run_episode(self, env, maze_id):
                return EpisodeRecord(
                    maze_id=maze_id, success=True, steps=1, reward=1.0, trajectory=[]
                )

        runner = MyRunner()
        assert isinstance(runner, EpisodeRunner)

    def test_non_conforming_class_fails(self):
        class NotARunner:
            pass

        assert not isinstance(NotARunner(), EpisodeRunner)


# ─── Harness ─────────────────────────────────────────────────────


class _DummyRunner:
    """A deterministic runner that always navigates right to the goal."""

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord:
        from openenv.core.env_server import CallToolAction

        env.reset()
        # Navigate right twice to reach goal in a 1x3 maze
        obs = env.step(CallToolAction(tool_name="navigate", arguments={"directions": "RR"}))
        data = obs.result.structured_content
        return EpisodeRecord(
            maze_id=maze_id,
            success=data.get("finished", False),
            steps=1,
            reward=data.get("reward", 0.0),
            trajectory=[
                StepRecord(
                    action={"directions": "RR"},
                    tool_name="navigate",
                    reward=data.get("reward", 0.0),
                    valid=data.get("valid", True),
                )
            ],
        )


class TestRunEval:
    def test_run_eval_single_instance(self):
        instance = _make_simple_instance(rows=1, cols=3)
        instance["id"] = "linear_maze"
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "maze.json")
            with open(path, "w") as f:
                json.dump(instance, f)

            runner = _DummyRunner()
            result = run_eval(
                runner=runner,
                instance_paths=[path],
                mode="text_grid",
                run_id="test_run",
                model="dummy",
            )

        assert result.run_id == "test_run"
        assert result.num_episodes == 1
        assert result.records[0].success is True
        assert result.metrics["success_rate"] == 1.0

    def test_run_eval_num_episodes_limits(self):
        instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                inst = _make_simple_instance(rows=1, cols=3)
                inst["id"] = f"maze_{i}"
                inst["shortest_path_cells"] = ["0,0", "0,1", "0,2"]
                path = os.path.join(tmpdir, f"maze_{i}.json")
                with open(path, "w") as f:
                    json.dump(inst, f)
                instances.append(path)

            runner = _DummyRunner()
            result = run_eval(
                runner=runner,
                instance_paths=instances,
                mode="text_grid",
                num_episodes=2,
            )

        assert result.num_episodes == 2
        assert len(result.records) == 2

    def test_run_eval_generates_run_id(self):
        instance = _make_simple_instance(rows=1, cols=3)
        instance["shortest_path_cells"] = ["0,0", "0,1", "0,2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "maze.json")
            with open(path, "w") as f:
                json.dump(instance, f)

            result = run_eval(
                runner=_DummyRunner(),
                instance_paths=[path],
                mode="text_grid",
            )

        assert len(result.run_id) == 12  # uuid hex[:12]
