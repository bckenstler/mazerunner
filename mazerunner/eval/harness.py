"""Eval harness — orchestrates episodes across maze instances."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from mazerunner.eval.metrics import compute_metrics
from mazerunner.eval.protocol import EpisodeRunner, EvalResult
from mazerunner.openenv.server.maze_environment import MazeEnvironment
from mazerunner.renderer.base import load_instance


def run_eval(
    runner: EpisodeRunner,
    instance_paths: list[str],
    mode: str,
    max_steps: int = 100,
    reward_mode: str = "sparse",
    num_episodes: int | None = None,
    run_id: str | None = None,
    model: str = "",
) -> EvalResult:
    """Run evaluation across maze instances.

    Args:
        runner: An EpisodeRunner that plays a single maze.
        instance_paths: Paths to maze instance JSON files.
        mode: Rendering mode (text_grid, vision_grid, vision_drag).
        max_steps: Maximum steps per episode.
        reward_mode: Reward function (sparse, shaped, efficiency).
        num_episodes: Limit number of episodes (default: all instances).
        run_id: Optional run identifier (auto-generated if None).
        model: Model name for metadata.

    Returns:
        EvalResult with records and computed metrics.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    paths = instance_paths[:num_episodes] if num_episodes is not None else instance_paths
    instances: dict[str, dict] = {}
    records = []

    for path in paths:
        instance = load_instance(path)
        maze_id = instance.get("id", Path(path).stem)
        instances[maze_id] = instance

        env = MazeEnvironment(
            mode=mode,
            instance=instance,
            reward_mode=reward_mode,
            max_steps=max_steps,
        )
        record = runner.run_episode(env, maze_id)
        records.append(record)

    metrics = compute_metrics(records, instances)

    return EvalResult(
        run_id=run_id,
        mode=mode,
        model=model,
        num_episodes=len(records),
        records=records,
        metrics=metrics,
    )
