"""Eval protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mazerunner.openenv.server.maze_environment import MazeEnvironment


@runtime_checkable
class EpisodeRunner(Protocol):
    """Protocol for running a single maze episode."""

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord: ...


@dataclass
class StepRecord:
    """Record of a single step in an episode."""

    action: dict
    tool_name: str
    reward: float
    valid: bool
    reasoning: str = ""
    raw_result: dict = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """Record of a complete episode."""

    maze_id: str
    success: bool
    steps: int
    reward: float
    trajectory: list[StepRecord]
    mode: str = ""
    initial_observation: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a full evaluation run."""

    run_id: str
    mode: str
    model: str
    num_episodes: int
    records: list[EpisodeRecord]
    metrics: dict[str, float]
