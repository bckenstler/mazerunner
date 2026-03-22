"""Agent dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for an agent episode."""

    model: str
    mode: str
    max_turns: int = 100
    system_prompt: str = ""
    temperature: float = 0.0
    reasoning_effort: str = "medium"


@dataclass
class TurnRecord:
    """Record of a single agent turn."""

    turn_number: int
    tool_name: str
    tool_arguments: dict
    raw_result: dict
    transformed_output: str | list
    reward: float
    done: bool
    reasoning: str = ""


@dataclass
class EpisodeResult:
    """Result of a complete agent episode."""

    maze_id: str
    mode: str
    success: bool
    total_turns: int
    total_reward: float
    turns: list[TurnRecord]
    maze_info: dict
    initial_observation: dict = field(default_factory=dict)
