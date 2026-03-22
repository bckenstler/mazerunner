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
    provider: str = "openai"
    # OpenAI-specific
    reasoning_effort: str = "medium"       # "none"|"minimal"|"low"|"medium"|"high"|"xhigh" (or "" to disable)
    reasoning_summary: str = "auto"        # "auto"|"concise"|"detailed"
    # Anthropic-specific
    thinking_type: str = "adaptive"        # "adaptive"|"enabled"|"disabled"
    thinking_budget_tokens: int | None = None  # Required when type="enabled"; must be < max_tokens
    thinking_display: str | None = None    # "summarized" (default) or "omitted"
    max_tokens: int = 16000                # Required for Anthropic API
    effort: str | None = None              # "low"|"medium"|"high"|"max" (None = don't send)
    # Gemini-specific
    thinking_budget: int | None = None     # Gemini 2.5: token count (0=off for flash, 128-32768, -1=dynamic)
    thinking_level: str | None = None      # Gemini 3: "minimal"|"low"|"medium"|"high"
    # Navigation mode
    single_step: bool = False              # If True, navigate accepts only one direction per call


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
