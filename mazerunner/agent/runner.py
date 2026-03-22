"""Agent runners — bridge agent loops to the eval protocol."""

from __future__ import annotations

from typing import Any

from mazerunner.agent.types import AgentConfig, EpisodeResult
from mazerunner.eval.protocol import EpisodeRecord, StepRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _result_to_record(result: EpisodeResult, maze_id: str) -> EpisodeRecord:
    """Convert an EpisodeResult to an EpisodeRecord."""
    trajectory = [
        StepRecord(
            action=t.tool_arguments,
            tool_name=t.tool_name,
            reward=t.reward,
            valid=t.raw_result.get("valid", True),
            reasoning=t.reasoning,
            raw_result=t.raw_result,
        )
        for t in result.turns
    ]

    return EpisodeRecord(
        maze_id=maze_id,
        success=result.success,
        steps=result.total_turns,
        reward=result.total_reward,
        trajectory=trajectory,
        mode=result.mode,
        initial_observation=result.initial_observation,
    )


class OpenAIAgentRunner:
    """Adapts run_openai_episode to the EpisodeRunner protocol."""

    def __init__(
        self,
        config: AgentConfig,
        client: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._client = client
        self._verbose = verbose

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord:
        """Run a single episode and return an EpisodeRecord."""
        from mazerunner.agent.openai_loop import run_openai_episode

        result = run_openai_episode(
            self._config, env, self._client, verbose=self._verbose,
        )
        return _result_to_record(result, maze_id)


class GeminiAgentRunner:
    """Adapts run_gemini_episode to the EpisodeRunner protocol."""

    def __init__(
        self,
        config: AgentConfig,
        client: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._client = client
        self._verbose = verbose

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord:
        """Run a single episode and return an EpisodeRecord."""
        from mazerunner.agent.gemini_loop import run_gemini_episode

        result = run_gemini_episode(
            self._config, env, self._client, verbose=self._verbose,
        )
        return _result_to_record(result, maze_id)


class FireworksAgentRunner:
    """Adapts run_fireworks_episode to the EpisodeRunner protocol."""

    def __init__(
        self,
        config: AgentConfig,
        client: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._client = client
        self._verbose = verbose

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord:
        """Run a single episode and return an EpisodeRecord."""
        from mazerunner.agent.fireworks_loop import run_fireworks_episode

        result = run_fireworks_episode(
            self._config, env, self._client, verbose=self._verbose,
        )
        return _result_to_record(result, maze_id)


class AnthropicAgentRunner:
    """Adapts run_anthropic_episode to the EpisodeRunner protocol."""

    def __init__(
        self,
        config: AgentConfig,
        client: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._client = client
        self._verbose = verbose

    def run_episode(self, env: MazeEnvironment, maze_id: str) -> EpisodeRecord:
        """Run a single episode and return an EpisodeRecord."""
        from mazerunner.agent.anthropic_loop import run_anthropic_episode

        result = run_anthropic_episode(
            self._config, env, self._client, verbose=self._verbose,
        )
        return _result_to_record(result, maze_id)


def get_runner(
    config: AgentConfig,
    client: Any | None = None,
    verbose: bool = False,
) -> OpenAIAgentRunner | GeminiAgentRunner | FireworksAgentRunner | AnthropicAgentRunner:
    """Factory that returns the appropriate runner for the configured provider."""
    if config.provider == "anthropic":
        return AnthropicAgentRunner(config, client, verbose)
    if config.provider == "gemini":
        return GeminiAgentRunner(config, client, verbose)
    if config.provider == "fireworks":
        return FireworksAgentRunner(config, client, verbose)
    return OpenAIAgentRunner(config, client, verbose)
