"""OpenAI agent runner — bridges agent loop to eval protocol."""

from __future__ import annotations

from typing import Any

from mazerunner.agent.openai_loop import run_openai_episode
from mazerunner.agent.types import AgentConfig
from mazerunner.eval.protocol import EpisodeRecord, StepRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


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
        result = run_openai_episode(
            self._config, env, self._client, verbose=self._verbose,
        )

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
