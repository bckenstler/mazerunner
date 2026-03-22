"""Fireworks AI agent loop for maze navigation using Chat Completions API."""

from __future__ import annotations

import json
from typing import Any

from openenv.core.env_server import CallToolAction

from mazerunner.agent.chat_context import ChatCompletionsContext
from mazerunner.agent.openai_loop import _DEFAULT_SYSTEM_PROMPT
from mazerunner.agent.tool_defs import get_chat_tool_schemas
from mazerunner.agent.tool_transform import transform_tool_output
from mazerunner.agent.types import AgentConfig, EpisodeResult, TurnRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def run_fireworks_episode(
    config: AgentConfig,
    env: MazeEnvironment,
    client: Any | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single maze episode using the Fireworks Chat Completions API.

    Args:
        config: Agent configuration (model, mode, max_turns, etc.).
        env: A MazeEnvironment instance (already configured).
        client: A Fireworks client instance. If None, creates one.
        verbose: If True, print each step's observation, reasoning, and tool call.

    Returns:
        EpisodeResult with full trajectory.
    """
    if client is None:
        from fireworks.client import Fireworks
        client = Fireworks()

    # Reset environment and get initial observation
    obs = env.reset()
    meta = obs.metadata
    maze_id = meta.get("maze_id", "unknown")
    rendered = meta["rendered"]
    mode = config.mode

    # Build initial context
    context = ChatCompletionsContext(mode)
    system_text = config.system_prompt or _DEFAULT_SYSTEM_PROMPT

    # For drag mode, append image resolution and starting position
    if mode == "vision_drag":
        import base64
        from io import BytesIO
        from PIL import Image as _PILImage

        img = _PILImage.open(BytesIO(base64.b64decode(rendered)))
        pos = meta["position"]
        system_text += (
            f" The maze image is {img.width}x{img.height} pixels."
            f" Your starting position is [{pos[0]:.0f}, {pos[1]:.0f}]."
            f" Your drag path must start at your current position and be contiguous"
            f" — no teleporting. Coordinates must be within the image resolution."
        )

    context.add_system(system_text)

    # Add initial maze state as a user message
    if mode == "text_grid":
        context.add_user_message(
            f"Here is the maze. Navigate from X to G.\n\n{rendered}"
        )
    else:
        context.add_user_message([
            {"type": "text", "text": "Here is the maze. Navigate from X to G."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{rendered}"},
            },
        ])

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if mode == "text_grid":
        _log(f"--- Initial observation ---\n{rendered}\n")
    else:
        _log(f"--- Initial observation --- [image {len(rendered)} chars b64]\n")

    tools = get_chat_tool_schemas(mode, single_step=config.single_step)
    turns: list[TurnRecord] = []
    total_reward = 0.0
    success = False
    maze_info: dict = {}

    # Build optional API kwargs
    api_kwargs: dict[str, Any] = {}
    if config.thinking_budget:
        api_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }
    elif config.reasoning_effort:
        api_kwargs["reasoning_effort"] = config.reasoning_effort
    # temperature — only when not using reasoning/thinking
    if not config.reasoning_effort and not config.thinking_budget:
        if config.temperature != 0.0:
            api_kwargs["temperature"] = config.temperature

    for turn_num in range(config.max_turns):
        # Call the model
        response = client.chat.completions.create(
            model=config.model,
            messages=context.messages,
            tools=tools,
            **api_kwargs,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract reasoning content (Fireworks reasoning models)
        reasoning_text = getattr(message, "reasoning_content", None) or ""
        if reasoning_text:
            _log(f"--- Reasoning (turn {turn_num}) ---\n{reasoning_text}\n")

        # Check for tool calls
        tool_calls = message.tool_calls or []

        # Build assistant message dict for context passback
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        else:
            assistant_msg["content"] = None
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        # Preserve reasoning_content for multi-turn passback
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text

        if not tool_calls:
            _log(f"--- No tool call (turn {turn_num}), ending ---")
            context.add_assistant_message(assistant_msg)
            break

        # Enforce single tool call per turn — take only the first
        tc = tool_calls[0]
        if len(tool_calls) > 1:
            # Trim assistant message to only include the first tool call
            assistant_msg["tool_calls"] = [assistant_msg["tool_calls"][0]]

        context.add_assistant_message(assistant_msg)

        # Process the tool call
        tool_name = tc.function.name
        tool_arguments = json.loads(tc.function.arguments)

        # Call the environment
        env_obs = env.step(
            CallToolAction(tool_name=tool_name, arguments=tool_arguments)
        )
        raw_result = env_obs.result.structured_content

        # Track maze_info calls
        if tool_name == "get_maze_info":
            maze_info = raw_result

        reward = raw_result.get("reward", 0.0)
        total_reward += reward
        is_done = raw_result.get("done", False)

        _log(f"--- Tool call (turn {turn_num}) ---")
        _log(f"  {tool_name}({json.dumps(tool_arguments)})")
        _log(f"  valid={raw_result.get('valid')} finished={raw_result.get('finished')} reward={reward} done={is_done}")

        # Transform for the model (chat format)
        transformed = transform_tool_output(
            tool_name, tool_arguments, raw_result, mode, format="chat"
        )

        # Show observation in verbose mode
        if isinstance(transformed, str):
            _log(f"--- Observation ---\n{transformed}\n")
        else:
            text_parts = [b["text"] for b in transformed if b.get("type") == "text"]
            _log(f"--- Observation ---\n{''.join(text_parts)} [+ image]\n")

        context.add_tool_result(tc.id, transformed)

        turns.append(TurnRecord(
            turn_number=turn_num,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            raw_result=raw_result,
            transformed_output=transformed,
            reward=reward,
            done=is_done,
            reasoning=reasoning_text,
        ))

        done = False
        if raw_result.get("finished", False):
            success = True
            done = True
        elif is_done:
            done = True

        if done:
            break

    return EpisodeResult(
        maze_id=maze_id,
        mode=mode,
        success=success,
        total_turns=len(turns),
        total_reward=total_reward,
        turns=turns,
        maze_info=maze_info,
        initial_observation=meta,
    )
