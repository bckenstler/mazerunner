"""Anthropic Messages API agent loop for maze navigation."""

from __future__ import annotations

import json
from typing import Any

from openenv.core.env_server import CallToolAction

from mazerunner.agent.openai_loop import _DEFAULT_SYSTEM_PROMPT
from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
from mazerunner.agent.tool_transform import transform_tool_output
from mazerunner.agent.types import AgentConfig, EpisodeResult, TurnRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _extract_thinking(content_blocks: list) -> str:
    """Extract thinking text from Anthropic response content blocks."""
    parts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "thinking":
            text = getattr(block, "thinking", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _transform_to_anthropic_content(
    transformed: str | list[dict],
    rendered_b64: str | None = None,
) -> str | list[dict]:
    """Convert tool_transform output to Anthropic tool_result content format.

    For text_grid: returns the string as-is.
    For vision modes: converts OpenAI-format content blocks to Anthropic format.
    """
    if isinstance(transformed, str):
        return transformed

    # Convert OpenAI content blocks to Anthropic format
    result = []
    for block in transformed:
        if block.get("type") == "input_text":
            result.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "input_image":
            # Extract base64 data from data URI
            url = block.get("image_url", "")
            if url.startswith("data:image/png;base64,"):
                b64_data = url[len("data:image/png;base64,"):]
            else:
                b64_data = url
            result.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_data,
                },
            })
    return result


def _apply_image_window(messages: list[dict], mode: str) -> None:
    """Replace older images in tool_result messages, keeping only the most recent.

    Mutates messages in place. No-op for text_grid mode.
    """
    if mode == "text_grid":
        return

    # Find indices of user messages containing tool_result with images
    image_msg_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tr_content = block.get("content", [])
                if isinstance(tr_content, list) and any(
                    isinstance(b, dict) and b.get("type") == "image"
                    for b in tr_content
                ):
                    image_msg_indices.append(i)
                    break

    if len(image_msg_indices) <= 1:
        return

    # Replace images in all but the last
    for idx in image_msg_indices[:-1]:
        content = messages[idx]["content"]
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tr_content = block.get("content", [])
            if not isinstance(tr_content, list):
                continue
            new_content = []
            for b in tr_content:
                if isinstance(b, dict) and b.get("type") == "image":
                    new_content.append({
                        "type": "text",
                        "text": "[Previous maze image omitted]",
                    })
                else:
                    new_content.append(b)
            block["content"] = new_content


def run_anthropic_episode(
    config: AgentConfig,
    env: MazeEnvironment,
    client: Any | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single maze episode using the Anthropic Messages API.

    Args:
        config: Agent configuration.
        env: A MazeEnvironment instance (already configured).
        client: An Anthropic client instance. If None, creates one.
        verbose: If True, print each step's observation, reasoning, and tool call.

    Returns:
        EpisodeResult with full trajectory.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    # Reset environment and get initial observation
    obs = env.reset()
    meta = obs.metadata
    maze_id = meta.get("maze_id", "unknown")
    rendered = meta["rendered"]
    mode = config.mode

    # Build system prompt
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

    # Build initial user message
    if mode == "text_grid":
        initial_user = {
            "role": "user",
            "content": f"Here is the maze. Navigate from X to G.\n\n{rendered}",
        }
    else:
        initial_user = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the maze. Navigate from X to G."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": rendered,
                    },
                },
            ],
        }

    messages: list[dict] = [initial_user]

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if mode == "text_grid":
        _log(f"--- Initial observation ---\n{rendered}\n")
    else:
        _log(f"--- Initial observation --- [image {len(rendered)} chars b64]\n")

    tools = get_anthropic_tool_schemas(mode)
    turns: list[TurnRecord] = []
    total_reward = 0.0
    success = False
    maze_info: dict = {}

    # Build API kwargs
    api_kwargs: dict[str, Any] = {}

    # Thinking configuration
    if config.thinking_type == "adaptive":
        thinking_param: dict[str, Any] = {"type": "adaptive"}
        if config.thinking_display:
            thinking_param["display"] = config.thinking_display
        api_kwargs["thinking"] = thinking_param
    elif config.thinking_type == "enabled" and config.thinking_budget_tokens:
        thinking_param = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget_tokens,
        }
        if config.thinking_display:
            thinking_param["display"] = config.thinking_display
        api_kwargs["thinking"] = thinking_param
    elif config.thinking_type == "disabled":
        api_kwargs["thinking"] = {"type": "disabled"}

    # Effort configuration (only send when explicitly set)
    if config.effort is not None:
        api_kwargs["output_config"] = {"effort": config.effort}

    # Temperature — not supported with thinking
    if config.thinking_type not in ("adaptive", "enabled") and config.temperature != 0.0:
        api_kwargs["temperature"] = config.temperature

    # Enforce single tool per turn
    api_kwargs["tool_choice"] = {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }

    for turn_num in range(config.max_turns):
        # Apply image windowing before API call
        _apply_image_window(messages, mode)

        # Call the model
        response = client.messages.create(
            model=config.model,
            system=system_text,
            max_tokens=config.max_tokens,
            tools=tools,
            messages=messages,
            **api_kwargs,
        )

        content_blocks = response.content

        # Extract thinking
        reasoning_text = _extract_thinking(content_blocks)
        if reasoning_text:
            _log(f"--- Reasoning (turn {turn_num}) ---\n{reasoning_text}\n")

        # Check if model wants to use a tool
        if response.stop_reason != "tool_use":
            # Model output text without a tool call — append and prompt to continue
            messages.append({"role": "assistant", "content": content_blocks})
            messages.append({"role": "user", "content": "Please use the tool to navigate."})
            _log(f"--- No tool call (turn {turn_num}), prompting to use tool ---")
            continue

        # Find the first tool_use block
        tool_block = None
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                tool_block = block
                break

        if tool_block is None:
            break

        tool_name = tool_block.name
        tool_arguments = tool_block.input
        tool_use_id = tool_block.id

        # Append assistant response (full content including thinking blocks)
        messages.append({"role": "assistant", "content": content_blocks})

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

        # Transform for the model (OpenAI format)
        transformed = transform_tool_output(
            tool_name, tool_arguments, raw_result, mode
        )

        # Convert to Anthropic content format
        anthropic_content = _transform_to_anthropic_content(transformed)

        # Show observation in verbose mode
        if isinstance(transformed, str):
            _log(f"--- Observation ---\n{transformed}\n")
        else:
            text_parts = [b["text"] for b in transformed if b.get("type") == "input_text"]
            _log(f"--- Observation ---\n{''.join(text_parts)} [+ image]\n")

        # Build tool_result and append as user message
        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": anthropic_content,
        }
        messages.append({"role": "user", "content": [tool_result]})

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

        if raw_result.get("finished", False):
            success = True
            break
        elif is_done:
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
