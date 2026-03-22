"""Gemini agent loop for maze navigation."""

from __future__ import annotations

import base64
import json
from typing import Any

from openenv.core.env_server import CallToolAction

from mazerunner.agent.openai_loop import _DEFAULT_SYSTEM_PROMPT
from mazerunner.agent.tool_defs import get_gemini_tools
from mazerunner.agent.tool_transform import _format_action_str
from mazerunner.agent.types import AgentConfig, EpisodeResult, TurnRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _extract_thinking(parts: list) -> str:
    """Extract thinking text from Gemini response parts."""
    texts = []
    for part in parts:
        if getattr(part, "thought", False) and getattr(part, "text", None):
            texts.append(part.text)
    return "\n".join(texts)


def _build_tool_feedback_text(
    tool_name: str,
    tool_arguments: dict,
    raw_result: dict,
    mode: str,
) -> str:
    """Build the text portion of tool feedback (shared logic with tool_transform)."""
    valid = raw_result.get("valid", False)
    finished = raw_result.get("finished", False)

    if valid and finished:
        return "Maze complete! You reached the goal."

    if valid:
        position = raw_result.get("position", [])
        if mode == "vision_drag" and position:
            return f"Valid action. Your position is now [{position[0]:.0f}, {position[1]:.0f}]. Current maze state:"
        return "Valid action. Current maze state:"

    action_str = _format_action_str(tool_name, tool_arguments)
    if mode == "vision_drag":
        return f"Invalid action: {action_str}. Your position has not changed. Current maze state:"
    return f"Invalid action: {action_str}. Current maze state:"


def _apply_image_window(contents: list, mode: str) -> None:
    """Replace older images in user Contents, keeping only the most recent.

    Mutates contents in place. Never modifies model Content objects.
    No-op for text_grid mode.
    """
    if mode == "text_grid":
        return

    from google.genai import types

    # Find indices of user Contents that contain inline_data (image) parts
    image_indices = []
    for i, content in enumerate(contents):
        if not hasattr(content, "role") or content.role != "user":
            continue
        if any(
            getattr(p, "inline_data", None) is not None
            for p in (content.parts or [])
        ):
            image_indices.append(i)

    if len(image_indices) <= 1:
        return

    # Replace images in all but the last
    for idx in image_indices[:-1]:
        new_parts = []
        for part in contents[idx].parts:
            if getattr(part, "inline_data", None) is not None:
                new_parts.append(types.Part.from_text(text="[Previous maze image omitted]"))
            else:
                new_parts.append(part)
        contents[idx] = types.Content(role="user", parts=new_parts)


def run_gemini_episode(
    config: AgentConfig,
    env: MazeEnvironment,
    client: Any | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single maze episode using the Gemini API.

    Args:
        config: Agent configuration.
        env: A MazeEnvironment instance (already configured).
        client: A google.genai.Client instance. If None, creates one.
        verbose: If True, print each step's observation, reasoning, and tool call.

    Returns:
        EpisodeResult with full trajectory.
    """
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client()

    # Reset environment and get initial observation
    obs = env.reset()
    meta = obs.metadata
    maze_id = meta.get("maze_id", "unknown")
    rendered = meta["rendered"]
    mode = config.mode

    # Build system prompt (same as OpenAI)
    system_text = config.system_prompt or _DEFAULT_SYSTEM_PROMPT

    # For drag mode, append image resolution and starting position
    if mode == "vision_drag":
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
        initial_parts = [types.Part.from_text(text=
            f"Here is the maze. Navigate from X to G.\n\n{rendered}"
        )]
    else:
        initial_parts = [
            types.Part.from_text(text="Here is the maze. Navigate from X to G."),
            types.Part.from_bytes(data=base64.b64decode(rendered), mime_type="image/png"),
        ]

    contents: list[types.Content] = [
        types.Content(role="user", parts=initial_parts),
    ]

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if mode == "text_grid":
        _log(f"--- Initial observation ---\n{rendered}\n")
    else:
        _log(f"--- Initial observation --- [image {len(rendered)} chars b64]\n")

    # Build tools
    tool = get_gemini_tools(mode)

    # Build generation config
    thinking_config = None
    if config.thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=config.thinking_budget,
            include_thoughts=True,
        )
    elif config.thinking_level is not None:
        thinking_config = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, config.thinking_level.upper()),
            include_thoughts=True,
        )

    gen_config = types.GenerateContentConfig(
        tools=[tool],
        system_instruction=system_text,
        temperature=config.temperature,
        thinking_config=thinking_config,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    turns: list[TurnRecord] = []
    total_reward = 0.0
    success = False
    maze_info: dict = {}

    for turn_num in range(config.max_turns):
        # Apply image windowing before API call
        _apply_image_window(contents, mode)

        # Call the model
        response = client.models.generate_content(
            model=config.model,
            contents=contents,
            config=gen_config,
        )

        # Extract thinking from response parts
        response_parts = response.candidates[0].content.parts if response.candidates else []
        reasoning_text = _extract_thinking(response_parts)
        if reasoning_text:
            _log(f"--- Reasoning (turn {turn_num}) ---\n{reasoning_text}\n")

        # Check for function calls
        function_calls = response.function_calls or []

        if not function_calls:
            _log(f"--- No tool call (turn {turn_num}), ending ---")
            break

        # Preserve model's full Content (includes thought signatures)
        model_content = response.candidates[0].content
        contents.append(model_content)

        # Enforce single tool call per turn — take only the first
        fc = function_calls[0]
        tool_name = fc.name
        tool_arguments = dict(fc.args) if fc.args else {}

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

        # Build feedback text
        feedback_text = _build_tool_feedback_text(
            tool_name, tool_arguments, raw_result, mode
        )
        rendered_state = raw_result.get("rendered", "")
        finished = raw_result.get("valid", False) and raw_result.get("finished", False)

        # Build tool result Content
        tool_result_content = types.Content(
            role="tool",
            parts=[types.Part.from_function_response(
                name=tool_name, response={"result": feedback_text},
            )],
        )
        contents.append(tool_result_content)

        # For vision modes with non-finished results, add image as user message
        if mode != "text_grid" and not finished and rendered_state:
            image_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=feedback_text),
                    types.Part.from_bytes(
                        data=base64.b64decode(rendered_state),
                        mime_type="image/png",
                    ),
                ],
            )
            contents.append(image_content)
        elif mode == "text_grid" and not finished and rendered_state:
            # For text mode, append ASCII rendering as user message
            text_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{feedback_text}\n{rendered_state}")],
            )
            contents.append(text_content)

        # Build transformed output for TurnRecord (reuse OpenAI format for storage)
        from mazerunner.agent.tool_transform import transform_tool_output
        transformed = transform_tool_output(tool_name, tool_arguments, raw_result, mode)

        # Show observation in verbose mode
        if isinstance(transformed, str):
            _log(f"--- Observation ---\n{transformed}\n")
        else:
            text_parts = [b["text"] for b in transformed if b.get("type") == "input_text"]
            _log(f"--- Observation ---\n{''.join(text_parts)} [+ image]\n")

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
