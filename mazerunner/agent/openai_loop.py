"""OpenAI Responses API agent loop for maze navigation."""

from __future__ import annotations

import json
from typing import Any

from openenv.core.env_server import CallToolAction

from mazerunner.agent.context_manager import SlidingWindowContext
from mazerunner.agent.tool_defs import get_tool_schemas
from mazerunner.agent.tool_transform import transform_tool_output
from mazerunner.agent.types import AgentConfig, EpisodeResult, TurnRecord
from mazerunner.openenv.server.maze_environment import MazeEnvironment

def _extract_reasoning_summary(output_items: list) -> str:
    """Extract reasoning summary text from response output items.

    The Responses API returns reasoning items with summary content blocks
    when the model uses reasoning. This extracts and concatenates them.
    """
    parts = []
    for item in output_items:
        if getattr(item, "type", None) == "reasoning":
            for block in getattr(item, "summary", []):
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
    return "\n".join(parts)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a maze-solving agent. Navigate to the goal (G). "
    "X marks your current position in the maze. "
    "After you move, your starting cell will be marked S. "
    "You may only call one tool per turn. After each tool call you will see the "
    "updated maze state. Plan your moves carefully and avoid hitting walls."
)


def run_openai_episode(
    config: AgentConfig,
    env: MazeEnvironment,
    client: Any | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single maze episode using the OpenAI Responses API.

    Args:
        config: Agent configuration (model, mode, max_turns, etc.).
        env: A MazeEnvironment instance (already configured).
        client: An OpenAI client instance. If None, creates one.
        verbose: If True, print each step's observation, reasoning, and tool call.

    Returns:
        EpisodeResult with full trajectory.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    # Reset environment and get initial observation
    obs = env.reset()
    meta = obs.metadata
    maze_id = meta.get("maze_id", "unknown")
    rendered = meta["rendered"]
    mode = config.mode

    # Build initial context
    context = SlidingWindowContext(mode)
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
        initial_content = f"Here is the maze. Navigate from X to G.\n\n{rendered}"
        context._items.append({"role": "user", "content": initial_content})
    else:
        context._items.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Here is the maze. Navigate from X to G."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{rendered}",
                    "detail": "auto",
                },
            ],
        })

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if mode == "text_grid":
        _log(f"--- Initial observation ---\n{rendered}\n")
    else:
        _log(f"--- Initial observation --- [image {len(rendered)} chars b64]\n")

    tools = get_tool_schemas(mode)
    turns: list[TurnRecord] = []
    total_reward = 0.0
    success = False
    maze_info: dict = {}

    # Build optional API kwargs
    api_kwargs: dict[str, Any] = {}
    if config.reasoning_effort:
        api_kwargs["reasoning"] = {
            "effort": config.reasoning_effort,
            "summary": "auto",
        }
    # temperature is not supported with reasoning models
    if not config.reasoning_effort and config.temperature != 0.0:
        api_kwargs["temperature"] = config.temperature

    for turn_num in range(config.max_turns):
        # Call the model
        response = client.responses.create(
            model=config.model,
            input=context.items,
            tools=tools,
            **api_kwargs,
        )

        # Extract output items
        output_items = response.output

        # Extract reasoning summary from this response
        reasoning_text = _extract_reasoning_summary(output_items)
        if reasoning_text:
            _log(f"--- Reasoning (turn {turn_num}) ---\n{reasoning_text}\n")

        # Find function_call items
        function_calls = [
            item for item in output_items
            if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            # Model didn't call a tool — add its output and break
            _log(f"--- No tool call (turn {turn_num}), ending ---")
            context.add_response_items(output_items)
            break

        # Enforce single tool call per turn — take only the first
        fc = function_calls[0]

        # Filter output_items to only include items up to and including the first function_call
        # (drop any extra function_call items the model emitted)
        filtered_items = []
        for item in output_items:
            filtered_items.append(item)
            if item is fc:
                break
        context.add_response_items(filtered_items)

        # Process the single function call
        done = False
        for fc in [fc]:
            tool_name = fc.name
            tool_arguments = json.loads(fc.arguments)

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

            # Transform for the model
            transformed = transform_tool_output(
                tool_name, tool_arguments, raw_result, mode
            )

            # Show observation in verbose mode
            if isinstance(transformed, str):
                _log(f"--- Observation ---\n{transformed}\n")
            else:
                text_parts = [b["text"] for b in transformed if b.get("type") == "input_text"]
                _log(f"--- Observation ---\n{''.join(text_parts)} [+ image]\n")

            context.add_tool_output(fc.call_id, transformed)

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
