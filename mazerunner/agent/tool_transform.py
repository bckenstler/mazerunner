"""Transform raw tool results into model-facing content.

Strips internal fields (reward, step_count, position) and provides clean
feedback with only the rendered maze state.
"""

from __future__ import annotations

import json
from typing import Any


def _format_action_str(tool_name: str, tool_arguments: dict) -> str:
    """Format a tool call as a readable string for error messages."""
    if tool_name == "navigate":
        return f"navigate('{tool_arguments.get('directions', '')}')"
    elif tool_name == "drag":
        points = tool_arguments.get("points", [])
        return f"drag({points})"
    return f"{tool_name}()"


def _make_image_content(text: str, rendered_b64: str) -> list[dict]:
    """Build content blocks with text and a base64 PNG image."""
    return [
        {"type": "input_text", "text": text},
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{rendered_b64}",
            "detail": "auto",
        },
    ]


def transform_tool_output(
    tool_name: str,
    tool_arguments: dict,
    raw_result: dict,
    mode: str,
) -> str | list[dict]:
    """Transform a raw tool result into model-facing content.

    Args:
        tool_name: Name of the tool that was called.
        tool_arguments: Arguments passed to the tool.
        raw_result: Raw dict result from MazeEnvironment.
        mode: Rendering mode (text_grid, vision_grid, vision_drag).

    Returns:
        For text_grid: a plain string.
        For vision modes: a list of content blocks with text and image_url.
    """
    # get_maze_info: return formatted text regardless of mode
    if tool_name == "get_maze_info":
        if "error" in raw_result:
            return raw_result["error"]
        lines = [
            f"Grid: {raw_result['grid_rows']}x{raw_result['grid_cols']}",
            f"Start: {raw_result['start']}",
            f"Goal: {raw_result['goal']}",
            f"Mode: {raw_result['mode']}",
        ]
        return "\n".join(lines)

    # Navigate/drag actions
    valid = raw_result.get("valid", False)
    finished = raw_result.get("finished", False)
    rendered = raw_result.get("rendered", "")

    # Finished successfully
    if valid and finished:
        return "Maze complete! You reached the goal."

    # Build feedback text
    if valid:
        position = raw_result.get("position", [])
        if mode == "vision_drag" and position:
            text = f"Valid action. Your position is now [{position[0]:.0f}, {position[1]:.0f}]. Current maze state:"
        else:
            text = "Valid action. Current maze state:"
    else:
        action_str = _format_action_str(tool_name, tool_arguments)
        if mode == "vision_drag":
            text = f"Invalid action: {action_str}. Your position has not changed. Current maze state:"
        else:
            text = f"Invalid action: {action_str}. Current maze state:"

    # Text mode: append ASCII rendering
    if mode == "text_grid":
        return f"{text}\n{rendered}"

    # Vision modes: return content blocks with image
    return _make_image_content(text, rendered)
