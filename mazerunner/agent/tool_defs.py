"""OpenAI Responses API tool schemas for maze navigation."""

from __future__ import annotations


_NAVIGATE_SCHEMA = {
    "type": "function",
    "name": "navigate",
    "description": (
        "Move through the maze using direction characters. "
        "Pass a string of U (up), D (down), L (left), R (right) characters. "
        "Example: 'RRDD' moves right twice then down twice."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directions": {
                "type": "string",
                "description": "String of direction characters (U/D/L/R).",
            }
        },
        "required": ["directions"],
        "additionalProperties": False,
    },
}

_DRAG_SCHEMA = {
    "type": "function",
    "name": "drag",
    "description": (
        "Drag through the maze along a pixel coordinate path. "
        "The path must start at your current position and each segment must be "
        "contiguous (no teleporting). Pass a list of [x, y] coordinate pairs "
        "defining the drag path. The path will be rejected if it crosses a wall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "List of [x, y] pixel coordinate pairs.",
            }
        },
        "required": ["points"],
        "additionalProperties": False,
    },
}

_GET_MAZE_INFO_SCHEMA = {
    "type": "function",
    "name": "get_maze_info",
    "description": "Get information about the current maze (dimensions, start, goal, mode).",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}


def get_tool_schemas(mode: str) -> list[dict]:
    """Get tool schemas for the given mode.

    Args:
        mode: One of text_grid, vision_grid, vision_drag.

    Returns:
        List of OpenAI Responses API tool definitions.
    """
    if mode in ("text_grid", "vision_grid"):
        return [_NAVIGATE_SCHEMA]
    elif mode == "vision_drag":
        return [_DRAG_SCHEMA]
    else:
        raise ValueError(f"Unknown mode: {mode}")
