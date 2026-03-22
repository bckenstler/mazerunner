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

_NAVIGATE_SINGLE_SCHEMA = {
    "type": "function",
    "name": "navigate",
    "description": (
        "Move one step in the maze. "
        "Pass a single direction: U (up), D (down), L (left), or R (right)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directions": {
                "type": "string",
                "enum": ["U", "D", "L", "R"],
                "description": "A single direction character.",
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


def get_tool_schemas(mode: str, single_step: bool = False) -> list[dict]:
    """Get tool schemas for the given mode (OpenAI Responses API format).

    Args:
        mode: One of text_grid, vision_grid, vision_drag.
        single_step: If True, use single-step navigate schema for grid modes.

    Returns:
        List of OpenAI Responses API tool definitions.
    """
    if mode in ("text_grid", "vision_grid"):
        return [_NAVIGATE_SINGLE_SCHEMA if single_step else _NAVIGATE_SCHEMA]
    elif mode == "vision_drag":
        return [_DRAG_SCHEMA]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_anthropic_tool_schemas(mode: str, single_step: bool = False) -> list[dict]:
    """Get tool schemas in Anthropic Messages API format.

    Uses ``input_schema`` instead of ``parameters`` and omits the
    ``type: "function"`` wrapper.

    Args:
        mode: One of text_grid, vision_grid, vision_drag.
        single_step: If True, use single-step navigate schema for grid modes.

    Returns:
        List of Anthropic tool definitions with input_schema.
    """
    schemas = get_tool_schemas(mode, single_step=single_step)
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in schemas
    ]


def get_chat_tool_schemas(mode: str, single_step: bool = False) -> list[dict]:
    """Get tool schemas in Chat Completions format (Fireworks / OpenAI Chat).

    Wraps the Responses API schemas in the ``{"type":"function","function":{...}}``
    envelope expected by the Chat Completions API.

    Args:
        mode: One of text_grid, vision_grid, vision_drag.
        single_step: If True, use single-step navigate schema for grid modes.

    Returns:
        List of Chat Completions tool definitions.
    """
    schemas = get_tool_schemas(mode, single_step=single_step)
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in schemas
    ]


def get_gemini_tools(mode: str, single_step: bool = False):
    """Get Gemini tool declarations for the given mode.

    Converts the existing schemas into Gemini FunctionDeclaration objects
    wrapped in a Tool.

    Args:
        mode: One of text_grid, vision_grid, vision_drag.
        single_step: If True, use single-step navigate schema for grid modes.

    Returns:
        A google.genai.types.Tool with appropriate function declarations.
    """
    from google.genai import types

    schemas = get_tool_schemas(mode, single_step=single_step)
    declarations = []
    for schema in schemas:
        declarations.append(types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters_json_schema=schema["parameters"],
        ))
    return types.Tool(function_declarations=declarations)
