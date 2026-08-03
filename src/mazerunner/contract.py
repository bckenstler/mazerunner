"""The shared agent contract: one forced submit_drag_path tool call.

Every provider receives the same prompt text, the same JSON schema, and the
same normalized-coordinate convention. Provider adapters only translate this
contract into their API's shape.
"""

from __future__ import annotations

import math

TOOL_NAME = "submit_drag_path"

MIN_POINTS = 2
MAX_POINTS = 512

TOOL_DESCRIPTION = (
    "Submit one continuous drag trajectory through the maze as an ordered list "
    "of normalized image coordinates. (0,0) is the top-left corner of the "
    "image and (1,1) is the bottom-right corner. Consecutive points are "
    "connected by straight line segments, so include enough points to follow "
    "every bend of the route."
)

TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "minItems": MIN_POINTS,
            "maxItems": MAX_POINTS,
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["points"],
    "additionalProperties": False,
}

def prompt_text(width: int | None = None, height: int | None = None) -> str:
    """The shared task prompt.

    The default is dimension-free: coordinate grounding (including inferring
    the aspect ratio) is part of the task. Passing width/height produces the
    dimensions-disclosed ablation variant — models that plan in pixel space
    no longer have to estimate the canvas size before normalizing.
    """
    coordinates = (
        "Coordinates are normalized to the image: (0,0) is the top-left "
        "corner, (1,1) is the bottom-right corner."
    )
    if width is not None and height is not None:
        coordinates += (
            f" The image is {width} pixels wide and {height} pixels tall, so "
            f"a pixel position (px, py) normalizes to (px/{width}, py/{height})."
        )
    return (
        "You are looking at a maze rendered as an image. Somewhere in the maze is "
        "a START marker drawn as a cyan circular badge with a white play arrow, "
        "and a GOAL marker drawn as an amber circular badge containing a small "
        "treasure chest.\n\n"
        "Trace one continuous drag path from the start marker to the goal marker:\n"
        "- Begin at the cyan badge and end at the amber badge.\n"
        "- Stay on the traversable route and follow the center of the corridor.\n"
        "- Never cross walls or leave the open route.\n"
        "- Consecutive points are joined by straight segments, so add enough "
        "points that every bend and curve of the route is followed.\n\n"
        f"{coordinates}\n\n"
        "Answer only by calling the submit_drag_path tool exactly once with your "
        "full trajectory. Do not answer in prose."
    )


PROMPT_TEXT = prompt_text()


def validate_submission(arguments: object) -> tuple[list[tuple[float, float]] | None, str | None]:
    """Validate raw tool arguments against the contract.

    Returns (points, None) on success or (None, reason) on rejection. The
    reason is recorded verbatim in the attempt as `schema_error`, so it has to
    name the offending point rather than just the rule.

    Booleans are rejected explicitly: `isinstance(True, int)` is true in
    Python, so a submitted `true` would otherwise validate as the coordinate 1.
    """
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    if "points" not in arguments:
        return None, "missing required 'points' array"
    raw = arguments["points"]
    if not isinstance(raw, list):
        return None, "'points' must be an array"
    if len(raw) < MIN_POINTS:
        return None, f"too few points ({len(raw)} < {MIN_POINTS})"
    if len(raw) > MAX_POINTS:
        return None, f"too many points ({len(raw)} > {MAX_POINTS})"
    points: list[tuple[float, float]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, f"point {i} is not an object"
        for key in ("x", "y"):
            if key not in entry:
                return None, f"point {i} is missing '{key}'"
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"point {i} '{key}' is not a number"
            if not math.isfinite(value):
                return None, f"point {i} '{key}' is not finite"
            if value < 0.0 or value > 1.0:
                return None, f"point {i} '{key}'={value} is outside [0, 1]"
        points.append((float(entry["x"]), float(entry["y"])))
    return points, None


FEEDBACK_CATEGORY_TEXT = {
    "collision": "your path crossed a wall",
    "wrong_start": "your path did not begin on the cyan START badge",
    "stopped_short": "your path never reached the amber GOAL badge",
    "schema_invalid": "your submission did not match the required format",
    "no_tool_call": "you did not call the tool",
    "other": "your path was not accepted",
}


def feedback_text(category: str, stop_xy: tuple[float, float] | None = None) -> str:
    """What the model is told between attempts.

    Deliberately withholds any oracle geometry -- no distance to the true
    route, no direction to move, no hint about where the corridor actually
    goes. It reports only what a real failed drag would reveal: the attempt was
    rejected, why, and where it stopped. Anything more would test instruction
    following rather than closed-loop visual correction.
    """
    reason = FEEDBACK_CATEGORY_TEXT.get(category, FEEDBACK_CATEGORY_TEXT["other"])
    where = ""
    if stop_xy is not None:
        where = (
            f" It stopped at approximately x={stop_xy[0]:.3f}, y={stop_xy[1]:.3f} "
            "in normalized coordinates, marked with a red ⊗."
        )

    return (
        f"That attempt was not accepted: {reason}.{where}\n\n"
        "The second image shows the same maze with your submitted path drawn "
        "over it in red.\n\n"
        "Look again at where your path left the open corridor, then submit a "
        "corrected path by calling submit_drag_path exactly once."
    )
