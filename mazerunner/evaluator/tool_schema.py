"""LLM tool definitions for agentic maze navigation."""

from mazerunner.evaluator.session import SegmentResult, SegmentStatus, SessionResult


SUBMIT_SEGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_segment",
        "description": (
            "Submit a path segment for validation. The segment is a list of [x, y] pixel "
            "coordinates. The first segment must start within the green START circle. "
            "Subsequent segments must continue from near the last accepted point. "
            "Segments that hit walls are rejected."
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
                    "description": (
                        "List of [x, y] pixel coordinates forming the path segment."
                    ),
                }
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Signal that path navigation is complete. Call this when you believe your "
            "path has reached the red GOAL circle. The accumulated path will be scored."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

AGENTIC_TOOLS = [SUBMIT_SEGMENT_TOOL, FINISH_TOOL]


def format_tool_result(result: SegmentResult) -> str:
    """Format a SegmentResult into a human-readable string for the LLM."""
    lines = [f"Status: {result.status.value}"]
    lines.append(f"Reason: {result.reason}")
    if result.violation_point is not None:
        lines.append(
            f"Violation at: ({result.violation_point[0]:.1f}, "
            f"{result.violation_point[1]:.1f})"
        )
    lines.append(
        f"Path so far: {result.num_points_so_far} points, "
        f"{result.path_length_so_far:.1f}px length"
    )
    return "\n".join(lines)


def format_finish_result(result: SessionResult) -> str:
    """Format a SessionResult into a human-readable string for the LLM."""
    lines = [f"Finish reason: {result.finish_reason}"]
    lines.append(f"Path: {len(result.accepted_path)} points")

    stats = result.stats
    lines.append(
        f"Stats: {stats.segments_accepted} accepted, "
        f"{stats.segments_rejected} rejected "
        f"({stats.wall_rejections} wall, {stats.contiguity_rejections} contiguity)"
    )

    if result.eval_result is not None:
        er = result.eval_result
        lines.append(f"Success@0: {er.success.get('0', False)}")
        lines.append(f"Start OK: {er.start_ok}, Goal OK: {er.goal_ok}")
        lines.append(f"Valid fraction@0: {er.valid_frac.get('0', 0.0):.4f}")
        lines.append(f"Min clearance: {er.min_clearance:.2f}")
        lines.append(f"Goal distance: {er.goal_distance:.2f}")
        lines.append(f"Path length: {er.path_length:.2f}")
    else:
        lines.append("No evaluation (path too short or empty)")

    return "\n".join(lines)
