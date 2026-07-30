"""Derived per-attempt metrics: route progress and failure category.

Both take plain dicts rather than the Evaluation dataclass, so the identical
code serves two callers: the runner computing them inline, and the analysis
loader back-filling them onto runs that were collected before these existed.
That is what makes the pre-registered headline metric recoverable from stored
data instead of requiring a re-run.

Route progress answers "how far along the certified route did this attempt get
before it went wrong" — a partial-credit companion to pass@1, never blended
with it.
"""

from __future__ import annotations

from .geometry import densify_polyline, polyline_length, project_onto_polyline
from .io import reference_points_px

# Ordered by precedence: the first condition that holds names the failure.
FAILURE_CATEGORIES = (
    "pass",
    "no_tool_call",
    "schema_invalid",
    "wrong_start",
    "collision",
    "stopped_short",
    "other",
)

DENSIFY_SPACING_PX = 2.0


def _reference_polyline(task: dict) -> list[tuple[float, float]]:
    """The certified route in pixels, densified so projection is smooth.

    The stored path is downsampled to <=200 points, so its own polyline length
    is not `optimal_length_px_geometric`; progress is normalized by this
    polyline's length to keep a completed route at exactly 1.0.
    """
    return densify_polyline(reference_points_px(task), DENSIFY_SPACING_PX)


def _submitted_points_px(task: dict, submission: dict | None) -> list[tuple[float, float]]:
    if not isinstance(submission, dict):
        return []
    points = submission.get("points")
    if not isinstance(points, list):
        return []
    width, height = task["width"], task["height"]
    out = []
    for point in points:
        if isinstance(point, dict) and "x" in point and "y" in point:
            try:
                out.append((float(point["x"]) * (width - 1), float(point["y"]) * (height - 1)))
            except (TypeError, ValueError):
                continue
    return out


def route_progress(
    task: dict,
    evaluation: dict | None,
    submission: dict | None = None,
) -> float:
    """Fraction of the certified route completed before the attempt failed.

    Success is 1.0. Anything that never got started -- no tool call, invalid
    schema, or a path that does not begin at the start badge -- is 0.0, because
    partial credit for a route you never entered would reward guessing.
    """
    if evaluation is None:
        return 0.0
    if evaluation.get("success"):
        return 1.0
    if not evaluation.get("schema_valid") or not evaluation.get("starts_correctly"):
        return 0.0

    reference = _reference_polyline(task)
    total = polyline_length(reference)
    if total <= 0:
        return 0.0

    collision = evaluation.get("first_collision")
    if isinstance(collision, dict) and "x_px" in collision:
        stop = (float(collision["x_px"]), float(collision["y_px"]))
    else:
        # Collision-free but never reached the goal: credit how far the
        # submitted path itself got before stopping.
        points = _submitted_points_px(task, submission)
        if not points:
            return 0.0
        stop = points[-1]

    arclength, _perpendicular, _segment = project_onto_polyline(stop, reference)
    return max(0.0, min(1.0, arclength / total))


def failure_category(
    evaluation: dict | None,
    provider_error: str | None = None,
) -> str:
    """One label per attempt, for the failure-taxonomy breakdown."""
    if evaluation is None:
        return "no_tool_call"
    if evaluation.get("success"):
        return "pass"
    if not evaluation.get("schema_valid"):
        return "schema_invalid"
    if not evaluation.get("starts_correctly"):
        return "wrong_start"
    if not evaluation.get("collision_free"):
        return "collision"
    if not evaluation.get("ends_correctly"):
        return "stopped_short"
    return "other"


def derive(task: dict, row: dict) -> dict:
    """Derived metrics for one attempt row. Safe to call on any row."""
    evaluation = row.get("evaluation")
    return {
        "route_progress": route_progress(task, evaluation, row.get("submission")),
        "failure_category": failure_category(evaluation, row.get("provider_error")),
    }
