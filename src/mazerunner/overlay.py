"""Overlays: submitted or reference trajectory drawn over the maze.

Used two ways. For scoring runs these are evaluator-only artifacts. In feedback
mode the model is shown the overlay of its *own* failed attempt — the red path
it drew and a ⊗ where it left the corridor — which is what a real drag would
show, and carries no oracle geometry.
"""

from __future__ import annotations

import io as _io

from PIL import Image, ImageDraw

GREEN = (46, 204, 64)
RED = (255, 65, 54)
WHITE = (255, 255, 255)


def render_overlay(
    base: Image.Image,
    points_px: list[tuple[float, float]],
    *,
    success: bool,
    collision_px: tuple[float, float] | None = None,
) -> Image.Image:
    img = base.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    color = GREEN if success else RED
    if len(points_px) >= 2:
        # White outer stroke keeps the trajectory visible across styles.
        draw.line(points_px, fill=WHITE, width=7, joint="curve")
        draw.line(points_px, fill=color, width=3, joint="curve")
    for point, ring in ((points_px[0], WHITE), (points_px[-1], WHITE)):
        x, y = point
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=ring, width=2)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=2)
    if collision_px is not None:
        x, y = collision_px
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=WHITE, width=4)
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=RED, width=2)
        draw.line([(x - 6, y - 6), (x + 6, y + 6)], fill=RED, width=3)
        draw.line([(x - 6, y + 6), (x + 6, y - 6)], fill=RED, width=3)
    return img


def submitted_points_px(task: dict, arguments: dict | None) -> list[tuple[float, float]] | None:
    """Denormalize a submission to pixel space, or None if it is unusable."""
    if not isinstance(arguments, dict):
        return None
    points = arguments.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        width, height = task["width"], task["height"]
        return [(p["x"] * (width - 1), p["y"] * (height - 1)) for p in points]
    except (TypeError, KeyError):
        return None


def attempt_overlay(
    base: Image.Image, task: dict, arguments: dict | None, evaluation
) -> Image.Image | None:
    """The model's own attempt drawn over the maze, with ⊗ at the stop point."""
    points_px = submitted_points_px(task, arguments)
    if points_px is None:
        return None
    collision = None
    first = getattr(evaluation, "first_collision", None)
    if isinstance(first, dict):
        collision = (first["x_px"], first["y_px"])
    return render_overlay(
        base,
        points_px,
        success=getattr(evaluation, "success", False),
        collision_px=collision,
    )


def attempt_overlay_bytes(
    base: Image.Image, task: dict, arguments: dict | None, evaluation
) -> bytes | None:
    """PNG bytes of the attempt overlay, for sending back to the model."""
    overlay = attempt_overlay(base, task, arguments, evaluation)
    if overlay is None:
        return None
    buffer = _io.BytesIO()
    overlay.save(buffer, format="PNG")
    return buffer.getvalue()
