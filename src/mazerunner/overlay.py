"""Evaluator-only overlays: submitted or reference trajectory over the maze.

Never shown to models.
"""

from __future__ import annotations

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
