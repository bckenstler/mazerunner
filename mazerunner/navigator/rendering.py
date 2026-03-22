"""Rendering overlays for navigator state — X marker and breadcrumbs."""

from typing import Any, Dict, List, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from mazerunner.renderer.base import (
    DragRenderConfig,
    GridRenderConfig,
    get_color_schema,
    hex_to_rgb,
    parse_cell,
)
from mazerunner.renderer.text_grid import render_text_grid
from mazerunner.renderer.vision_drag import cell_to_pixel_center, render_vision_drag
from mazerunner.renderer.vision_grid import render_vision_grid


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font, falling back through available options."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _marker_color(
    position: Tuple[int, int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    schema: Dict[str, str],
) -> Tuple[int, int, int]:
    """Choose X marker color based on position."""
    if position == start:
        return hex_to_rgb(schema["start"])
    elif position == goal:
        return hex_to_rgb(schema["goal"])
    else:
        return hex_to_rgb(schema["wall"])


def render_grid_state(
    instance: Dict[str, Any],
    position: Tuple[int, int],
    mode: str = "text_grid",
    config: GridRenderConfig | None = None,
) -> Union[str, Image.Image]:
    """Render current grid navigator state with X marker overlay.

    Args:
        instance: Maze instance dict (as loaded from JSON).
        position: Current cell position as (row, col).
        mode: "text_grid" for ASCII output, "vision_grid" for PIL Image.
        config: Optional GridRenderConfig for vision_grid mode.

    Returns:
        ASCII string (text_grid) or PIL Image (vision_grid) with X at position.
    """
    if mode == "text_grid":
        return _render_text_grid_state(instance, position)
    else:
        return _render_vision_grid_state(instance, position, config)


def _render_text_grid_state(
    instance: Dict[str, Any],
    position: Tuple[int, int],
) -> str:
    """Overlay X marker on text grid rendering."""
    base = render_text_grid(instance)
    lines = list(base.split("\n"))
    r, c = position
    line_idx = 2 * r + 1
    col_start = 4 * c + 1
    line = lines[line_idx]
    lines[line_idx] = line[:col_start] + " X " + line[col_start + 3:]
    return "\n".join(lines)


def _render_vision_grid_state(
    instance: Dict[str, Any],
    position: Tuple[int, int],
    config: GridRenderConfig | None = None,
) -> Image.Image:
    """Overlay X marker on vision grid rendering."""
    if config is None:
        config = GridRenderConfig()
    img = render_vision_grid(instance, config)
    draw = ImageDraw.Draw(img)

    start = parse_cell(instance["start"])
    goal = parse_cell(instance["goal"])
    schema = get_color_schema(instance)

    cs = config.cell_size
    wt = config.wall_thickness
    m = config.margin

    cx = m + wt + position[1] * (cs + wt) + cs / 2
    cy = m + wt + position[0] * (cs + wt) + cs / 2

    # Erase S/G letter if X is on start or goal cell
    if position == start or position == goal:
        corridor_rgb = hex_to_rgb(schema["corridor"])
        x0 = m + wt + position[1] * (cs + wt)
        y0 = m + wt + position[0] * (cs + wt)
        draw.rectangle([x0, y0, x0 + cs - 1, y0 + cs - 1], fill=corridor_rgb)

    font_size = int(cs * 0.7)
    font = _load_font(font_size)
    color = _marker_color(position, start, goal, schema)
    draw.text((cx, cy), "X", fill=color, font=font, anchor="mm")
    return img


def render_drag_state(
    instance: Dict[str, Any],
    path: List[Tuple[float, float]],
    position: Tuple[float, float],
    config: DragRenderConfig | None = None,
) -> Image.Image:
    """Render current drag navigator state with breadcrumbs and X marker.

    Draws dotted breadcrumbs along the accumulated path and an X marker
    at the current pixel position.

    Args:
        instance: Maze instance dict (as loaded from JSON).
        path: List of (x, y) pixel coordinates visited so far.
        position: Current (x, y) pixel position.
        config: Optional DragRenderConfig. Uses defaults if None.

    Returns:
        PIL Image with breadcrumb trail and X marker.
    """
    if config is None:
        config = DragRenderConfig()
    img = render_vision_drag(instance, config)
    draw = ImageDraw.Draw(img)

    start = parse_cell(instance["start"])
    goal = parse_cell(instance["goal"])
    schema = get_color_schema(instance)
    breadcrumb_rgb = hex_to_rgb(schema["solution_path"])

    # Path coordinates are in logical (final-image) space — no scaling needed.
    # render_vision_drag internally renders at 2x then downsamples back to logical size.

    # Draw dotted breadcrumbs along path
    dot_spacing = max(config.corridor_width // 2, 3)
    dot_radius = 2
    if len(path) >= 2:
        accumulated = 0.0
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = (dx * dx + dy * dy) ** 0.5
            if seg_len == 0:
                continue
            # Walk along segment placing dots
            dist = dot_spacing - accumulated if accumulated < dot_spacing else 0.0
            while dist <= seg_len:
                t = dist / seg_len
                px = x0 + t * dx
                py = y0 + t * dy
                draw.ellipse(
                    [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
                    fill=breadcrumb_rgb,
                )
                dist += dot_spacing
            accumulated = seg_len - (dist - dot_spacing)

    # Determine which cell the position is in for color selection
    wt = config.wall_thickness
    cw = config.corridor_width
    m = config.margin
    col = int((position[0] - m - wt) / config.cell_size)
    row = int((position[1] - m - wt) / config.cell_size)

    # Erase S/G letter if X is on start or goal cell
    if (row, col) == start or (row, col) == goal:
        corridor_rgb = hex_to_rgb(schema["corridor"])
        x0 = m + wt + col * config.cell_size
        y0 = m + wt + row * config.cell_size
        draw.rectangle([x0, y0, x0 + cw - 1, y0 + cw - 1], fill=corridor_rgb)

    # Draw X marker at current position
    fx = position[0]
    fy = position[1]

    font_size = int(cw * 0.7)
    font = _load_font(font_size)
    color = _marker_color((row, col), start, goal, schema)
    draw.text((fx, fy), "X", fill=color, font=font, anchor="mm")
    return img
