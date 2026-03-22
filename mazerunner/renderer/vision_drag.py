"""Vision drag mode renderer — classic maze look with corridors and walls."""

from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

from mazerunner.renderer.base import (
    DragRenderConfig,
    get_color_schema,
    get_opening_side,
    hex_to_rgb,
    parse_cell,
)


def cell_to_pixel_center(
    row: int, col: int, config: DragRenderConfig, margin: int = 0
) -> Tuple[float, float]:
    """Return the pixel center of a cell's corridor area."""
    wt = config.wall_thickness
    cw = config.corridor_width
    cs = config.cell_size
    x = margin + wt + col * cs + cw / 2
    y = margin + wt + row * cs + cw / 2
    return (x, y)


def cell_to_pixel_rect(
    row: int, col: int, config: DragRenderConfig, margin: int = 0
) -> Tuple[int, int, int, int]:
    """Return the bounding box (x0, y0, x1, y1) of a cell's corridor area."""
    wt = config.wall_thickness
    cw = config.corridor_width
    cs = config.cell_size
    x0 = margin + wt + col * cs
    y0 = margin + wt + row * cs
    return (x0, y0, x0 + cw - 1, y0 + cw - 1)


def _image_dimensions(rows: int, cols: int, config: DragRenderConfig) -> Tuple[int, int]:
    """Compute final image dimensions."""
    wt = config.wall_thickness
    cs = config.cell_size
    m = config.margin
    width = 2 * m + wt + cols * cs
    height = 2 * m + wt + rows * cs
    return (width, height)


def render_vision_drag(
    instance: Dict[str, Any], config: DragRenderConfig | None = None
) -> Image.Image:
    """Render a maze instance as a corridor-style PIL Image."""
    if config is None:
        config = DragRenderConfig()

    rows = instance["grid_rows"]
    cols = instance["grid_cols"]
    adjacency = instance["adjacency"]
    start = parse_cell(instance["start"])
    goal = parse_cell(instance["goal"])
    schema = get_color_schema(instance)

    wall_rgb = hex_to_rgb(schema["wall"])
    corridor_rgb = hex_to_rgb(schema["corridor"])
    start_rgb = hex_to_rgb(schema["start"])
    goal_rgb = hex_to_rgb(schema["goal"])
    bg_rgb = hex_to_rgb(schema["background"])

    wt = config.wall_thickness
    cw = config.corridor_width
    cs = config.cell_size
    m = config.margin

    scale = 2 if config.antialias else 1
    sw = wt * scale
    scw = cw * scale
    scs = cs * scale
    sm = m * scale

    final_w, final_h = _image_dimensions(rows, cols, config)
    render_w, render_h = final_w * scale, final_h * scale

    img = Image.new("RGB", (render_w, render_h), wall_rgb)
    draw = ImageDraw.Draw(img)

    # Draw margin border in background color
    if sm > 0:
        # Top
        draw.rectangle([0, 0, render_w - 1, sm - 1], fill=bg_rgb)
        # Bottom
        draw.rectangle([0, render_h - sm, render_w - 1, render_h - 1], fill=bg_rgb)
        # Left
        draw.rectangle([0, 0, sm - 1, render_h - 1], fill=bg_rgb)
        # Right
        draw.rectangle([render_w - sm, 0, render_w - 1, render_h - 1], fill=bg_rgb)

    # Draw corridor for each cell
    for r in range(rows):
        for c in range(cols):
            x0 = sm + sw + c * scs
            y0 = sm + sw + r * scs
            draw.rectangle([x0, y0, x0 + scw - 1, y0 + scw - 1], fill=corridor_rgb)

    # Draw passages (erase walls between connected cells)
    for cell_key, neighbors in adjacency.items():
        r, c = parse_cell(cell_key)
        for n_key in neighbors:
            nr, nc = parse_cell(n_key)
            # Only process right and down to avoid double-drawing
            if nr == r and nc == c + 1:
                # Horizontal passage (right)
                x0 = sm + sw + c * scs + scw
                y0 = sm + sw + r * scs
                draw.rectangle([x0, y0, x0 + sw - 1, y0 + scw - 1], fill=corridor_rgb)
            elif nr == r + 1 and nc == c:
                # Vertical passage (down)
                x0 = sm + sw + c * scs
                y0 = sm + sw + r * scs + scw
                draw.rectangle([x0, y0, x0 + scw - 1, y0 + sw - 1], fill=corridor_rgb)

    # Draw border openings for edge start/goal (one opening per endpoint)
    for cell in [start, goal]:
        r, c = cell
        side = get_opening_side(r, c, rows, cols)
        if side is not None:
            cell_x0 = sm + sw + c * scs
            cell_y0 = sm + sw + r * scs
            if side == "top":
                draw.rectangle([cell_x0, 0, cell_x0 + scw - 1, cell_y0 - 1], fill=corridor_rgb)
            elif side == "bottom":
                draw.rectangle([cell_x0, cell_y0 + scw, cell_x0 + scw - 1, render_h - 1], fill=corridor_rgb)
            elif side == "left":
                draw.rectangle([0, cell_y0, cell_x0 - 1, cell_y0 + scw - 1], fill=corridor_rgb)
            elif side == "right":
                draw.rectangle([cell_x0 + scw, cell_y0, render_w - 1, cell_y0 + scw - 1], fill=corridor_rgb)

    # Draw start and goal letter markers
    font_size = int(scw * 0.7)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    for cell, color, letter in [(start, start_rgb, "S"), (goal, goal_rgb, "G")]:
        cx = sm + sw + cell[1] * scs + scw / 2
        cy = sm + sw + cell[0] * scs + scw / 2
        draw.text((cx, cy), letter, fill=color, font=font, anchor="mm")

    # Downsample if antialiased
    if config.antialias:
        img = img.resize((final_w, final_h), Image.LANCZOS)

    return img
