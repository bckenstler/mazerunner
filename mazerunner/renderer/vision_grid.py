"""Vision grid mode renderer — cell-and-wall grid image."""

from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

from mazerunner.renderer.base import (
    GridRenderConfig,
    get_color_schema,
    get_opening_side,
    hex_to_rgb,
    parse_cell,
)


def _image_dimensions(rows: int, cols: int, config: GridRenderConfig) -> Tuple[int, int]:
    """Compute final image dimensions."""
    wt = config.wall_thickness
    cs = config.cell_size
    m = config.margin
    grid_width = wt * (cols + 1) + cs * cols
    grid_height = wt * (rows + 1) + cs * rows
    return (2 * m + grid_width, 2 * m + grid_height)


def render_vision_grid(
    instance: Dict[str, Any], config: GridRenderConfig | None = None
) -> Image.Image:
    """Render a maze instance as a cell-and-wall grid PIL Image."""
    if config is None:
        config = GridRenderConfig()

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
    cs = config.cell_size
    m = config.margin

    scale = 2 if config.antialias else 1
    sw = wt * scale
    scs = cs * scale
    sm = m * scale

    final_w, final_h = _image_dimensions(rows, cols, config)
    render_w, render_h = final_w * scale, final_h * scale

    # Fill with background, then draw grid area in wall color
    img = Image.new("RGB", (render_w, render_h), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Grid area
    grid_x0 = sm
    grid_y0 = sm
    grid_x1 = render_w - sm - 1
    grid_y1 = render_h - sm - 1
    draw.rectangle([grid_x0, grid_y0, grid_x1, grid_y1], fill=wall_rgb)

    # Draw each cell (all cells in corridor color; markers drawn later as letters)
    for r in range(rows):
        for c in range(cols):
            x0 = sm + sw + c * (scs + sw)
            y0 = sm + sw + r * (scs + sw)
            draw.rectangle([x0, y0, x0 + scs - 1, y0 + scs - 1], fill=corridor_rgb)

    # Draw passages (erase walls between connected cells)
    for cell_key, neighbors in adjacency.items():
        r, c = parse_cell(cell_key)
        for n_key in neighbors:
            nr, nc = parse_cell(n_key)
            # Only process right and down
            if nr == r and nc == c + 1:
                # Horizontal passage
                x0 = sm + sw + c * (scs + sw) + scs
                y0 = sm + sw + r * (scs + sw)
                draw.rectangle([x0, y0, x0 + sw - 1, y0 + scs - 1], fill=corridor_rgb)
            elif nr == r + 1 and nc == c:
                # Vertical passage
                x0 = sm + sw + c * (scs + sw)
                y0 = sm + sw + r * (scs + sw) + scs
                draw.rectangle([x0, y0, x0 + scs - 1, y0 + sw - 1], fill=corridor_rgb)

    # Draw thin gridlines over passages
    sgl = config.gridline_thickness * scale
    for cell_key, neighbors in adjacency.items():
        r, c = parse_cell(cell_key)
        for n_key in neighbors:
            nr, nc = parse_cell(n_key)
            if nr == r and nc == c + 1:
                # Vertical gridline in horizontal passage
                wall_x0 = sm + sw + c * (scs + sw) + scs
                wall_y0 = sm + sw + r * (scs + sw)
                mid_x = wall_x0 + sw // 2 - sgl // 2
                draw.rectangle([mid_x, wall_y0, mid_x + sgl - 1, wall_y0 + scs - 1], fill=wall_rgb)
            elif nr == r + 1 and nc == c:
                # Horizontal gridline in vertical passage
                wall_x0 = sm + sw + c * (scs + sw)
                wall_y0 = sm + sw + r * (scs + sw) + scs
                mid_y = wall_y0 + sw // 2 - sgl // 2
                draw.rectangle([wall_x0, mid_y, wall_x0 + scs - 1, mid_y + sgl - 1], fill=wall_rgb)

    # Draw border openings for edge start/goal (one opening per endpoint)
    for cell in [start, goal]:
        r, c = cell
        side = get_opening_side(r, c, rows, cols)
        if side is not None:
            cell_x0 = sm + sw + c * (scs + sw)
            cell_y0 = sm + sw + r * (scs + sw)
            if side == "top":
                draw.rectangle([cell_x0, 0, cell_x0 + scs - 1, cell_y0 - 1], fill=corridor_rgb)
            elif side == "bottom":
                draw.rectangle([cell_x0, cell_y0 + scs, cell_x0 + scs - 1, render_h - 1], fill=corridor_rgb)
            elif side == "left":
                draw.rectangle([0, cell_y0, cell_x0 - 1, cell_y0 + scs - 1], fill=corridor_rgb)
            elif side == "right":
                draw.rectangle([cell_x0 + scs, cell_y0, render_w - 1, cell_y0 + scs - 1], fill=corridor_rgb)

    # Draw start and goal letter markers
    font_size = int(scs * 0.7)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    for cell, color, letter in [(start, start_rgb, "S"), (goal, goal_rgb, "G")]:
        cx = sm + sw + cell[1] * (scs + sw) + scs / 2
        cy = sm + sw + cell[0] * (scs + sw) + scs / 2
        draw.text((cx, cy), letter, fill=color, font=font, anchor="mm")

    # Downsample if antialiased
    if config.antialias:
        img = img.resize((final_w, final_h), Image.LANCZOS)

    return img
