"""PIL-based maze renderer."""

from PIL import Image, ImageDraw, ImageFont

from mazerunner.common.types import MazeGrid, RenderConfig
from mazerunner.generator.placement import opening_center, opening_pixel_rect
from mazerunner.generator.themes import get_theme


class MazeRenderer:
    def render(self, maze: MazeGrid, config: RenderConfig, antialias: bool = True) -> Image.Image:
        scale = 2 if antialias else 1

        # Working config at scale
        w_width = config.image_width * scale
        w_height = config.image_height * scale
        cw = config.corridor_width * scale
        wt = config.wall_thickness * scale
        chrome_top = config.chrome_height_top * scale
        chrome_left = config.chrome_width_left * scale

        theme = get_theme(config.theme_name)

        img = Image.new("RGBA", (w_width, w_height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        if chrome_top > 0:
            # Draw chrome bar
            draw.rectangle([0, 0, w_width - 1, chrome_top - 1], fill=theme["chrome_bg"] + (255,))
            # Chrome bottom border
            border_y = chrome_top - 1
            draw.line([(0, border_y), (w_width - 1, border_y)], fill=theme["chrome_border"] + (255,), width=max(1, scale))

            # Traffic light circles
            circle_diameter = 12 * scale
            circle_radius = circle_diameter // 2
            spacing = 8 * scale
            cy = chrome_top // 2  # vertically centered in chrome bar
            colors = [theme["traffic_close"], theme["traffic_minimize"], theme["traffic_maximize"]]
            for i, color in enumerate(colors):
                cx = spacing + circle_radius + i * (circle_diameter + spacing)
                draw.ellipse(
                    [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
                    fill=color + (255,),
                )

            # Title text
            try:
                font_size = max(10, 14 * scale)
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
            title = "Maze Puzzle"
            bbox = draw.textbbox((0, 0), title, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = (w_width - text_w) // 2
            text_y = (chrome_top - text_h) // 2
            draw.text((text_x, text_y), title, fill=theme["title_text"] + (255,), font=font)

        # Fill maze area with wall color
        draw.rectangle(
            [chrome_left, chrome_top, w_width - 1, w_height - 1],
            fill=theme["wall"] + (255,),
        )

        # Carve corridors
        cell_size = cw + wt
        maze_origin_x = chrome_left + wt
        maze_origin_y = chrome_top + wt

        for r in range(maze.rows):
            for c in range(maze.cols):
                x_min = maze_origin_x + c * cell_size
                y_min = maze_origin_y + r * cell_size
                draw.rectangle(
                    [x_min, y_min, x_min + cw - 1, y_min + cw - 1],
                    fill=theme["corridor"] + (255,),
                )

        # Carve passages
        for passage in maze.passages:
            cells = list(passage)
            r1, c1 = cells[0]
            r2, c2 = cells[1]

            if r1 == r2:
                # Horizontal passage
                min_col = min(c1, c2)
                x_start = maze_origin_x + min_col * cell_size + cw
                x_end = x_start + wt - 1
                y_start = maze_origin_y + r1 * cell_size
                y_end = y_start + cw - 1
                draw.rectangle([x_start, y_start, x_end, y_end], fill=theme["corridor"] + (255,))
            else:
                # Vertical passage
                min_row = min(r1, r2)
                col = c1 if c1 == c2 else c2
                y_start = maze_origin_y + min_row * cell_size + cw
                y_end = y_start + wt - 1
                x_start = maze_origin_x + col * cell_size
                x_end = x_start + cw - 1
                draw.rectangle([x_start, y_start, x_end, y_end], fill=theme["corridor"] + (255,))

        # Carve outer wall openings for start/goal
        def _draw_opening(cell, edge):
            y_min, y_max, x_min, x_max = opening_pixel_rect(cell, edge, config, maze.rows, maze.cols)
            draw.rectangle(
                [x_min * scale, y_min * scale, x_max * scale - 1, y_max * scale - 1],
                fill=theme["corridor"] + (255,),
            )

        if maze.start_edge:
            _draw_opening(maze.start, maze.start_edge)
        if maze.goal_edge:
            _draw_opening(maze.goal, maze.goal_edge)

        # Draw start marker (circle at opening center if edge, else cell center)
        marker_radius = cw * 0.35
        if maze.start_edge:
            oc = opening_center(maze.start, maze.start_edge, config, maze.rows, maze.cols)
            sx, sy = oc[0] * scale, oc[1] * scale
        else:
            sr, sc = maze.start
            sx = maze_origin_x + sc * cell_size + cw / 2.0
            sy = maze_origin_y + sr * cell_size + cw / 2.0
        draw.ellipse(
            [sx - marker_radius, sy - marker_radius, sx + marker_radius, sy + marker_radius],
            fill=theme["start_marker"] + (255,),
        )

        # Draw goal marker (circle at opening center if edge, else cell center)
        if maze.goal_edge:
            oc = opening_center(maze.goal, maze.goal_edge, config, maze.rows, maze.cols)
            gx, gy = oc[0] * scale, oc[1] * scale
        else:
            gr, gc = maze.goal
            gx = maze_origin_x + gc * cell_size + cw / 2.0
            gy = maze_origin_y + gr * cell_size + cw / 2.0
        draw.ellipse(
            [gx - marker_radius, gy - marker_radius, gx + marker_radius, gy + marker_radius],
            fill=theme["goal_marker"] + (255,),
        )

        # Downscale if antialias
        if antialias:
            img = img.resize(
                (config.image_width, config.image_height),
                Image.LANCZOS,
            )

        return img
