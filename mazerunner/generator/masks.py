"""Convert logical maze to pixel masks."""

from typing import List, Tuple

import numpy as np

from mazerunner.common.types import Cell, DifficultyConfig, MazeGrid, RenderConfig
from mazerunner.generator.placement import opening_center, opening_pixel_rect


def compute_cell_pixel_bounds(
    row: int, col: int, config: RenderConfig, grid_rows: int, grid_cols: int
) -> Tuple[int, int, int, int]:
    """Returns (y_min, y_max, x_min, x_max) for the cell interior."""
    cell_size = config.corridor_width + config.wall_thickness
    maze_origin_x = config.chrome_width_left + config.wall_thickness
    maze_origin_y = config.chrome_height_top + config.wall_thickness

    x_min = maze_origin_x + col * cell_size
    y_min = maze_origin_y + row * cell_size

    x_max = x_min + config.corridor_width
    y_max = y_min + config.corridor_width

    return (y_min, y_max, x_min, x_max)


def compute_cell_center(
    row: int, col: int, config: RenderConfig, grid_rows: int, grid_cols: int
) -> Tuple[float, float]:
    """Returns (x, y) pixel center of cell."""
    y_min, y_max, x_min, x_max = compute_cell_pixel_bounds(row, col, config, grid_rows, grid_cols)
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def generate_wall_mask(maze: MazeGrid, config: RenderConfig) -> np.ndarray:
    """Returns (H, W) bool where True = wall."""
    h, w = config.image_height, config.image_width
    mask = np.ones((h, w), dtype=bool)

    # Carve cell interiors
    for r in range(maze.rows):
        for c in range(maze.cols):
            y_min, y_max, x_min, x_max = compute_cell_pixel_bounds(
                r, c, config, maze.rows, maze.cols
            )
            mask[y_min:y_max, x_min:x_max] = False

    # Carve passages between cells
    cell_size = config.corridor_width + config.wall_thickness
    maze_origin_x = config.chrome_width_left + config.wall_thickness
    maze_origin_y = config.chrome_height_top + config.wall_thickness

    for passage in maze.passages:
        cells = list(passage)
        r1, c1 = cells[0]
        r2, c2 = cells[1]

        if r1 == r2:
            # Horizontal passage (same row, adjacent columns)
            min_col = min(c1, c2)
            # Carve the wall strip between the two cells
            x_start = maze_origin_x + min_col * cell_size + config.corridor_width
            x_end = x_start + config.wall_thickness
            y_start = maze_origin_y + r1 * cell_size
            y_end = y_start + config.corridor_width
            mask[y_start:y_end, x_start:x_end] = False
        else:
            # Vertical passage (same col, adjacent rows)
            min_row = min(r1, r2)
            y_start = maze_origin_y + min_row * cell_size + config.corridor_width
            y_end = y_start + config.wall_thickness
            x_start = maze_origin_x + c1 * cell_size
            x_end = x_start + config.corridor_width
            mask[y_start:y_end, x_start:x_end] = False

    return mask


def carve_outer_openings(
    wall_mask: np.ndarray, maze: MazeGrid, config: RenderConfig
) -> None:
    """Carve corridor-width gaps through the outer wall at start/goal edge positions.

    Modifies wall_mask in place.
    """
    if maze.start_edge:
        y_min, y_max, x_min, x_max = opening_pixel_rect(
            maze.start, maze.start_edge, config, maze.rows, maze.cols
        )
        wall_mask[y_min:y_max, x_min:x_max] = False

    if maze.goal_edge:
        y_min, y_max, x_min, x_max = opening_pixel_rect(
            maze.goal, maze.goal_edge, config, maze.rows, maze.cols
        )
        wall_mask[y_min:y_max, x_min:x_max] = False


def generate_free_space_mask(wall_mask: np.ndarray) -> np.ndarray:
    """Returns inverse of wall mask."""
    return ~wall_mask


def generate_region_mask(
    center: Tuple[float, float], radius: float, shape: Tuple[int, int]
) -> np.ndarray:
    """Circular bool mask. center is (x, y), shape is (H, W)."""
    h, w = shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)
    return dist <= radius


def solution_cells_to_polyline(
    solution_path: List[Cell],
    config: RenderConfig,
    grid_rows: int,
    grid_cols: int,
    start_edge: str = "",
    goal_edge: str = "",
) -> List[Tuple[float, float]]:
    """Convert cell path to pixel polyline through cell centers.

    When start_edge/goal_edge are set, prepend/append the opening center
    so the polyline extends to the wall opening.
    """
    polyline = []

    if start_edge and solution_path:
        pt = opening_center(solution_path[0], start_edge, config, grid_rows, grid_cols)
        polyline.append(pt)

    for cell in solution_path:
        center = compute_cell_center(cell[0], cell[1], config, grid_rows, grid_cols)
        polyline.append(center)

    if goal_edge and solution_path:
        pt = opening_center(solution_path[-1], goal_edge, config, grid_rows, grid_cols)
        polyline.append(pt)

    return polyline


def compute_image_size(
    config_difficulty: DifficultyConfig,
    chrome_height_top: int,
    chrome_width_left: int,
    target_width: int = 1920,
    target_height: int = 1080,
) -> Tuple[int, int, RenderConfig]:
    """Compute image dimensions to fill target resolution (default 1920x1080).

    Scales corridor_width and wall_thickness so the maze fills the target
    resolution as closely as possible while keeping integer pixel values.
    The image is fit to the target so that neither dimension exceeds it.
    """
    rows = config_difficulty.grid_rows
    cols = config_difficulty.grid_cols
    cw = config_difficulty.corridor_width
    wt = config_difficulty.wall_thickness

    # Original (unscaled) maze dimensions
    orig_maze_w = cols * (cw + wt) + wt
    orig_maze_h = rows * (cw + wt) + wt
    orig_w = chrome_width_left + orig_maze_w
    orig_h = chrome_height_top + orig_maze_h

    # Compute scale factor to fit target resolution
    scale = min(target_width / orig_w, target_height / orig_h)

    # Scale wall thickness first (at least 2px), then derive corridor width
    scaled_wt = max(2, round(wt * scale))

    # Compute max corridor_width that fits within target for both dimensions
    avail_w = target_width - chrome_width_left
    avail_h = target_height - chrome_height_top
    max_cw_from_w = (avail_w - scaled_wt) // cols - scaled_wt
    max_cw_from_h = (avail_h - scaled_wt) // rows - scaled_wt
    scaled_cw = max(4, min(max_cw_from_w, max_cw_from_h))

    # Recompute actual image size from scaled values
    maze_width = cols * (scaled_cw + scaled_wt) + scaled_wt
    maze_height = rows * (scaled_cw + scaled_wt) + scaled_wt
    image_width = chrome_width_left + maze_width
    image_height = chrome_height_top + maze_height

    render_config = RenderConfig(
        image_width=image_width,
        image_height=image_height,
        corridor_width=scaled_cw,
        wall_thickness=scaled_wt,
        chrome_height_top=chrome_height_top,
        chrome_width_left=chrome_width_left,
        theme_name="",
    )

    return image_width, image_height, render_config
