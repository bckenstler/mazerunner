"""Drag-based maze navigator using pixel coordinate paths."""

import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from mazerunner.navigator.base import HistoryEntry, InteractionResult, MazeNavigator
from mazerunner.navigator.rendering import render_drag_state
from mazerunner.renderer.base import DragRenderConfig, get_color_schema, hex_to_rgb, parse_cell
from mazerunner.renderer.vision_drag import (
    cell_to_pixel_center,
    cell_to_pixel_rect,
    render_vision_drag,
)


class DragNavigator(MazeNavigator):
    """Navigate a maze using pixel coordinate drag paths.

    Actions are lists of [x, y] coordinate pairs representing a drag path.
    A collision mask is built from a non-antialiased render — any pixel that
    is not wall-colored is passable.

    The first interaction must start within the start cell's corridor rectangle.
    Subsequent interactions must continue from within 1px of the current position.
    All path segments must pass through passable pixels only.

    Args:
        instance: Maze instance dict (as loaded from JSON).
        config: Optional DragRenderConfig. Uses defaults if None.
    """

    def __init__(
        self,
        instance: Dict[str, Any],
        config: DragRenderConfig | None = None,
    ) -> None:
        super().__init__(instance)
        self._config = config or DragRenderConfig()
        self._position_px = cell_to_pixel_center(
            *self._start_cell, self._config, self._config.margin
        )
        self._path: List[Tuple[float, float]] = [self._position_px]
        self._interaction_count = 0
        self._mask: np.ndarray | None = None

    def _build_mask(self) -> np.ndarray:
        """Build collision mask from a non-antialiased render."""
        no_aa_config = DragRenderConfig(
            wall_thickness=self._config.wall_thickness,
            corridor_width=self._config.corridor_width,
            margin=self._config.margin,
            marker_radius_frac=self._config.marker_radius_frac,
            antialias=False,
        )
        img = render_vision_drag(self._instance, no_aa_config)
        arr = np.array(img)
        schema = get_color_schema(self._instance)
        wall_rgb = hex_to_rgb(schema["wall"])
        # Passable = NOT wall color
        passable = ~(
            (arr[:, :, 0] == wall_rgb[0])
            & (arr[:, :, 1] == wall_rgb[1])
            & (arr[:, :, 2] == wall_rgb[2])
        )
        return passable

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = self._build_mask()
        return self._mask

    @property
    def position(self) -> Tuple[float, float]:
        return self._position_px

    def _segment_valid(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if a line segment between two points is entirely passable."""
        mask = self.mask
        h, w = mask.shape
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        steps = max(int(length * 2), 1)
        for i in range(steps + 1):
            t = i / steps
            ix = int(round(p1[0] + t * dx))
            iy = int(round(p1[1] + t * dy))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return False
            if not mask[iy, ix]:
                return False
        return True

    def _point_in_cell_rect(self, px: Tuple[float, float], cell: Tuple[int, int]) -> bool:
        """Check if a pixel point is within a cell's corridor rectangle."""
        x0, y0, x1, y1 = cell_to_pixel_rect(*cell, self._config, self._config.margin)
        return x0 <= px[0] <= x1 and y0 <= px[1] <= y1

    def _make_invalid(self, action: Any) -> InteractionResult:
        result = InteractionResult(
            valid=False,
            position=self._position_px,
            finished=self._finished,
            steps_applied=0,
        )
        self._history.append(HistoryEntry(action=action, result=result))
        return result

    def interact(self, action: List[List[float]]) -> InteractionResult:
        """Process a drag path action as a list of [x, y] coordinate pairs.

        Validates that: the path has at least 2 points, the first point is in
        a valid starting location, and all segments pass through passable pixels.

        Args:
            action: List of [x, y] coordinate pairs defining the drag path.

        Returns:
            InteractionResult with the outcome. ``steps_applied`` is the number
            of path segments (points - 1) if valid, 0 otherwise.
        """
        if len(action) < 2:
            return self._make_invalid(action)

        first_point = (action[0][0], action[0][1])

        # Validate start point
        if self._interaction_count == 0:
            # First interaction: must start in start cell rect
            if not self._point_in_cell_rect(first_point, self._start_cell):
                return self._make_invalid(action)
        else:
            # Subsequent: must be within 1px of current position
            dx = first_point[0] - self._position_px[0]
            dy = first_point[1] - self._position_px[1]
            if math.sqrt(dx * dx + dy * dy) > 1.0:
                return self._make_invalid(action)

        # Validate all segments
        points = [(p[0], p[1]) for p in action]
        for i in range(len(points) - 1):
            if not self._segment_valid(points[i], points[i + 1]):
                return self._make_invalid(action)

        # All valid — update state
        self._interaction_count += 1
        self._path.extend(points[1:])
        self._position_px = points[-1]

        # Check if position is in goal cell rect
        if self._point_in_cell_rect(self._position_px, self._goal_cell):
            self._finished = True

        result = InteractionResult(
            valid=True,
            position=self._position_px,
            finished=self._finished,
            steps_applied=len(points) - 1,
        )
        self._history.append(HistoryEntry(action=action, result=result))
        return result

    def render(self) -> Image.Image:
        """Render the maze with breadcrumb trail and X marker at current position.

        Returns:
            PIL Image with dotted breadcrumbs along the path and an X at the
            current position.
        """
        return render_drag_state(
            self._instance,
            self._path,
            self._position_px,
            self._config,
        )
