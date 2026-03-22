"""Grid-based maze navigator using L/R/U/D cell steps."""

from typing import Any, Dict, Tuple, Union

from PIL import Image

from mazerunner.navigator.base import HistoryEntry, InteractionResult, MazeNavigator
from mazerunner.navigator.rendering import render_grid_state
from mazerunner.renderer.base import GridRenderConfig, has_wall


DIRECTION_MAP = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


class GridNavigator(MazeNavigator):
    """Navigate a maze using L/R/U/D cell steps.

    Actions are strings of direction characters (e.g. "RRD" for right, right, down).
    Movement has atomic semantics: if any step in a multi-character action hits a wall
    or goes out of bounds, the entire action is rejected and no steps are applied.

    Args:
        instance: Maze instance dict (as loaded from JSON).
        render_mode: Rendering mode for ``render()`` — "text_grid" or "vision_grid".
        config: Optional GridRenderConfig for vision_grid rendering.
    """

    def __init__(
        self,
        instance: Dict[str, Any],
        render_mode: str = "text_grid",
        config: GridRenderConfig | None = None,
    ) -> None:
        super().__init__(instance)
        self._position_cell = self._start_cell
        self._render_mode = render_mode
        self._config = config

    @property
    def position(self) -> Tuple[int, int]:
        return self._position_cell

    def interact(self, action: str) -> InteractionResult:
        """Process a direction string action (e.g. "RRDU").

        All steps are validated before any are applied. If any character is
        invalid, hits a wall, or goes out of bounds, the entire action is
        rejected with ``steps_applied=0``.

        Args:
            action: String of direction characters (U/D/L/R).

        Returns:
            InteractionResult with the outcome.
        """
        # Simulate all steps first
        r, c = self._position_cell
        for ch in action:
            dr, dc = DIRECTION_MAP.get(ch, (0, 0))
            if ch not in DIRECTION_MAP:
                result = InteractionResult(
                    valid=False,
                    position=self._position_cell,
                    finished=self._finished,
                    steps_applied=0,
                )
                self._history.append(HistoryEntry(action=action, result=result))
                return result
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= self._rows or nc < 0 or nc >= self._cols:
                result = InteractionResult(
                    valid=False,
                    position=self._position_cell,
                    finished=self._finished,
                    steps_applied=0,
                )
                self._history.append(HistoryEntry(action=action, result=result))
                return result
            if has_wall(self._adjacency, (r, c), (nr, nc)):
                result = InteractionResult(
                    valid=False,
                    position=self._position_cell,
                    finished=self._finished,
                    steps_applied=0,
                )
                self._history.append(HistoryEntry(action=action, result=result))
                return result
            r, c = nr, nc

        # All steps valid
        self._position_cell = (r, c)
        if self._position_cell == self._goal_cell:
            self._finished = True

        result = InteractionResult(
            valid=True,
            position=self._position_cell,
            finished=self._finished,
            steps_applied=len(action),
        )
        self._history.append(HistoryEntry(action=action, result=result))
        return result

    def render(self) -> Union[str, Image.Image]:
        """Render the maze with an X marker at the current position.

        Returns:
            ASCII string (text_grid mode) or PIL Image (vision_grid mode).
        """
        return render_grid_state(
            self._instance,
            self._position_cell,
            self._render_mode,
            self._config,
        )
