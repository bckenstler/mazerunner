"""Base types and abstract class for maze navigation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from PIL import Image

from mazerunner.renderer.base import has_wall, parse_cell


@dataclass
class InteractionResult:
    """Result of a navigation interaction attempt.

    Attributes:
        valid: Whether the action was accepted (no wall collisions or out-of-bounds).
        position: Current position after the interaction (cell or pixel coordinates).
        finished: Whether the agent has reached the goal.
        steps_applied: Number of individual steps/segments applied (0 if invalid).
    """

    valid: bool
    position: Union[Tuple[int, int], Tuple[float, float]]
    finished: bool
    steps_applied: int


@dataclass
class HistoryEntry:
    """Record of one interaction in the navigation history.

    Attributes:
        action: The action that was attempted (string for grid, coordinate list for drag).
        result: The InteractionResult from processing the action.
    """

    action: Any
    result: InteractionResult


class MazeNavigator(ABC):
    """Abstract base class for maze navigators.

    Navigators provide a stateful interaction interface for agent evaluation.
    Each navigator tracks position, history, and whether the goal has been reached.

    Args:
        instance: Maze instance dict (as loaded from JSON).
    """

    def __init__(self, instance: Dict[str, Any]) -> None:
        self._instance = instance
        self._start_cell = parse_cell(instance["start"])
        self._goal_cell = parse_cell(instance["goal"])
        self._adjacency = instance["adjacency"]
        self._rows = instance["grid_rows"]
        self._cols = instance["grid_cols"]
        self._finished = False
        self._history: List[HistoryEntry] = []

    @property
    def finished(self) -> bool:
        """Whether the agent has reached the goal cell."""
        return self._finished

    @property
    def history(self) -> List[HistoryEntry]:
        """Copy of the interaction history."""
        return list(self._history)

    @property
    @abstractmethod
    def position(self) -> Union[Tuple[int, int], Tuple[float, float]]:
        """Current agent position (cell coordinates or pixel coordinates)."""
        ...

    @abstractmethod
    def interact(self, action: Any) -> InteractionResult:
        """Process an agent action and return the result.

        Args:
            action: The action to attempt. Format depends on navigator type.

        Returns:
            An InteractionResult indicating validity, new position, and goal status.
        """
        ...

    @abstractmethod
    def render(self) -> Union[str, Image.Image]:
        """Render the current maze state with the agent's position marked.

        Returns:
            A text string or PIL Image showing the maze with an X at the
            agent's current position.
        """
        ...
