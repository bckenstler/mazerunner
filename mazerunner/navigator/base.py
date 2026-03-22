"""Base types and abstract class for maze navigation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from PIL import Image

from mazerunner.renderer.base import has_wall, parse_cell


@dataclass
class InteractionResult:
    valid: bool
    position: Union[Tuple[int, int], Tuple[float, float]]
    finished: bool
    steps_applied: int


@dataclass
class HistoryEntry:
    action: Any
    result: InteractionResult


class MazeNavigator(ABC):
    """Abstract base class for maze navigators."""

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
        return self._finished

    @property
    def history(self) -> List[HistoryEntry]:
        return list(self._history)

    @property
    @abstractmethod
    def position(self) -> Union[Tuple[int, int], Tuple[float, float]]:
        ...

    @abstractmethod
    def interact(self, action: Any) -> InteractionResult:
        ...

    @abstractmethod
    def render(self) -> Union[str, Image.Image]:
        ...
