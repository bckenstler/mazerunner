"""Observation model for maze OpenEnv environment."""

from pydantic import BaseModel


class MazeObservation(BaseModel):
    rendered: str
    mode: str
    position: list[float]
    valid: bool
    finished: bool
    steps_applied: int
    reward: float
    maze_id: str
    step_count: int
    done: bool
