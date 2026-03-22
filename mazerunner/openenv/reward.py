"""Reward computation for maze environments."""

from typing import Any, Dict, Tuple, Union

from mazerunner.navigator.base import InteractionResult, MazeNavigator
from mazerunner.renderer.base import parse_cell


def _bfs_distance(adjacency: Dict[str, list], start: str, goal: str) -> int:
    """BFS shortest distance between two cells in the adjacency graph."""
    if start == goal:
        return 0
    visited = {start}
    queue = [(start, 0)]
    while queue:
        cell, dist = queue.pop(0)
        for neighbor in adjacency.get(cell, []):
            if neighbor == goal:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


def _cell_key(position: Union[Tuple[int, int], Tuple[float, float]]) -> str:
    """Convert a position to an adjacency-dict key string."""
    return f"{int(position[0])},{int(position[1])}"


def _position_to_cell_key(
    position: Union[Tuple[int, int], Tuple[float, float]],
    mode: str,
    instance: Dict[str, Any],
) -> str:
    """Convert navigator position to a cell key for BFS lookup."""
    if mode == "vision_drag":
        # For drag mode, we need to map pixel coords back to grid cell.
        # Use the same logic as DragNavigator._point_in_cell_rect but simplified.
        from mazerunner.renderer.base import DragRenderConfig

        config = DragRenderConfig()
        wt = config.wall_thickness
        m = config.margin
        cs = config.cell_size
        col = int((position[0] - m - wt) / cs)
        row = int((position[1] - m - wt) / cs)
        return f"{row},{col}"
    else:
        return _cell_key(position)


def compute_reward(
    navigator: MazeNavigator,
    instance: Dict[str, Any],
    result: InteractionResult,
    prev_position: Union[Tuple[int, int], Tuple[float, float]],
    total_interactions: int,
    reward_mode: str = "sparse",
    mode: str = "text_grid",
    max_steps: int | None = None,
) -> float:
    """Compute reward based on the reward mode.

    Args:
        navigator: Current maze navigator.
        instance: Maze instance dict.
        result: InteractionResult from the latest action.
        prev_position: Position before the action.
        total_interactions: Total interactions so far (including current).
        reward_mode: One of 'sparse', 'shaped', 'efficiency'.
        mode: Environment mode ('text_grid', 'vision_grid', 'vision_drag').
        max_steps: Max allowed steps (for timeout penalty).

    Returns:
        Float reward value.
    """
    if reward_mode == "sparse":
        if result.finished:
            return 1.0
        if max_steps is not None and total_interactions >= max_steps:
            return -1.0
        return 0.0

    elif reward_mode == "shaped":
        if not result.valid:
            return 0.0
        adjacency = instance["adjacency"]
        goal = instance["goal"]
        optimal_len = len(instance["shortest_path_cells"]) - 1
        if optimal_len <= 0:
            return 0.0

        prev_key = _position_to_cell_key(prev_position, mode, instance)
        curr_key = _position_to_cell_key(result.position, mode, instance)

        prev_dist = _bfs_distance(adjacency, prev_key, goal)
        curr_dist = _bfs_distance(adjacency, curr_key, goal)

        if prev_dist < 0 or curr_dist < 0:
            return 0.0
        return (prev_dist - curr_dist) / optimal_len

    elif reward_mode == "efficiency":
        if result.finished:
            optimal_len = len(instance["shortest_path_cells"]) - 1
            if optimal_len <= 0:
                return 1.0
            return min(1.0, optimal_len / total_interactions)
        if max_steps is not None and total_interactions >= max_steps:
            return 0.0
        return 0.0

    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
