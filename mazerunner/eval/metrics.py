"""Metric computation for eval results."""

from __future__ import annotations

from mazerunner.eval.protocol import EpisodeRecord


def compute_metrics(
    records: list[EpisodeRecord],
    instances: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Compute aggregate metrics from episode records.

    Args:
        records: List of completed episode records.
        instances: Optional mapping of maze_id -> instance dict for efficiency calc.

    Returns:
        Dict with success_rate, avg_steps, avg_steps_successful, avg_reward,
        efficiency, and invalid_action_rate.
    """
    if not records:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_steps_successful": 0.0,
            "avg_reward": 0.0,
            "efficiency": 0.0,
            "invalid_action_rate": 0.0,
        }

    successes = [r for r in records if r.success]
    total_steps = sum(r.steps for r in records)
    total_actions = sum(len(r.trajectory) for r in records)
    total_invalid = sum(1 for r in records for s in r.trajectory if not s.valid)

    success_rate = len(successes) / len(records)
    avg_steps = total_steps / len(records)
    avg_steps_successful = (
        sum(r.steps for r in successes) / len(successes) if successes else 0.0
    )
    avg_reward = sum(r.reward for r in records) / len(records)

    # Efficiency: ratio of optimal path length to actual steps for successes
    efficiency = 0.0
    if successes and instances is not None:
        efficiencies = []
        for r in successes:
            inst = instances.get(r.maze_id)
            if inst is not None:
                optimal = len(inst.get("shortest_path_cells", [])) - 1
                if optimal > 0 and r.steps > 0:
                    efficiencies.append(min(1.0, optimal / r.steps))
        if efficiencies:
            efficiency = sum(efficiencies) / len(efficiencies)

    invalid_action_rate = total_invalid / total_actions if total_actions > 0 else 0.0

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_steps_successful": avg_steps_successful,
        "avg_reward": avg_reward,
        "efficiency": efficiency,
        "invalid_action_rate": invalid_action_rate,
    }
