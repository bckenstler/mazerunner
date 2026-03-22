"""JSON serialization for eval results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mazerunner.eval.protocol import EpisodeRecord, EvalResult, StepRecord


def save_eval_result(result: EvalResult, path: str | Path) -> None:
    """Save an EvalResult to a JSON file."""
    Path(path).write_text(json.dumps(asdict(result), indent=2))


def load_eval_result(path: str | Path) -> EvalResult:
    """Load an EvalResult from a JSON file."""
    data = json.loads(Path(path).read_text())
    records = [
        EpisodeRecord(
            maze_id=r["maze_id"],
            success=r["success"],
            steps=r["steps"],
            reward=r["reward"],
            trajectory=[
                StepRecord(
                    action=s["action"],
                    tool_name=s["tool_name"],
                    reward=s["reward"],
                    valid=s["valid"],
                    reasoning=s.get("reasoning", ""),
                    raw_result=s.get("raw_result", {}),
                )
                for s in r["trajectory"]
            ],
            mode=r.get("mode", ""),
            initial_observation=r.get("initial_observation", {}),
        )
        for r in data["records"]
    ]
    return EvalResult(
        run_id=data["run_id"],
        mode=data["mode"],
        model=data["model"],
        num_episodes=data["num_episodes"],
        records=records,
        metrics=data["metrics"],
    )
