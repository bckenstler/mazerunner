"""Closed-loop feedback mode (H6): can a model correct its own failed drag?

Each episode is a real multi-turn conversation. The model attempts the maze;
if it fails, it is shown its own path drawn over the same image with a ⊗ where
it left the corridor, told only the failure category, and asked again. It stops
at the first success or after `max_attempts` turns.

This is a separate leaderboard from the one-shot benchmark — a model that gets
four tries is answering an easier question — so results are written as
episodes, not attempts.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

from PIL import Image

from .contract import PROMPT_TEXT, feedback_text
from .evaluator import evaluate_task
from .metrics import derive
from .overlay import attempt_overlay_bytes
from .providers.base import ProviderError, is_retryable, serving_stack

MAX_ATTEMPTS = 4


def _stop_point_normalized(task: dict, evaluation) -> tuple[float, float] | None:
    first = getattr(evaluation, "first_collision", None)
    if not isinstance(first, dict):
        return None
    width, height = task["width"], task["height"]
    return (first["x_px"] / (width - 1), first["y_px"] / (height - 1))


def _call(adapter, *, prior=None, feedback=None, png_bytes=None, prompt=None, tries=3):
    """One turn, with retries scoped to that turn.

    Retrying at the episode level would replay and re-bill every earlier turn,
    so a transport failure on turn 3 must only re-send turn 3.
    """
    last = None
    for attempt in range(tries):
        try:
            if prior is None:
                return adapter.run(png_bytes, prompt), None
            return adapter.continue_run(prior, feedback, png_bytes), None
        except ProviderError as exc:
            last = str(exc)
            if not is_retryable(exc) or attempt == tries - 1:
                return None, last
            time.sleep(2**attempt)
    return None, last


def run_episode(
    adapter,
    provider_name: str,
    task_id: str,
    task_dir: Path,
    task: dict,
    mask,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> list[dict]:
    """Run one task to first success or `max_attempts`. Returns one row per turn."""
    base_png = (task_dir / task["image_file"]).read_bytes()
    base_image = Image.open(task_dir / task["image_file"])
    episode_id = f"{provider_name}/{task_id}"
    rows: list[dict] = []
    prior = None
    feedback = None

    for turn in range(max_attempts):
        if turn == 0:
            response, transport_error = _call(adapter, png_bytes=base_png, prompt=PROMPT_TEXT)
        else:
            response, transport_error = _call(
                adapter, prior=prior, feedback=feedback, png_bytes=overlay_png
            )

        row = {
            "episode_id": episode_id,
            "provider": provider_name,
            "model": adapter.model,
            "maze": task_id,
            "turn": turn,
            "task_dir": str(task_dir),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        if response is None:
            row["error"] = f"transport failure: {transport_error}"
            row["stop_reason"] = "transport"
            rows.append(row)
            break

        evaluation = (
            evaluate_task(task, mask, response.tool_arguments)
            if response.tool_arguments is not None
            else None
        )
        row.update(
            {
                "latency_s": round(response.latency_s, 2),
                "usage": response.usage,
                "response_id": response.response_id,
                "provider_error": response.error,
                "submission": response.tool_arguments,
                "reasoning": response.reasoning,
                "raw_response": response.raw,
                "serving_stack": serving_stack(response.raw),
                "evaluation": evaluation.to_dict() if evaluation is not None else None,
            }
        )
        row["derived"] = derive(task, row)
        rows.append(row)

        if evaluation is not None and evaluation.success:
            row["stop_reason"] = "success"
            break
        if turn == max_attempts - 1:
            row["stop_reason"] = "exhausted"
            break

        # Build the next turn's feedback from this failure.
        overlay_png = attempt_overlay_bytes(
            base_image, task, response.tool_arguments, evaluation
        )
        if overlay_png is None:
            # Nothing to show back (no usable path); the episode cannot continue
            # as a *visual* correction loop, which is what H6 is about.
            row["stop_reason"] = "no_overlay"
            break
        category = row["derived"]["failure_category"]
        feedback = feedback_text(category, _stop_point_normalized(task, evaluation))
        prior = response
        row["stop_reason"] = "retry"

    return rows


def episode_summary(rows: list[dict]) -> dict:
    """Per-episode outcome: did it ever succeed, and after how many turns."""
    turns = [r for r in rows if not r.get("error")]
    success_turn = next(
        (r["turn"] for r in turns if (r.get("evaluation") or {}).get("success")), None
    )
    return {
        "episode_id": rows[0]["episode_id"] if rows else None,
        "maze": rows[0]["maze"] if rows else None,
        "turns_used": len(turns),
        "solved": success_turn is not None,
        "solved_on_turn": success_turn,
        "route_progress_by_turn": [
            (r.get("derived") or {}).get("route_progress") for r in turns
        ],
        "stop_reason": rows[-1].get("stop_reason") if rows else None,
    }


def write_episodes(rows: list[dict], summaries: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.jsonl").open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
    with (out_dir / "episode-summaries.jsonl").open("a") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary) + "\n")
