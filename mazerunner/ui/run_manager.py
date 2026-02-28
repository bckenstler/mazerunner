"""Live run engine for agentic maze navigation with turn-by-turn streaming."""

import asyncio
import base64
import json
import os
import random
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import io

import numpy as np
import litellm
from PIL import Image, ImageDraw

from mazerunner.common.rle import decode_rle
from mazerunner.evaluator.session import MazeSession, SegmentStatus
from mazerunner.evaluator.tool_schema import (
    AGENTIC_TOOLS,
    format_finish_result,
    format_tool_result,
)

# Reuse the system prompt template from the agentic runner
AGENTIC_SYSTEM_PROMPT = """\
You are navigating a maze. The image is {width}x{height} pixels \
(origin top-left, X right, Y down).
The maze is a {rows}x{cols} grid with:
- GREEN circle = START (centered near ({start_x}, {start_y}))
- RED circle = GOAL (centered near ({goal_x}, {goal_y}))
- White corridors to travel through, dark walls to avoid

Navigate from the green START to the red GOAL by submitting path segments \
using the submit_segment tool.
Each segment is a list of [x, y] pixel coordinates.

Rules:
- Your first segment must start within the green START circle.
- Each subsequent segment must continue from near your last accepted point.
- Segments that pass through walls will be rejected. Use the feedback to \
adjust your path.
- Stay centered in corridors and include waypoints at every turn.
- When you believe you've reached the red GOAL, call the finish tool.

Strategy:
- Break the path into manageable segments rather than one long path.
- If a segment is rejected, try a different route from your last accepted \
position.
- Pay attention to wall violation coordinates in rejection feedback."""


def _encode_image_base64(path: str) -> str:
    """Read a PNG file and return a base64-encoded data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _render_progress_image(
    base_image: Image.Image,
    accepted_segments: list[list[list[float]]],
    rejected_segments: list[list[list[float]]],
    violation_points: list[list[float]],
) -> str:
    """Render path progress onto the maze image and return as base64 data URI.

    Draws accepted segments in green, rejected in red, and violation points
    as red circles — matching the UI canvas overlay.
    """
    img = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw accepted segments (green)
    for seg in accepted_segments:
        if len(seg) >= 2:
            flat = [(int(p[0]), int(p[1])) for p in seg]
            draw.line(flat, fill=(78, 204, 163, 220), width=3)

    # Draw rejected segments (red)
    for seg in rejected_segments:
        if len(seg) >= 2:
            flat = [(int(p[0]), int(p[1])) for p in seg]
            draw.line(flat, fill=(233, 69, 96, 180), width=2)

    # Draw violation points (red circles)
    for vp in violation_points:
        x, y = int(vp[0]), int(vp[1])
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(233, 69, 96, 230))

    img = Image.alpha_composite(img, overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _compute_region_center(mask_rle: dict) -> tuple[int, int]:
    """Compute (x, y) center of a region from its RLE-encoded mask."""
    mask = decode_rle(mask_rle)
    ys, xs = np.where(mask)
    return int(xs.mean()), int(ys.mean())


def _format_system_prompt(gt_data: dict, image_size: dict) -> str:
    """Fill in the system prompt template from GT metadata + verified image size."""
    width = image_size["w"]
    height = image_size["h"]
    rows = gt_data["difficulty"]["grid_rows"]
    cols = gt_data["difficulty"]["grid_cols"]
    start_x, start_y = _compute_region_center(gt_data["regions"]["start_mask_rle"])
    goal_x, goal_y = _compute_region_center(gt_data["regions"]["goal_mask_rle"])
    return AGENTIC_SYSTEM_PROMPT.format(
        width=width, height=height, rows=rows, cols=cols,
        start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y,
    )


async def run_maze_live(
    maze_id: str,
    model: str,
    image_dir: str,
    gt_dir: str,
    on_turn: Callable[[dict], Awaitable[None]],
    max_turns: int = 30,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    api_base: Optional[str] = None,
) -> dict:
    """Run a maze in agentic mode, streaming turns via on_turn callback.

    Returns the complete run log dict.
    """
    image_path = os.path.join(image_dir, f"{maze_id}.png")
    gt_path = os.path.join(gt_dir, f"{maze_id}.json")

    with open(gt_path) as f:
        gt_data = json.load(f)

    # PIL-verified image dimensions
    base_image = Image.open(image_path)
    actual_w, actual_h = base_image.size
    image_size = {"w": actual_w, "h": actual_h}

    system_prompt = _format_system_prompt(gt_data, image_size)
    image_uri = _encode_image_base64(image_path)

    session = MazeSession(gt_data)

    # Track segments for progress image rendering
    accepted_segments: list[list[list[float]]] = []
    rejected_segments: list[list[list[float]]] = []
    violation_points: list[list[float]] = []

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {
                    "type": "text",
                    "text": (
                        "Navigate this maze from START (green) to GOAL "
                        "(red). Use submit_segment to submit path "
                        "segments and finish when done."
                    ),
                },
            ],
        },
    ]

    # Thinking models need higher token budgets
    model_lower = model.lower()
    is_thinking_model = any(
        model_lower.startswith(p) for p in ("gpt-5", "o3", "o4", "o1")
    )
    effective_max_tokens = max(max_tokens, 65536) if is_thinking_model else max_tokens

    timestamp = datetime.now(timezone.utc).isoformat()
    run_log = {
        "maze_id": maze_id,
        "model": model,
        "timestamp": timestamp,
        "image_path": image_path,
        "gt_path": gt_path,
        "image_size": image_size,
        "grid": {
            "rows": gt_data["difficulty"]["grid_rows"],
            "cols": gt_data["difficulty"]["grid_cols"],
        },
        "system_prompt": system_prompt,
        "turns": [],
        "final_result": None,
    }

    session_result = None
    turn = 0
    accepted_path_so_far: list[list[float]] = []
    progress_msg_idx: int | None = None  # index of last progress image in messages

    while turn < max_turns:
        turn += 1

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "tools": AGENTIC_TOOLS,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if api_base:
            kwargs["api_base"] = api_base

        # Retry with exponential backoff
        max_retries = 5
        base_delay = 2.0
        response = None
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(**kwargs)
                break
            except litellm.RateLimitError as e:
                last_error = e
                if attempt == max_retries:
                    break
                delay = min(base_delay * (2 ** attempt), 60.0) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 60.0) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    break

        if response is None:
            if last_error is not None:
                raise RuntimeError(f"API error: {last_error}")
            break

        message = response.choices[0].message
        messages.append(message)

        content = getattr(message, "content", "") or ""
        tool_calls = getattr(message, "tool_calls", None)

        turn_data: dict = {
            "turn": turn,
            "role": "assistant",
            "content": content,
            "tool_calls": [],
            "tool_results": [],
        }

        if not tool_calls:
            # Model didn't use tools — stream the turn and break
            run_log["turns"].append(turn_data)
            await on_turn(turn_data)
            break

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            tc_data = {
                "id": tc.id,
                "name": fn_name,
                "arguments": fn_args,
            }
            turn_data["tool_calls"].append(tc_data)

            if fn_name == "submit_segment":
                points = fn_args.get("points", [])
                seg_result = session.submit_segment(points)
                tool_response = format_tool_result(seg_result)

                # Track segments for progress image and path overlay
                if seg_result.status == SegmentStatus.ACCEPTED:
                    accepted_segments.append(points)
                    accepted_path_so_far.extend(points)
                else:
                    rejected_segments.append(points)
                if seg_result.violation_point is not None:
                    violation_points.append(list(seg_result.violation_point))
                segment_points = points

                tr_data = {
                    "tool_call_id": tc.id,
                    "status": seg_result.status.value,
                    "reason": seg_result.reason,
                    "violation_point": (
                        list(seg_result.violation_point)
                        if seg_result.violation_point
                        else None
                    ),
                    "segment_points": segment_points,
                    "path_so_far": list(accepted_path_so_far),
                    "num_points": seg_result.num_points_so_far,
                    "path_length": seg_result.path_length_so_far,
                }
                turn_data["tool_results"].append(tr_data)

            elif fn_name == "finish":
                session_result = session.finish()
                tool_response = format_finish_result(session_result)

                stats = session_result.stats
                eval_dict = None
                if session_result.eval_result is not None:
                    er = session_result.eval_result
                    eval_dict = {
                        "success": er.success,
                        "start_ok": er.start_ok,
                        "goal_ok": er.goal_ok,
                        "valid_frac": er.valid_frac,
                        "min_clearance": er.min_clearance,
                        "goal_distance": er.goal_distance,
                        "path_length": er.path_length,
                    }

                tr_data = {
                    "tool_call_id": tc.id,
                    "status": "FINISHED",
                    "reason": session_result.finish_reason,
                    "stats": {
                        "segments_accepted": stats.segments_accepted,
                        "segments_rejected": stats.segments_rejected,
                        "wall_rejections": stats.wall_rejections,
                        "contiguity_rejections": stats.contiguity_rejections,
                    },
                    "eval_result": eval_dict,
                }
                turn_data["tool_results"].append(tr_data)
            else:
                tool_response = f"Unknown tool: {fn_name}"
                tr_data = {
                    "tool_call_id": tc.id,
                    "status": "ERROR",
                    "reason": tool_response,
                }
                turn_data["tool_results"].append(tr_data)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_response,
            })

        # Re-inject maze image with progress overlay (replace previous to avoid
        # accumulating base64 images in the messages list which eats memory)
        if session_result is None:
            progress_uri = _render_progress_image(
                base_image, accepted_segments, rejected_segments, violation_points,
            )
            progress_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": progress_uri}},
                    {"type": "text", "text": "Here is the maze with your progress so far. Green lines are accepted path segments. Red lines are rejected segments. Continue navigating."},
                ],
            }
            if progress_msg_idx is not None:
                messages[progress_msg_idx] = progress_msg
            else:
                messages.append(progress_msg)
                progress_msg_idx = len(messages) - 1

        run_log["turns"].append(turn_data)
        await on_turn(turn_data)

        if session_result is not None:
            break

    # Force-finish if turns exhausted
    if session_result is None:
        try:
            session_result = session.finish()
        except RuntimeError:
            pass

    # Build final result
    if session_result is not None:
        stats = session_result.stats
        eval_dict = None
        if session_result.eval_result is not None:
            er = session_result.eval_result
            eval_dict = {
                "success": er.success,
                "start_ok": er.start_ok,
                "goal_ok": er.goal_ok,
                "valid_frac": er.valid_frac,
                "min_clearance": er.min_clearance,
                "goal_distance": er.goal_distance,
                "path_length": er.path_length,
            }
        run_log["final_result"] = {
            "finish_reason": session_result.finish_reason,
            "stats": {
                "segments_accepted": stats.segments_accepted,
                "segments_rejected": stats.segments_rejected,
                "wall_rejections": stats.wall_rejections,
                "contiguity_rejections": stats.contiguity_rejections,
            },
            "eval_result": eval_dict,
        }

    return run_log


def save_run_log(run_log: dict, runs_dir: str) -> str:
    """Save a run log to runs/{model_name}/{maze_id}_{timestamp}.json.

    Returns the path to the saved file.
    """
    model_name = run_log["model"].replace("/", "_").replace(":", "_")
    maze_id = run_log["maze_id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(runs_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{maze_id}_{ts}.json"
    filepath = os.path.join(model_dir, filename)

    with open(filepath, "w") as f:
        json.dump(run_log, f, indent=2)

    return filepath
