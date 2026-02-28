"""Run vision LLMs against MazeRunner in agentic (multi-turn) mode.

Usage:
    PYTHONPATH=. python scripts/run_models_agentic.py \
        --image-dir data/dev/images \
        --gt-dir data/dev/gt \
        --model gpt-4o \
        --output results/gpt-4o-agentic.jsonl \
        --concurrency 5 \
        --evaluate --verbose
"""

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Optional

import base64
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
from scripts.run_models import (
    append_jsonl,
    discover_mazes,
    encode_image_base64,
    load_existing_ids,
    run_evaluation,
    visualize_paths,
)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

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


def _compute_region_center(mask_rle: dict) -> tuple[int, int]:
    """Compute (x, y) center of a region from its RLE-encoded mask."""
    mask = decode_rle(mask_rle)
    ys, xs = np.where(mask)
    return int(xs.mean()), int(ys.mean())


def format_system_prompt(gt_data: dict) -> str:
    """Fill in the system prompt template from GT metadata."""
    width = gt_data["image_size"]["w"]
    height = gt_data["image_size"]["h"]
    rows = gt_data["difficulty"]["grid_rows"]
    cols = gt_data["difficulty"]["grid_cols"]
    start_x, start_y = _compute_region_center(gt_data["regions"]["start_mask_rle"])
    goal_x, goal_y = _compute_region_center(gt_data["regions"]["goal_mask_rle"])
    return AGENTIC_SYSTEM_PROMPT.format(
        width=width, height=height, rows=rows, cols=cols,
        start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y,
    )


def render_progress_image(
    base_image: Image.Image,
    accepted_segments: list[list[list[float]]],
    rejected_segments: list[list[list[float]]],
    violation_points: list[list[float]],
) -> str:
    """Render path progress onto the maze image and return as base64 data URI."""
    img = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for seg in accepted_segments:
        if len(seg) >= 2:
            flat = [(int(p[0]), int(p[1])) for p in seg]
            draw.line(flat, fill=(78, 204, 163, 220), width=3)

    for seg in rejected_segments:
        if len(seg) >= 2:
            flat = [(int(p[0]), int(p[1])) for p in seg]
            draw.line(flat, fill=(233, 69, 96, 180), width=2)

    for vp in violation_points:
        x, y = int(vp[0]), int(vp[1])
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(233, 69, 96, 230))

    img = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Async per-maze execution
# ---------------------------------------------------------------------------


async def run_maze_agentic(
    sem: asyncio.Semaphore,
    maze_id: str,
    image_dir: str,
    gt_dir: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_turns: int,
    api_base: Optional[str],
    verbose: bool,
) -> Optional[dict]:
    """Run a single maze in agentic multi-turn mode."""
    async with sem:
        image_path = os.path.join(image_dir, f"{maze_id}.png")
        gt_path = os.path.join(gt_dir, f"{maze_id}.json")

        with open(gt_path) as f:
            gt_data = json.load(f)

        # Use actual PNG dimensions (PIL-verified) for system prompt
        base_image = Image.open(image_path)
        actual_w, actual_h = base_image.size
        gt_data["image_size"] = {"w": actual_w, "h": actual_h}

        system_prompt = format_system_prompt(gt_data)
        image_uri = encode_image_base64(image_path)

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

        # Thinking/reasoning models need higher token budgets
        model_lower = model.lower()
        is_thinking_model = any(
            model_lower.startswith(p) for p in ("gpt-5", "o3", "o4", "o1")
        )
        effective_max_tokens = (
            max(max_tokens, 65536) if is_thinking_model else max_tokens
        )

        session_result = None
        turn = 0
        progress_msg_idx: int | None = None  # track progress image message

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

            # Retry with exponential backoff for rate limiting
            max_retries = 5
            base_delay = 2.0
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = await litellm.acompletion(**kwargs)
                    break
                except litellm.RateLimitError as e:
                    if attempt == max_retries:
                        print(
                            f"  [{maze_id}] Rate limit exhausted: {e}",
                            file=sys.stderr,
                        )
                        break
                    delay = min(base_delay * (2**attempt), 60.0) + random.uniform(
                        0, 1
                    )
                    if verbose:
                        print(
                            f"  [{maze_id}] Rate limited, retrying in {delay:.1f}s",
                            file=sys.stderr,
                        )
                    await asyncio.sleep(delay)
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status == 429 and attempt < max_retries:
                        delay = min(
                            base_delay * (2**attempt), 60.0
                        ) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                    else:
                        print(
                            f"  [{maze_id}] API error: {e}", file=sys.stderr
                        )
                        break

            if response is None:
                break

            message = response.choices[0].message
            messages.append(message)

            # Check for tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                # Model didn't use tools
                if verbose:
                    content = getattr(message, "content", "") or ""
                    print(
                        f"  [{maze_id}] Turn {turn}: no tool call, "
                        f"text: {content[:100]}",
                        file=sys.stderr,
                    )
                break

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "submit_segment":
                    points = fn_args.get("points", [])
                    seg_result = session.submit_segment(points)
                    tool_response = format_tool_result(seg_result)
                    # Track segments for progress image
                    if seg_result.status == SegmentStatus.ACCEPTED:
                        accepted_segments.append(points)
                    else:
                        rejected_segments.append(points)
                    if seg_result.violation_point is not None:
                        violation_points.append(list(seg_result.violation_point))
                    if verbose:
                        print(
                            f"  [{maze_id}] Turn {turn}: submit_segment "
                            f"-> {seg_result.status.value}",
                            file=sys.stderr,
                        )

                elif fn_name == "finish":
                    session_result = session.finish()
                    tool_response = format_finish_result(session_result)
                    if verbose:
                        print(
                            f"  [{maze_id}] Turn {turn}: finish "
                            f"-> {session_result.finish_reason}",
                            file=sys.stderr,
                        )

                else:
                    tool_response = f"Unknown tool: {fn_name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_response,
                    }
                )

            # Re-inject maze image with progress overlay (replace previous to
            # avoid accumulating base64 images which eats memory)
            if session_result is None:
                progress_uri = render_progress_image(
                    base_image, accepted_segments, rejected_segments, violation_points,
                )
                progress_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": progress_uri}},
                        {
                            "type": "text",
                            "text": "Here is the maze with your progress so far. Green lines are accepted path segments. Red lines are rejected segments. Continue navigating.",
                        },
                    ],
                }
                if progress_msg_idx is not None:
                    messages[progress_msg_idx] = progress_msg
                else:
                    messages.append(progress_msg)
                    progress_msg_idx = len(messages) - 1

            if session_result is not None:
                break

        # Force-finish if turns exhausted
        if session_result is None:
            try:
                session_result = session.finish()
            except RuntimeError:
                pass  # Already finished

            if session_result and verbose:
                print(
                    f"  [{maze_id}] Force-finished after {turn} turns",
                    file=sys.stderr,
                )

        if session_result is None or len(session_result.accepted_path) < 2:
            reason = session_result.finish_reason if session_result else "no result"
            if verbose:
                print(
                    f"  [{maze_id}] No valid path ({reason})", file=sys.stderr
                )
            return None

        # Build output entry (backward compatible with single-shot evaluator)
        prediction = {
            "encoding": "polyline",
            "data": {
                "points": [
                    [p[0], p[1]] for p in session_result.accepted_path
                ]
            },
        }

        entry = {
            "id": maze_id,
            "prediction": prediction,
            "mode": "agentic",
            "agentic_stats": {
                "turns": turn,
                "segments_submitted": session_result.stats.total_segments_submitted,
                "segments_accepted": session_result.stats.segments_accepted,
                "segments_rejected": session_result.stats.segments_rejected,
                "wall_rejections": session_result.stats.wall_rejections,
                "contiguity_rejections": session_result.stats.contiguity_rejections,
                "finish_reason": session_result.finish_reason,
            },
        }

        if verbose:
            stats = session_result.stats
            print(
                f"  [{maze_id}] Done -- {stats.segments_accepted} accepted, "
                f"{stats.segments_rejected} rejected, "
                f"{len(session_result.accepted_path)} points"
            )

        return entry


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> list[dict]:
    """Run all mazes and return collected results."""
    maze_ids = discover_mazes(
        args.image_dir,
        args.gt_dir,
        args.maze_ids.split(",") if args.maze_ids else None,
        args.max_mazes,
    )

    if not maze_ids:
        print("No mazes found.", file=sys.stderr)
        return []

    # Resume support
    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = load_existing_ids(args.output)
        if existing_ids:
            print(
                f"Resuming: skipping {len(existing_ids)} already-completed mazes"
            )
        maze_ids = [m for m in maze_ids if m not in existing_ids]

    if not maze_ids:
        print("All mazes already completed.")
        return []

    print(f"Model: {args.model}")
    print(f"Mode: agentic (multi-turn)")
    print(f"Max turns per maze: {args.max_turns}")
    print(f"Mazes: {len(maze_ids)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_maze_agentic(
            sem=sem,
            maze_id=maze_id,
            image_dir=args.image_dir,
            gt_dir=args.gt_dir,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            api_base=args.api_base,
            verbose=args.verbose,
        )
        for maze_id in maze_ids
    ]

    results: list[dict] = []
    succeeded = 0
    failed = 0

    for coro in asyncio.as_completed(tasks):
        entry = await coro
        if entry is not None:
            append_jsonl(args.output, entry)
            results.append(entry)
            succeeded += 1
        else:
            failed += 1

    print()
    print(
        f"Done: {succeeded} succeeded, {failed} failed "
        f"out of {len(maze_ids)} mazes"
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vision LLMs against MazeRunner in agentic mode"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory with maze PNG images"
    )
    parser.add_argument(
        "--gt-dir", required=True, help="Directory with GT JSON files"
    )
    parser.add_argument(
        "--model", required=True, help="litellm model string"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL path"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--max-mazes",
        type=int,
        default=None,
        help="Limit number of mazes to process",
    )
    parser.add_argument(
        "--maze-ids",
        default=None,
        help="Comma-separated specific maze IDs",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Max turns per maze (default: 30)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluator on output after completion",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-maze progress"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max response tokens (default: 8192)",
    )
    parser.add_argument(
        "--api-base", default=None, help="Custom API base URL"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip mazes already in output file",
    )
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="Directory to write path visualization images",
    )

    args = parser.parse_args()

    results = asyncio.run(run_all(args))

    if args.evaluate and results:
        run_evaluation(args.output, args.gt_dir, args.verbose)

    if args.visualize_dir and os.path.exists(args.output):
        visualize_paths(args.output, args.image_dir, args.visualize_dir)


if __name__ == "__main__":
    main()
