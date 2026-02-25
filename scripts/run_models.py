"""Run vision LLMs against MazeRunner maze images and collect path outputs.

Usage:
    PYTHONPATH=. python scripts/run_models.py \
        --image-dir data/dev/images \
        --gt-dir data/dev/gt \
        --model gpt-4o \
        --output results/gpt-4o.jsonl \
        --encoding polyline \
        --concurrency 5 \
        --evaluate --verbose
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import litellm
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# JSON Schemas for structured output
# ---------------------------------------------------------------------------

POLYLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "encoding": {"type": "string", "enum": ["polyline"]},
        "data": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                }
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    },
    "required": ["encoding", "data"],
    "additionalProperties": False,
}

CELL_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "encoding": {"type": "string", "enum": ["cell_route"]},
        "data": {
            "type": "object",
            "properties": {
                "cells": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                }
            },
            "required": ["cells"],
            "additionalProperties": False,
        },
    },
    "required": ["encoding", "data"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

POLYLINE_PROMPT_STRUCTURED = """\
You are navigating a maze. The image is {width}x{height} pixels (origin top-left, X right, Y down).
The maze is a {rows}x{cols} grid with:
- GREEN circle = START, RED circle = GOAL
- White corridors to travel through, dark walls to avoid

Find a path from the green START to the red GOAL staying within corridors.
Include a waypoint at every turn so the path accurately follows the corridors. Stay centered in corridors."""

POLYLINE_PROMPT_FALLBACK = """\
You are navigating a maze. The image is {width}x{height} pixels (origin top-left, X right, Y down).
The maze is a {rows}x{cols} grid with:
- GREEN circle = START, RED circle = GOAL
- White corridors to travel through, dark walls to avoid

Find a path from the green START to the red GOAL staying within corridors.
Include a waypoint at every turn so the path accurately follows the corridors. Stay centered in corridors.

Output ONLY a JSON object in this exact format, no other text:
{{"encoding": "polyline", "data": {{"points": [[x1, y1], [x2, y2], ...]}}}}"""

CELL_ROUTE_PROMPT_STRUCTURED = """\
You are navigating a maze on a {rows}x{cols} grid (row 0 = top, col 0 = left).
- GREEN circle = START, RED circle = GOAL
- Move only through adjacent open corridors (up/down/left/right)

List every grid cell from START to GOAL in order. Each consecutive pair must be adjacent."""

CELL_ROUTE_PROMPT_FALLBACK = """\
You are navigating a maze on a {rows}x{cols} grid (row 0 = top, col 0 = left).
- GREEN circle = START, RED circle = GOAL
- Move only through adjacent open corridors (up/down/left/right)

List every grid cell from START to GOAL in order. Each consecutive pair must be adjacent.

Output ONLY a JSON object in this exact format, no other text:
{{"encoding": "cell_route", "data": {{"cells": [[row1, col1], [row2, col2], ...]}}}}"""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_base64(path: str) -> str:
    """Read a PNG file and return a base64-encoded data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(gt_data: dict, encoding: str, use_structured: bool) -> str:
    """Fill in the prompt template from GT metadata."""
    width = gt_data["image_size"]["w"]
    height = gt_data["image_size"]["h"]
    rows = gt_data["difficulty"]["grid_rows"]
    cols = gt_data["difficulty"]["grid_cols"]

    vars_ = {"width": width, "height": height, "rows": rows, "cols": cols}

    if encoding == "polyline":
        template = POLYLINE_PROMPT_STRUCTURED if use_structured else POLYLINE_PROMPT_FALLBACK
    else:
        template = CELL_ROUTE_PROMPT_STRUCTURED if use_structured else CELL_ROUTE_PROMPT_FALLBACK

    return template.format(**vars_)


# ---------------------------------------------------------------------------
# Response parsing (fallback)
# ---------------------------------------------------------------------------

def parse_model_response(text: str) -> Optional[dict]:
    """Parse model response text into a prediction dict.

    Tries three strategies in order:
    1. Direct json.loads
    2. Extract from ```json ... ``` code block
    3. Find first balanced { ... } substring
    """
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if _validate_prediction(result):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if _validate_prediction(result):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: first balanced braces
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start : i + 1])
                        if _validate_prediction(result):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return None


def _validate_prediction(d: dict) -> bool:
    """Check that a parsed dict has the required prediction structure."""
    return (
        isinstance(d, dict)
        and "encoding" in d
        and "data" in d
        and isinstance(d["data"], dict)
    )


# ---------------------------------------------------------------------------
# Maze discovery
# ---------------------------------------------------------------------------

def discover_mazes(
    image_dir: str, gt_dir: str, maze_ids: Optional[list[str]], max_mazes: Optional[int]
) -> list[str]:
    """Find maze IDs that have both an image and GT file.

    Returns sorted list of IDs, filtered/limited by arguments.
    """
    image_files = {Path(f).stem for f in os.listdir(image_dir) if f.endswith(".png")}
    gt_files = {Path(f).stem for f in os.listdir(gt_dir) if f.endswith(".json")}
    available = sorted(image_files & gt_files)

    if maze_ids is not None:
        requested = set(maze_ids)
        missing = requested - set(available)
        if missing:
            print(f"Warning: maze IDs not found: {sorted(missing)}", file=sys.stderr)
        available = [m for m in available if m in requested]

    if max_mazes is not None:
        available = available[:max_mazes]

    return available


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def append_jsonl(path: str, entry: dict) -> None:
    """Append a single JSON object as a line to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_existing_ids(path: str) -> set[str]:
    """Load maze IDs already present in an output JSONL file."""
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "id" in entry:
                    ids.add(entry["id"])
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def _resolve_structured_mode(mode: str, model: str) -> str:
    """Resolve 'auto' to a concrete structured output tier.

    Returns one of: 'json_schema', 'json_object', 'off'.
    """
    if mode != "auto":
        return mode

    try:
        if litellm.supports_response_schema(model=model):
            return "json_schema"
    except Exception:
        pass

    return "off"


def _build_response_format(structured_mode: str, encoding: str) -> Optional[dict]:
    """Build the response_format dict for litellm, or None."""
    if structured_mode == "json_schema":
        schema = POLYLINE_SCHEMA if encoding == "polyline" else CELL_ROUTE_SCHEMA
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "maze_path",
                "schema": schema,
                "strict": True,
            },
        }
    elif structured_mode == "json_object":
        return {"type": "json_object"}
    return None


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------

async def run_maze(
    sem: asyncio.Semaphore,
    maze_id: str,
    image_dir: str,
    gt_dir: str,
    model: str,
    encoding: str,
    temperature: float,
    max_tokens: int,
    api_base: Optional[str],
    response_format: Optional[dict],
    use_structured_prompt: bool,
    verbose: bool,
    reasoning_effort: Optional[str] = None,
) -> Optional[dict]:
    """Run a single maze through the model and return a submission entry."""
    async with sem:
        image_path = os.path.join(image_dir, f"{maze_id}.png")
        gt_path = os.path.join(gt_dir, f"{maze_id}.json")

        with open(gt_path) as f:
            gt_data = json.load(f)

        prompt = format_prompt(gt_data, encoding, use_structured_prompt)
        image_uri = encode_image_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Thinking/reasoning models include reasoning tokens in max_tokens,
        # so we need a much higher budget to leave room for the actual
        # output after internal reasoning.
        is_thinking_model = litellm.supports_reasoning(model=model)
        effective_max_tokens = max(max_tokens, 65536) if is_thinking_model else max_tokens

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if response_format:
            kwargs["response_format"] = response_format
        if reasoning_effort and is_thinking_model:
            kwargs["reasoning_effort"] = reasoning_effort

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
                    print(f"  [{maze_id}] Rate limit exhausted after {max_retries} retries: {e}", file=sys.stderr)
                    return None
                delay = min(base_delay * (2 ** attempt), 60.0) + random.uniform(0, 1)
                if verbose:
                    print(f"  [{maze_id}] Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                await asyncio.sleep(delay)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 60.0) + random.uniform(0, 1)
                    if verbose:
                        print(f"  [{maze_id}] 429 error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                    await asyncio.sleep(delay)
                else:
                    print(f"  [{maze_id}] API error: {e}", file=sys.stderr)
                    return None

        if response is None:
            print(f"  [{maze_id}] No response after retries", file=sys.stderr)
            return None

        # Extract text from response — thinking models may return content
        # in different fields when using structured output.
        text = ""
        message = response.choices[0].message

        # Strategy 1: check .parsed for structured output (thinking models)
        parsed = getattr(message, "parsed", None)
        if parsed is not None and isinstance(parsed, dict):
            text = json.dumps(parsed)
        # Strategy 2: content is a list of content blocks (some thinking models)
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block["text"]
                    break
        # Strategy 3: standard string content
        elif message.content:
            text = message.content
        # Strategy 4: tool_calls fallback
        elif getattr(message, "tool_calls", None):
            try:
                text = message.tool_calls[0].function.arguments
            except (AttributeError, IndexError):
                pass

        # Extract reasoning/thinking content if present
        reasoning_content = getattr(message, "reasoning_content", None) or None
        reasoning_tokens = None
        usage = getattr(response, "usage", None)
        if usage:
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

        prediction = parse_model_response(text)
        if prediction is None:
            if verbose:
                print(f"  [{maze_id}] Failed to parse response", file=sys.stderr)
                # Print first 200 chars of response for debugging
                preview = text[:200].replace("\n", " ")
                print(f"           Response preview: {preview}", file=sys.stderr)
            return None

        entry = {"id": maze_id, "prediction": prediction}
        if reasoning_content is not None:
            entry["reasoning_content"] = reasoning_content
        if reasoning_tokens is not None:
            entry["reasoning_tokens"] = reasoning_tokens

        if verbose:
            enc = prediction.get("encoding", "?")
            data = prediction.get("data", {})
            if "points" in data:
                n = len(data["points"])
                print(f"  [{maze_id}] OK — {enc}, {n} points")
            elif "cells" in data:
                n = len(data["cells"])
                print(f"  [{maze_id}] OK — {enc}, {n} cells")
            else:
                print(f"  [{maze_id}] OK — {enc}")

        return entry


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
            print(f"Resuming: skipping {len(existing_ids)} already-completed mazes")
        maze_ids = [m for m in maze_ids if m not in existing_ids]

    if not maze_ids:
        print("All mazes already completed.")
        return []

    # Resolve structured output mode
    structured_mode = _resolve_structured_mode(args.structured_output, args.model)
    response_format = _build_response_format(structured_mode, args.encoding)
    use_structured_prompt = structured_mode == "json_schema"

    mode_label = {
        "json_schema": "json_schema (server-side constrained decoding)",
        "json_object": "json_object (valid JSON, prompt-guided format)",
        "off": "off (prompt-only, client-side parsing)",
    }

    reasoning_effort = getattr(args, "reasoning_effort", None)

    print(f"Model: {args.model}")
    print(f"Encoding: {args.encoding}")
    print(f"Structured output: {mode_label.get(structured_mode, structured_mode)}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Mazes: {len(maze_ids)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_maze(
            sem=sem,
            maze_id=maze_id,
            image_dir=args.image_dir,
            gt_dir=args.gt_dir,
            model=args.model,
            encoding=args.encoding,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_base=args.api_base,
            response_format=response_format,
            use_structured_prompt=use_structured_prompt,
            verbose=args.verbose,
            reasoning_effort=reasoning_effort,
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
    print(f"Done: {succeeded} succeeded, {failed} failed out of {len(maze_ids)} mazes")

    return results


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------

def run_evaluation(output_path: str, gt_dir: str, verbose: bool) -> None:
    """Run the MazeRunner evaluator on the output file."""
    from mazerunner.evaluator.evaluate import BUFFER_RADII, evaluate_dataset

    results, summary = evaluate_dataset(
        submission_path=output_path,
        gt_dir=gt_dir,
    )

    print()
    print("=" * 60)
    print("MazeRunner Evaluation Summary")
    print("=" * 60)
    print(f"  Mazes evaluated: {summary.get('num_mazes', 0)}")
    print()

    print("  Success rates (success@r):")
    for r in BUFFER_RADII:
        key = f"success@{r}"
        val = summary.get(key, 0.0)
        print(f"    r={r}: {val:.4f}")
    print()

    print("  Valid fraction (valid_frac@r):")
    for r in BUFFER_RADII:
        key = f"valid_frac@{r}"
        val = summary.get(key, 0.0)
        print(f"    r={r}: {val:.4f}")
    print()

    print(f"  Mean min clearance:  {summary.get('mean_min_clearance', 0.0):.2f}")
    print(f"  Mean goal distance:  {summary.get('mean_goal_distance', 0.0):.2f}")
    print(f"  Mean path length:    {summary.get('mean_path_length', 0.0):.2f}")
    print(f"  Mean length regret:  {summary.get('mean_length_regret', 0.0):.4f}")
    print(f"  Mean mono score:     {summary.get('mean_mono_score', 0.0):.4f}")
    print(f"  Start OK rate:       {summary.get('start_ok_rate', 0.0):.4f}")
    print(f"  Goal OK rate:        {summary.get('goal_ok_rate', 0.0):.4f}")
    print("=" * 60)

    if verbose and results:
        print()
        print("Per-maze results:")
        print("-" * 60)
        for res in results:
            print(f"  Maze: {res.maze_id}")
            print(f"    success@0: {res.success.get('0', False)}")
            print(f"    min_clearance: {res.min_clearance:.2f}")
            print(f"    goal_distance: {res.goal_distance:.2f}")
            print(f"    path_length: {res.path_length:.2f}")
            print(f"    length_regret: {res.length_regret:.4f}")
            print(f"    mono_score: {res.mono_score:.4f}")
            print(f"    start_ok: {res.start_ok}, goal_ok: {res.goal_ok}")
            print()


# ---------------------------------------------------------------------------
# Path visualization
# ---------------------------------------------------------------------------

def visualize_paths(output_path: str, image_dir: str, visualize_dir: str) -> None:
    """Draw predicted paths on maze images and save to visualize_dir."""
    os.makedirs(visualize_dir, exist_ok=True)

    with open(output_path) as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        maze_id = entry.get("id")
        prediction = entry.get("prediction", {})
        if not maze_id:
            continue

        # Only handle polyline encoding
        if prediction.get("encoding") != "polyline":
            continue

        points = prediction.get("data", {}).get("points")
        if not points or len(points) < 2:
            continue

        image_path = os.path.join(image_dir, f"{maze_id}.png")
        if not os.path.exists(image_path):
            print(f"  [{maze_id}] Image not found, skipping visualization", file=sys.stderr)
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw the path as a blue line
        line_coords = [(p[0], p[1]) for p in points]
        draw.line(line_coords, fill="#2196F3", width=3)

        # Mark start point (green circle)
        sx, sy = points[0][0], points[0][1]
        draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill="#4CAF50", outline="#4CAF50")

        # Mark end point (red circle)
        ex, ey = points[-1][0], points[-1][1]
        draw.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill="#F44336", outline="#F44336")

        out_path = os.path.join(visualize_dir, f"{maze_id}.png")
        img.save(out_path)
        count += 1

    print(f"Saved {count} visualization(s) to {visualize_dir}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(args: argparse.Namespace) -> None:
    """Print the prompt for the first maze without calling the API."""
    maze_ids = discover_mazes(
        args.image_dir,
        args.gt_dir,
        args.maze_ids.split(",") if args.maze_ids else None,
        max_mazes=1,
    )
    if not maze_ids:
        print("No mazes found.", file=sys.stderr)
        sys.exit(1)

    maze_id = maze_ids[0]
    gt_path = os.path.join(args.gt_dir, f"{maze_id}.json")
    with open(gt_path) as f:
        gt_data = json.load(f)

    structured_mode = _resolve_structured_mode(args.structured_output, args.model)
    use_structured_prompt = structured_mode == "json_schema"
    response_format = _build_response_format(structured_mode, args.encoding)

    prompt = format_prompt(gt_data, args.encoding, use_structured_prompt)

    print(f"=== Dry Run: maze {maze_id} ===")
    print(f"Model: {args.model}")
    print(f"Encoding: {args.encoding}")
    print(f"Structured output mode: {structured_mode}")
    print(f"Image: {os.path.join(args.image_dir, f'{maze_id}.png')}")
    print()
    print("--- Prompt ---")
    print(prompt)
    print()
    if response_format:
        print("--- response_format ---")
        print(json.dumps(response_format, indent=2))
    else:
        print("--- response_format: None (prompt-only mode) ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vision LLMs against MazeRunner maze images"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory with maze PNG images"
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Directory with GT JSON files",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="litellm model string (e.g. gpt-4o, claude-sonnet-4-20250514, gemini/gemini-2.0-flash)",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL path"
    )
    parser.add_argument(
        "--encoding",
        default="polyline",
        choices=["polyline", "cell_route"],
        help="Encoding to request (default: polyline)",
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
        "--evaluate",
        action="store_true",
        help="Run evaluator on output after completion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-maze progress",
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
        "--api-base",
        default=None,
        help="Custom API base URL for hosted models",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip mazes already in output file",
    )
    parser.add_argument(
        "--structured-output",
        default="auto",
        choices=["auto", "json_schema", "json_object", "off"],
        help="Structured output mode (default: auto)",
    )
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="Directory to write path visualization images",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort level for thinking models (default: None)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt for first maze without calling API",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    results = asyncio.run(run_all(args))

    if args.evaluate and results:
        run_evaluation(args.output, args.gt_dir, args.verbose)

    if args.visualize_dir and os.path.exists(args.output):
        visualize_paths(args.output, args.image_dir, args.visualize_dir)


if __name__ == "__main__":
    main()
