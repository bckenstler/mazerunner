#!/usr/bin/env python
"""Run agent eval across all three modes against mazes in data/dev.

Loads OPENAI_API_KEY from .env (overriding any existing env var), then runs
a small eval (3 episodes each) in text_grid, vision_grid, and vision_drag
modes using gpt-5.4 with medium reasoning effort. Saves results to
data/eval_results/ and prints a summary.

Usage:
    uv run python scripts/e2e_agent_eval.py
    uv run python scripts/e2e_agent_eval.py -v          # verbose trajectory
    uv run python scripts/e2e_agent_eval.py --modes text_grid --num-episodes 1 -v
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

# Unbuffered print so output streams in real-time
print = partial(print, flush=True)

# Load .env — override existing env vars so .env key takes precedence
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()

from mazerunner.agent.runner import get_runner
from mazerunner.agent.types import AgentConfig
from mazerunner.eval.harness import run_eval
from mazerunner.eval.io import save_eval_result

INSTANCE_DIR = Path(__file__).resolve().parent.parent / "data" / "dev"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results"
ALL_MODES = ["text_grid", "vision_grid", "vision_drag"]
MODEL = "gpt-5.4"
REASONING_EFFORT = "medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent eval across maze modes")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show real-time trajectory (observation, reasoning, tool calls)")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        choices=ALL_MODES, help="Modes to evaluate")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT,
                        choices=["low", "medium", "high"])
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "fireworks"])
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Fireworks thinking budget_tokens")
    return parser.parse_args()


def discover_instances() -> list[str]:
    """Find maze instance JSON files."""
    inst_dir = INSTANCE_DIR / "instances"
    if inst_dir.is_dir():
        return sorted(str(p) for p in inst_dir.glob("*.json"))
    return sorted(str(p) for p in INSTANCE_DIR.glob("*.json"))


def run_mode_eval(
    mode: str,
    instance_paths: list[str],
    args: argparse.Namespace,
) -> dict:
    """Run eval for a single mode and return metrics."""
    config = AgentConfig(
        model=args.model,
        mode=mode,
        max_turns=args.max_turns,
        temperature=0.0,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
        thinking_budget=args.thinking_budget,
    )
    runner = get_runner(config, verbose=args.verbose)

    result = run_eval(
        runner=runner,
        instance_paths=instance_paths,
        mode=mode,
        max_steps=args.max_steps,
        reward_mode="sparse",
        num_episodes=args.num_episodes,
        model=args.model,
    )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"eval_{mode}.json"
    save_eval_result(result, output_path)

    return {
        "mode": mode,
        "num_episodes": result.num_episodes,
        "metrics": result.metrics,
        "output_path": str(output_path),
        "records_summary": [
            {
                "maze_id": r.maze_id,
                "success": r.success,
                "steps": r.steps,
                "reward": r.reward,
                "has_reasoning": any(s.reasoning for s in r.trajectory),
            }
            for r in result.records
        ],
    }


def main():
    args = parse_args()
    instance_paths = discover_instances()
    if not instance_paths:
        print(f"ERROR: No maze instances found in {INSTANCE_DIR}", file=sys.stderr)
        sys.exit(1)

    # Validate API key for the chosen provider
    if args.provider == "fireworks":
        if not os.environ.get("FIREWORKS_API_KEY"):
            print("ERROR: FIREWORKS_API_KEY not found in .env or environment", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in .env or environment", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(instance_paths)} maze instances in {INSTANCE_DIR}")
    print(f"Running {args.num_episodes} episodes per mode with {args.model} (provider: {args.provider}, reasoning: {args.reasoning_effort})")
    print(f"Modes: {', '.join(args.modes)}")
    if args.verbose:
        print("Verbose trajectory output enabled")
    print()

    all_results = []
    total_start = time.monotonic()

    for mode in args.modes:
        print(f"{'=' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}")

        start = time.monotonic()
        try:
            result = run_mode_eval(mode, instance_paths, args)
            elapsed = time.monotonic() - start
            all_results.append((mode, result, elapsed, None))

            # Print per-episode results
            for rec in result["records_summary"]:
                status = "SUCCESS" if rec["success"] else "FAIL"
                reasoning_flag = " [reasoning]" if rec["has_reasoning"] else ""
                print(f"  {rec['maze_id']:20s} {status:7s}  steps={rec['steps']:3d}  reward={rec['reward']:.2f}{reasoning_flag}")

            print(f"\n  Metrics:")
            for k, v in result["metrics"].items():
                print(f"    {k:30s} {v:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Saved: {result['output_path']}")

        except Exception as e:
            elapsed = time.monotonic() - start
            all_results.append((mode, None, elapsed, str(e)))
            print(f"  ERROR: {e}")

        print()

    total_elapsed = time.monotonic() - total_start

    # Summary
    print(f"{'=' * 60}")
    print("  Agent Eval Summary")
    print(f"{'=' * 60}")
    all_passed = True
    for mode, result, elapsed, error in all_results:
        if error:
            print(f"  x {mode:20s} ERROR  ({elapsed:.1f}s) - {error}")
            all_passed = False
        else:
            sr = result["metrics"]["success_rate"]
            print(f"  * {mode:20s} success_rate={sr:.2%}  ({elapsed:.1f}s)")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    if not all_passed:
        print("\n  Some evals had errors")
        sys.exit(1)
    else:
        print("\n  All evals completed successfully")


if __name__ == "__main__":
    main()
