"""CLI for MazeRunner evaluation."""

import argparse
import json
from dataclasses import asdict

from mazerunner.evaluator.evaluate import BUFFER_RADII, evaluate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MazeRunner benchmark submissions"
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission file (JSON or JSONL)",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Path to ground truth directory",
    )
    parser.add_argument(
        "--buffer-radii",
        default="0,1,2,4,8",
        help="Comma-separated list of buffer radii (default: 0,1,2,4,8)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=2.0,
        help="Densification max gap in pixels (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-maze results",
    )

    args = parser.parse_args()

    buffer_radii = [int(r.strip()) for r in args.buffer_radii.split(",")]

    results, summary = evaluate_dataset(
        submission_path=args.submission,
        gt_dir=args.gt_dir,
        buffer_radii=buffer_radii,
        max_gap=args.max_gap,
    )

    # Print summary
    print("=" * 60)
    print("MazeRunner Evaluation Summary")
    print("=" * 60)
    print(f"  Mazes evaluated: {summary.get('num_mazes', 0)}")
    print()

    print("  Success rates (success@r):")
    for r in buffer_radii:
        key = f"success@{r}"
        val = summary.get(key, 0.0)
        print(f"    r={r}: {val:.4f}")
    print()

    print("  Valid fraction (valid_frac@r):")
    for r in buffer_radii:
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

    # Verbose: per-maze results
    if args.verbose and results:
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

    # Optionally save results
    if args.output:
        output_data = {
            "summary": summary,
            "per_maze": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
