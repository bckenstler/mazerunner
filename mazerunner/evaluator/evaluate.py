"""Main evaluation pipeline for MazeRunner benchmark."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from mazerunner.common.rle import decode_rle
from mazerunner.common.types import EvalResult
from mazerunner.evaluator.canonicalize import canonicalize
from mazerunner.evaluator.metrics import (
    compute_clearance_map,
    compute_goal_distance,
    compute_length_regret,
    compute_min_clearance,
    compute_mono_score,
    compute_path_length,
    compute_success_at_r,
    compute_valid_frac,
)
from mazerunner.evaluator.schemas import load_gt, load_submission

BUFFER_RADII = [0, 1, 2, 4, 8]


def evaluate_single(
    prediction: dict,
    gt_data: dict,
    buffer_radii: Optional[List[int]] = None,
    max_gap: float = 2.0,
) -> EvalResult:
    """Evaluate a single prediction against ground truth.

    1. Parse image_size, decode all RLE masks from gt_data
    2. Load render_config from gt_data (if present)
    3. Canonicalize the prediction
    4. Compute clearance_map once
    5. For each radius in buffer_radii: compute success@r and valid_frac@r
    6. Compute all secondary metrics
    7. Return EvalResult
    """
    if buffer_radii is None:
        buffer_radii = BUFFER_RADII

    maze_id = gt_data["id"]
    image_size = gt_data["image_size"]

    # Decode masks
    regions = gt_data["regions"]
    free_space_mask = decode_rle(regions["free_space_mask_rle"])
    start_mask = decode_rle(regions["start_mask_rle"])
    goal_mask = decode_rle(regions["goal_mask_rle"])

    # Render config (optional)
    render_config = gt_data.get("render_config", None)

    # Canonicalize prediction to dense polyline
    polyline = canonicalize(prediction, image_size, render_config, max_gap)

    # Compute clearance map once
    clearance_map = compute_clearance_map(free_space_mask)

    # Success@r and valid_frac@r for each radius
    success_dict: Dict[str, bool] = {}
    valid_frac_dict: Dict[str, float] = {}
    start_ok = False
    goal_ok = False

    for r in buffer_radii:
        key = str(r)
        s, s_ok, g_ok = compute_success_at_r(
            polyline, clearance_map, start_mask, goal_mask, r
        )
        success_dict[key] = s
        valid_frac_dict[key] = compute_valid_frac(polyline, clearance_map, r)
        # Use the most permissive (r=0) endpoint check
        if r == 0:
            start_ok = s_ok
            goal_ok = g_ok

    # Secondary metrics
    min_clearance = compute_min_clearance(polyline, clearance_map)
    goal_distance = compute_goal_distance(polyline, goal_mask)
    path_length = compute_path_length(polyline)

    gt_length = float(gt_data["gt"]["solution_length"])
    length_regret = compute_length_regret(path_length, gt_length)

    # GT polyline for mono score
    gt_polyline_raw = gt_data["gt"]["solution_polyline"]
    gt_polyline = [(float(p[0]), float(p[1])) for p in gt_polyline_raw]
    mono_score = compute_mono_score(polyline, gt_polyline)

    return EvalResult(
        maze_id=maze_id,
        success=success_dict,
        valid_frac=valid_frac_dict,
        min_clearance=min_clearance,
        goal_distance=goal_distance,
        path_length=path_length,
        length_regret=length_regret,
        mono_score=mono_score,
        start_ok=start_ok,
        goal_ok=goal_ok,
    )


def evaluate_dataset(
    submission_path: str,
    gt_dir: str,
    buffer_radii: Optional[List[int]] = None,
    max_gap: float = 2.0,
) -> Tuple[List[EvalResult], dict]:
    """Evaluate a full submission against a ground truth directory.

    1. Load submission
    2. For each entry, load corresponding GT file from gt_dir/{id}.json
    3. Evaluate each
    4. Compute aggregate summary
    5. Return (per_maze_results, summary_dict)
    """
    if buffer_radii is None:
        buffer_radii = BUFFER_RADII

    entries = load_submission(submission_path)
    results: List[EvalResult] = []

    for entry in entries:
        maze_id = entry["id"]
        gt_path = os.path.join(gt_dir, f"{maze_id}.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        gt_data = load_gt(gt_path)
        result = evaluate_single(entry["prediction"], gt_data, buffer_radii, max_gap)
        results.append(result)

    # Compute aggregate summary
    summary: dict = {}
    n = len(results)
    if n == 0:
        return results, summary

    # Mean success@r
    for r in buffer_radii:
        key = str(r)
        summary[f"success@{key}"] = sum(
            1 for res in results if res.success.get(key, False)
        ) / n

    # Mean valid_frac@r
    for r in buffer_radii:
        key = str(r)
        summary[f"valid_frac@{key}"] = sum(
            res.valid_frac.get(key, 0.0) for res in results
        ) / n

    summary["mean_min_clearance"] = sum(r.min_clearance for r in results) / n
    summary["mean_goal_distance"] = sum(r.goal_distance for r in results) / n
    summary["mean_path_length"] = sum(r.path_length for r in results) / n
    summary["mean_length_regret"] = sum(r.length_regret for r in results) / n
    summary["mean_mono_score"] = sum(r.mono_score for r in results) / n
    summary["start_ok_rate"] = sum(1 for r in results if r.start_ok) / n
    summary["goal_ok_rate"] = sum(1 for r in results if r.goal_ok) / n
    summary["num_mazes"] = n

    return results, summary
