"""All metric computations for MazeRunner evaluation."""

import math
from typing import List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_clearance_map(free_space_mask: np.ndarray) -> np.ndarray:
    """Return distance transform of free_space_mask.

    Each free pixel gets its Euclidean distance to the nearest wall (non-free pixel).
    Wall pixels get 0.
    """
    return distance_transform_edt(free_space_mask)


def check_endpoint(
    point: Tuple[float, float],
    region_mask: np.ndarray,
    tolerance: int = 4,
) -> bool:
    """Check if a point falls within a dilated version of region_mask.

    Dilate by computing distance_transform_edt on region_mask and
    thresholding at tolerance.
    """
    dilated = distance_transform_edt(~region_mask) <= tolerance
    x, y = point
    col = int(round(x))
    row = int(round(y))
    H, W = region_mask.shape
    if row < 0 or row >= H or col < 0 or col >= W:
        return False
    return bool(dilated[row, col])


def compute_success_at_r(
    polyline: List[Tuple[float, float]],
    clearance_map: np.ndarray,
    start_mask: np.ndarray,
    goal_mask: np.ndarray,
    radius: float,
    tolerance: int = 4,
) -> Tuple[bool, bool, bool]:
    """Compute success@r metric.

    Returns (success, start_ok, goal_ok).
    success = start_ok AND goal_ok AND all points have clearance >= radius.
    """
    start_ok = check_endpoint(polyline[0], start_mask, tolerance)
    goal_ok = check_endpoint(polyline[-1], goal_mask, tolerance)

    H, W = clearance_map.shape
    all_clear = True
    for x, y in polyline:
        col = int(round(x))
        row = int(round(y))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        if clearance_map[row, col] < radius:
            all_clear = False
            break

    success = start_ok and goal_ok and all_clear
    return (success, start_ok, goal_ok)


def compute_valid_frac(
    polyline: List[Tuple[float, float]],
    clearance_map: np.ndarray,
    radius: float,
) -> float:
    """Fraction of points with clearance >= radius."""
    if len(polyline) == 0:
        return 0.0

    H, W = clearance_map.shape
    valid_count = 0
    for x, y in polyline:
        col = int(round(x))
        row = int(round(y))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        if clearance_map[row, col] >= radius:
            valid_count += 1

    return valid_count / len(polyline)


def compute_min_clearance(
    polyline: List[Tuple[float, float]],
    clearance_map: np.ndarray,
) -> float:
    """Minimum clearance along the path."""
    if len(polyline) == 0:
        return 0.0

    H, W = clearance_map.shape
    min_c = float("inf")
    for x, y in polyline:
        col = int(round(x))
        row = int(round(y))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        c = clearance_map[row, col]
        if c < min_c:
            min_c = c

    return float(min_c)


def compute_goal_distance(
    polyline: List[Tuple[float, float]],
    goal_mask: np.ndarray,
) -> float:
    """Euclidean distance from last point to nearest True pixel in goal_mask."""
    dist_map = distance_transform_edt(~goal_mask)
    x, y = polyline[-1]
    col = int(round(x))
    row = int(round(y))
    H, W = goal_mask.shape
    col = max(0, min(W - 1, col))
    row = max(0, min(H - 1, row))
    return float(dist_map[row, col])


def compute_path_length(polyline: List[Tuple[float, float]]) -> float:
    """Sum of euclidean distances between consecutive points."""
    total = 0.0
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i - 1][0]
        dy = polyline[i][1] - polyline[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def compute_length_regret(pred_length: float, gt_length: float) -> float:
    """Compute (pred_length - gt_length) / gt_length."""
    if gt_length == 0:
        return 0.0
    return (pred_length - gt_length) / gt_length


def compute_mono_score(
    polyline: List[Tuple[float, float]],
    gt_polyline: List[Tuple[float, float]],
    lambda_val: float = 0.1,
) -> float:
    """Compute monotonicity score measuring how consistently the predicted
    path progresses along the GT path without backtracking.

    1. Compute GT cumulative arc lengths for each GT point
    2. For each predicted point, find closest GT point, get its arc length = "progress"
    3. Walk progress values. Track max_progress. Accumulate backtracking.
    4. Return exp(-lambda_val * backtrack_total / total_gt_length)
    """
    if len(gt_polyline) < 2 or len(polyline) < 2:
        return 1.0

    # Step 1: GT cumulative arc lengths
    gt_arc = [0.0]
    for i in range(1, len(gt_polyline)):
        dx = gt_polyline[i][0] - gt_polyline[i - 1][0]
        dy = gt_polyline[i][1] - gt_polyline[i - 1][1]
        gt_arc.append(gt_arc[-1] + math.sqrt(dx * dx + dy * dy))

    total_gt_length = gt_arc[-1]
    if total_gt_length == 0:
        return 1.0

    # Convert GT points to numpy for efficient nearest-neighbor lookup
    gt_arr = np.array(gt_polyline, dtype=np.float64)
    gt_arc_arr = np.array(gt_arc, dtype=np.float64)

    # Step 2: For each predicted point, find closest GT point's arc length
    progress_values = []
    for px, py in polyline:
        dists_sq = (gt_arr[:, 0] - px) ** 2 + (gt_arr[:, 1] - py) ** 2
        nearest_idx = int(np.argmin(dists_sq))
        progress_values.append(gt_arc_arr[nearest_idx])

    # Step 3: Walk progress values, accumulate backtracking
    max_progress = progress_values[0]
    backtrack_total = 0.0
    for p in progress_values[1:]:
        if p > max_progress:
            max_progress = p
        else:
            backtrack_total += max_progress - p

    # Step 4: Return score
    return math.exp(-lambda_val * backtrack_total / total_gt_length)
