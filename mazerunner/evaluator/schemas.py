"""Validation for submission and ground truth JSON."""

import json


VALID_ENCODINGS = {"polyline", "delta", "cell_route", "multi_segment"}

REQUIRED_REGION_KEYS = {
    "free_space_mask_rle",
    "wall_mask_rle",
    "start_mask_rle",
    "goal_mask_rle",
}


def validate_submission_entry(entry: dict) -> dict:
    """Validate a single submission entry.

    Must have "id" (str) and "prediction" (dict).
    Prediction must have "encoding" (one of the valid encodings) and "data".
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Submission entry must be a dict, got {type(entry).__name__}")

    if "id" not in entry:
        raise ValueError("Submission entry missing required key 'id'")
    if not isinstance(entry["id"], str):
        raise ValueError(f"Submission entry 'id' must be a string, got {type(entry['id']).__name__}")

    if "prediction" not in entry:
        raise ValueError(f"Submission entry '{entry['id']}' missing required key 'prediction'")
    pred = entry["prediction"]
    if not isinstance(pred, dict):
        raise ValueError(f"Submission entry '{entry['id']}' prediction must be a dict")

    if "encoding" not in pred:
        raise ValueError(f"Submission entry '{entry['id']}' prediction missing 'encoding'")
    if pred["encoding"] not in VALID_ENCODINGS:
        raise ValueError(
            f"Submission entry '{entry['id']}' has unknown encoding '{pred['encoding']}'. "
            f"Valid: {sorted(VALID_ENCODINGS)}"
        )

    if "data" not in pred:
        raise ValueError(f"Submission entry '{entry['id']}' prediction missing 'data'")

    return entry


def validate_gt_entry(gt: dict) -> dict:
    """Validate ground truth JSON.

    Must have "id", "image_size" (with "w" and "h"), "regions" (with all
    required mask RLEs), and "gt" (with "solution_polyline", "solution_length").
    """
    if not isinstance(gt, dict):
        raise ValueError(f"Ground truth entry must be a dict, got {type(gt).__name__}")

    if "id" not in gt:
        raise ValueError("Ground truth entry missing required key 'id'")

    # image_size
    if "image_size" not in gt:
        raise ValueError(f"GT '{gt['id']}' missing 'image_size'")
    img = gt["image_size"]
    if not isinstance(img, dict):
        raise ValueError(f"GT '{gt['id']}' image_size must be a dict")
    for k in ("w", "h"):
        if k not in img:
            raise ValueError(f"GT '{gt['id']}' image_size missing '{k}'")

    # regions
    if "regions" not in gt:
        raise ValueError(f"GT '{gt['id']}' missing 'regions'")
    regions = gt["regions"]
    if not isinstance(regions, dict):
        raise ValueError(f"GT '{gt['id']}' regions must be a dict")
    for k in REQUIRED_REGION_KEYS:
        if k not in regions:
            raise ValueError(f"GT '{gt['id']}' regions missing '{k}'")

    # gt section
    if "gt" not in gt:
        raise ValueError(f"GT '{gt['id']}' missing 'gt'")
    gt_sec = gt["gt"]
    if not isinstance(gt_sec, dict):
        raise ValueError(f"GT '{gt['id']}' gt must be a dict")
    for k in ("solution_polyline", "solution_length"):
        if k not in gt_sec:
            raise ValueError(f"GT '{gt['id']}' gt missing '{k}'")

    return gt


def load_submission(path: str) -> list:
    """Load submission file. Supports JSON (list) and JSONL (one JSON per line)."""
    with open(path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Submission file is empty: {path}")

    # Try JSON first (list)
    if content.startswith("["):
        entries = json.loads(content)
        if not isinstance(entries, list):
            raise ValueError(f"Submission JSON must be a list, got {type(entries).__name__}")
    else:
        # JSONL: one JSON object per line
        entries = []
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")

    validated = []
    for i, entry in enumerate(entries):
        try:
            validated.append(validate_submission_entry(entry))
        except ValueError as e:
            raise ValueError(f"Submission entry {i}: {e}")

    return validated


def load_gt(path: str) -> dict:
    """Load a single GT JSON file. Validate and return."""
    with open(path, "r") as f:
        gt = json.load(f)
    return validate_gt_entry(gt)
