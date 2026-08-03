"""Continuous-path evaluator.

Scores a submitted normalized trajectory against the hidden traversability
mask. Every straight segment between consecutive waypoints is sampled at
sub-pixel spacing and a swept pointer disk is required to stay entirely inside
the mask, so listing legal waypoints while cutting through a wall between them
always fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .contract import validate_submission

SAMPLE_SPACING_PX = 0.75
# The efficiency denominator is the mask-certified geometric optimum, so a
# legally shorter submission should be impossible; the small margin covers the
# string-pulling approximation in geodesic.py. A raw efficiency above this
# signals a mask more permissive than certification believed — the
# permissive-mask canary in the hardening checks (docs/USAGE.md).
EFFICIENCY_CANARY_THRESHOLD = 1.02


@dataclass
class Evaluation:
    """One submission's verdict, and the `evaluation` record in attempts.jsonl.

    The three success gates are stored separately rather than collapsed into
    `success` alone, because failure-mode classification needs to tell "never
    left the start" from "drove through a wall"; the taxonomy in
    results/failure-modes.md is built on that separation.
    """

    success: bool = False
    schema_valid: bool = False
    schema_error: str | None = None
    starts_correctly: bool = False
    ends_correctly: bool = False
    collision_free: bool = False
    first_collision: dict | None = None
    point_count: int = 0
    path_length_px: float = 0.0
    reference_length_px: float = 0.0
    efficiency: float = 0.0
    efficiency_raw: float = 0.0
    efficiency_canary: bool = False
    min_clearance_px: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The serialized record. Rounding is deliberate, not cosmetic:
        fixing pixel values at 2dp and efficiency at 4dp keeps a re-score
        byte-comparable against the stored run across platforms."""
        return {
            "success": self.success,
            "schema_valid": self.schema_valid,
            "schema_error": self.schema_error,
            "starts_correctly": self.starts_correctly,
            "ends_correctly": self.ends_correctly,
            "collision_free": self.collision_free,
            "first_collision": self.first_collision,
            "point_count": self.point_count,
            "path_length_px": round(self.path_length_px, 2),
            "reference_length_px": round(self.reference_length_px, 2),
            "efficiency": round(self.efficiency, 4),
            "efficiency_raw": round(self.efficiency_raw, 4),
            "efficiency_canary": self.efficiency_canary,
            "min_clearance_px": (
                None if self.min_clearance_px is None else round(self.min_clearance_px, 2)
            ),
            "warnings": self.warnings,
        }


def disk_offsets(radius: int) -> np.ndarray:
    """Integer (dy, dx) offsets covering a disk of the given pixel radius."""
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    keep = dy * dy + dx * dx <= radius * radius
    return np.stack([dy[keep], dx[keep]], axis=1)


def to_pixels(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    """Normalized [0,1] coordinate to pixel space. The model's coordinates are
    normalized; everything inside the scorer is pixels."""
    return (point[0] * (width - 1), point[1] * (height - 1))


def sample_segment(
    a: tuple[float, float], b: tuple[float, float], spacing: float = SAMPLE_SPACING_PX
) -> list[tuple[float, float]]:
    """Samples at t in {1/n, ..., 1}; the segment start is the previous sample."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, math.ceil(length / spacing))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]


def _disk_clear(
    mask: np.ndarray, center: tuple[float, float], offsets: np.ndarray
) -> bool:
    h, w = mask.shape
    cy = int(round(center[1]))
    cx = int(round(center[0]))
    ys = offsets[:, 0] + cy
    xs = offsets[:, 1] + cx
    if ys.min() < 0 or xs.min() < 0 or ys.max() >= h or xs.max() >= w:
        return False
    return bool(mask[ys, xs].all())


def check_path_collision(
    mask: np.ndarray,
    pixel_points: list[tuple[float, float]],
    pointer_radius: int,
) -> tuple[bool, dict | None]:
    """Return (collision_free, first_collision_info).

    Checks the swept disk at the first point and then along every segment, so
    a path whose waypoints are all legal but whose straight line between two
    of them crosses a wall still fails. Reporting the *first* collision rather
    than a count is what makes the replay's ⊗ marker meaningful.
    """
    offsets = disk_offsets(pointer_radius)
    if not _disk_clear(mask, pixel_points[0], offsets):
        return False, {"segment_index": 0, "x_px": pixel_points[0][0], "y_px": pixel_points[0][1]}
    for i in range(len(pixel_points) - 1):
        for sample in sample_segment(pixel_points[i], pixel_points[i + 1]):
            if not _disk_clear(mask, sample, offsets):
                return False, {
                    "segment_index": i,
                    "x_px": round(sample[0], 2),
                    "y_px": round(sample[1], 2),
                }
    return True, None


def min_clearance(mask: np.ndarray, pixel_points: list[tuple[float, float]]) -> float:
    """Approximate minimum distance from the sampled path to the nearest wall."""
    clearance_map = ndimage.distance_transform_edt(mask)
    samples = [pixel_points[0]]
    for i in range(len(pixel_points) - 1):
        samples.extend(sample_segment(pixel_points[i], pixel_points[i + 1], spacing=2.0))
    h, w = mask.shape
    best = float("inf")
    for x, y in samples:
        cy = min(max(int(round(y)), 0), h - 1)
        cx = min(max(int(round(x)), 0), w - 1)
        best = min(best, float(clearance_map[cy, cx]))
    return best


def evaluate(
    arguments: object,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    start_px: tuple[float, float],
    goal_px: tuple[float, float],
    start_radius_px: float,
    goal_radius_px: float,
    pointer_radius_px: int,
    reference_length_px: float,
    compute_clearance: bool = True,
) -> Evaluation:
    """Score one submitted path. The benchmark's scoring core, frozen for v1.

    Success is the conjunction of three independent gates — starts in the
    start badge, ends in the goal badge, and never collides — each recorded
    separately so a failure can be attributed rather than just counted.

    A submission that fails schema validation returns immediately with every
    gate false: a malformed path is a *failure*, not an error. There is no
    partial credit and no retry.

    Efficiency is computed only on success. `efficiency` is capped at 1.0 for
    reporting while `efficiency_raw` is left uncapped, which is what lets the
    canary above 1.02 fire — a legal path shorter than the certified optimum
    means the mask is more permissive than certification believed.

    `compute_clearance=False` skips a full distance transform per call; the
    tolerance sweep re-scores thousands of stored submissions and never reads
    the clearance.
    """
    ev = Evaluation(reference_length_px=reference_length_px)

    points, error = validate_submission(arguments)
    if points is None:
        ev.schema_error = error
        return ev
    ev.schema_valid = True
    ev.point_count = len(points)

    pixel_points = [to_pixels(p, width, height) for p in points]
    ev.path_length_px = sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pixel_points, pixel_points[1:])
    )

    ev.starts_correctly = (
        math.hypot(pixel_points[0][0] - start_px[0], pixel_points[0][1] - start_px[1])
        <= start_radius_px
    )
    ev.ends_correctly = (
        math.hypot(pixel_points[-1][0] - goal_px[0], pixel_points[-1][1] - goal_px[1])
        <= goal_radius_px
    )

    ev.collision_free, ev.first_collision = check_path_collision(
        mask, pixel_points, pointer_radius_px
    )
    # A full distance transform over the mask on every call; the tolerance
    # sweep re-scores thousands of stored submissions and does not need it.
    if compute_clearance:
        ev.min_clearance_px = min_clearance(mask, pixel_points)

    ev.success = ev.starts_correctly and ev.ends_correctly and ev.collision_free
    if ev.success and ev.path_length_px > 0:
        ev.efficiency_raw = reference_length_px / ev.path_length_px
        ev.efficiency = min(1.0, ev.efficiency_raw)
        if ev.efficiency_raw > EFFICIENCY_CANARY_THRESHOLD:
            ev.efficiency_canary = True
            ev.warnings.append(
                f"efficiency_raw={ev.efficiency_raw:.3f} exceeds "
                f"{EFFICIENCY_CANARY_THRESHOLD}: a valid path is much shorter than "
                "the reference — the mask may be more permissive than the graph"
            )
    return ev


def evaluate_task(task: dict, mask: np.ndarray, arguments: object) -> Evaluation:
    """Convenience wrapper taking a task.json record.

    Prefers the mask-certified geometric optimum as the efficiency
    denominator; falls back to the graph route length on older tasks.
    """
    width, height = task["width"], task["height"]
    reference = task["reference"]
    return evaluate(
        arguments,
        mask,
        width=width,
        height=height,
        start_px=to_pixels((task["start"]["x"], task["start"]["y"]), width, height),
        goal_px=to_pixels((task["goal"]["x"], task["goal"]["y"]), width, height),
        start_radius_px=task["start_radius_px"],
        goal_radius_px=task["goal_radius_px"],
        pointer_radius_px=task["pointer_radius_px"],
        reference_length_px=reference.get(
            "optimal_length_px_geometric", reference["optimal_length_px"]
        ),
    )
