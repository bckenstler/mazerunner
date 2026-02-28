"""Agentic session manager for incremental maze path building."""

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from mazerunner.common.rle import decode_rle
from mazerunner.common.types import EvalResult
from mazerunner.evaluator.canonicalize import clamp_points, densify
from mazerunner.evaluator.evaluate import evaluate_single
from mazerunner.evaluator.metrics import check_endpoint, compute_clearance_map


class SegmentStatus(enum.Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED_WALL = "REJECTED_WALL"
    REJECTED_CONTIGUITY = "REJECTED_CONTIGUITY"
    REJECTED_EMPTY = "REJECTED_EMPTY"
    REJECTED_NOT_IN_START = "REJECTED_NOT_IN_START"


@dataclass
class SegmentResult:
    status: SegmentStatus
    segment_index: int
    reason: str
    violation_point: Optional[Tuple[float, float]] = None
    path_length_so_far: float = 0.0
    num_points_so_far: int = 0


@dataclass
class SessionStats:
    total_segments_submitted: int = 0
    segments_accepted: int = 0
    segments_rejected: int = 0
    wall_rejections: int = 0
    contiguity_rejections: int = 0
    total_tool_calls: int = 0


@dataclass
class SessionResult:
    eval_result: Optional[EvalResult]
    stats: SessionStats
    accepted_path: List[Tuple[float, float]]
    finish_reason: str


class MazeSession:
    """Core session manager for incremental path building.

    Agents submit path segments incrementally via submit_segment(). Each
    segment is validated (contiguity, wall avoidance) and either accepted
    or rejected with feedback. Call finish() to score the accumulated path.
    """

    def __init__(
        self,
        gt_data: dict,
        contiguity_tolerance: float | None = None,
        wall_check_densify_gap: float = 2.0,
        endpoint_tolerance: int | None = None,
    ):
        self._gt_data = gt_data

        # Scale tolerances based on rendered corridor width
        cw = gt_data.get("render_config", {}).get("corridor_width", 30)
        self._contiguity_tolerance = contiguity_tolerance if contiguity_tolerance is not None else max(3.0, cw * 0.1)
        self._wall_check_densify_gap = wall_check_densify_gap
        self._endpoint_tolerance = endpoint_tolerance if endpoint_tolerance is not None else max(10, int(cw * 0.3))

        # Decode masks
        regions = gt_data["regions"]
        self._wall_mask = decode_rle(regions["wall_mask_rle"])
        self._free_space_mask = decode_rle(regions["free_space_mask_rle"])
        self._start_mask = decode_rle(regions["start_mask_rle"])
        self._goal_mask = decode_rle(regions["goal_mask_rle"])

        # Image bounds
        self._W = gt_data["image_size"]["w"]
        self._H = gt_data["image_size"]["h"]

        # Pre-compute clearance map for final scoring
        self._clearance_map = compute_clearance_map(self._free_space_mask)

        # Accumulated path and stats
        self._path: List[Tuple[float, float]] = []
        self._stats = SessionStats()
        self._segment_index = 0
        self._finished = False

    def submit_segment(self, points: List[List[float]]) -> SegmentResult:
        """Validate and append a path segment.

        Validation pipeline:
        1. Non-empty check
        2. Clamp to image bounds
        3. Start region check (first segment only)
        4. Contiguity check (subsequent segments)
        5. Densify for wall detection
        6. Wall check on densified points
        7. Accept and append clamped points
        """
        if self._finished:
            raise RuntimeError("Session already finished")

        self._stats.total_segments_submitted += 1
        self._stats.total_tool_calls += 1
        idx = self._segment_index

        # 1. Non-empty check
        if not points:
            self._stats.segments_rejected += 1
            return SegmentResult(
                status=SegmentStatus.REJECTED_EMPTY,
                segment_index=idx,
                reason="Segment is empty (no points provided).",
                path_length_so_far=self._path_length(),
                num_points_so_far=len(self._path),
            )

        # Parse to tuples
        parsed = [(float(p[0]), float(p[1])) for p in points]

        # 2. Clamp to image bounds
        clamped = clamp_points(parsed, self._W, self._H)

        # 3. Start region check (first segment only)
        if not self._path:
            first_pt = clamped[0]
            if not check_endpoint(first_pt, self._start_mask, tolerance=self._endpoint_tolerance):
                self._stats.segments_rejected += 1
                return SegmentResult(
                    status=SegmentStatus.REJECTED_NOT_IN_START,
                    segment_index=idx,
                    reason=(
                        f"First point ({first_pt[0]:.1f}, {first_pt[1]:.1f}) "
                        f"is not within the start region."
                    ),
                    violation_point=first_pt,
                    path_length_so_far=self._path_length(),
                    num_points_so_far=len(self._path),
                )

        # 4. Contiguity check (subsequent segments)
        if self._path:
            last_accepted = self._path[-1]
            first_new = clamped[0]
            gap = (
                (last_accepted[0] - first_new[0]) ** 2
                + (last_accepted[1] - first_new[1]) ** 2
            ) ** 0.5
            if gap > self._contiguity_tolerance:
                self._stats.segments_rejected += 1
                self._stats.contiguity_rejections += 1
                return SegmentResult(
                    status=SegmentStatus.REJECTED_CONTIGUITY,
                    segment_index=idx,
                    reason=(
                        f"Gap of {gap:.1f}px between last accepted point "
                        f"and first point of segment exceeds tolerance "
                        f"({self._contiguity_tolerance}px)."
                    ),
                    violation_point=first_new,
                    path_length_so_far=self._path_length(),
                    num_points_so_far=len(self._path),
                )

        # 5. Densify for wall checking
        densified = densify(clamped, self._wall_check_densify_gap)

        # 6. Wall check
        for dx, dy in densified:
            col = int(round(dx))
            row = int(round(dy))
            col = max(0, min(self._W - 1, col))
            row = max(0, min(self._H - 1, row))
            if self._wall_mask[row, col]:
                self._stats.segments_rejected += 1
                self._stats.wall_rejections += 1
                return SegmentResult(
                    status=SegmentStatus.REJECTED_WALL,
                    segment_index=idx,
                    reason=f"Path hits a wall at ({dx:.1f}, {dy:.1f}).",
                    violation_point=(dx, dy),
                    path_length_so_far=self._path_length(),
                    num_points_so_far=len(self._path),
                )

        # 7. Accept — append clamped (not densified) points
        self._path.extend(clamped)
        self._stats.segments_accepted += 1
        self._segment_index += 1

        return SegmentResult(
            status=SegmentStatus.ACCEPTED,
            segment_index=idx,
            reason=f"Segment accepted ({len(clamped)} points).",
            path_length_so_far=self._path_length(),
            num_points_so_far=len(self._path),
        )

    def finish(self) -> SessionResult:
        """Score the accumulated path and return results."""
        if self._finished:
            raise RuntimeError("Session already finished")

        self._finished = True
        self._stats.total_tool_calls += 1

        if not self._path:
            return SessionResult(
                eval_result=None,
                stats=self._stats,
                accepted_path=[],
                finish_reason="empty_path",
            )

        if len(self._path) < 2:
            return SessionResult(
                eval_result=None,
                stats=self._stats,
                accepted_path=list(self._path),
                finish_reason="single_point",
            )

        # Build prediction in standard format
        prediction = {
            "encoding": "polyline",
            "data": {"points": [[p[0], p[1]] for p in self._path]},
        }

        eval_result = evaluate_single(prediction, self._gt_data)

        return SessionResult(
            eval_result=eval_result,
            stats=self._stats,
            accepted_path=list(self._path),
            finish_reason="completed",
        )

    def _path_length(self) -> float:
        """Compute accumulated path length in pixels."""
        total = 0.0
        for i in range(1, len(self._path)):
            dx = self._path[i][0] - self._path[i - 1][0]
            dy = self._path[i][1] - self._path[i - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
        return total
