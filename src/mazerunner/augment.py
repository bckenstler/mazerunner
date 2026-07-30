"""Geometry-level augmentation.

Transforms are applied to world geometry *before* rasterization, so the mask,
render, reference route, and geodesic all derive from the same transformed
world — augmentation can never desynchronize ground truth. Every sampled
value is returned as a JSON dict and recorded with the task; the augmented
world is re-validated by the full fail-closed pipeline afterwards.

Pipeline per transform: optional mirror flip → quarter turn → (curved
families only) small free rotation → fit-to-canvas with margin. The fit step
guarantees geometry stays in bounds for any target canvas and any rotation.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .geometry import Point
from .world import World

CANVAS_PRESETS = [(800, 740), (740, 800), (900, 700), (960, 780)]
FIT_MARGIN = 46.0
# Families whose look is axis-aligned keep 90° symmetry only; curved families
# also take a small free rotation.
FREE_ROTATION_FAMILIES = {"organic", "island", "radial"}
WIDTH_JITTER = (0.9, 1.12)


def sample_augmentation(rng: np.random.Generator, family: str) -> dict:
    return {
        "flip": bool(rng.integers(2)),
        "quarter_turns": int(rng.integers(4)),
        "rotation_deg": (
            float(np.round(rng.uniform(-8, 8), 2)) if family in FREE_ROTATION_FAMILIES else 0.0
        ),
        "canvas": list(CANVAS_PRESETS[int(rng.integers(len(CANVAS_PRESETS)))]),
        "width_scale": float(np.round(rng.uniform(*WIDTH_JITTER), 3)),
    }


def _transform_factory(params: dict, src_w: float, src_h: float):
    flip = params["flip"]
    quarters = params["quarter_turns"] % 4
    theta = math.radians(params["rotation_deg"])
    cx, cy = src_w / 2, src_h / 2

    def transform(p: Point) -> Point:
        x, y = p
        if flip:
            x = src_w - x
        for _ in range(quarters):
            x, y = src_h - y, x  # 90° CW in a (w,h) -> (h,w) frame
        if theta:
            dx, dy = x - cx, y - cy
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            x = cx + dx * cos_t - dy * sin_t
            y = cy + dx * sin_t + dy * cos_t
        return x, y

    return transform


def apply_augmentation(world: World, params: dict) -> World:
    out = copy.deepcopy(world)
    transform = _transform_factory(params, world.width, world.height)

    points: list[Point] = []

    def collect(p: Point) -> Point:
        q = transform(p)
        points.append(q)
        return q

    out.nodes = {n: collect(p) for n, p in out.nodes.items()}
    for e in out.edges:
        e.geometry = [collect(p) for p in e.geometry]
        e.width_px = e.width_px * params["width_scale"]
    rect_nodes = list(out.node_rects.items())
    rect_corners = {
        n: [collect((x0, y0)), collect((x1, y1))] for n, (x0, y0, x1, y1) in rect_nodes
    }
    prim_records = []
    for prim in out.extra_open:
        if prim[0] == "rect":
            _, x0, y0, x1, y1 = prim
            prim_records.append(("rect", collect((x0, y0)), collect((x1, y1))))
        elif prim[0] == "disk":
            _, x, y, r = prim
            prim_records.append(("disk", collect((x, y)), r))

    # Fit everything into the target canvas with a fixed margin.
    target_w, target_h = params["canvas"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = max(e.width_px for e in out.edges) / 2 + max(
        list(out.node_pad_radius.values()) or [0]
    )
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    scale = min(
        (target_w - 2 * FIT_MARGIN) / max(1.0, max_x - min_x),
        (target_h - 2 * FIT_MARGIN) / max(1.0, max_y - min_y),
    )
    scale = min(max(scale, 0.8), 1.3)
    off_x = FIT_MARGIN + (target_w - 2 * FIT_MARGIN - (max_x - min_x) * scale) / 2
    off_y = FIT_MARGIN + (target_h - 2 * FIT_MARGIN - (max_y - min_y) * scale) / 2

    def fit(p: Point) -> Point:
        return ((p[0] - min_x) * scale + off_x, (p[1] - min_y) * scale + off_y)

    out.nodes = {n: fit(p) for n, p in out.nodes.items()}
    for e in out.edges:
        e.geometry = [fit(p) for p in e.geometry]
        e.width_px = max(e.width_px * scale, 2 * out.pointer_radius_px + 4)
    out.node_pad_radius = {n: r * scale for n, r in out.node_pad_radius.items()}
    out.node_rects = {}
    for n, (c0, c1) in rect_corners.items():
        (x0, y0), (x1, y1) = fit(c0), fit(c1)
        out.node_rects[n] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    out.extra_open = []
    for record in prim_records:
        if record[0] == "rect":
            (x0, y0), (x1, y1) = fit(record[1]), fit(record[2])
            out.extra_open.append(("rect", min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        else:
            (x, y) = fit(record[1])
            out.extra_open.append(("disk", x, y, record[2] * scale))

    out.width, out.height = int(target_w), int(target_h)
    return out
