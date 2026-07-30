"""Task artifact persistence: input.png, mask.png, task.json, ground-truth.json.

Masks are the scoring source of truth: their hash is computed over the raw
binary array (shape + packed bits) and verified byte-identically on every
validate (hardening fix 5). Styled renders may drift across platforms; masks
must not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Point, polyline_length
from .world import World, WorldValidation


def mask_sha256(mask: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{mask.shape[0]}x{mask.shape[1]};".encode())
    h.update(np.packbits(mask).tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(point: Point, width: int, height: int) -> dict:
    return {"x": round(point[0] / (width - 1), 4), "y": round(point[1] / (height - 1), 4)}


def _downsample(points: list[Point], max_points: int = 200) -> list[Point]:
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]


def save_task(
    task_dir: Path,
    world: World,
    mask: np.ndarray,
    image: Image.Image,
    validation: WorldValidation,
    style_record: dict | None = None,
) -> dict:
    task_dir.mkdir(parents=True, exist_ok=True)
    image.save(task_dir / "input.png")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(task_dir / "mask.png")

    w, h = world.width, world.height
    task = {
        "id": world.id,
        "type": world.type,
        "style": world.style,
        "state_representation": world.state_representation,
        "width": w,
        "height": h,
        "image_file": "input.png",
        "mask_file": "mask.png",
        "seed": world.seed,
        "generator": world.generator,
        "style_record": style_record,
        "pointer_radius_px": world.pointer_radius_px,
        "start_radius_px": round(validation.start_radius_px, 1),
        "goal_radius_px": round(validation.goal_radius_px, 1),
        "start": _norm(world.start_px, w, h),
        "goal": _norm(world.goal_px, w, h),
        "reference": {
            "solver": world.solver,
            "optimal_length_steps": validation.reference_steps,
            "optimal_length_px": round(validation.reference_length_px, 2),
            "optimal_length_px_geometric": round(validation.geodesic_length_px, 2),
            "corner_cut_headroom_px": round(
                validation.reference_length_px - validation.geodesic_length_px, 2
            ),
            "min_clearance_px": round(validation.min_clearance_px, 2),
            "optimal_path": [
                _norm(p, w, h) for p in _downsample(validation.geodesic_points_px)
            ],
        },
    }
    (task_dir / "task.json").write_text(json.dumps(task, indent=2))

    ground_truth = {
        "id": world.id,
        "seed": world.seed,
        "generator": world.generator,
        "nodes": {str(n): [round(x, 2), round(y, 2)] for n, (x, y) in world.nodes.items()},
        "edges": [
            {"a": e.a, "b": e.b, "width_px": e.width_px, "length_px": round(polyline_length(e.geometry), 2)}
            for e in world.edges
        ],
        "start_node": world.start_node,
        "goal_node": world.goal_node,
        "optimal_path_nodes": world.path_nodes,
    }
    (task_dir / "ground-truth.json").write_text(json.dumps(ground_truth, indent=2))
    return task


def load_task(task_dir: Path) -> tuple[dict, np.ndarray]:
    task = json.loads((task_dir / "task.json").read_text())
    mask = np.asarray(Image.open(task_dir / task["mask_file"]).convert("L")) > 127
    return task, mask


def reference_points_px(task: dict) -> list[Point]:
    w, h = task["width"], task["height"]
    return [(p["x"] * (w - 1), p["y"] * (h - 1)) for p in task["reference"]["optimal_path"]]
