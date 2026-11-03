"""The v2 difficulty ladder: every family at four structural size levels, plus
warped twins of the grid families.

v1 samples a tier per slot and accepts whatever the measured tier comes out
as. v2 holds the far-endpoint band fixed and varies only size: each family
has levels L1..L4 (L1 = v1's medium settings, so v1 results can be placed on
the new scale), and every level of a grid family (rectilinear, braided,
pipes, island) also gets a warped twin labelled e.g. `L3w`.

A twin is built from the *same* attempt as its straight task: same seeds,
same overrides, same archetype and style seed, with only the warp added.
Generators pick endpoints before warping, so the pair shares topology,
start, and goal, and the score gap between them isolates what irregular
geometry costs. A pair is accepted only when both halves pass every
fail-closed stage; otherwise the slot moves to its next attempt.

Every task must also be drawable within the submission cap: its stored
`reference.optimal_path` is the fewest-point simplification of the certified
geodesic that still passes the scorer, and an attempt with no such path
under MAX_POINTS is rejected. (v1 stores a uniform 200-point thinning for
display, which cuts corners on v2's longest routes.)

Task ids read `<family>-<level>-s<slot>`, and twins share the slot number
(`rectilinear-L3-s0042` / `rectilinear-L3w-s0042`). The tier field carries the
level label; the v1 measured tier is kept as `measures.measured_tier`.
Verification, stats, and QC sheets are the v1 ones in dataset.py.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dataset import (
    MAX_ATTEMPTS_PER_SLOT,
    _attempt,
    _draw,
    _environment_record,
    _index_row,
    _Rejected,
    _save,
    _seed_record,
    _slot_seeds,
    _supports,
)
from .contract import MAX_POINTS
from .evaluator import evaluate_task
from .generators import FAMILIES
from .geometry import simplify_polyline
from .io import _norm

# Simplification tolerances tried coarsest first; the first passing one wins.
DRAWABLE_EPSILONS_PX = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25)


def ladder_plan(config: dict) -> list[tuple[int, str, str]]:
    """(slot, family, level) for every straight task, in presentation order.
    A warped twin shares its straight task's slot."""
    plan = []
    for family in FAMILIES:
        for level in sorted(config["levels"][family]):
            for _ in range(config["per_cell"]):
                plan.append((len(plan), family, level))
    return plan


def drawable_reference(built: dict) -> tuple[list[dict], float]:
    """Fewest-point simplification of the geodesic that passes the scorer, in
    normalized coordinates, and the tolerance that produced it. Scored exactly
    as a model submission would be, against the task record save_task writes.
    Raises _Rejected when nothing under MAX_POINTS passes."""
    world, validation = built["world"], built["validation"]
    w, h = world.width, world.height
    record = {
        "width": w, "height": h,
        "start": _norm(world.start_px, w, h), "goal": _norm(world.goal_px, w, h),
        "start_radius_px": round(validation.start_radius_px, 1),
        "goal_radius_px": round(validation.goal_radius_px, 1),
        "pointer_radius_px": world.pointer_radius_px,
        "reference": {"optimal_length_px": validation.reference_length_px,
                      "optimal_length_px_geometric": validation.geodesic_length_px},
    }
    for epsilon in DRAWABLE_EPSILONS_PX:
        points = simplify_polyline(validation.geodesic_points_px, epsilon)
        if len(points) > MAX_POINTS:
            break
        path = [_norm(p, w, h) for p in points]
        if evaluate_task(record, built["mask"], {"points": path}).success:
            return path, epsilon
    raise _Rejected("drawable", f"no simplification within {MAX_POINTS} points passes the scorer")


def build_ladder_slot(config: dict, slot: int, family: str, level: str, attempt_offset: int = 0) -> list[dict]:
    """Build one straight task and, where the family warps, its twin from the
    same attempt. Returns their index rows, or one error row."""
    spec = {**config.get("mask", {}), **config["levels"][family][level]}
    warp = config.get("warp", {}).get(family, {}).get(level)
    band = tuple(config["endpoint_band"])
    allowed = [a for a in config["archetypes"] if _supports(a, family)]
    split = config["split"]
    rejections: list[dict] = []
    for attempt in range(attempt_offset, attempt_offset + MAX_ATTEMPTS_PER_SLOT):
        seeds = _slot_seeds(config["master_seed"], config["seed_base"], slot, attempt)
        overrides, archetype = _draw(seeds, [spec], band, allowed)
        variants = [(level, overrides)]
        if warp:
            variants.append((level + "w", {**overrides, "warp": list(warp)}))
        built = []
        try:
            for label, variant in variants:
                task = _attempt(family, seeds, variant, archetype)
                task["drawable"] = drawable_reference(task)
                built.append((label, task))
        except _Rejected as rej:
            rejections.append({"attempt": attempt, "variant": label, "stage": rej.stage,
                               "detail": rej.detail[:300], "seeds": seeds})
            continue

        ids = {label: f"{family}-{label}-s{slot:04d}" for label, _ in built}
        rows = []
        for label, task in built:
            task["measures"] = {**task["measures"], "tier": label,
                                "measured_tier": task["measures"]["tier"]}
            provenance = {
                **_seed_record(split, slot, attempt, config["master_seed"], config["seed_base"], seeds),
                "family": family,
                "tier_target": label,
                "tier_relaxed": False,
                "level": level,
                "warped": label != level,
            }
            twin = [tid for other, tid in ids.items() if other != label]
            if twin:
                provenance["twin"] = twin[0]
            task_dir, task_json = _save(task, ids[label], config["out_dir"], split, provenance)
            path, epsilon = task["drawable"]
            task_json["reference"]["optimal_path"] = path
            task_json["reference"]["optimal_path_epsilon_px"] = epsilon
            provenance["rejections"] = rejections
            provenance["environment"] = _environment_record()
            task_json["provenance"] = provenance
            (task_dir / "task.json").write_text(json.dumps(task_json, indent=2))
            row = _index_row(ids[label], task_dir, family, task, label, label, False,
                             attempt - attempt_offset + 1, len(rejections))
            rows.append({**row, "level": level, "warped": label != level, "twin": twin[0] if twin else None})
        return rows
    return [{"error": f"slot {slot} ({family}/{level}): no pair passed in {MAX_ATTEMPTS_PER_SLOT}",
             "rejections": rejections}]


def _known_hashes(paths: list[str]) -> dict[str, str]:
    """Semantic hashes of already-built datasets the ladder must not repeat."""
    known: dict[str, str] = {}
    for root in paths:
        if root.endswith(".txt"):
            # A published hash list: dedup against a dataset no longer shipped.
            for line in Path(root).read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    known[line.strip()] = root
            continue
        for index in sorted(Path(root).glob("*/index.jsonl")):
            for line in index.read_text().splitlines():
                row = json.loads(line)
                known[row["graph_hash"]] = f"{index.parent}/{row['task_id']}"
    return known


def build_ladder(config_path: Path, workers: int = 8) -> dict:
    """Build the whole ladder split, dedup it, and write index + manifest.

    Dedup runs on straight tasks only: a twin repeats its straight task's
    graph by design. A duplicate slot is rebuilt from fresh attempts, pair
    and all.
    """
    config = json.loads(config_path.read_text())
    out_dir = Path(config["out_dir"])
    split_dir = out_dir / config["split"]
    if split_dir.exists():
        shutil.rmtree(split_dir)
    plan = ladder_plan(config)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_ladder_slot, config, *job) for job in plan]
        results = []
        for i, future in enumerate(futures):
            rows = future.result()
            if "error" in rows[0]:
                raise RuntimeError(rows[0]["error"])
            results.append(rows)
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(plan)} slots built")

    seen = _known_hashes(config.get("dedup_against", []))
    for i, rows in enumerate(results):
        offset = MAX_ATTEMPTS_PER_SLOT
        while rows[0]["graph_hash"] in seen:
            slot, family, level = plan[i]
            print(f"  slot {slot} duplicates {seen[rows[0]['graph_hash']]}, resampling")
            for row in rows:
                shutil.rmtree(row["dir"])
            rows = build_ladder_slot(config, slot, family, level, attempt_offset=offset)
            if "error" in rows[0]:
                raise RuntimeError(rows[0]["error"])
            offset += MAX_ATTEMPTS_PER_SLOT
            results[i] = rows
        seen[rows[0]["graph_hash"]] = f"{config['split']}/{rows[0]['task_id']}"

    flat = [row for rows in results for row in rows]
    (split_dir / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in flat))
    summary = {
        "split": config["split"],
        "size": len(flat),
        "total_rejections": sum(rows[0]["rejections"] for rows in results),
        "tiers": dict(Counter(row["tier"] for row in flat)),
        "archetypes": dict(Counter(row["archetype"] for row in flat)),
    }
    manifest = {"config": config, "environment": _environment_record(),
                "summaries": {config["split"]: summary}, "total_semantic_hashes": len(seen)}
    (out_dir / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
