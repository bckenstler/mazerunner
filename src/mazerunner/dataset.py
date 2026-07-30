"""Dataset construction: factorized sampling, dedup, splits, reproducibility.

Every random decision derives from the master seed via a documented
SeedSequence chain: (master, split_base, slot, attempt) spawns the topology,
style, augmentation, and difficulty streams for that attempt. A task's
provenance block records the seeds, the resolved parameter values, the
rejection history, certification thresholds/metrics, and environment — enough
to rebuild the task byte-identically from task.json alone.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from .augment import apply_augmentation, sample_augmentation
from .certify import Thresholds, certified_render
from .generators import FAMILIES
from .io import mask_sha256, save_task
from .measure import measure_task
from .overlay import render_overlay
from .styles import ARCHETYPES
from .world import adjacency, open_mask, validate_world

MAX_ATTEMPTS_PER_SLOT = 30

# Per-family difficulty spaces: coherent override tuples the sampler chooses
# from, plus an endpoint-quantile band per tier (the route-length lever).
QUANTILE_BANDS = {"easy": (0.4, 0.6), "medium": (0.6, 0.85), "hard": (0.9, 1.0)}

DIFFICULTY_SPACES: dict[str, dict[str, list[dict]]] = {
    "rectilinear": {
        "easy": [{"cols": 5, "rows": 5, "corridor": 30}, {"cols": 6, "rows": 5, "corridor": 28}],
        "medium": [{"cols": 7, "rows": 6, "corridor": 26}, {"cols": 8, "rows": 7, "corridor": 26}],
        "hard": [{"cols": 9, "rows": 8, "corridor": 22}, {"cols": 10, "rows": 9, "corridor": 20}],
    },
    "braided": {
        "easy": [{"cols": 5, "rows": 5, "corridor": 30, "loops": 5}],
        "medium": [{"cols": 7, "rows": 6, "corridor": 26, "loops": 9}, {"cols": 8, "rows": 7, "corridor": 26, "loops": 12}],
        "hard": [{"cols": 9, "rows": 8, "corridor": 22, "loops": 14}, {"cols": 10, "rows": 9, "corridor": 20, "loops": 18}],
    },
    "rooms": {
        "easy": [{"corridor": 22, "extra_edges": 1}],
        "medium": [{"corridor": 18, "extra_edges": 1}, {"corridor": 18, "extra_edges": 2}],
        "hard": [{"corridor": 16, "extra_edges": 0}],
    },
    "organic": {
        "easy": [{"nodes": 18, "corridor": 24, "min_dist": 118, "loops": 1}],
        "medium": [{"nodes": 30, "corridor": 22, "min_dist": 92, "loops": 1}],
        "hard": [{"nodes": 38, "corridor": 16, "min_dist": 72, "loops": 2}, {"nodes": 44, "corridor": 13, "min_dist": 62, "loops": 2}],
    },
    "cave": {
        # Cellular-automata viability drops fast below p≈0.6 (measured: 5% of
        # seeds pass at 0.55, 70% at 0.6); difficulty comes mostly from the
        # endpoint quantile, so the bands stay in the viable region.
        "easy": [{"open_probability": 0.66}],
        "medium": [{"open_probability": 0.62}],
        "hard": [{"open_probability": 0.6}],
    },
    "radial": {
        "easy": [{"rings": 3, "corridor": 26}],
        "medium": [{"rings": 4, "corridor": 24}],
        "hard": [{"rings": 5, "corridor": 22, "loops": 2}],
    },
    "island": {
        "easy": [{"islands": 11, "corridor": 18, "min_dist": 150, "extra_edges": 2}],
        "medium": [{"islands": 16, "corridor": 16, "min_dist": 128, "extra_edges": 2}],
        "hard": [{"islands": 20, "corridor": 14, "min_dist": 108, "extra_edges": 3}],
    },
    "pipes": {
        "easy": [{"cols": 5, "rows": 4, "corridor": 20, "loops": 2}],
        "medium": [{"cols": 6, "rows": 5, "corridor": 18, "loops": 4}],
        "hard": [{"cols": 7, "rows": 6, "corridor": 16, "loops": 6}, {"cols": 8, "rows": 6, "corridor": 15, "loops": 7}],
    },
}


def graph_hash(world) -> str:
    """Canonical semantic hash: Weisfeiler-Lehman over the adjacency with
    degree seeds, plus endpoint step-eccentricities. Invariant to node
    relabeling and to style/augmentation."""
    adj = adjacency(world)
    labels = {n: str(len(neighbors)) for n, neighbors in adj.items()}
    for _ in range(3):
        labels = {
            n: hashlib.sha256(
                (labels[n] + "|" + ",".join(sorted(labels[m] for m, _w in adj[n]))).encode()
            ).hexdigest()[:12]
            for n in adj
        }
    from . import solver as solver_mod

    start_ecc = len(solver_mod.bfs_path(adj, world.start_node, world.goal_node) or [])
    digest = hashlib.sha256()
    digest.update(world.id.encode())
    digest.update(",".join(sorted(labels.values())).encode())
    digest.update(f"|{start_ecc}|{labels[world.start_node]}|{labels[world.goal_node]}".encode())
    return digest.hexdigest()


def _environment_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            pkg: importlib_metadata.version(pkg)
            for pkg in ("numpy", "pillow", "scipy")
        },
    }


def _slot_seeds(master_seed: int, seed_base: int, slot: int, attempt: int) -> dict:
    seq = np.random.SeedSequence([master_seed, seed_base, slot, attempt])
    topo, style, aug, sampler = seq.spawn(4)
    return {
        "topo_seed": int(topo.generate_state(1)[0] % 2**31),
        "style_seed": int(style.generate_state(1)[0] % 2**31),
        "augmentation_seed": int(aug.generate_state(1)[0] % 2**31),
        "sampler_seed": int(sampler.generate_state(1)[0] % 2**31),
    }


def build_slot(
    master_seed: int,
    seed_base: int,
    slot: int,
    family: str,
    tier: str,
    archetype_options: list[str],
    out_dir: str,
    split: str,
    attempt_offset: int = 0,
) -> dict:
    """Build one task slot with fail-closed retries.

    Strict pass first: the measured tier must equal the target. If every
    attempt fails only on tier, a relaxed pass accepts the measured tier and
    records `tier_relaxed` so `dataset stats` shows the true distribution.
    """
    rejections: list[dict] = []
    attempts = [
        (a, _slot_seeds(master_seed, seed_base, slot, a))
        for a in range(attempt_offset, attempt_offset + MAX_ATTEMPTS_PER_SLOT)
    ]
    for relax_tier in (False, True):
        for attempt, seeds in attempts:
            sampler_rng = np.random.default_rng(seeds["sampler_seed"])

            options = DIFFICULTY_SPACES[family][tier]
            overrides = dict(options[int(sampler_rng.integers(len(options)))])
            q_lo, q_hi = QUANTILE_BANDS[tier]
            overrides["endpoint_quantile"] = float(np.round(sampler_rng.uniform(q_lo, q_hi), 3))
            archetype_name = archetype_options[int(sampler_rng.integers(len(archetype_options)))]

            def reject(stage: str, detail: str) -> None:
                rejections.append(
                    {"attempt": attempt, "relaxed": relax_tier, "stage": stage,
                     "detail": detail[:300], "seeds": seeds}
                )

            try:
                world = FAMILIES[family].build(seeds["topo_seed"], overrides)
            except ValueError as exc:
                reject("generation", str(exc))
                continue
            aug_params = sample_augmentation(
                np.random.default_rng(seeds["augmentation_seed"]), family
            )
            world = apply_augmentation(world, aug_params)
            mask = open_mask(world)
            try:
                validation = validate_world(world, mask)
            except ValueError as exc:
                reject("validation", str(exc))
                continue
            measures = measure_task(world, validation)
            if measures["tier"] != tier and not relax_tier:
                reject("tier", f"targeted {tier}, measured {measures['tier']}")
                continue
            semantic_hash = graph_hash(world)
            try:
                image, style_record, certification, _style_rej = certified_render(
                    world, mask, ARCHETYPES[archetype_name], seeds["style_seed"]
                )
            except ValueError as exc:
                reject("style_certification", str(exc))
                continue

            task_id = f"{family}-{measures['tier']}-s{slot:04d}"
            task_dir = Path(out_dir) / split / task_id
            task = save_task(task_dir, world, mask, image, validation, style_record)
            overlay = render_overlay(image, validation.geodesic_points_px, success=True)
            overlay.save(task_dir / "reference-overlay.png")

            provenance = {
                "dataset_split": split,
                "slot": slot,
                "attempt": attempt,
                "master_seed": master_seed,
                "seed_base": seed_base,
                "seed_derivation": "SeedSequence([master_seed, seed_base, slot, attempt]).spawn(4) -> topo, style, augmentation, sampler",
                **seeds,
                "family": family,
                "tier_target": tier,
                "tier_relaxed": relax_tier,
                "difficulty_overrides": overrides,
                "augmentation": aug_params,
                "measures": measures,
                "graph_hash": semantic_hash,
                "mask_sha256": mask_sha256(mask),
                "certification": {
                    "metrics": certification.metrics,
                    "thresholds": certification.thresholds,
                },
                "rejections": rejections,
                "environment": _environment_record(),
            }
            task["provenance"] = provenance
            (task_dir / "task.json").write_text(json.dumps(task, indent=2))
            return {
                "task_id": task_id,
                "dir": str(task_dir),
                "family": family,
                "archetype": archetype_name,
                "tier": measures["tier"],
                "tier_target": tier,
                "tier_relaxed": relax_tier,
                "graph_hash": semantic_hash,
                "mask_sha256": provenance["mask_sha256"],
                "measures": measures,
                "attempts_used": attempt - attempt_offset + 1,
                "rejections": len(rejections),
            }
    return {
        "error": f"slot {slot} ({family}/{tier}): no attempt passed in {MAX_ATTEMPTS_PER_SLOT}",
        "rejections": rejections,
    }


def _slot_plan(config: dict, split_name: str) -> list[tuple]:
    """Deterministic (slot, family, tier, archetype_options) plan for a split."""
    split = config["splits"][split_name]
    size = split["size"]
    mix = config["tier_mix"]
    plan_rng = np.random.default_rng(
        np.random.SeedSequence([config["master_seed"], split["seed_base"], 0xB1A])
    )
    tiers = (
        ["easy"] * round(size * mix["easy"])
        + ["medium"] * round(size * mix["medium"])
        + ["hard"] * (size - round(size * mix["easy"]) - round(size * mix["medium"]))
    )
    plan_rng.shuffle(tiers)
    families = sorted(FAMILIES)
    forbidden_cells = {tuple(c) for c in config.get("public_forbidden_cells", [])}
    plan = []
    for slot in range(size):
        family = families[slot % len(families)]
        allowed = [
            a
            for a in split["archetypes"]
            if (split_name == "test-hidden" or (family, a) not in forbidden_cells)
            and _supports(a, family)
        ]
        plan.append((slot, family, tiers[slot], allowed))
    return plan


def _supports(archetype_name: str, family: str) -> bool:
    rep = "RASTER" if family == "cave" else "GRAPH"
    return rep in ARCHETYPES[archetype_name].supports


def build_split(
    config: dict,
    split_name: str,
    workers: int = 8,
    registry: dict[str, str] | None = None,
) -> dict:
    """Build one split. `registry` carries semantic hashes from other splits
    so dedup resampling covers cross-split leakage, not just within-split."""
    split = config["splits"][split_name]
    out_dir = config["out_dir"]
    plan = _slot_plan(config, split_name)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                build_slot,
                config["master_seed"],
                split["seed_base"],
                slot,
                family,
                tier,
                allowed,
                out_dir,
                split_name,
            )
            for slot, family, tier, allowed in plan
        ]
        for i, future in enumerate(futures):
            result = future.result()
            if "error" in result:
                raise RuntimeError(f"{split_name}: {result['error']}")
            results.append(result)
            if (i + 1) % 25 == 0:
                print(f"  {split_name}: {i + 1}/{len(plan)} tasks built")

    # Semantic dedup against BOTH this split and every hash in the registry
    # (other splits) — a duplicate is resampled, and its superseded task dir
    # removed if the measured tier (and so the dir name) changed.
    import shutil

    seen: dict[str, str] = dict(registry or {})
    retry_offset = MAX_ATTEMPTS_PER_SLOT
    for i, result in enumerate(results):
        while result["graph_hash"] in seen:
            slot, family, tier, allowed = plan[i]
            print(f"  {split_name}: slot {slot} duplicate of {seen[result['graph_hash']]}, resampling")
            old_dir = Path(result["dir"])
            result = build_slot(
                config["master_seed"], split["seed_base"], slot, family, tier,
                allowed, out_dir, split_name, attempt_offset=retry_offset,
            )
            if "error" in result:
                raise RuntimeError(f"{split_name}: {result['error']}")
            retry_offset += MAX_ATTEMPTS_PER_SLOT
            if old_dir != Path(result["dir"]) and old_dir.exists():
                shutil.rmtree(old_dir)
            results[i] = result
        seen[result["graph_hash"]] = f"{split_name}/{result['task_id']}"

    index_path = Path(out_dir) / split_name / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w") as fh:
        for result in results:
            fh.write(json.dumps(result) + "\n")
    return {
        "split": split_name,
        "size": len(results),
        "total_rejections": sum(r["rejections"] for r in results),
        "tiers": dict(Counter(r["tier"] for r in results)),
        "archetypes": dict(Counter(r["archetype"] for r in results)),
    }


def build_all(config_path: Path, workers: int = 8, splits: list[str] | None = None) -> dict:
    config = json.loads(config_path.read_text())
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    # Seed the dedup registry with any splits that already exist on disk and
    # are not being rebuilt, so partial rebuilds stay leakage-free.
    registry: dict[str, str] = {}
    for split_name in config["splits"]:
        if splits is not None and split_name not in splits:
            index = out_dir / split_name / "index.jsonl"
            if index.exists():
                for line in index.read_text().splitlines():
                    row = json.loads(line)
                    registry[row["graph_hash"]] = f"{split_name}/{row['task_id']}"
    for split_name in config["splits"]:
        if splits is not None and split_name not in splits:
            continue
        print(f"building split {split_name} ({config['splits'][split_name]['size']} tasks)")
        summary = build_split(config, split_name, workers, registry=registry)
        summaries[split_name] = summary
        index = out_dir / split_name / "index.jsonl"
        for line in index.read_text().splitlines():
            row = json.loads(line)
            registry[row["graph_hash"]] = f"{split_name}/{row['task_id']}"

    # Cross-split leakage check over everything built so far.
    hashes: dict[str, str] = {}
    for split_name in config["splits"]:
        index = out_dir / split_name / "index.jsonl"
        if not index.exists():
            continue
        for line in index.read_text().splitlines():
            row = json.loads(line)
            key = row["graph_hash"]
            if key in hashes and not hashes[key].startswith(split_name):
                raise RuntimeError(
                    f"leakage: {row['task_id']} in {split_name} duplicates {hashes[key]}"
                )
            hashes[key] = f"{split_name}/{row['task_id']}"

    manifest = {
        "config": config,
        "environment": _environment_record(),
        "summaries": summaries,
        "total_semantic_hashes": len(hashes),
    }
    (out_dir / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --- verification, stats, QC sheets ---


def verify_split(out_dir: Path, split_name: str, sample: int | None = 25) -> list[str]:
    """Prove reproducibility and integrity for a (sample of a) built split.

    Per sampled task: rebuild the world from logged seeds/params → demand a
    byte-identical mask and the same semantic hash; re-certify the stored
    render; replay the stored reference path through the scorer.
    """
    from PIL import Image

    from .evaluator import evaluate_task
    from .io import load_task

    failures: list[str] = []
    index = out_dir / split_name / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rng = np.random.default_rng(0)
    if sample is not None and sample < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=sample, replace=False)]

    for row in rows:
        task_dir = Path(row["dir"])
        task, mask = load_task(task_dir)
        prov = task["provenance"]

        world = FAMILIES[prov["family"]].build(prov["topo_seed"], prov["difficulty_overrides"])
        world = apply_augmentation(world, prov["augmentation"])
        rebuilt_mask = open_mask(world)
        if mask_sha256(rebuilt_mask) != prov["mask_sha256"]:
            failures.append(f"{row['task_id']}: mask not reproducible from provenance")
            continue
        if graph_hash(world) != prov["graph_hash"]:
            failures.append(f"{row['task_id']}: semantic hash drifted")

        image = Image.open(task_dir / task["image_file"])
        from .certify import certify_render

        cert = certify_render(image, mask, world, task["style_record"])
        if not cert.ok:
            failures.append(f"{row['task_id']}: stored render fails certification: {cert.failures}")

        result = evaluate_task(task, mask, {"points": task["reference"]["optimal_path"]})
        if not result.success:
            failures.append(f"{row['task_id']}: stored reference fails the scorer")
    return failures


def split_stats(out_dir: Path, split_name: str) -> dict:
    index = out_dir / split_name / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    coverage: dict[str, Counter] = {}
    for row in rows:
        coverage.setdefault(row["family"], Counter())[row["archetype"]] += 1
    lengths = [row["measures"]["geodesic_length_px"] for row in rows]
    return {
        "size": len(rows),
        "tiers": dict(Counter(row["tier"] for row in rows)),
        "families": dict(Counter(row["family"] for row in rows)),
        "archetypes": dict(Counter(row["archetype"] for row in rows)),
        "coverage": {f: dict(c) for f, c in coverage.items()},
        "geodesic_length_px": {
            "min": min(lengths),
            "median": float(np.median(lengths)),
            "max": max(lengths),
        },
        "mean_rejections_per_task": round(
            sum(row["rejections"] for row in rows) / len(rows), 2
        ),
    }


def qc_sheet(out_dir: Path, split_name: str, count: int, seed: int = 0) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    index = out_dir / split_name / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rng = np.random.default_rng(seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(count, len(rows)), replace=False)]
    font = ImageFont.load_default(size=14)
    thumbs = []
    for row in picks:
        img = Image.open(Path(row["dir"]) / "input.png").convert("RGB")
        scale = 340 / img.width
        thumb = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        w, h = thumb.size
        framed = Image.new("RGB", (w + 6, h + 24), (24, 24, 28))
        framed.paste(thumb, (3, 3))
        ImageDraw.Draw(framed).text(
            (6, h + 5), f"{row['task_id']} · {row['archetype']}", fill=(230, 230, 235), font=font
        )
        thumbs.append(framed)
    cols = 4
    tw = max(t.size[0] for t in thumbs)
    th = max(t.size[1] for t in thumbs)
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows_n * th), (24, 24, 28))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    out = out_dir / f"qc-{split_name}-{seed}.png"
    sheet.save(out)
    return out
