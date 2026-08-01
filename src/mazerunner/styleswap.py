"""Style-swap sets: identical topology, different rendering.

H3 asks whether visual style is a difficulty axis separable from maze topology.
Answering it needs pairs that differ in *nothing else* — same corridors, same
route, same mask, same canvas — which the four-layer separation already makes
possible: rebuild the world from its recorded seeds, then render it through a
different archetype.

This deliberately bypasses `dataset.build_split`, whose semantic hash is
style-invariant by design and would reject every variant in a pair-group as a
duplicate of its siblings.
"""

from __future__ import annotations

import json
from pathlib import Path

from .certify import certified_render
from .dataset import rebuild_from_provenance, supports_archetype
from .evaluator import evaluate_task
from .io import load_task, mask_sha256, save_task
from .styles import ARCHETYPES
from .world import validate_world

# The five public archetypes that appear in no `public_forbidden_cells` pair,
# so every (family, archetype) cell here is legal for the public splits and the
# hidden split's compositional holdouts stay untouched.
SWAP_ARCHETYPES = (
    "forest-path",
    "glow-cavern",
    "candy-pastel",
    "pencil-sketch",
    "desert-canyon",
)


class SwapFailure(RuntimeError):
    """A variant could not be produced fairly; the pair-group is incomplete."""


def build_variant(
    source_task: dict,
    world,
    mask,
    archetype_name: str,
    out_dir: Path,
    *,
    style_seed: int,
) -> dict:
    """Render one archetype over an already-rebuilt world and save it."""
    validation = validate_world(world, mask)
    image, style_record, certification, rejections = certified_render(
        world, mask, ARCHETYPES[archetype_name], style_seed
    )
    task = save_task(out_dir, world, mask, image, validation, style_record)

    prov = dict(source_task["provenance"])
    prov.update(
        {
            "style_swap": {
                "source_task_id": source_task["id"],
                "source_archetype": source_task["style_record"]["archetype"],
                "archetype": archetype_name,
                "style_seed": style_seed,
            },
            "certification": {
                "metrics": certification.metrics,
                "thresholds": certification.thresholds,
            },
            "rejections": rejections,
        }
    )
    task["provenance"] = prov
    (out_dir / "task.json").write_text(json.dumps(task, indent=2))
    return task


def assert_fair(source_prov: dict, out_dir: Path) -> None:
    """The variant must differ from its source in style and nothing else.

    Two checks carry the whole H3 claim: the scored mask is byte-identical, and
    the certified route still passes the scorer under the new rendering. If
    either fails, the pair is not a controlled comparison.
    """
    task, mask = load_task(out_dir)
    digest = mask_sha256(mask)
    if digest != source_prov["mask_sha256"]:
        raise SwapFailure(
            f"{out_dir.name}: mask differs from source ({digest[:12]} != "
            f"{source_prov['mask_sha256'][:12]})"
        )
    result = evaluate_task(task, mask, {"points": task["reference"]["optimal_path"]})
    if not result.success:
        raise SwapFailure(f"{out_dir.name}: reference route no longer scores as a pass")


def build_style_swap_set(
    dataset_dir: Path,
    task_ids: list[str],
    out_dir: Path,
    *,
    archetypes: tuple[str, ...] = SWAP_ARCHETYPES,
    seed: int = 0,
) -> dict:
    """Build `len(task_ids)` pair-groups, each rendered in every archetype.

    Groups are all-or-nothing: certification can legitimately refuse a
    (topology, style) combination, and a half-populated group would unbalance
    the variance decomposition H3 depends on.
    """
    index = {
        json.loads(line)["task_id"]: json.loads(line)
        for line in (dataset_dir / "index.jsonl").read_text().splitlines()
        if line.strip()
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, rows, skipped = [], [], []
    for group_index, task_id in enumerate(task_ids):
        source_task, _mask = load_task(Path(index[task_id]["dir"]))
        prov = source_task["provenance"]
        family = prov["family"]
        world, mask = rebuild_from_provenance(prov)

        if mask_sha256(mask) != prov["mask_sha256"]:
            skipped.append({"task_id": task_id, "reason": "mask not reproducible"})
            continue

        staged, failure = [], None
        for archetype in archetypes:
            if not supports_archetype(archetype, family):
                failure = f"{archetype} does not support {family}"
                break
            variant_dir = out_dir / f"{task_id}--{archetype}"
            # Vary the style seed per (group, archetype) so two variants never
            # share a palette draw, while staying a pure function of `seed`.
            style_seed = seed + group_index * 1_000_003 + archetypes.index(archetype)
            try:
                task = build_variant(
                    source_task, world, mask, archetype, variant_dir, style_seed=style_seed
                )
                assert_fair(prov, variant_dir)
            except (ValueError, SwapFailure) as exc:
                failure = f"{archetype}: {exc}"
                break
            staged.append((variant_dir, archetype, task))

        if failure is not None:
            for variant_dir, _a, _t in staged:
                for path in sorted(variant_dir.rglob("*"), reverse=True):
                    path.unlink() if path.is_file() else path.rmdir()
                variant_dir.rmdir()
            skipped.append({"task_id": task_id, "reason": failure})
            continue

        groups.append(task_id)
        for variant_dir, archetype, task in staged:
            rows.append(
                {
                    "task_id": variant_dir.name,
                    "dir": str(variant_dir),
                    "pair_group": task_id,
                    "source_task_id": task_id,
                    "archetype": archetype,
                    "family": family,
                    "tier": index[task_id]["tier"],
                    "graph_hash": prov["graph_hash"],
                    "mask_sha256": prov["mask_sha256"],
                    "measures": prov["measures"],
                }
            )

    (out_dir / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "source_dataset": str(dataset_dir),
        "requested_groups": len(task_ids),
        "complete_groups": len(groups),
        "archetypes": list(archetypes),
        "seed": seed,
        "variants": len(rows),
        "groups": groups,
        "skipped": skipped,
    }
    (out_dir / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
