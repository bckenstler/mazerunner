"""Frozen evaluation subsets, selected by seeded integer program.

The main run scores 100 of the 200 dev tasks, stratified so that no family,
tier, or archetype is under-represented. Those constraints are tight against a
skewed split -- cave has no easy tasks at all, and blueprint-rooms has only 9
instances against a floor of 6 -- so seeded rejection sampling would search
essentially forever. Instead the selection is a binary program solved by HiGHS
with a seeded-random objective: deterministic given the seed, feasible by
construction, and re-solvable by anyone with the seed and the index file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

DEFAULT_TIER_TARGETS = {"easy": 30, "medium": 40, "hard": 30}


@dataclass(frozen=True)
class SelectionSpec:
    size: int = 100
    per_family_min: int = 12
    per_family_max: int = 13
    tier_targets: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TIER_TARGETS)
    )
    archetype_floor: int = 6


class InfeasibleSelection(RuntimeError):
    """The constraint set admits no selection from this pool."""


def read_task_list(path: Path) -> list[str]:
    """Task ids from a frozen list file.

    Accepts newline-separated (evals/*.txt) or the single comma-separated line
    used by results/pilot-tasks.txt, so both survive.
    """
    text = Path(path).read_text()
    parts = text.replace(",", "\n").split("\n")
    return [p.strip() for p in parts if p.strip() and not p.startswith("#")]


def write_task_list(path: Path, task_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(task_ids) + "\n")


def _indicator(values: list[str], key: str) -> np.ndarray:
    return np.array([1.0 if v == key else 0.0 for v in values])


def select_eval_set(
    rows: list[dict],
    *,
    seed: int,
    spec: SelectionSpec | None = None,
) -> tuple[list[str], dict]:
    """Choose a stratified subset. Returns (sorted task_ids, manifest)."""
    spec = spec or SelectionSpec()
    n = len(rows)
    families = [r["family"] for r in rows]
    tiers = [r["tier"] for r in rows]
    archetypes = [r["archetype"] for r in rows]

    constraints = [LinearConstraint(np.ones(n), spec.size, spec.size)]
    for family in sorted(set(families)):
        constraints.append(
            LinearConstraint(_indicator(families, family), spec.per_family_min, spec.per_family_max)
        )
    for tier, target in spec.tier_targets.items():
        constraints.append(LinearConstraint(_indicator(tiers, tier), target, target))
    for archetype in sorted(set(archetypes)):
        constraints.append(
            LinearConstraint(_indicator(archetypes, archetype), spec.archetype_floor, np.inf)
        )

    # Seeded-random objective: any feasible point is equally acceptable, so the
    # seed alone determines which one HiGHS returns.
    objective = np.random.default_rng(seed).random(n)
    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if not result.success:
        raise InfeasibleSelection(
            f"no selection satisfies the constraints (HiGHS: {result.message}). "
            f"pool={n}, size={spec.size}, tiers={spec.tier_targets}, "
            f"archetype_floor={spec.archetype_floor}"
        )

    chosen = [rows[i] for i in range(n) if result.x[i] > 0.5]
    task_ids = sorted(r["task_id"] for r in chosen)

    def counts(key: str) -> dict:
        out: dict[str, int] = {}
        for row in chosen:
            out[row[key]] = out.get(row[key], 0) + 1
        return dict(sorted(out.items()))

    manifest = {
        "seed": seed,
        "spec": asdict(spec),
        "pool_size": n,
        "selected": len(task_ids),
        "task_ids": task_ids,
        "achieved": {
            "family": counts("family"),
            "tier": counts("tier"),
            "archetype": counts("archetype"),
        },
        "solver": "scipy.optimize.milp (HiGHS)",
    }
    return task_ids, manifest


def build_eval_set(
    dataset_dir: Path,
    out_path: Path,
    *,
    seed: int,
    spec: SelectionSpec | None = None,
    pool_path: Path | None = None,
) -> dict:
    """Select, write the frozen id list, and record a reproducibility manifest.

    `pool_path` restricts the candidate pool to an existing frozen list, which
    is how the ablation subsets are nested inside dev-eval-100: every ablation
    attempt is then paired with that task's main-run attempts.
    """
    index = Path(dataset_dir) / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]

    pool_ids = None
    if pool_path is not None:
        pool_ids = read_task_list(Path(pool_path))
        keep = set(pool_ids)
        rows = [r for r in rows if r["task_id"] in keep]
        missing = keep - {r["task_id"] for r in rows}
        if missing:
            raise InfeasibleSelection(
                f"{len(missing)} pool ids are absent from {index}: {sorted(missing)[:3]}"
            )

    task_ids, manifest = select_eval_set(rows, seed=seed, spec=spec)

    manifest["dataset_dir"] = str(dataset_dir)
    manifest["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    if pool_ids is not None:
        manifest["pool_path"] = str(pool_path)
        manifest["pool_size"] = len(pool_ids)
        manifest["pool_sha256"] = hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()

    write_task_list(Path(out_path), task_ids)
    Path(out_path).with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_eval_set(out_path: Path) -> list[str]:
    """Re-solve from the recorded seed and confirm the same ids come back."""
    manifest_path = Path(out_path).with_suffix(".json")
    if not manifest_path.exists():
        return [f"no manifest at {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())

    index = Path(manifest["dataset_dir"]) / "index.jsonl"
    failures = []
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    if digest != manifest["index_sha256"]:
        failures.append(f"index.jsonl changed since selection ({digest[:12]} != "
                        f"{manifest['index_sha256'][:12]})")

    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]

    pool_path = manifest.get("pool_path")
    if pool_path:
        pool = Path(pool_path)
        if not pool.exists():
            failures.append(f"pool list {pool} is missing")
        else:
            digest = hashlib.sha256(pool.read_bytes()).hexdigest()
            if digest != manifest.get("pool_sha256"):
                failures.append(f"pool list {pool} changed since selection")
            pool_ids = set(read_task_list(pool))
            if not set(manifest["task_ids"]) <= pool_ids:
                failures.append(f"selection is not a subset of {pool}")
            rows = [r for r in rows if r["task_id"] in pool_ids]

    spec_fields = dict(manifest["spec"])
    task_ids, _ = select_eval_set(rows, seed=manifest["seed"], spec=SelectionSpec(**spec_fields))
    if task_ids != manifest["task_ids"]:
        failures.append("re-solve produced a different selection")
    if read_task_list(out_path) != manifest["task_ids"]:
        failures.append(f"{out_path} does not match its manifest")
    return failures
