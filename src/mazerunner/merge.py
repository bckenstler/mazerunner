"""Consolidate sharded runs into one canonical result set.

A 57-shard run leaves 57 attempts files. Merging them is not just concatenation:
resumed shards can re-record an attempt, and a mis-specified shard range can
have two processes doing the same work. Duplicates that *agree* are harmless;
duplicates that disagree about success mean the sharding was wrong, and that
must stop the pipeline rather than be averaged away.

Everything streams. A full main run is ~400 MB of provider traces, so nothing
here loads a file into memory, and no source file is ever modified or removed.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

ATTEMPT_KEY = ("provider", "maze", "trial")


def _key(row: dict) -> tuple:
    return tuple(row.get(field) for field in ATTEMPT_KEY)


def _is_evaluated(row: dict) -> bool:
    return row.get("evaluation") is not None


def _succeeded(row: dict) -> bool:
    evaluation = row.get("evaluation")
    return bool(evaluation and evaluation.get("success"))


def _prefer(existing: dict, candidate: dict) -> dict:
    """Which of two rows for the same attempt to keep.

    An evaluated row beats a transport failure (the retry succeeded); otherwise
    the later timestamp wins.
    """
    if _is_evaluated(candidate) and not _is_evaluated(existing):
        return candidate
    if _is_evaluated(existing) and not _is_evaluated(candidate):
        return existing
    return candidate if candidate.get("timestamp", "") > existing.get("timestamp", "") else existing


def merge_runs(
    paths: list[Path],
    out_dir: Path,
    *,
    expected_units: int | None = None,
) -> dict:
    """Merge attempts.jsonl files into out_dir/attempts.jsonl + a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "attempts.jsonl"

    kept: dict[tuple, dict] = {}
    duplicates = 0
    conflicts: list[dict] = []
    malformed = 0
    sources: list[dict] = []

    for path in sorted(paths):
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                count += 1
                key = _key(row)
                if key in kept:
                    duplicates += 1
                    previous = kept[key]
                    if _is_evaluated(previous) and _is_evaluated(row):
                        if _succeeded(previous) != _succeeded(row):
                            conflicts.append(
                                {
                                    "key": dict(zip(ATTEMPT_KEY, key)),
                                    "a": {
                                        "success": _succeeded(previous),
                                        "shard": previous.get("shard"),
                                        "timestamp": previous.get("timestamp"),
                                    },
                                    "b": {
                                        "success": _succeeded(row),
                                        "shard": row.get("shard"),
                                        "timestamp": row.get("timestamp"),
                                    },
                                }
                            )
                    kept[key] = _prefer(previous, row)
                else:
                    kept[key] = row
        sources.append({"path": str(path), "rows": count})

    ordered = sorted(kept.values(), key=lambda r: (r.get("provider") or "", r.get("maze") or "", r.get("trial") or 0))
    with merged_path.open("w") as out:
        for row in ordered:
            out.write(json.dumps(row) + "\n")

    per_leg: dict[str, dict] = {}
    canary_tasks: set[str] = set()
    for row in ordered:
        leg = per_leg.setdefault(
            row.get("provider") or "?",
            {"attempts": 0, "evaluated": 0, "successes": 0, "transport_failures": 0},
        )
        leg["attempts"] += 1
        if row.get("error"):
            leg["transport_failures"] += 1
        if _is_evaluated(row):
            leg["evaluated"] += 1
            if _succeeded(row):
                leg["successes"] += 1
            if row["evaluation"].get("efficiency_canary"):
                canary_tasks.add(row.get("maze"))

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "sources": sources,
        "rows_in": sum(s["rows"] for s in sources),
        "rows_out": len(ordered),
        "duplicates_collapsed": duplicates,
        "conflicts": conflicts,
        "malformed_lines": malformed,
        "expected_units": expected_units,
        "per_leg": per_leg,
        "efficiency_canary_tasks": sorted(t for t in canary_tasks if t),
    }
    (out_dir / "merge-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def missing_units(
    merged_path: Path,
    task_ids: list[str],
    providers: list[str],
    trials: int,
) -> list[tuple[str, str, int]]:
    """Attempts that were planned but never recorded, for the requeue loop."""
    present = set()
    if merged_path.exists():
        with merged_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # A recorded transport failure still counts as missing work.
                if row.get("error"):
                    continue
                present.add(_key(row))
    out = []
    for provider in providers:
        for task_id in task_ids:
            for trial in range(trials):
                if (provider, task_id, trial) not in present:
                    out.append((provider, task_id, trial))
    return out


def write_missing_task_list(missing: list[tuple[str, str, int]], path: Path) -> int:
    """Task ids needing another pass, one per line. Returns the count."""
    task_ids = sorted({task_id for _provider, task_id, _trial in missing})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(task_ids) + ("\n" if task_ids else ""))
    return len(task_ids)
