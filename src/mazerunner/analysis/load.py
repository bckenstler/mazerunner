"""Loading attempt rows and joining them to task metadata.

Streams: a full run is ~400 MB of provider traces, and nothing here needs the
raw payloads. This is also the single place `derived` gets back-filled, so runs
collected before route progress existed analyse identically to later ones.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..metrics import derive

# Fields worth keeping in memory; `raw_response` is deliberately dropped.
KEEP = (
    "provider", "model", "maze", "trial", "ordinal", "run_id", "shard",
    "prompt_variant", "image_variant", "latency_s", "usage", "submission",
    "evaluation", "derived", "error", "provider_error", "serving_stack",
    "task_dir", "timestamp",
)


def load_index(dataset_dir: Path) -> dict[str, dict]:
    """task_id -> its dataset index row (family, archetype, tier, measures)."""
    return {
        json.loads(line)["task_id"]: json.loads(line)
        for line in (Path(dataset_dir) / "index.jsonl").read_text().splitlines()
        if line.strip()
    }


def load_attempts(
    paths: list[Path],
    dataset_dir: Path | None = None,
    *,
    backfill: bool = True,
) -> list[dict]:
    """Attempt rows, slimmed, with task metadata joined and `derived` ensured."""
    index = load_index(dataset_dir) if dataset_dir else {}
    tasks: dict[str, dict] = {}
    rows: list[dict] = []

    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = {k: raw.get(k) for k in KEEP if k in raw}

                meta = index.get(row.get("maze"))
                if meta:
                    row["family"] = meta.get("family")
                    row["tier"] = meta.get("tier")
                    row["archetype"] = meta.get("archetype")
                    row["measures"] = meta.get("measures")

                if backfill and row.get("derived") is None and not row.get("error"):
                    task_dir = raw.get("task_dir") or (meta or {}).get("dir")
                    if task_dir:
                        if task_dir not in tasks:
                            tasks[task_dir] = json.loads(
                                (Path(task_dir) / "task.json").read_text()
                            )
                        row["derived"] = derive(tasks[task_dir], raw)
                rows.append(row)
    return rows


def scored(rows: list[dict]) -> list[dict]:
    """Attempts that reached the model. Transport failures are excluded from
    denominators per the pre-registered scoring rules."""
    return [r for r in rows if not r.get("error")]


def by_task(rows: list[dict], key=lambda r: bool((r.get("evaluation") or {}).get("success"))):
    """provider -> task -> [values]."""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in scored(rows):
        out[row["provider"]][row["maze"]].append(key(row))
    return out


def task_means(rows: list[dict], key=None) -> dict[str, dict[str, float]]:
    """provider -> task -> mean over that task's attempts."""
    grouped = by_task(rows, key) if key else by_task(rows)
    return {
        provider: {task: sum(v) / len(v) for task, v in tasks.items()}
        for provider, tasks in grouped.items()
    }
