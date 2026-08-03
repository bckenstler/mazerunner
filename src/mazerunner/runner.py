"""Live provider smoke runner.

Retries transport/provider exceptions with exponential backoff; a successfully
returned but invalid maze path is never retried — that would change the task
from one-shot planning into best-of-N search.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .contract import prompt_text
from .evaluator import evaluate_task
from .generators import FAMILIES
from .imagesrc import ImageSource, ImageSpec
from .io import load_task
from .metrics import derive
from .overlay import render_overlay
from .providers import ENV_KEYS, PROVIDERS
from .providers.base import ProviderError, is_retryable, serving_stack

# Rate limits get a long, jittered ladder: dozens of shards hitting the same
# provider recover only if they back off by *different* amounts.
RATE_LIMIT_TRIES = 6
RATE_LIMIT_BASE_S = 8.0
RATE_LIMIT_CAP_S = 120.0
# Server errors and connection failures are usually brief.
TRANSIENT_TRIES = 4
TRANSIENT_BASE_S = 2.0
TRANSIENT_CAP_S = 60.0


def _backoff_seconds(error: ProviderError, attempt: int, rng: random.Random) -> float:
    """Full-jitter exponential backoff, honoring Retry-After when the API sent one."""
    if error.retry_after is not None:
        return min(error.retry_after, RATE_LIMIT_CAP_S)
    if error.status == 429:
        base, cap = RATE_LIMIT_BASE_S, RATE_LIMIT_CAP_S
    else:
        base, cap = TRANSIENT_BASE_S, TRANSIENT_CAP_S
    return rng.uniform(0.0, min(cap, base * (2**attempt)))


def _attempt_with_retries(
    provider,
    png_bytes: bytes,
    prompt: str,
    *,
    rng: random.Random | None = None,
) -> tuple[object | None, str | None, list[dict]]:
    """Call the provider, retrying only what is worth retrying.

    Returns (response, last_error, attempt_history). The history records every
    try — status, retryability, and how long we waited — so a leg that limped
    through rate limits is distinguishable afterwards from one that sailed.
    """
    rng = rng or random.Random()
    history: list[dict] = []
    last_error = None
    attempt = 0
    while True:
        try:
            return provider.run(png_bytes, prompt), None, history
        except ProviderError as exc:
            last_error = str(exc)
            retryable = is_retryable(exc)
            limit = RATE_LIMIT_TRIES if exc.status == 429 else TRANSIENT_TRIES
            record = {
                "attempt": attempt + 1,
                "status": exc.status,
                "retryable": retryable,
                "error": last_error[:500],
                "waited_s": 0.0,
            }
            if not retryable or attempt + 1 >= limit:
                history.append(record)
                return None, last_error, history
            wait = _backoff_seconds(exc, attempt, rng)
            record["waited_s"] = round(wait, 2)
            history.append(record)
            time.sleep(wait)
            attempt += 1


def _task_sources(mazes_dir: Path, dataset_dir: Path | None, mazes: list[str] | None):
    """(name, task_dir) pairs from either the smoke set or a dataset split."""
    if dataset_dir is not None:
        index = dataset_dir / "index.jsonl"
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        return [
            (row["task_id"], Path(row["dir"]))
            for row in rows
            if mazes is None or row["task_id"] in mazes
        ]
    return [(name, mazes_dir / name) for name in FAMILIES if mazes is None or name in mazes]


@dataclass(frozen=True)
class AttemptUnit:
    """One (provider, task, trial) call — the atom of work that gets sharded."""

    provider: str
    task_id: str
    task_dir: Path
    trial: int
    ordinal: int  # position within this provider's leg, after shuffling


def plan_units(
    sources: list[tuple[str, Path]],
    provider_names: list[str],
    trials: int,
    *,
    order_seed: int | None = None,
) -> list[AttemptUnit]:
    """Every attempt a run will make, with per-leg task order randomized.

    Each provider gets an independently shuffled task order so that position in
    the leg is uncorrelated with task identity — otherwise a leg cut short (or
    inspected mid-flight) reports whatever the index file happened to list
    first, which is grouped by family and tier.
    """
    units: list[AttemptUnit] = []
    for leg_index, provider in enumerate(provider_names):
        order = list(range(len(sources)))
        if order_seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence([order_seed, leg_index]))
            rng.shuffle(order)
        ordinal = 0
        for index in order:
            task_id, task_dir = sources[index]
            for trial in range(trials):
                units.append(AttemptUnit(provider, task_id, task_dir, trial, ordinal))
                ordinal += 1
    return units


def shard(units: list[AttemptUnit], index: int, count: int) -> list[AttemptUnit]:
    """Stride-slice the flattened unit list.

    Striding after the shuffle keeps every shard stratified across tasks and
    trials; sharding by task instead would let one pathological task's whole
    trial block land in a single process and stall it.
    """
    if count <= 1:
        return list(units)
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for {count} shards")
    return units[index::count]


def completed_keys(attempts_path: Path) -> set[tuple]:
    """(provider, task_id, trial) already *answered*, for --resume.

    Transport failures are deliberately not counted as answered: the retry
    policy requeues them once, so a resumed shard replays exactly those units. The merge prefers the
    evaluated row over the recorded failure, so nothing is double-counted.
    """
    keys: set[tuple] = set()
    if not attempts_path.exists():
        return keys
    with attempts_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            keys.add((row.get("provider"), row.get("maze"), row.get("trial")))
    return keys


def _task_tiers(dataset_dir: Path | None) -> dict[str, str]:
    """task_id -> measured difficulty tier, from a split's index."""
    if dataset_dir is None:
        return {}
    index = dataset_dir / "index.jsonl"
    if not index.exists():
        return {}
    tiers = {}
    for line in index.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tiers[row["task_id"]] = row.get("tier", "unknown")
    return tiers


def _redacted(settings: dict) -> dict:
    """Provider settings safe to write into a manifest."""
    return {k: v for k, v in settings.items() if "key" not in k.lower()}


def _code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_smoke(
    config_path: Path,
    providers: list[str] | None,
    mazes: list[str] | None,
    trials: int | None,
    mazes_dir: Path,
    results_dir: Path,
    dataset_dir: Path | None = None,
    dry_run: bool = False,
    include_dimensions: bool = False,
    run_id: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = False,
    order_seed: int | None = None,
    image_spec: "ImageSpec | None" = None,
) -> int:
    """Execute a live run and write its attempts, config snapshot, and summary.

    A provider whose env key is unset is skipped with a message rather than
    failing the run — a partial leg is recoverable, an aborted one wastes the
    calls already paid for.

    Work is planned first and only then sharded, so `shard_index`/`shard_count`
    split one deterministic unit list across machines instead of each machine
    inventing its own. `order_seed` fixes the interleaving so a rerun issues
    calls in the same order.

    `resume` skips units already *answered* in the shard's attempts file;
    transport failures are not counted as answered, so they replay. `dry_run`
    prints the plan and spends nothing — the cheap way to confirm a leg is
    what you meant before it bills.
    """
    config = json.loads(config_path.read_text())
    trial_count = trials if trials is not None else config.get("trials", 1)

    selected_providers = []
    for name, settings in config.get("providers", {}).items():
        if providers is not None and name not in providers:
            continue
        if providers is None and not settings.get("enabled", True):
            continue
        provider_type = settings.get("type", name)
        if provider_type not in PROVIDERS:
            print(f"skipping unknown provider {name!r} (type {provider_type!r})")
            continue
        env_key = settings.get("env_key") or ENV_KEYS.get(name)
        if not env_key or not os.environ.get(env_key):
            print(f"skipping {name}: {env_key or 'API key env var'} is not set")
            continue
        selected_providers.append((name, settings))
    if not selected_providers:
        print("no providers to run — set API keys in the environment")
        return 1

    sources = _task_sources(mazes_dir, dataset_dir, mazes)
    if not sources:
        print(f"no tasks matched {mazes}")
        return 1
    provider_names = [name for name, _settings in selected_providers]
    all_units = plan_units(sources, provider_names, trial_count, order_seed=order_seed)
    units = shard(all_units, shard_index, shard_count)

    if dry_run:
        print(
            f"dry run: {len(sources)} tasks × {len(selected_providers)} providers × "
            f"{trial_count} trials = {len(all_units)} API calls"
            + (f" ({len(units)} in shard {shard_index}/{shard_count})" if shard_count > 1 else "")
        )
        return 0

    if run_id:
        run_dir = results_dir / run_id / f"shard-{shard_index:02d}"
    else:
        run_dir = results_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"

    already = completed_keys(attempts_path) if resume else set()
    if already:
        before = len(units)
        units = [u for u in units if (u.provider, u.task_id, u.trial) not in already]
        print(f"resuming: {before - len(units)} attempts already recorded, {len(units)} to go")

    (run_dir / "run-manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "shard": {"index": shard_index, "count": shard_count},
                "started": datetime.datetime.now().isoformat(),
                "config_file": str(config_path),
                "dataset": str(dataset_dir) if dataset_dir else None,
                "task_count": len(sources),
                "task_ids": [name for name, _dir in sources],
                "trials": trial_count,
                "order_seed": order_seed,
                "prompt_variant": "with-dimensions" if include_dimensions else "frozen",
                "image_spec": asdict(image_spec) if image_spec else None,
                "planned_units": len(all_units),
                "shard_units": len(units),
                "code_version": _code_version(),
                "providers": {
                    name: _redacted(settings) for name, settings in selected_providers
                },
            },
            indent=2,
        )
    )

    # The mismatched variant needs the whole pool up front (and each task's
    # tier) so the derangement can be tier-matched.
    spec = image_spec or ImageSpec()
    tiers = _task_tiers(dataset_dir) if spec.mode == "mismatched" else None
    image_source = ImageSource(spec, sources=sources, tiers=tiers)

    adapters: dict[str, object] = {}
    settings_by_name = dict(selected_providers)
    aggregates: dict[str, dict] = {}

    # Append, never truncate: a crash must leave every completed attempt on disk.
    with attempts_path.open("a") as attempts_file:
        for unit in units:
            name = unit.provider
            if name not in adapters:
                settings = settings_by_name[name]
                adapters[name] = PROVIDERS[settings.get("type", name)](
                    **{k: v for k, v in settings.items() if k not in ("enabled", "type")}
                )
            adapter = adapters[name]
            agg = aggregates.setdefault(
                f"{name}/{adapter.model}",
                {
                    "requested": 0,
                    "completed": 0,
                    "successes": 0,
                    "efficiencies": [],
                    "latencies": [],
                },
            )

            task, mask = load_task(unit.task_dir)
            png_bytes, image_variant = image_source.bytes_for(unit.task_id, unit.task_dir, task)
            prompt = (
                prompt_text(task["width"], task["height"])
                if include_dimensions
                else prompt_text()
            )

            agg["requested"] += 1
            print(f"{name} · {unit.task_id} · trial {unit.trial + 1} ... ", end="", flush=True)
            response, transport_error, transport_history = _attempt_with_retries(
                adapter, png_bytes, prompt
            )

            row = {
                "provider": name,
                "model": adapter.model,
                "maze": unit.task_id,
                "trial": unit.trial,
                "ordinal": unit.ordinal,
                "run_id": run_id,
                "shard": shard_index,
                "order_seed": order_seed,
                "task_dir": str(unit.task_dir),
                "prompt_variant": "with-dimensions" if include_dimensions else "frozen",
                "image_variant": image_variant,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if transport_history:
                row["transport_errors"] = transport_history
            if response is None:
                row["error"] = f"transport failure: {transport_error}"
                print("TRANSPORT FAILURE")
            else:
                agg["completed"] += 1
                agg["latencies"].append(response.latency_s)
                row.update(
                    {
                        "latency_s": round(response.latency_s, 2),
                        "usage": response.usage,
                        "response_id": response.response_id,
                        "response_model": response.model,
                        "provider_error": response.error,
                        "submission": response.tool_arguments,
                        "reasoning": response.reasoning,
                        "raw_response": response.raw,
                        "serving_stack": serving_stack(response.raw),
                    }
                )
                evaluation = evaluate_task(
                    task, mask, response.tool_arguments
                ) if response.tool_arguments is not None else None
                if evaluation is None:
                    row["evaluation"] = None
                    print(f"NO TOOL CALL ({response.error})")
                else:
                    row["evaluation"] = evaluation.to_dict()
                    if evaluation.success:
                        agg["successes"] += 1
                        agg["efficiencies"].append(evaluation.efficiency)
                    for warning in evaluation.warnings:
                        print(f"\n  WARNING: {warning}")
                    print("PASS" if evaluation.success else "FAIL")
                    _write_overlay(
                        overlays_dir,
                        unit.task_dir,
                        task,
                        response.tool_arguments,
                        evaluation,
                        f"{name}-{unit.task_id}-{unit.trial}",
                    )
                row["derived"] = derive(task, row)
            attempts_file.write(json.dumps(row) + "\n")
            attempts_file.flush()

    summary = {
        "config_file": str(config_path),
        "trials": trial_count,
        "mazes": [name for name, _dir in sources],
        "dataset": str(dataset_dir) if dataset_dir else None,
        "prompt_variant": "with-dimensions" if include_dimensions else "frozen",
        "results": {},
    }
    for key, agg in aggregates.items():
        eff = agg["efficiencies"]
        lat = agg["latencies"]
        summary["results"][key] = {
            "requested": agg["requested"],
            "completed": agg["completed"],
            "successes": agg["successes"],
            "pass_rate": round(agg["successes"] / agg["requested"], 4) if agg["requested"] else 0,
            "mean_efficiency_on_success": round(sum(eff) / len(eff), 4) if eff else None,
            "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nrun written to {run_dir}/")
    for key, result in summary["results"].items():
        print(
            f"  {key:<32} pass {result['successes']}/{result['requested']} "
            f"eff={result['mean_efficiency_on_success']} "
            f"latency={result['mean_latency_s']}s"
        )
    return 0


def _write_overlay(overlays_dir, task_dir, task, arguments, evaluation, label):
    points = arguments.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return
    try:
        w, h = task["width"], task["height"]
        points_px = [(p["x"] * (w - 1), p["y"] * (h - 1)) for p in points]
    except (TypeError, KeyError):
        return
    collision = None
    if evaluation.first_collision is not None:
        collision = (
            evaluation.first_collision["x_px"],
            evaluation.first_collision["y_px"],
        )
    base = Image.open(task_dir / task["image_file"])
    overlay = render_overlay(base, points_px, success=evaluation.success, collision_px=collision)
    overlay.save(overlays_dir / f"{label}.png")
