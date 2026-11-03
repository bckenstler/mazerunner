"""The v2 leaderboard: macro pass@1 over equally weighted ladder cells.

The v2 set is a grid of cells — maze family × size level, with warped twins of
the four grid families — and every cell holds the same number of tasks. The
headline is the mean over cells of the mean over a cell's tasks of each task's
pass rate, so no family or level outweighs another.

Two rules decide what a row contributes:

- An evaluated attempt is scored from its evaluation; any non-success
  (collision, malformed submission, no tool call) is a fail.
- An attempt left with only a transport failure is a fail once the model was
  really asked twice and never answered (`unanswered_attempts`, written by
  merge). An attempt that only ever saw connection errors never reached the
  model and is excluded as missing work.

Intervals resample tasks within each cell (task-clustered, as in stats.py),
and comparisons between two configurations are paired by task: both legs are
resampled with the same task draw, so task difficulty cancels.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import pass_at_k

TASK_ID = re.compile(r"^(?P<family>[a-z]+)-(?P<level>L\d)(?P<warped>w?)-s(?P<slot>\d+)$")
LEVELS = ("L1", "L2", "L3", "L4")
UPPER = ("L3", "L4")
MAX_UNANSWERED = 2
RESAMPLES = 4000
SEED = 20261031


def parse_task(task_id: str) -> dict:
    """family, level, warped, slot from a v2 task id like braided-L3w-s0040."""
    m = TASK_ID.match(task_id)
    if not m:
        raise ValueError(f"not a v2 ladder task id: {task_id}")
    return {
        "family": m["family"],
        "level": m["level"],
        "warped": bool(m["warped"]),
        "slot": m["slot"],
        "cell": (m["family"], m["level"] + m["warped"]),
    }


def twin_of(task_id: str) -> str:
    """The straight task a warped task was built from (same slot)."""
    t = parse_task(task_id)
    return f"{t['family']}-{t['level']}-s{t['slot']}"


def score_row(row: dict) -> tuple[bool | None, float | None]:
    """(success, route_progress) for one merged row; (None, None) if missing."""
    if row.get("error"):
        if (row.get("unanswered_attempts") or 0) >= MAX_UNANSWERED:
            return False, 0.0
        return None, None
    ok = bool((row.get("evaluation") or {}).get("success"))
    progress = (row.get("derived") or {}).get("route_progress")
    return ok, 1.0 if ok else float(progress or 0.0)


def load_board_rows(path: Path) -> dict:
    """Per leg: task -> [success per trial], route progress, and counts."""
    legs: dict[str, dict] = defaultdict(lambda: {
        "tasks": defaultdict(list), "progress": [], "missing": 0, "unanswered": 0,
        "output_tokens": [], "output_tokens_upper": [], "latency": [],
    })
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            leg = legs[row["provider"]]
            ok, progress = score_row(row)
            if ok is None:
                leg["missing"] += 1
                continue
            if row.get("error"):
                leg["unanswered"] += 1
            leg["tasks"][row["maze"]].append(ok)
            leg["progress"].append(progress)
            usage = row.get("usage") or {}
            out = usage.get("completion_tokens", usage.get("output_tokens"))
            if out:
                leg["output_tokens"].append(out)
                if parse_task(row["maze"])["level"] in UPPER:
                    leg["output_tokens_upper"].append(out)
            if row.get("latency_s"):
                leg["latency"].append(row["latency_s"])
    return legs


def _macro_cells(cells: dict, rng: np.random.Generator) -> dict:
    arrays = [np.asarray(v, dtype=float) for _, v in sorted(cells.items())]
    if not arrays:
        return {"score": float("nan"), "lo": float("nan"), "hi": float("nan"), "cells": 0}
    point = float(np.mean([a.mean() for a in arrays]))
    boots = np.zeros(RESAMPLES)
    for a in arrays:
        idx = rng.integers(0, a.size, size=(RESAMPLES, a.size))
        boots += a[idx].mean(axis=1)
    boots /= len(arrays)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"score": point, "lo": float(lo), "hi": float(hi), "cells": len(arrays)}


def macro(tasks: dict, levels=LEVELS, *, seed: int = SEED) -> dict:
    """Macro pass@1 and its 95% CI over the cells at the given levels."""
    cells = defaultdict(list)
    for task, trials in tasks.items():
        t = parse_task(task)
        if t["level"] in levels:
            cells[t["cell"]].append(np.mean(trials))
    return _macro_cells(cells, np.random.default_rng(seed))


def paired(a: dict, b: dict, levels=LEVELS, *, seed: int = SEED) -> dict:
    """Macro difference a - b over shared tasks, resampled jointly."""
    cells = defaultdict(list)
    for task in sorted(a.keys() & b.keys()):  # sorted: set order varies by hash seed
        t = parse_task(task)
        if t["level"] in levels:
            cells[t["cell"]].append(np.mean(a[task]) - np.mean(b[task]))
    return _macro_cells(cells, np.random.default_rng(seed))


def by_cell(tasks: dict) -> dict[tuple, float]:
    """Mean pass rate per (family, level[w]) cell."""
    cells = defaultdict(list)
    for task, trials in tasks.items():
        cells[parse_task(task)["cell"]].append(np.mean(trials))
    return {cell: float(np.mean(v)) for cell, v in sorted(cells.items())}


def by_level(tasks: dict) -> dict[str, float]:
    """Mean over cells at each level label (L1..L4, L1w..L4w)."""
    levels = defaultdict(list)
    for (_family, level), value in by_cell(tasks).items():
        levels[level].append(value)
    return {level: float(np.mean(v)) for level, v in sorted(levels.items())}


def warp_pairs(tasks: dict) -> dict[str, dict[str, float]]:
    """Straight vs warped pass rate per level, over the warped families only.

    Each warped task is compared with the straight task it was built from, so
    topology, endpoints and style are held fixed and only the geometry bends.
    """
    straight, warped = defaultdict(list), defaultdict(list)
    for task, trials in tasks.items():
        t = parse_task(task)
        twin = twin_of(task)
        if t["warped"] and twin in tasks:
            straight[t["level"]].append(np.mean(tasks[twin]))
            warped[t["level"]].append(np.mean(trials))
    return {
        level: {"straight": float(np.mean(straight[level])),
                "warped": float(np.mean(warped[level])),
                "pairs": len(warped[level])}
        for level in sorted(warped)
    }


def pass_at_k_macro(tasks: dict, k: int, levels=LEVELS) -> float:
    """Macro pass@k over cells, from the unbiased per-task estimator."""
    cells = defaultdict(list)
    for task, trials in tasks.items():
        t = parse_task(task)
        if t["level"] in levels and len(trials) >= k:
            cells[t["cell"]].append(pass_at_k(len(trials), sum(trials), k))
    return float(np.mean([np.mean(v) for v in cells.values()])) if cells else float("nan")


def build_board(path: Path, upper_only: frozenset[str] = frozenset()) -> tuple[dict, dict]:
    """Every leg's scores, plus each leg's task -> trials map for paired
    comparisons. Legs in `upper_only` ran only L3/L4 and are scored on those
    cells; every leg also gets an L3/L4 score for comparison."""
    legs = load_board_rows(path)
    board = {}
    for name, leg in legs.items():
        tasks = leg["tasks"]
        upper = name in upper_only
        trials = max(len(v) for v in tasks.values())
        board[name] = {
            "upper_only": upper,
            "headline": macro(tasks, UPPER if upper else LEVELS),
            "upper": macro(tasks, UPPER),
            "pass_at_k": {"k": trials, "score": pass_at_k_macro(tasks, trials, UPPER if upper else LEVELS)},
            "route_progress": float(np.mean(leg["progress"])),
            "attempts": sum(len(v) for v in tasks.values()),
            "missing": leg["missing"],
            "unanswered": leg["unanswered"],
            "trials": trials,
            "by_level": by_level(tasks),
            "by_cell": {f"{f}/{lv}": v for (f, lv), v in by_cell(tasks).items()},
            "warp": warp_pairs(tasks),
            "median_output_tokens": float(np.median(leg["output_tokens"])) if leg["output_tokens"] else None,
            "median_output_tokens_upper": (float(np.median(leg["output_tokens_upper"]))
                                           if leg["output_tokens_upper"] else None),
            "median_latency_s": float(np.median(leg["latency"])) if leg["latency"] else None,
        }
    return board, {name: dict(leg["tasks"]) for name, leg in legs.items()}
