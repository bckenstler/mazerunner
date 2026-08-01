"""Statistics for the leaderboard and the paired model comparisons.

Two choices here are load-bearing and easy to get wrong:

Resampling is **clustered over tasks**, not over attempts. Eight attempts on
one maze are not eight independent observations — they share a topology, a
style, and whatever makes that maze hard. Resampling attempts would shrink
intervals by roughly the square root of the trial count and manufacture
significance that is not there.

Model comparisons are **paired per task**. Every model saw the identical task
set, so the between-model variance that matters is within-task, and an
unpaired test throws that structure away.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_RESAMPLES = 10000


def bootstrap_ci(
    per_task_values: list[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling whole tasks."""
    values = np.asarray(per_task_values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: probability that k draws from n attempts contain a pass.

    The estimator (Chen et al.) rather than the naive "did any of the first k
    succeed", which is noisier and depends on trial ordering.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} attempts, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def paired_bootstrap(
    a_per_task: dict[str, float],
    b_per_task: dict[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Difference in means over tasks both models attempted."""
    shared = sorted(set(a_per_task) & set(b_per_task))
    if not shared:
        return {"n_tasks": 0, "difference": float("nan"), "ci": (float("nan"),) * 2, "p": 1.0}
    diffs = np.array([a_per_task[t] - b_per_task[t] for t in shared])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    observed = float(diffs.mean())
    # Two-sided p by sign flipping: how often does a null with no direction
    # produce a difference at least this large?
    flips = rng.choice([-1.0, 1.0], size=(resamples, diffs.size))
    null = (diffs * flips).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())
    return {
        "n_tasks": len(shared),
        "difference": observed,
        "ci": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "p": p,
    }


def mcnemar(a_any: dict[str, bool], b_any: dict[str, bool]) -> dict:
    """Exact McNemar on per-task any-success, for discordant pairs only."""
    shared = sorted(set(a_any) & set(b_any))
    b01 = sum(1 for t in shared if not a_any[t] and b_any[t])
    b10 = sum(1 for t in shared if a_any[t] and not b_any[t])
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    # Two-sided exact binomial against p=0.5.
    tail = sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1)) / (2**n)
    return {"a_only": b10, "b_only": b01, "p": min(1.0, 2 * tail)}
