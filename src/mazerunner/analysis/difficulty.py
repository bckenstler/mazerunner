"""H1 (does measured difficulty predict success?) and H3 (variance components).

Both fits use resampling over *tasks* rather than parametric standard errors.
Eight attempts on one maze are correlated by construction, so the usual
model-based intervals would be too narrow; a cluster bootstrap makes no
independence claim we cannot support.

Logistic regression is fitted by IRLS here rather than pulled from statsmodels:
the model is four covariates on a few thousand rows, and avoiding the
dependency keeps the analysis reproducible from the committed lockfile alone.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

FEATURES = ("normalized_length", "turns", "route_branches", "min_clearance_px")


def _design(rows: list[dict], tasks: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X standardized with intercept, y, task ids)."""
    X, y, groups = [], [], []
    for row in rows:
        if row.get("error"):
            continue
        measures = row.get("measures")
        task = tasks.get(row.get("maze"))
        if not measures or task is None:
            continue
        clearance = task["reference"].get("min_clearance_px")
        if clearance is None:
            continue
        X.append([
            measures["normalized_length"],
            measures["turns"],
            measures["route_branches"],
            clearance,
        ])
        y.append(1.0 if (row.get("evaluation") or {}).get("success") else 0.0)
        groups.append(row["maze"])
    X = np.asarray(X, dtype=float)
    # Standardize so coefficients are comparable across units (px vs counts).
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.column_stack([np.ones(len(X)), X])
    return X, np.asarray(y), groups


def fit_logistic(X: np.ndarray, y: np.ndarray, *, iterations: int = 50, ridge: float = 1e-6):
    """Newton-Raphson / IRLS. Ridge term only to keep the Hessian invertible."""
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-8, None)
        gradient = X.T @ (y - p) - ridge * beta
        hessian = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def logistic_with_cluster_ci(
    rows: list[dict],
    tasks: dict[str, dict],
    *,
    resamples: int = 400,
    seed: int = 0,
) -> dict:
    """Standardized coefficients with a task-clustered bootstrap CI."""
    X, y, groups = _design(rows, tasks)
    if len(y) == 0:
        return {"n": 0, "coefficients": {}}
    beta = fit_logistic(X, y)

    by_task: dict[str, list[int]] = defaultdict(list)
    for i, task_id in enumerate(groups):
        by_task[task_id].append(i)
    task_ids = list(by_task)
    rng = np.random.default_rng(seed)

    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, len(task_ids), len(task_ids))
        idx = np.concatenate([by_task[task_ids[k]] for k in picked])
        try:
            draws.append(fit_logistic(X[idx], y[idx]))
        except np.linalg.LinAlgError:
            continue
    draws = np.asarray(draws)

    names = ("intercept",) + FEATURES
    return {
        "n_attempts": int(len(y)),
        "n_tasks": len(task_ids),
        "coefficients": {
            name: {
                "beta": float(beta[i]),
                "odds_ratio": float(np.exp(beta[i])),
                "ci": (
                    float(np.percentile(draws[:, i], 2.5)),
                    float(np.percentile(draws[:, i], 97.5)),
                ),
            }
            for i, name in enumerate(names)
        },
    }


def variance_components(
    grid: dict[tuple[str, str], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Two-way decomposition over a (topology, style) grid, bootstrapped by topology."""
    topologies = sorted({t for t, _ in grid})
    styles = sorted({s for _, s in grid})

    def decompose(topo_subset):
        M = np.full((len(topo_subset), len(styles)), np.nan)
        for i, t in enumerate(topo_subset):
            for j, s in enumerate(styles):
                if (t, s) in grid:
                    M[i, j] = grid[(t, s)]
        grand = np.nanmean(M)
        row = np.nanmean(M, axis=1) - grand
        col = np.nanmean(M, axis=0) - grand
        ss_topo = len(styles) * np.nansum(row**2)
        ss_style = len(topo_subset) * np.nansum(col**2)
        ss_res = np.nansum((M - grand - row[:, None] - col[None, :]) ** 2)
        total = ss_topo + ss_style + ss_res
        if total <= 0:
            return None
        return (ss_topo / total, ss_style / total, ss_res / total)

    point = decompose(topologies)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sample = [topologies[k] for k in rng.integers(0, len(topologies), len(topologies))]
        result = decompose(sample)
        if result:
            draws.append(result)
    draws = np.asarray(draws)

    labels = ("topology", "style", "interaction")
    return {
        "n_topologies": len(topologies),
        "n_styles": len(styles),
        "components": {
            label: {
                "share": float(point[i]),
                "ci": (
                    float(np.percentile(draws[:, i], 2.5)),
                    float(np.percentile(draws[:, i], 97.5)),
                ),
            }
            for i, label in enumerate(labels)
        },
    }
