"""Analysis statistics — the places a wrong choice would manufacture significance."""

from __future__ import annotations

import json

import pytest

from mazerunner.analysis.load import load_attempts, scored, task_means
from mazerunner.analysis.quantization import grid_snap, localization_error, percentiles
from mazerunner.analysis.stats import bootstrap_ci, mcnemar, paired_bootstrap, pass_at_k


def _row(provider="a", maze="t1", trial=0, success=True, points=None, error=None):
    row = {"provider": provider, "maze": maze, "trial": trial}
    if error:
        row["error"] = error
        return row
    row["evaluation"] = {"success": success}
    row["submission"] = {"points": points or [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.8}]}
    return row


# ---------- pass@k ----------

def test_pass_at_k_edges():
    assert pass_at_k(8, 0, 8) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 1, 8) == 1.0   # the single pass is always drawn


def test_pass_at_k_is_monotone_in_k():
    values = [pass_at_k(8, 2, k) for k in range(1, 9)]
    assert values == sorted(values)


def test_pass_at_k_beats_naive_estimate():
    """Unbiased estimator, not 'did any of the first k succeed'."""
    assert 0.0 < pass_at_k(8, 1, 2) < 1.0


def test_pass_at_k_rejects_impossible_k():
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 8)


# ---------- bootstrap ----------

def test_bootstrap_ci_brackets_the_mean():
    values = [0.0, 0.25, 0.5, 0.75, 1.0] * 8
    lo, hi = bootstrap_ci(values, resamples=2000)
    assert lo < 0.5 < hi


def test_bootstrap_ci_narrows_with_more_tasks():
    few = bootstrap_ci([0.0, 1.0] * 5, resamples=2000)
    many = bootstrap_ci([0.0, 1.0] * 200, resamples=2000)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_bootstrap_ci_is_degenerate_on_a_constant():
    lo, hi = bootstrap_ci([1.0] * 20, resamples=500)
    assert lo == hi == 1.0


def test_bootstrap_ci_is_seed_reproducible():
    v = [0.0, 0.5, 1.0] * 10
    assert bootstrap_ci(v, resamples=500, seed=3) == bootstrap_ci(v, resamples=500, seed=3)


# ---------- paired tests ----------

def test_paired_bootstrap_detects_a_consistent_edge():
    a = {f"t{i}": 0.8 for i in range(40)}
    b = {f"t{i}": 0.3 for i in range(40)}
    result = paired_bootstrap(a, b, resamples=2000)
    assert result["difference"] == pytest.approx(0.5)
    assert result["p"] < 0.01
    assert result["ci"][0] > 0


def test_paired_bootstrap_reports_a_tie_as_a_tie():
    a = {f"t{i}": 0.5 + (0.1 if i % 2 else -0.1) for i in range(40)}
    b = {f"t{i}": 0.5 for i in range(40)}
    result = paired_bootstrap(a, b, resamples=2000)
    assert result["p"] > 0.05
    assert result["ci"][0] < 0 < result["ci"][1]


def test_paired_bootstrap_uses_only_shared_tasks():
    a = {"t1": 1.0, "t2": 1.0, "only_a": 1.0}
    b = {"t1": 0.0, "t2": 0.0}
    assert paired_bootstrap(a, b, resamples=200)["n_tasks"] == 2


def test_paired_bootstrap_handles_no_overlap():
    result = paired_bootstrap({"a": 1.0}, {"b": 1.0}, resamples=100)
    assert result["n_tasks"] == 0


def test_mcnemar_counts_discordant_pairs_only():
    a = {"t1": True, "t2": True, "t3": False, "t4": True}
    b = {"t1": True, "t2": False, "t3": True, "t4": False}
    result = mcnemar(a, b)
    assert result["a_only"] == 2   # t2, t4
    assert result["b_only"] == 1   # t3


def test_mcnemar_identical_models_are_not_significant():
    a = {f"t{i}": i % 2 == 0 for i in range(20)}
    assert mcnemar(a, dict(a))["p"] == 1.0


def test_mcnemar_detects_a_one_sided_advantage():
    a = {f"t{i}": True for i in range(12)}
    b = {f"t{i}": False for i in range(12)}
    assert mcnemar(a, b)["p"] < 0.01


# ---------- loading ----------

def test_scored_excludes_transport_failures():
    rows = [_row(), _row(maze="t2", error="transport failure")]
    assert len(scored(rows)) == 1


def test_task_means_averages_within_task():
    rows = [_row(maze="t1", trial=0, success=True), _row(maze="t1", trial=1, success=False)]
    assert task_means(rows)["a"]["t1"] == 0.5


def test_loader_drops_raw_payloads(tmp_path):
    """A full run is ~400MB of traces; analysis must not hold them."""
    path = tmp_path / "attempts.jsonl"
    row = _row()
    row["raw_response"] = {"huge": "x" * 10000}
    path.write_text(json.dumps(row) + "\n")

    loaded = load_attempts([path], None, backfill=False)
    assert "raw_response" not in loaded[0]


# ---------- fingerprints ----------

def test_grid_snap_detects_round_coordinates():
    snapped = [_row(points=[{"x": 0.10, "y": 0.20}, {"x": 0.50, "y": 0.75}])]
    irregular = [_row(provider="b", points=[{"x": 0.1037, "y": 0.2192}, {"x": 0.5231, "y": 0.7614}])]
    result = grid_snap(snapped + irregular)
    assert result["a"][0.05] == 1.0
    assert result["b"][0.05] == 0.0


def test_localization_error_measures_distance_to_the_start_badge():
    task = {"width": 101, "height": 101, "start": {"x": 0.0, "y": 0.0},
            "goal": {"x": 1.0, "y": 1.0}}
    rows = [_row(points=[{"x": 0.1, "y": 0.0}, {"x": 0.9, "y": 0.9}])]
    errors = localization_error(rows, {"t1": task})
    assert errors["a"][0] == pytest.approx(10.0)


def test_percentiles_are_ordered():
    q = percentiles([float(i) for i in range(100)])
    assert q[50] < q[75] < q[90] < q[95]
