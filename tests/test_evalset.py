"""Eval-set selection: constraints actually hold, and the seed reproduces it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mazerunner.evalset import (
    InfeasibleSelection,
    SelectionSpec,
    read_task_list,
    select_eval_set,
    write_task_list,
)

DEV_INDEX = Path("datasets/v1/dev/index.jsonl")


def _dev_rows():
    if not DEV_INDEX.exists():
        pytest.skip("v1 dev split not built")
    return [json.loads(line) for line in DEV_INDEX.read_text().splitlines() if line.strip()]


def _counts(rows, ids, key):
    chosen = [r for r in rows if r["task_id"] in set(ids)]
    out = {}
    for row in chosen:
        out[row[key]] = out.get(row[key], 0) + 1
    return out


def test_selection_satisfies_every_constraint():
    rows = _dev_rows()
    ids, manifest = select_eval_set(rows, seed=20260730)

    assert len(ids) == 100
    assert len(set(ids)) == 100

    families = _counts(rows, ids, "family")
    assert all(12 <= n <= 13 for n in families.values()), families
    assert set(families) == {r["family"] for r in rows}

    assert _counts(rows, ids, "tier") == {"easy": 30, "medium": 40, "hard": 30}

    archetypes = _counts(rows, ids, "archetype")
    assert all(n >= 6 for n in archetypes.values()), archetypes
    # Every public archetype must appear; a missing one would silently bias the run.
    assert set(archetypes) == {r["archetype"] for r in rows}


def test_selection_is_deterministic_for_a_seed():
    rows = _dev_rows()
    first, _ = select_eval_set(rows, seed=7)
    second, _ = select_eval_set(rows, seed=7)
    assert first == second


def test_different_seeds_give_different_sets():
    rows = _dev_rows()
    a, _ = select_eval_set(rows, seed=1)
    b, _ = select_eval_set(rows, seed=2)
    assert a != b


def test_ids_are_sorted_for_stable_diffs():
    rows = _dev_rows()
    ids, _ = select_eval_set(rows, seed=3)
    assert ids == sorted(ids)


def test_manifest_records_achieved_composition():
    rows = _dev_rows()
    ids, manifest = select_eval_set(rows, seed=11)
    assert manifest["seed"] == 11
    assert manifest["selected"] == len(ids)
    assert sum(manifest["achieved"]["tier"].values()) == 100


def test_infeasible_constraints_raise_rather_than_hang():
    """Rejection sampling would spin forever here; the ILP reports impossibility."""
    rows = _dev_rows()
    impossible = SelectionSpec(size=100, archetype_floor=50)
    with pytest.raises(InfeasibleSelection):
        select_eval_set(rows, seed=0, spec=impossible)


def test_read_task_list_handles_both_formats(tmp_path):
    newline = tmp_path / "a.txt"
    newline.write_text("alpha\nbeta\ngamma\n")
    assert read_task_list(newline) == ["alpha", "beta", "gamma"]

    # results/pilot-tasks.txt is one comma-separated line.
    comma = tmp_path / "b.txt"
    comma.write_text("alpha,beta,gamma")
    assert read_task_list(comma) == ["alpha", "beta", "gamma"]


def test_read_task_list_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("# frozen set\nalpha\n\nbeta\n")
    assert read_task_list(path) == ["alpha", "beta"]


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "nested" / "list.txt"
    write_task_list(path, ["x", "y"])
    assert read_task_list(path) == ["x", "y"]


def test_frozen_eval_set_on_disk_matches_manifest():
    """The committed list is the one the recorded seed produces."""
    out = Path("evals/dev-eval-100.txt")
    if not out.exists():
        pytest.skip("eval set not built")
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert read_task_list(out) == manifest["task_ids"]
    rows = _dev_rows()
    ids, _ = select_eval_set(rows, seed=manifest["seed"], spec=SelectionSpec(**manifest["spec"]))
    assert ids == manifest["task_ids"]
