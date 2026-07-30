"""Unit planning, sharding, and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mazerunner.runner import AttemptUnit, completed_keys, plan_units, shard

SOURCES = [(f"task-{i:02d}", Path(f"/tasks/task-{i:02d}")) for i in range(10)]


def test_plan_covers_every_provider_task_trial():
    units = plan_units(SOURCES, ["a", "b"], 3, order_seed=1)
    assert len(units) == 10 * 2 * 3
    keys = {(u.provider, u.task_id, u.trial) for u in units}
    assert len(keys) == 60


def test_ordinals_are_dense_per_leg():
    units = plan_units(SOURCES, ["a", "b"], 2, order_seed=1)
    for provider in ("a", "b"):
        ordinals = sorted(u.ordinal for u in units if u.provider == provider)
        assert ordinals == list(range(20))


def test_seed_makes_order_reproducible():
    first = plan_units(SOURCES, ["a"], 1, order_seed=7)
    second = plan_units(SOURCES, ["a"], 1, order_seed=7)
    assert [u.task_id for u in first] == [u.task_id for u in second]


def test_legs_get_independent_orders():
    """Otherwise every model meets the tasks in the same sequence."""
    units = plan_units(SOURCES, ["a", "b"], 1, order_seed=3)
    order_a = [u.task_id for u in units if u.provider == "a"]
    order_b = [u.task_id for u in units if u.provider == "b"]
    assert order_a != order_b
    assert sorted(order_a) == sorted(order_b)


def test_shuffle_actually_reorders():
    units = plan_units(SOURCES, ["a"], 1, order_seed=5)
    assert [u.task_id for u in units] != [name for name, _ in SOURCES]


def test_no_seed_preserves_source_order():
    units = plan_units(SOURCES, ["a"], 1, order_seed=None)
    assert [u.task_id for u in units] == [name for name, _ in SOURCES]


def test_trials_stay_adjacent_within_a_task():
    units = [u for u in plan_units(SOURCES, ["a"], 3, order_seed=2) if u.task_id == "task-00"]
    assert [u.trial for u in units] == [0, 1, 2]


# ---------- sharding ----------

def test_shards_partition_the_work_exactly():
    units = plan_units(SOURCES, ["a", "b"], 4, order_seed=1)
    pieces = [shard(units, i, 7) for i in range(7)]
    assert sum(len(p) for p in pieces) == len(units)
    rejoined = {(u.provider, u.task_id, u.trial) for piece in pieces for u in piece}
    assert rejoined == {(u.provider, u.task_id, u.trial) for u in units}


def test_shards_are_disjoint():
    units = plan_units(SOURCES, ["a"], 3, order_seed=1)
    seen = set()
    for i in range(5):
        for unit in shard(units, i, 5):
            key = (unit.provider, unit.task_id, unit.trial)
            assert key not in seen
            seen.add(key)


def test_shards_are_balanced_within_one():
    units = plan_units(SOURCES, ["a"], 3, order_seed=1)
    sizes = [len(shard(units, i, 7)) for i in range(7)]
    assert max(sizes) - min(sizes) <= 1


def test_striding_spreads_a_task_across_shards():
    """Sharding by task would strand one task's whole trial block in a process."""
    units = plan_units(SOURCES, ["a"], 8, order_seed=1)
    holders = {
        i for i in range(4) if any(u.task_id == "task-03" for u in shard(units, i, 4))
    }
    assert len(holders) > 1


def test_single_shard_returns_everything():
    units = plan_units(SOURCES, ["a"], 2, order_seed=1)
    assert shard(units, 0, 1) == units


def test_bad_shard_index_raises():
    units = plan_units(SOURCES, ["a"], 1, order_seed=1)
    with pytest.raises(ValueError):
        shard(units, 5, 5)


# ---------- resume ----------

def test_completed_keys_reads_recorded_attempts(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"provider": "a", "maze": f"task-{i:02d}", "trial": 0}) for i in range(3)
        )
        + "\n"
    )
    assert completed_keys(path) == {("a", "task-00", 0), ("a", "task-01", 0), ("a", "task-02", 0)}


def test_completed_keys_on_missing_file(tmp_path):
    assert completed_keys(tmp_path / "nope.jsonl") == set()


def test_completed_keys_survives_a_truncated_final_line(tmp_path):
    """A crash mid-write must not make the whole leg unresumable."""
    path = tmp_path / "attempts.jsonl"
    path.write_text(
        json.dumps({"provider": "a", "maze": "task-00", "trial": 0})
        + "\n"
        + '{"provider": "a", "maze": "task-01", "tri'
    )
    assert completed_keys(path) == {("a", "task-00", 0)}


def test_resume_filter_leaves_only_outstanding_work():
    units = plan_units(SOURCES, ["a"], 2, order_seed=1)
    done = {("a", "task-00", 0), ("a", "task-01", 1)}
    remaining = [u for u in units if (u.provider, u.task_id, u.trial) not in done]
    assert len(remaining) == len(units) - 2
