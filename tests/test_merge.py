"""Shard merging: dedup policy, conflict detection, requeue accounting."""

from __future__ import annotations

import json
from pathlib import Path

from mazerunner.merge import merge_runs, missing_units, write_missing_task_list


def _row(provider="a", maze="t1", trial=0, success=None, error=None, timestamp="2026-01-01T00:00:00",
         canary=False, shard=0):
    row = {
        "provider": provider,
        "maze": maze,
        "trial": trial,
        "timestamp": timestamp,
        "shard": shard,
    }
    if error:
        row["error"] = error
    if success is not None:
        row["evaluation"] = {"success": success, "efficiency_canary": canary}
    return row


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_merges_disjoint_shards(tmp_path):
    a = _write(tmp_path / "s0" / "attempts.jsonl", [_row(maze="t1", success=True)])
    b = _write(tmp_path / "s1" / "attempts.jsonl", [_row(maze="t2", success=False)])

    manifest = merge_runs([a, b], tmp_path / "merged")

    assert manifest["rows_out"] == 2
    assert manifest["duplicates_collapsed"] == 0
    lines = (tmp_path / "merged" / "attempts.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_agreeing_duplicate_is_collapsed(tmp_path):
    a = _write(tmp_path / "s0" / "attempts.jsonl", [_row(success=True)])
    b = _write(tmp_path / "s1" / "attempts.jsonl", [_row(success=True)])

    manifest = merge_runs([a, b], tmp_path / "merged")

    assert manifest["rows_out"] == 1
    assert manifest["duplicates_collapsed"] == 1
    assert manifest["conflicts"] == []


def test_disagreeing_duplicate_is_reported(tmp_path):
    """Two shards scoring the same attempt differently means the split was wrong."""
    a = _write(tmp_path / "s0" / "attempts.jsonl", [_row(success=True, shard=0)])
    b = _write(tmp_path / "s1" / "attempts.jsonl", [_row(success=False, shard=1)])

    manifest = merge_runs([a, b], tmp_path / "merged")

    assert len(manifest["conflicts"]) == 1
    conflict = manifest["conflicts"][0]
    assert conflict["key"] == {"provider": "a", "maze": "t1", "trial": 0}
    assert {conflict["a"]["success"], conflict["b"]["success"]} == {True, False}


def test_evaluated_row_beats_transport_failure(tmp_path):
    """The resumed attempt that actually reached the model is the real one."""
    a = _write(tmp_path / "s0" / "attempts.jsonl", [_row(error="transport failure: 429")])
    b = _write(tmp_path / "s1" / "attempts.jsonl", [_row(success=True)])

    merge_runs([a, b], tmp_path / "merged")

    merged = json.loads((tmp_path / "merged" / "attempts.jsonl").read_text().strip())
    assert merged["evaluation"]["success"] is True
    assert "error" not in merged


def test_transport_failure_does_not_overwrite_an_evaluated_row(tmp_path):
    a = _write(tmp_path / "s0" / "attempts.jsonl", [_row(success=True, timestamp="2026-01-01T00:00:00")])
    b = _write(tmp_path / "s1" / "attempts.jsonl", [_row(error="boom", timestamp="2026-01-02T00:00:00")])

    merge_runs([a, b], tmp_path / "merged")

    merged = json.loads((tmp_path / "merged" / "attempts.jsonl").read_text().strip())
    assert merged["evaluation"]["success"] is True


def test_later_timestamp_wins_between_equals(tmp_path):
    a = _write(tmp_path / "s0" / "attempts.jsonl",
               [_row(success=True, timestamp="2026-01-01T00:00:00", shard=0)])
    b = _write(tmp_path / "s1" / "attempts.jsonl",
               [_row(success=True, timestamp="2026-01-05T00:00:00", shard=9)])

    merge_runs([a, b], tmp_path / "merged")

    merged = json.loads((tmp_path / "merged" / "attempts.jsonl").read_text().strip())
    assert merged["shard"] == 9


def test_malformed_lines_are_counted_not_fatal(tmp_path):
    path = tmp_path / "s0" / "attempts.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_row(success=True)) + "\n" + '{"provider": "a", "ma')

    manifest = merge_runs([path], tmp_path / "merged")

    assert manifest["rows_out"] == 1
    assert manifest["malformed_lines"] == 1


def test_per_leg_accounting(tmp_path):
    rows = [
        _row(provider="gpt", maze="t1", trial=0, success=True),
        _row(provider="gpt", maze="t2", trial=0, success=False),
        _row(provider="opus", maze="t1", trial=0, error="boom"),
    ]
    path = _write(tmp_path / "s0" / "attempts.jsonl", rows)

    manifest = merge_runs([path], tmp_path / "merged")

    assert manifest["per_leg"]["gpt"] == {
        "attempts": 2, "evaluated": 2, "successes": 1, "transport_failures": 0
    }
    assert manifest["per_leg"]["opus"]["transport_failures"] == 1


def test_efficiency_canary_tasks_are_surfaced(tmp_path):
    """Canary tasks are quarantined, not dropped; the merge must name them."""
    rows = [_row(maze="suspect", success=True, canary=True), _row(maze="fine", success=True)]
    path = _write(tmp_path / "s0" / "attempts.jsonl", rows)

    manifest = merge_runs([path], tmp_path / "merged")

    assert manifest["efficiency_canary_tasks"] == ["suspect"]


def test_missing_units_finds_gaps(tmp_path):
    rows = [_row(provider="a", maze="t1", trial=0, success=True),
            _row(provider="a", maze="t1", trial=1, success=True)]
    merged = _write(tmp_path / "merged" / "attempts.jsonl", rows)

    missing = missing_units(merged, ["t1", "t2"], ["a"], trials=2)

    assert set(missing) == {("a", "t2", 0), ("a", "t2", 1)}


def test_transport_failures_count_as_missing(tmp_path):
    """A recorded failure is still work owed — the retry policy requeues it."""
    rows = [_row(provider="a", maze="t1", trial=0, error="429")]
    merged = _write(tmp_path / "merged" / "attempts.jsonl", rows)

    missing = missing_units(merged, ["t1"], ["a"], trials=1)

    assert missing == [("a", "t1", 0)]


def test_write_missing_task_list_dedupes(tmp_path):
    missing = [("a", "t2", 0), ("a", "t2", 1), ("a", "t5", 0)]
    out = tmp_path / "missing.txt"

    count = write_missing_task_list(missing, out)

    assert count == 2
    assert out.read_text().split() == ["t2", "t5"]


def test_complete_run_reports_nothing_missing(tmp_path):
    rows = [_row(provider="a", maze="t1", trial=t, success=True) for t in range(3)]
    merged = _write(tmp_path / "merged" / "attempts.jsonl", rows)

    assert missing_units(merged, ["t1"], ["a"], trials=3) == []


def test_conflict_kinds_are_distinguished(tmp_path):
    """Two shards disagreeing is a bug; one shard disagreeing across time is a rerun."""
    cross_a = _write(tmp_path / "x0" / "attempts.jsonl", [_row(success=True, shard=0)])
    cross_b = _write(tmp_path / "x1" / "attempts.jsonl", [_row(success=False, shard=1)])
    manifest = merge_runs([cross_a, cross_b], tmp_path / "m1")
    assert len(manifest["cross_shard_conflicts"]) == 1
    assert not manifest["re_execution_conflicts"]

    rerun_a = _write(tmp_path / "y0" / "attempts.jsonl",
                     [_row(success=True, shard=3, timestamp="2026-01-01T00:00:00")])
    rerun_b = _write(tmp_path / "y1" / "attempts.jsonl",
                     [_row(success=False, shard=3, timestamp="2026-01-01T00:10:00")])
    manifest = merge_runs([rerun_a, rerun_b], tmp_path / "m2")
    assert len(manifest["re_execution_conflicts"]) == 1
    assert not manifest["cross_shard_conflicts"]
