"""Archival: unique paths per run, honest census, verification catches drift."""

from __future__ import annotations

import json

from mazerunner.archive import archive_runs, inventory, verify_archive


def _run(root, rel, rows):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_nested_shards_get_distinct_archive_paths(tmp_path):
    """Legs share shard names; taking only the parent dir would overwrite them."""
    results = tmp_path / "results"
    _run(results, "main/kimi/run-1/shard-03", [{"provider": "kimi", "maze": "t1", "trial": 0}])
    _run(results, "main/openai/run-1/shard-03", [{"provider": "openai", "maze": "t1", "trial": 0}])

    records = inventory(results)
    assert len({(r.leg, r.stamp) for r in records}) == 2

    archive_runs(records, tmp_path / "archive", results_root=results)
    archived = list((tmp_path / "archive" / "runs").rglob("attempts.jsonl.gz"))
    assert len(archived) == 2


def test_every_row_is_accounted_for(tmp_path):
    results = tmp_path / "results"
    _run(results, "leg/stamp", [
        {"provider": "a", "maze": "t1", "trial": 0, "raw_response": {"x": 1}, "reasoning": "r"},
        {"provider": "a", "maze": "t2", "trial": 0, "error": "transport failure"},
    ])
    records = inventory(results)
    assert records[0].rows == 2
    assert records[0].rows_with_raw == 1
    assert records[0].rows_with_error == 1


def test_empty_and_voided_runs_are_annotated_not_dropped(tmp_path):
    results = tmp_path / "results"
    _run(results, "killed/stamp", [])
    _run(results, "voided/stamp", [{"provider": "a", "maze": "t", "trial": 0, "error": "400"}])

    records = {r.leg: r for r in inventory(results)}

    assert "empty" in records["killed"].note
    assert "voided" in records["voided"].note


def test_verify_detects_a_modified_source(tmp_path):
    results = tmp_path / "results"
    run = _run(results, "leg/stamp", [{"provider": "a", "maze": "t1", "trial": 0}])
    archive_runs(inventory(results), tmp_path / "archive", results_root=results)
    assert verify_archive(tmp_path / "archive")["ok"]

    (run / "attempts.jsonl").write_text(json.dumps({"provider": "a", "maze": "CHANGED", "trial": 0}) + "\n")
    report = verify_archive(tmp_path / "archive")

    assert not report["ok"]
    assert report["mismatched"]


def test_verify_is_green_on_an_untouched_archive(tmp_path):
    results = tmp_path / "results"
    _run(results, "leg/stamp", [{"provider": "a", "maze": f"t{i}", "trial": 0} for i in range(5)])
    archive_runs(inventory(results), tmp_path / "archive", results_root=results)

    report = verify_archive(tmp_path / "archive")

    assert report["ok"]
    assert report["files_checked"] >= 1


def test_archive_is_idempotent(tmp_path):
    results = tmp_path / "results"
    _run(results, "leg/stamp", [{"provider": "a", "maze": "t1", "trial": 0}])
    out = tmp_path / "archive"

    first = archive_runs(inventory(results), out, results_root=results)
    second = archive_runs(inventory(results), out, results_root=results)

    assert first["coverage"]["rows_total"] == second["coverage"]["rows_total"]
    assert verify_archive(out)["ok"]


def test_archiving_a_growing_run_records_what_was_copied(tmp_path, monkeypatch):
    """A live run appends while we copy; the manifest must not contradict the file."""
    results = tmp_path / "results"
    run = _run(results, "leg/stamp", [{"provider": "a", "maze": "t1", "trial": 0}])
    records = inventory(results)
    assert records[0].rows == 1

    # Simulate the shard writing another attempt between scan and copy.
    with (run / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps({"provider": "a", "maze": "t2", "trial": 0}) + "\n")

    archive_runs(records, tmp_path / "archive", results_root=results)
    report = verify_archive(tmp_path / "archive")

    assert report["ok"], report
    manifest = json.loads((tmp_path / "archive" / "MANIFEST.json").read_text())
    assert manifest["records"][0]["rows"] == 2
    assert "in progress" in (manifest["records"][0]["note"] or "")


def test_appended_source_is_superseded_not_corrupt(tmp_path):
    """A run that kept writing leaves the archived bytes intact at the head."""
    results = tmp_path / "results"
    run = _run(results, "leg/stamp", [{"provider": "a", "maze": "t1", "trial": 0}])
    records = inventory(results)
    records[0].snapshot = True
    archive_runs(records, tmp_path / "archive", results_root=results)

    with (run / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps({"provider": "a", "maze": "t2", "trial": 0}) + "\n")

    report = verify_archive(tmp_path / "archive")
    assert report["ok"]
    assert report["superseded_snapshots"]


def test_rewritten_source_is_corrupt_even_if_marked_snapshot(tmp_path):
    """Timing must never excuse a rewrite: the archived bytes are gone."""
    results = tmp_path / "results"
    run = _run(results, "leg/stamp", [{"provider": "a", "maze": "t1", "trial": 0}])
    records = inventory(results)
    records[0].snapshot = True
    archive_runs(records, tmp_path / "archive", results_root=results)

    (run / "attempts.jsonl").write_text(
        json.dumps({"provider": "a", "maze": "REWRITTEN", "trial": 0}) + "\n"
    )

    report = verify_archive(tmp_path / "archive")
    assert not report["ok"]
    assert report["mismatched"]
