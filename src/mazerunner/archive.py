"""Durable archival of run artifacts.

`results/` is gitignored working space: ad-hoc shard dirs, partial legs from
killed runs, and 100s of MB of provider traces. This module freezes it into
`archive/runs/<leg>/<stamp>/` with checksums so nothing collected can be lost
or silently altered, and so the record survives outside a single machine.

Nothing here ever deletes or rewrites a source file. Legs that failed, were
killed mid-run, or predate trace capture are archived and *documented* as such
in the manifest — a voided leg is evidence, not garbage.
"""

from __future__ import annotations

import datetime
import gzip
import json
import shutil
import tarfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .io import file_sha256

MANIFEST_NAME = "MANIFEST.json"
# A run whose attempts file was touched this recently is treated as live.
LIVE_RUN_SECONDS = 180
ARCHIVE_VERSION = 1


@dataclass
class RunRecord:
    """One timestamped run directory under results/<leg>/<stamp>/."""

    leg: str
    stamp: str
    source_dir: str
    rows: int = 0
    rows_with_raw: int = 0
    rows_with_reasoning: int = 0
    rows_with_error: int = 0
    malformed_lines: int = 0
    providers: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    overlays: int = 0
    log_file: str | None = None
    files: dict[str, str] = field(default_factory=dict)  # relative path -> sha256
    bytes_total: int = 0
    note: str | None = None
    # True when the run was still appending as we copied. The archived payload
    # is a valid prefix, but the source will not match its recorded hash — a
    # later archive supersedes it. Verification must not call that corruption.
    snapshot: bool = False

    @property
    def is_empty(self) -> bool:
        return self.rows == 0


def _scan_attempts(path: Path) -> dict:
    """Row-level census of an attempts.jsonl. Never raises on bad lines."""
    stats = {
        "rows": 0,
        "rows_with_raw": 0,
        "rows_with_reasoning": 0,
        "rows_with_error": 0,
        "malformed_lines": 0,
        "providers": set(),
        "models": set(),
        "task_ids": set(),
    }
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_lines"] += 1
                continue
            stats["rows"] += 1
            if row.get("error"):
                stats["rows_with_error"] += 1
            if row.get("raw_response") is not None:
                stats["rows_with_raw"] += 1
            if row.get("reasoning"):
                stats["rows_with_reasoning"] += 1
            if row.get("provider"):
                stats["providers"].add(row["provider"])
            if row.get("model"):
                stats["models"].add(row["model"])
            if row.get("maze"):
                stats["task_ids"].add(row["maze"])
    for key in ("providers", "models", "task_ids"):
        stats[key] = sorted(stats[key])
    return stats


def inventory(results_root: Path) -> list[RunRecord]:
    """Every run directory under results/, newest last.

    A run directory is any dir containing attempts.jsonl. Its parent is the
    "leg" (e.g. sweep-gpt-xhigh) and its own name the timestamp. Legs written
    directly into results/ (the earliest smoke runs) get leg name "_root".
    """
    records: list[RunRecord] = []
    for attempts in sorted(results_root.rglob("attempts.jsonl")):
        run_dir = attempts.parent
        # The leg is the *full* relative path above the run dir, not just its
        # parent's name: sharded runs nest as
        # results/main/<provider>/<run-id>/shard-NN, so seven legs share the
        # name "shard-03" and taking only the parent would archive them all to
        # one path, silently overwriting six.
        relative = run_dir.relative_to(results_root)
        leg = str(relative.parent) if str(relative.parent) != "." else "_root"
        stamp = relative.name

        stats = _scan_attempts(attempts)
        record = RunRecord(
            leg=leg,
            stamp=stamp,
            source_dir=str(run_dir),
            rows=stats["rows"],
            rows_with_raw=stats["rows_with_raw"],
            rows_with_reasoning=stats["rows_with_reasoning"],
            rows_with_error=stats["rows_with_error"],
            malformed_lines=stats["malformed_lines"],
            providers=stats["providers"],
            models=stats["models"],
            task_ids=stats["task_ids"],
        )

        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                rel = str(path.relative_to(run_dir))
                record.files[rel] = file_sha256(path)
                record.bytes_total += path.stat().st_size
                if path.suffix == ".png":
                    record.overlays += 1

        # Launcher logs sit beside the tree as results/<leg>.log for flat runs
        # and results/<root>-<provider>-<shard>.log for sharded ones.
        for candidate in (
            results_root / f"{leg}.log",
            results_root / f"{leg.replace('/', '-')}.log",
        ):
            if candidate.exists():
                record.log_file = str(candidate)
                break

        # A directory touched moments ago is almost certainly still being
        # written. Its hashes will be stale before verification runs, so mark
        # it a snapshot up front rather than discovering the mismatch later.
        if time.time() - attempts.stat().st_mtime < LIVE_RUN_SECONDS:
            record.snapshot = True
            record.note = "archived while the run was still writing; re-archive once quiesced"

        if record.rows == 0:
            record.note = "empty — run produced no attempts (killed before first write)"
        elif record.rows_with_raw == 0 and record.rows_with_error == 0:
            record.note = "predates full trace capture — no raw_response stored"
        elif record.rows_with_error == record.rows:
            record.note = "all attempts failed transport — leg voided, retained as evidence"

        records.append(record)
    return records


def _coverage_table(records: list[RunRecord]) -> dict:
    total = sum(r.rows for r in records)
    with_raw = sum(r.rows_with_raw for r in records)
    with_reasoning = sum(r.rows_with_reasoning for r in records)
    errors = sum(r.rows_with_error for r in records)
    return {
        "rows_total": total,
        "rows_with_raw_response": with_raw,
        "rows_with_reasoning": with_reasoning,
        "rows_transport_error": errors,
        "rows_without_raw_response": total - with_raw,
        "malformed_lines": sum(r.malformed_lines for r in records),
        "empty_runs": [f"{r.leg}/{r.stamp}" for r in records if r.is_empty],
        "runs_without_traces": [
            f"{r.leg}/{r.stamp}" for r in records if r.rows and r.rows_with_raw == 0
        ],
        "voided_runs": [
            f"{r.leg}/{r.stamp}"
            for r in records
            if r.rows and r.rows_with_error == r.rows
        ],
    }


def archive_runs(
    records: list[RunRecord],
    out_dir: Path,
    *,
    results_root: Path,
    compress_overlays: bool = True,
) -> dict:
    """Copy every run into out_dir, compressed, with per-file checksums.

    Layout: archive/runs/<leg>/<stamp>/{attempts.jsonl.gz, summary.json,
    <leg>.log, overlays.tar.gz}. Idempotent — re-archiving an unchanged run
    rewrites identical bytes, and the manifest records the *source* hashes so
    verification compares against the originals.
    """
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    archived = []

    for record in records:
        source = Path(record.source_dir)
        dest = runs_dir / record.leg / record.stamp
        dest.mkdir(parents=True, exist_ok=True)

        attempts = source / "attempts.jsonl"
        if attempts.exists():
            target = dest / "attempts.jsonl.gz"
            with attempts.open("rb") as src, gzip.open(target, "wb", compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
            # A live run appends while we copy, so the census taken during
            # inventory can be stale by the time the bytes land. Re-count what
            # was actually archived and correct the record, rather than
            # recording a number the archived file contradicts.
            with gzip.open(target, "rt") as handle:
                archived_rows = sum(1 for line in handle if line.strip())
            if archived_rows != record.rows + record.malformed_lines:
                record.snapshot = True
                record.note = (
                    (record.note + "; " if record.note else "")
                    + f"snapshot of a run in progress ({archived_rows} rows at copy time, "
                    f"{record.rows} at scan time); re-archive once quiesced"
                )
                record.rows = archived_rows - record.malformed_lines
            record.files["attempts.jsonl"] = file_sha256(attempts)

        summary = source / "summary.json"
        if summary.exists():
            shutil.copy2(summary, dest / "summary.json")

        overlays = source / "overlays"
        if compress_overlays and overlays.is_dir() and any(overlays.iterdir()):
            with tarfile.open(dest / "overlays.tar.gz", "w:gz") as tar:
                tar.add(overlays, arcname="overlays")

        if record.log_file:
            log = Path(record.log_file)
            if log.exists():
                shutil.copy2(log, dest / log.name)

        (dest / "run-record.json").write_text(json.dumps(asdict(record), indent=2))
        archived.append(f"{record.leg}/{record.stamp}")

    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "runs": len(records),
        "archived": archived,
        "retention": {
            "attempts.jsonl.gz": "tracked in git — full traces, submissions, reasoning, raw responses",
            "summary.json / run-record.json / *.log": "tracked in git",
            "overlays.tar.gz": (
                "local only (gitignored) — derived artifact, regenerable from the "
                "stored submissions via overlay.render_overlay()"
            ),
            "deletions": "none; empty, voided, and pre-trace-capture runs are retained and annotated",
        },
        "coverage": _coverage_table(records),
        "records": [asdict(r) for r in records],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_archive(out_dir: Path) -> dict:
    """Re-hash every archived source file and compare against the manifest.

    Checks the *sources* still match what was archived, and that each archived
    payload is readable (gzip decompresses, tar lists). Returns a report; the
    caller decides the exit code.
    """
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"ok": False, "error": f"no manifest at {manifest_path}"}
    manifest = json.loads(manifest_path.read_text())

    mismatched: list[str] = []
    missing_source: list[str] = []
    unreadable: list[str] = []
    superseded: list[str] = []
    checked = 0

    for record in manifest["records"]:
        source = Path(record["source_dir"])
        is_snapshot = record.get("snapshot", False)
        for rel, digest in record["files"].items():
            path = source / rel
            if not path.exists():
                missing_source.append(str(path))
                continue
            checked += 1
            if file_sha256(path) == digest:
                continue
            # A run that was mid-flight when archived has legitimately grown.
            # That is a stale archive to refresh, not a corrupted one.
            (superseded if is_snapshot else mismatched).append(str(path))

        dest = out_dir / "runs" / record["leg"] / record["stamp"]
        gz = dest / "attempts.jsonl.gz"
        if gz.exists():
            try:
                with gzip.open(gz, "rt") as handle:
                    rows = sum(1 for line in handle if line.strip())
                if rows != record["rows"] + record["malformed_lines"]:
                    unreadable.append(f"{gz}: {rows} lines, expected {record['rows']}")
            except OSError as exc:
                unreadable.append(f"{gz}: {exc}")
        tar = dest / "overlays.tar.gz"
        if tar.exists():
            try:
                with tarfile.open(tar) as archive:
                    archive.getnames()
            except tarfile.TarError as exc:
                unreadable.append(f"{tar}: {exc}")

    ok = not (mismatched or unreadable)
    return {
        "ok": ok,
        "files_checked": checked,
        "mismatched": mismatched,
        "missing_source": missing_source,
        "unreadable_archives": unreadable,
        "superseded_snapshots": superseded,
        "coverage": manifest["coverage"],
    }
