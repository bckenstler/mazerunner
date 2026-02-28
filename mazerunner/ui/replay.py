"""Replay data loader for saved run logs and single-shot results."""

import json
import os
from pathlib import Path


def list_runs(runs_dir: str) -> list[dict]:
    """Scan runs/ for JSON files, return metadata summaries."""
    results = []
    if not os.path.isdir(runs_dir):
        return results

    for model_dir in sorted(os.listdir(runs_dir)):
        model_path = os.path.join(runs_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(model_path, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                results.append({
                    "run_id": f"{model_dir}/{Path(fname).stem}",
                    "path": fpath,
                    "maze_id": data.get("maze_id", ""),
                    "model": data.get("model", model_dir),
                    "timestamp": data.get("timestamp", ""),
                    "num_turns": len(data.get("turns", [])),
                    "finish_reason": data.get("final_result", {}).get("finish_reason", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue

    return results


def load_run(path: str) -> dict:
    """Load a complete run log from a JSON file."""
    with open(path) as f:
        return json.load(f)


def list_results(results_dir: str) -> list[dict]:
    """Scan results/ for JSONL files, return file metadata."""
    results = []
    if not os.path.isdir(results_dir):
        return results

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(results_dir, fname)
        # Count entries
        count = 0
        try:
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            continue

        results.append({
            "filename": fname,
            "path": fpath,
            "num_entries": count,
        })

    return results


def load_result_entry(results_dir: str, filename: str, maze_id: str) -> dict | None:
    """Load a single entry from a JSONL results file by maze ID."""
    fpath = os.path.join(results_dir, filename)
    if not os.path.exists(fpath):
        return None

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("id") == maze_id:
                    return entry
            except json.JSONDecodeError:
                continue

    return None


def list_result_mazes(results_dir: str, filename: str) -> list[str]:
    """List all maze IDs in a JSONL results file."""
    fpath = os.path.join(results_dir, filename)
    ids = []
    if not os.path.exists(fpath):
        return ids

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "id" in entry:
                    ids.append(entry["id"])
            except json.JSONDecodeError:
                continue

    return sorted(ids)
