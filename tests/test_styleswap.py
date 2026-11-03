"""Style-swap variants must differ in style and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mazerunner.dataset import rebuild_from_provenance
from mazerunner.evaluator import evaluate_task
from mazerunner.io import load_task, mask_sha256
from mazerunner.styleswap import SWAP_ARCHETYPES, build_style_swap_set

DEV = Path("datasets/v2/test")


def test_rebuild_from_provenance_reproduces_the_mask():
    rows = [json.loads(l) for l in (DEV / "index.jsonl").read_text().splitlines() if l.strip()]
    task, _mask = load_task(Path(rows[0]["dir"]))
    prov = task["provenance"]
    _world, mask = rebuild_from_provenance(prov)
    assert mask_sha256(mask) == prov["mask_sha256"]


def test_build_is_deterministic(tmp_path):
    """Same seed, same tasks -> same variants."""
    rows = [json.loads(l) for l in (DEV / "index.jsonl").read_text().splitlines() if l.strip()]
    task_ids = [rows[0]["task_id"]]
    a = build_style_swap_set(DEV, task_ids, tmp_path / "a", archetypes=("forest-path",), seed=7)
    b = build_style_swap_set(DEV, task_ids, tmp_path / "b", archetypes=("forest-path",), seed=7)
    assert a["variants"] == b["variants"] == 1
    name = f"{task_ids[0]}--forest-path"
    assert (tmp_path / "a" / name / "input.png").read_bytes() == (
        tmp_path / "b" / name / "input.png"
    ).read_bytes()
