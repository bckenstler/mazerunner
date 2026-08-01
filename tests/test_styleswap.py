"""Style-swap variants must differ in style and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mazerunner.dataset import rebuild_from_provenance
from mazerunner.evaluator import evaluate_task
from mazerunner.io import load_task, mask_sha256
from mazerunner.styleswap import SWAP_ARCHETYPES, build_style_swap_set

SWAP_DIR = Path("datasets/styleswap-v1")
DEV = Path("datasets/v1/dev")


def _rows():
    index = SWAP_DIR / "index.jsonl"
    if not index.exists():
        pytest.skip("style-swap set not built")
    return [json.loads(l) for l in index.read_text().splitlines() if l.strip()]


def test_swap_archetypes_avoid_every_forbidden_cell():
    """These five are legal for all families, so no group is unbalanced."""
    config = json.loads(Path("dataset.config.json").read_text())
    forbidden = {tuple(cell) for cell in config["public_forbidden_cells"]}
    forbidden_archetypes = {archetype for _family, archetype in forbidden}
    assert not (set(SWAP_ARCHETYPES) & forbidden_archetypes)


def test_rebuild_from_provenance_reproduces_the_mask():
    rows = [json.loads(l) for l in (DEV / "index.jsonl").read_text().splitlines() if l.strip()]
    task, _mask = load_task(Path(rows[0]["dir"]))
    prov = task["provenance"]
    _world, mask = rebuild_from_provenance(prov)
    assert mask_sha256(mask) == prov["mask_sha256"]


def test_every_group_is_complete():
    """A half-populated group would unbalance the H3 variance decomposition."""
    rows = _rows()
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_group"], set()).add(row["archetype"])
    assert groups, "no pair-groups built"
    for group, archetypes in groups.items():
        assert archetypes == set(SWAP_ARCHETYPES), f"{group} missing {set(SWAP_ARCHETYPES) - archetypes}"


def test_variants_within_a_group_share_one_mask():
    """The load-bearing check: same maze, different paint."""
    groups = {}
    for row in _rows():
        groups.setdefault(row["pair_group"], []).append(row)
    for group, rows in groups.items():
        digests = set()
        for row in rows:
            _task, mask = load_task(Path(row["dir"]))
            digests.add(mask_sha256(mask))
        assert len(digests) == 1, f"{group} has {len(digests)} distinct masks"


def test_variants_keep_the_source_topology_hash():
    for row in _rows():
        task, _mask = load_task(Path(row["dir"]))
        assert task["provenance"]["graph_hash"] == row["graph_hash"]


def test_reference_route_still_passes_under_every_style():
    """If a restyle broke the route, the pair would not be a fair comparison."""
    for row in _rows():
        task, mask = load_task(Path(row["dir"]))
        result = evaluate_task(task, mask, {"points": task["reference"]["optimal_path"]})
        assert result.success, f"{row['task_id']} reference route fails the scorer"


def test_archetype_recorded_matches_the_render():
    for row in _rows():
        task, _mask = load_task(Path(row["dir"]))
        assert task["style_record"]["archetype"] == row["archetype"]
        assert task["provenance"]["style_swap"]["archetype"] == row["archetype"]


def test_variants_within_a_group_have_distinct_images():
    """Different archetypes must actually look different."""
    groups = {}
    for row in _rows():
        groups.setdefault(row["pair_group"], []).append(row)
    group, rows = next(iter(groups.items()))
    images = {(Path(r["dir"]) / "input.png").read_bytes() for r in rows}
    assert len(images) == len(rows), f"{group} has duplicate renders"


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
