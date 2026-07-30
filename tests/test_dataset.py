"""Dataset pipeline: build, dedup, holdouts, leakage, reproducibility."""

import json
from pathlib import Path

import pytest

from mazerunner.dataset import build_all, split_stats, verify_split

PUBLIC_ARCHETYPES = ["notebook", "dungeon-pebble", "forest-path", "glow-cavern"]
HIDDEN_ONLY = ["metro-map", "volcanic"]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    config = {
        "version": "test",
        "master_seed": 424242,
        "out_dir": str(out),
        "tier_mix": {"easy": 0.25, "medium": 0.5, "hard": 0.25},
        "public_forbidden_cells": [["rectilinear", "dungeon-pebble"]],
        "splits": {
            "dev": {"size": 8, "seed_base": 0, "archetypes": PUBLIC_ARCHETYPES},
            "test-hidden": {
                "size": 8,
                "seed_base": 1000000,
                "archetypes": PUBLIC_ARCHETYPES + HIDDEN_ONLY,
                "holdout_archetypes": HIDDEN_ONLY,
            },
        },
    }
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config))
    manifest = build_all(config_path, workers=4)
    return out, config, manifest


def _rows(out: Path, split: str):
    return [json.loads(line) for line in (out / split / "index.jsonl").read_text().splitlines()]


def test_build_completes_with_manifest(tiny_dataset):
    out, _config, manifest = tiny_dataset
    assert manifest["summaries"]["dev"]["size"] == 8
    assert manifest["summaries"]["test-hidden"]["size"] == 8
    assert (out / "build-manifest.json").exists()


def test_no_semantic_leakage_across_splits(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    dev_hashes = {r["graph_hash"] for r in _rows(out, "dev")}
    hidden_hashes = {r["graph_hash"] for r in _rows(out, "test-hidden")}
    assert not dev_hashes & hidden_hashes
    assert len(dev_hashes) == 8  # unique within split too


def test_holdout_archetypes_absent_from_public(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    dev_archetypes = {r["archetype"] for r in _rows(out, "dev")}
    assert not dev_archetypes & set(HIDDEN_ONLY)


def test_forbidden_cell_absent_from_public(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    for row in _rows(out, "dev"):
        assert not (row["family"] == "rectilinear" and row["archetype"] == "dungeon-pebble")


def test_provenance_records_full_reproducibility_chain(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    row = _rows(out, "dev")[0]
    task = json.loads((Path(row["dir"]) / "task.json").read_text())
    prov = task["provenance"]
    for key in (
        "master_seed", "seed_base", "slot", "attempt", "seed_derivation",
        "topo_seed", "style_seed", "augmentation_seed",
        "difficulty_overrides", "augmentation", "measures", "graph_hash",
        "mask_sha256", "certification", "rejections", "environment",
    ):
        assert key in prov, key
    assert task["style_record"]["params"]["corridor_fill"] is not None


def test_verify_split_green(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    assert verify_split(out, "dev", sample=None) == []
    assert verify_split(out, "test-hidden", sample=4) == []


def test_stats_report(tiny_dataset):
    out, _config, _manifest = tiny_dataset
    stats = split_stats(out, "dev")
    assert stats["size"] == 8
    assert set(stats["families"]) <= set(
        ["braided", "cave", "island", "organic", "pipes", "radial", "rectilinear", "rooms"]
    )
