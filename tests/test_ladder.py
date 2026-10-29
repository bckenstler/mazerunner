"""v2 ladder: warped twins, pinhole fill, and the size overrides it relies on."""

import json
from pathlib import Path

import numpy as np
import pytest

from mazerunner.contract import MAX_POINTS
from mazerunner.dataset import verify_split
from mazerunner.generators import FAMILIES, organic
from mazerunner.ladder import build_ladder, ladder_plan
from mazerunner.world import fill_pinholes, open_mask

V2_CONFIG = Path(__file__).parent.parent / "dataset.v2.config.json"


@pytest.mark.parametrize("family", ["rectilinear", "braided", "pipes", "island"])
def test_warp_keeps_topology_and_endpoints(family):
    straight = FAMILIES[family].build(101, {"endpoint_quantile": 0.95})
    warped = FAMILIES[family].build(101, {"endpoint_quantile": 0.95, "warp": [12, 200]})
    assert {(e.a, e.b) for e in straight.edges} == {(e.a, e.b) for e in warped.edges}
    assert (straight.start_node, straight.goal_node) == (warped.start_node, warped.goal_node)
    assert straight.nodes != warped.nodes


def test_fill_pinholes_opens_specks_only():
    mask = np.ones((40, 40), dtype=bool)
    mask[0:40, 0:3] = False      # wall touching the border: kept
    mask[10:20, 10:20] = False   # a real enclosed wall block: kept
    mask[30, 30] = False         # a one-pixel speck: opened
    filled = fill_pinholes(mask)
    assert filled[30, 30]
    assert not filled[15, 15]
    assert not filled[5, 1]
    assert (filled == mask).sum() == mask.size - 1


def test_fill_pinholes_is_off_by_default():
    world = FAMILIES["rectilinear"].build()
    assert world.fill_pinholes is False
    world.fill_pinholes = True
    assert (open_mask(world) >= open_mask(FAMILIES["rectilinear"].build())).all()


def test_organic_max_edge_caps_corridor_length():
    world = organic.build(1000, {"nodes": 100, "corridor": 11, "min_dist": 42, "loops": 4, "max_edge": 100})
    for e in world.edges:
        (x0, y0), (x1, y1) = world.nodes[e.a], world.nodes[e.b]
        assert np.hypot(x1 - x0, y1 - y0) <= 100


def test_v2_config_plan_covers_every_cell():
    config = json.loads(V2_CONFIG.read_text())
    plan = ladder_plan(config)
    assert len(plan) == 8 * 4 * config["per_cell"]
    assert len({slot for slot, _f, _l in plan}) == len(plan)
    assert set(config["warp"]) == {"rectilinear", "braided", "pipes", "island"}


@pytest.fixture(scope="module")
def tiny_ladder(tmp_path_factory):
    out = tmp_path_factory.mktemp("ladder")
    config = json.loads(V2_CONFIG.read_text())
    config.update(out_dir=str(out), per_cell=1, dedup_against=[])
    # Two L1 cells: one warped family, one not.
    config["levels"] = {f: ({"L1": config["levels"][f]["L1"]} if f in ("pipes", "rooms") else {})
                        for f in FAMILIES}
    path = out / "config.json"
    path.write_text(json.dumps(config))
    build_ladder(path, workers=2)
    rows = [json.loads(line) for line in (out / "test" / "index.jsonl").read_text().splitlines()]
    return out, rows


def test_ladder_builds_twins_from_one_attempt(tiny_ladder):
    _out, rows = tiny_ladder
    by_id = {r["task_id"]: r for r in rows}
    assert set(by_id) == {"rooms-L1-s0000", "pipes-L1-s0001", "pipes-L1w-s0001"}
    straight, warped = by_id["pipes-L1-s0001"], by_id["pipes-L1w-s0001"]
    assert straight["twin"] == "pipes-L1w-s0001" and warped["twin"] == "pipes-L1-s0001"
    assert straight["archetype"] == warped["archetype"]
    assert straight["graph_hash"] == warped["graph_hash"]
    truths = [json.loads((Path(r["dir"]) / "ground-truth.json").read_text()) for r in (straight, warped)]
    assert (truths[0]["start_node"], truths[0]["goal_node"]) == (truths[1]["start_node"], truths[1]["goal_node"])
    tasks = [json.loads((Path(r["dir"]) / "task.json").read_text()) for r in (straight, warped)]
    p0, p1 = (t["provenance"] for t in tasks)
    assert (p0["attempt"], p0["topo_seed"], p0["style_seed"]) == (p1["attempt"], p1["topo_seed"], p1["style_seed"])
    assert p1["difficulty_overrides"]["warp"] and "warp" not in p0["difficulty_overrides"]
    assert p0["measures"]["tier"] == "L1" and "measured_tier" in p0["measures"]


def test_ladder_reference_is_drawable_within_cap(tiny_ladder):
    _out, rows = tiny_ladder
    for row in rows:
        reference = json.loads((Path(row["dir"]) / "task.json").read_text())["reference"]
        assert 2 <= len(reference["optimal_path"]) <= MAX_POINTS
        assert reference["optimal_path_epsilon_px"] > 0


def test_ladder_split_verifies(tiny_ladder):
    out, _rows = tiny_ladder
    assert verify_split(out, "test", sample=None) == []
