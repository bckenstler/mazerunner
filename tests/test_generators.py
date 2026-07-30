import numpy as np
import pytest

from mazerunner.evaluator import evaluate_task
from mazerunner.generators import FAMILIES
from mazerunner.io import load_task, mask_sha256, save_task
from mazerunner.render import render_world
from mazerunner.world import open_mask, validate_world


@pytest.fixture(scope="module")
def built():
    out = {}
    for name, module in FAMILIES.items():
        world = module.build()
        mask = open_mask(world)
        validation = validate_world(world, mask)
        out[name] = (world, mask, validation)
    return out


def test_all_families_present(built):
    assert set(built) == {
        "rectilinear", "braided", "rooms", "organic",
        "cave", "radial", "island", "pipes",
    }


def test_fail_closed_checks_pass(built):
    for name, (world, mask, validation) in built.items():
        assert validation.reference_steps >= 5, name
        assert validation.min_clearance_px > world.pointer_radius_px, name
        assert validation.start_radius_px >= 12, name
        assert validation.goal_radius_px >= 12, name


def test_masks_are_deterministic(built):
    for name, (world, mask, _v) in built.items():
        again = open_mask(FAMILIES[name].build())
        assert mask_sha256(again) == mask_sha256(mask), name


def test_saved_reference_passes_scorer(built, tmp_path):
    for name, (world, mask, validation) in built.items():
        image, _record = render_world(world, mask)
        save_task(tmp_path / name, world, mask, image, validation)
        task, loaded_mask = load_task(tmp_path / name)
        result = evaluate_task(task, loaded_mask, {"points": task["reference"]["optimal_path"]})
        assert result.success, (name, result.to_dict())
        assert not result.efficiency_canary, (name, result.efficiency_raw)


def test_direct_wall_shortcut_rejected(built, tmp_path):
    # A straight drag from start to goal must collide in every family: the
    # double-sweep endpoint heuristic always puts multiple bends between them.
    for name, (world, mask, validation) in built.items():
        image, _record = render_world(world, mask)
        save_task(tmp_path / name, world, mask, image, validation)
        task, loaded_mask = load_task(tmp_path / name)
        shortcut = {"points": [task["start"], task["goal"]]}
        result = evaluate_task(task, loaded_mask, shortcut)
        assert not result.collision_free, name


def test_render_is_deterministic_from_style_record(built):
    # Reproducibility: same world + mask + archetype + style seed must give a
    # byte-identical render (and the recorded params must match).
    world, mask, _v = built["rectilinear"]
    img_a, rec_a = render_world(world, mask)
    img_b, rec_b = render_world(world, open_mask(world), rec_a["archetype"], rec_a["style_seed"])
    assert rec_a == rec_b
    assert np.array_equal(np.asarray(img_a), np.asarray(img_b))


def test_any_archetype_renders_any_family(built):
    # Style is decoupled from topology: every archetype must render every
    # family it supports without error.
    from mazerunner.styles import ARCHETYPES

    for name, (world, mask, _v) in built.items():
        for archetype in ARCHETYPES.values():
            if world.state_representation not in archetype.supports:
                continue
            img, record = render_world(world, mask, archetype, style_seed=99)
            assert record["params"]["corridor_fill"] is not None
            assert img.size == (world.width, world.height)
