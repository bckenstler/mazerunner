"""Augmentation preserves ground truth: a transformed world still validates."""

import numpy as np

from mazerunner.augment import apply_augmentation, sample_augmentation
from mazerunner.generators import FAMILIES
from mazerunner.world import open_mask, validate_world

import json


def test_augmented_worlds_revalidate():
    """Flips, quarter turns, canvas changes, and width jitter must all pass
    the full fail-closed validation (mask, separation, clearance, geodesic)."""
    rng = np.random.default_rng(3)
    for family in ("rectilinear", "rooms", "cave", "organic"):
        world = FAMILIES[family].build()
        params = sample_augmentation(rng, family)
        augmented = apply_augmentation(world, params)
        mask = open_mask(augmented)
        validation = validate_world(augmented, mask)
        assert validation.geodesic_length_px > 0, (family, params)
        assert (augmented.width, augmented.height) == tuple(params["canvas"])


def test_augmentation_params_are_json_and_deterministic():
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    pa = sample_augmentation(rng_a, "organic")
    pb = sample_augmentation(rng_b, "organic")
    assert pa == pb
    json.dumps(pa)  # must be JSON-serializable

    world = FAMILIES["organic"].build()
    a = apply_augmentation(world, pa)
    b = apply_augmentation(world, pb)
    assert a.nodes == b.nodes


def test_grid_families_stay_axis_aligned():
    rng = np.random.default_rng(5)
    params = sample_augmentation(rng, "rectilinear")
    assert params["rotation_deg"] == 0.0
    world = FAMILIES["rectilinear"].build()
    augmented = apply_augmentation(world, params)
    # Every edge is still horizontal or vertical after the transform.
    for e in augmented.edges:
        (x0, y0), (x1, y1) = e.geometry[0], e.geometry[-1]
        assert abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6
