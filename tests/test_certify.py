"""Fairness certification: current archetypes pass; adversarial renders fail."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mazerunner.certify import Thresholds, certified_render, certify_render
from mazerunner.generators import FAMILIES
from mazerunner.render import render_world
from mazerunner.styles import ARCHETYPES
from mazerunner.world import open_mask


@pytest.fixture(scope="module")
def worlds():
    out = {}
    for name in ("rectilinear", "cave", "island"):
        world = FAMILIES[name].build()
        out[name] = (world, open_mask(world))
    return out


def test_all_archetypes_certify_on_sample_families(worlds):
    """Every archetype × a graph, a raster, and a sparse family certifies
    within the resample budget."""
    for name, (world, mask) in worlds.items():
        for archetype in ARCHETYPES.values():
            if world.state_representation not in archetype.supports:
                continue
            _img, record, cert, rejections = certified_render(world, mask, archetype, style_seed=7)
            assert cert.ok, (name, archetype.name, cert.failures)
            # Occasional resamples are fine; report chronic ones.
            assert len(rejections) < 4, (name, archetype.name, rejections)


def test_fake_corridor_extension_rejected(worlds):
    """A corridor-colored blob welded onto the corridor boundary must fail."""
    world, mask = worlds["rectilinear"]
    image, record = render_world(world, mask)
    draw = ImageDraw.Draw(image)
    # Weld a corridor-colored lobe onto the outside of the corridor at a wall
    # point: find a boundary pixel and paint outward from it.
    from scipy import ndimage

    ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
    ys, xs = np.nonzero(ring)
    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    fill = tuple(record["params"]["corridor_fill"])
    draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=fill)
    cert = certify_render(image, mask, world, record)
    assert not cert.ok
    assert any("extension" in f or "boundary" in f for f in cert.failures)


def test_invisible_wall_rejected(worlds):
    """Repainting the outline in the corridor color must fail the boundary check."""
    world, mask = worlds["rectilinear"]
    image, record = render_world(world, mask)
    from scipy import ndimage

    arr = np.asarray(image).copy()
    ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    arr[ring] = record["params"]["corridor_fill"]
    cert = certify_render(Image.fromarray(arr), mask, world, record)
    assert not cert.ok
    assert any("boundary" in f for f in cert.failures)


def test_wall_like_interior_decor_rejected(worlds):
    """Painting dark bars across corridors must fail the interior check."""
    world, mask = worlds["rectilinear"]
    image, record = render_world(world, mask)
    arr = np.asarray(image).copy()
    dark = np.zeros(3, dtype=np.uint8)
    stripe = np.zeros_like(mask)
    stripe[::4, :] = True  # every 4th row → ~25% of corridor pixels
    arr[mask & stripe] = dark
    cert = certify_render(Image.fromarray(arr), mask, world, record)
    assert not cert.ok
    assert any("interior" in f for f in cert.failures)


def test_hidden_marker_rejected(worlds):
    """Painting over a badge with the local background must fail marker check."""
    world, mask = worlds["rectilinear"]
    image, record = render_world(world, mask)
    draw = ImageDraw.Draw(image)
    x, y = world.start_px
    fill = tuple(record["params"]["corridor_fill"])
    draw.ellipse([x - 20, y - 20, x + 20, y + 20], fill=fill)
    cert = certify_render(image, mask, world, record)
    assert not cert.ok
    assert any("marker" in f for f in cert.failures)


def test_certification_records_thresholds_and_metrics(worlds):
    world, mask = worlds["rectilinear"]
    image, record = render_world(world, mask)
    cert = certify_render(image, mask, world, record, Thresholds())
    assert cert.ok, cert.failures
    assert "boundary_similar_fraction" in cert.metrics
    assert cert.thresholds["boundary_similar_tau"] == 32.0
