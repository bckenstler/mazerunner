"""Image substitution for the blind and resolution ablations."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from mazerunner.imagesrc import (
    ImageSource,
    ImageSpec,
    blank_png,
    rescale_png,
    spec_from_args,
    tier_matched_derangement,
)


def _png(width, height, color=(10, 200, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _task_dir(tmp_path, name, width=80, height=60) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.png").write_bytes(_png(width, height))
    return d


TASK = {"width": 80, "height": 60, "image_file": "input.png"}


# ---------- spec validation ----------

def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ImageSpec(mode="nonsense")


def test_rescale_without_a_scale_is_an_error():
    """Silently sending the original at 1.0x would fake a resolution result."""
    with pytest.raises(ValueError):
        ImageSpec(mode="rescale", scale=1.0)


def test_spec_from_args_infers_rescale():
    assert spec_from_args("real", 2.0, None).mode == "rescale"
    assert spec_from_args("real", 1.0, None).mode == "real"


def test_label_distinguishes_scales():
    assert ImageSpec(mode="rescale", scale=2.0).label == "rescale@2x"
    assert ImageSpec(mode="rescale", scale=0.5).label == "rescale@0.5x"
    assert ImageSpec(mode="blank").label == "blank"


# ---------- blank ----------

def test_blank_keeps_true_dimensions():
    """Holding aspect ratio constant isolates content, not shape cues."""
    image = Image.open(io.BytesIO(blank_png(123, 45)))
    assert image.size == (123, 45)


def test_blank_is_featureless():
    image = Image.open(io.BytesIO(blank_png(20, 20))).convert("RGB")
    assert image.getextrema() == ((128, 128), (128, 128), (128, 128))


def test_blank_via_source(tmp_path):
    d = _task_dir(tmp_path, "t1")
    data, prov = ImageSource(ImageSpec(mode="blank")).bytes_for("t1", d, TASK)
    assert Image.open(io.BytesIO(data)).size == (80, 60)
    assert prov["mode"] == "blank"


# ---------- rescale ----------

@pytest.mark.parametrize("scale,expected", [(2.0, (160, 120)), (0.5, (40, 30))])
def test_rescale_changes_size_both_directions(scale, expected):
    out = rescale_png(_png(80, 60), scale)
    assert Image.open(io.BytesIO(out)).size == expected


def test_rescale_never_produces_a_zero_dimension():
    out = rescale_png(_png(4, 4), 0.01)
    assert Image.open(io.BytesIO(out)).size == (1, 1)


def test_rescale_records_sent_and_true_dimensions(tmp_path):
    """The record must show what the model saw AND what the task really is."""
    d = _task_dir(tmp_path, "t1")
    source = ImageSource(ImageSpec(mode="rescale", scale=2.0))
    _data, prov = source.bytes_for("t1", d, TASK)
    assert (prov["sent_width"], prov["sent_height"]) == (160, 120)
    assert (prov["true_width"], prov["true_height"]) == (80, 60)


# ---------- mismatched ----------

def _tiers(ids, tier="medium"):
    return {t: tier for t in ids}


def test_derangement_has_no_fixed_points():
    ids = [f"t{i}" for i in range(12)]
    mapping = tier_matched_derangement(ids, _tiers(ids), seed=0)
    assert all(mapping[t] != t for t in ids)
    assert set(mapping) == set(ids)


def test_derangement_is_seed_reproducible():
    ids = [f"t{i}" for i in range(9)]
    a = tier_matched_derangement(ids, _tiers(ids), seed=5)
    b = tier_matched_derangement(ids, _tiers(ids), seed=5)
    assert a == b


def test_derangement_matches_tier():
    """A hard task must not be replaced by an easy image, or the floor inflates."""
    ids = [f"e{i}" for i in range(4)] + [f"h{i}" for i in range(4)]
    tiers = {**{f"e{i}": "easy" for i in range(4)}, **{f"h{i}": "hard" for i in range(4)}}
    mapping = tier_matched_derangement(ids, tiers, seed=1)
    assert all(tiers[src] == tiers[dst] for src, dst in mapping.items())


def test_singleton_tier_falls_back_without_self_assignment():
    ids = ["only-hard", "m1", "m2", "m3"]
    tiers = {"only-hard": "hard", "m1": "medium", "m2": "medium", "m3": "medium"}
    mapping = tier_matched_derangement(ids, tiers, seed=2)
    assert mapping["only-hard"] != "only-hard"


def test_mismatched_source_serves_another_tasks_image(tmp_path):
    dirs = {name: _task_dir(tmp_path, name) for name in ("a", "b", "c")}
    # give each a distinguishable image
    for i, (name, d) in enumerate(dirs.items()):
        (d / "input.png").write_bytes(_png(80, 60, (i * 40, 0, 0)))
    sources = [(name, d) for name, d in dirs.items()]

    src = ImageSource(ImageSpec(mode="mismatched", seed=3), sources=sources,
                      tiers=_tiers(list(dirs)))
    data, prov = src.bytes_for("a", dirs["a"], TASK)

    assert prov["source_task"] != "a"
    assert data == (dirs[prov["source_task"]] / "input.png").read_bytes()


def test_mismatched_requires_the_pool():
    with pytest.raises(ValueError):
        ImageSource(ImageSpec(mode="mismatched", seed=1))


# ---------- real ----------

def test_real_mode_is_byte_identical_to_the_task_image(tmp_path):
    """The main-run path must not change."""
    d = _task_dir(tmp_path, "t1")
    data, prov = ImageSource(ImageSpec()).bytes_for("t1", d, TASK)
    assert data == (d / "input.png").read_bytes()
    assert prov["mode"] == "real"
