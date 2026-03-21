"""Tests for color schema sampling and definitions."""

import re

import pytest

from mazerunner.generator.color_schemas import (
    COLOR_SCHEMAS,
    SCHEMA_NAMES,
    get_color_schema,
    sample_color_schema,
)
from mazerunner.generator.seed_utils import make_rng

HEX_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
REQUIRED_KEYS = {"name", "wall", "corridor", "start", "goal", "solution_path", "background"}


class TestColorSchemaDefinitions:
    def test_at_least_5_schemas(self):
        assert len(COLOR_SCHEMAS) >= 5

    def test_unique_names(self):
        assert len(SCHEMA_NAMES) == len(set(SCHEMA_NAMES))

    @pytest.mark.parametrize("schema", COLOR_SCHEMAS, ids=SCHEMA_NAMES)
    def test_required_keys(self, schema):
        assert set(schema.keys()) == REQUIRED_KEYS

    @pytest.mark.parametrize("schema", COLOR_SCHEMAS, ids=SCHEMA_NAMES)
    def test_valid_hex_colors(self, schema):
        for key in REQUIRED_KEYS - {"name"}:
            assert HEX_PATTERN.match(schema[key]), f"{schema['name']}.{key} = {schema[key]} is not valid hex"

    @pytest.mark.parametrize("schema", COLOR_SCHEMAS, ids=SCHEMA_NAMES)
    def test_start_goal_differ(self, schema):
        assert schema["start"] != schema["goal"]

    @pytest.mark.parametrize("schema", COLOR_SCHEMAS, ids=SCHEMA_NAMES)
    def test_wall_corridor_differ(self, schema):
        assert schema["wall"] != schema["corridor"]


class TestSampleColorSchema:
    def test_returns_valid_schema(self):
        rng = make_rng(42)
        schema = sample_color_schema(rng)
        assert set(schema.keys()) == REQUIRED_KEYS

    def test_deterministic(self):
        s1 = sample_color_schema(make_rng(42))
        s2 = sample_color_schema(make_rng(42))
        assert s1 == s2

    def test_returns_copy(self):
        rng = make_rng(42)
        s1 = sample_color_schema(rng)
        s1["wall"] = "#000000"
        s2 = get_color_schema(s1["name"])
        assert s2["wall"] != "#000000"

    def test_variety_across_seeds(self):
        names = set()
        for seed in range(50):
            schema = sample_color_schema(make_rng(seed))
            names.add(schema["name"])
        assert len(names) >= 3


class TestGetColorSchema:
    def test_get_existing(self):
        schema = get_color_schema("classic")
        assert schema["name"] == "classic"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown color schema"):
            get_color_schema("nonexistent")

    @pytest.mark.parametrize("name", SCHEMA_NAMES)
    def test_all_schemas_retrievable(self, name):
        schema = get_color_schema(name)
        assert schema["name"] == name
