"""Tests for Anthropic agent tool defs, config, and helpers."""

import pytest

from mazerunner.agent.types import AgentConfig


# ─── AgentConfig Anthropic Fields ────────────────────────────────


class TestAgentConfigAnthropic:
    def test_anthropic_provider(self):
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        assert config.provider == "anthropic"

    def test_thinking_type_default(self):
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        assert config.thinking_type == "adaptive"

    def test_thinking_type_enabled(self):
        config = AgentConfig(
            model="claude-sonnet-4-6", mode="text_grid",
            provider="anthropic", thinking_type="enabled", thinking_budget_tokens=8000,
        )
        assert config.thinking_type == "enabled"
        assert config.thinking_budget_tokens == 8000

    def test_max_tokens_default(self):
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        assert config.max_tokens == 16000

    def test_effort_default(self):
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        assert config.effort is None

    def test_thinking_display_default(self):
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        assert config.thinking_display is None

    def test_thinking_display_omitted(self):
        config = AgentConfig(
            model="claude-sonnet-4-6", mode="text_grid",
            provider="anthropic", thinking_display="omitted",
        )
        assert config.thinking_display == "omitted"

    def test_effort_custom(self):
        config = AgentConfig(
            model="claude-sonnet-4-6", mode="text_grid",
            provider="anthropic", effort="low",
        )
        assert config.effort == "low"


# ─── Anthropic Tool Schemas ──────────────────────────────────────


class TestAnthropicToolSchemas:
    def test_text_grid_schemas(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("text_grid")
        assert len(schemas) == 1
        assert schemas[0]["name"] == "navigate"
        assert "input_schema" in schemas[0]

    def test_vision_grid_schemas(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("vision_grid")
        assert schemas[0]["name"] == "navigate"

    def test_vision_drag_schemas(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("vision_drag")
        assert schemas[0]["name"] == "drag"

    def test_unknown_mode_raises(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        with pytest.raises(ValueError, match="Unknown mode"):
            get_anthropic_tool_schemas("unknown")

    def test_single_step_has_enum(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("text_grid", single_step=True)
        props = schemas[0]["input_schema"]["properties"]["directions"]
        assert "enum" in props
        assert set(props["enum"]) == {"U", "D", "L", "R"}

    def test_single_step_no_effect_on_drag(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("vision_drag", single_step=True)
        assert schemas[0]["name"] == "drag"

    def test_schema_has_description(self):
        from mazerunner.agent.tool_defs import get_anthropic_tool_schemas
        schemas = get_anthropic_tool_schemas("text_grid")
        assert schemas[0]["description"] is not None
        assert len(schemas[0]["description"]) > 0


# ─── Anthropic Loop Helpers ──────────────────────────────────────


class TestAnthropicHelpers:
    def test_extract_thinking_empty(self):
        from mazerunner.agent.anthropic_loop import _extract_thinking
        assert _extract_thinking([]) == ""

    def test_transform_to_anthropic_content_string(self):
        from mazerunner.agent.anthropic_loop import _transform_to_anthropic_content
        result = _transform_to_anthropic_content("Valid action.\nASCII maze")
        assert result == "Valid action.\nASCII maze"

    def test_transform_to_anthropic_content_vision(self):
        from mazerunner.agent.anthropic_loop import _transform_to_anthropic_content
        openai_blocks = [
            {"type": "input_text", "text": "Valid action."},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]
        result = _transform_to_anthropic_content(openai_blocks)
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Valid action."
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["data"] == "AAAA"


# ─── Runner Factory ──────────────────────────────────────────────


class TestAnthropicRunner:
    def test_get_runner_anthropic(self):
        from mazerunner.agent.runner import AnthropicAgentRunner, get_runner
        config = AgentConfig(model="claude-sonnet-4-6", mode="text_grid", provider="anthropic")
        runner = get_runner(config)
        assert isinstance(runner, AnthropicAgentRunner)
