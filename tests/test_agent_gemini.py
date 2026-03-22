"""Tests for Gemini agent tool defs and config extensions."""

import pytest

from mazerunner.agent.types import AgentConfig


# ─── AgentConfig Provider Fields ─────────────────────────────────


class TestAgentConfigProvider:
    def test_default_provider(self):
        config = AgentConfig(model="gpt-5.4", mode="text_grid")
        assert config.provider == "openai"

    def test_gemini_provider(self):
        config = AgentConfig(model="gemini-2.5-flash", mode="text_grid", provider="gemini")
        assert config.provider == "gemini"

    def test_thinking_budget(self):
        config = AgentConfig(
            model="gemini-2.5-flash", mode="text_grid",
            provider="gemini", thinking_budget=1024,
        )
        assert config.thinking_budget == 1024
        assert config.thinking_level is None

    def test_thinking_level(self):
        config = AgentConfig(
            model="gemini-3-flash-preview", mode="text_grid",
            provider="gemini", thinking_level="HIGH",
        )
        assert config.thinking_level == "HIGH"
        assert config.thinking_budget is None

    def test_default_thinking_fields_none(self):
        config = AgentConfig(model="gpt-5.4", mode="text_grid")
        assert config.thinking_budget is None
        assert config.thinking_level is None


# ─── Gemini Tool Defs ────────────────────────────────────────────


class TestGeminiToolDefs:
    def test_text_grid_tools(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        tool = get_gemini_tools("text_grid")
        names = [d.name for d in tool.function_declarations]
        assert "navigate" in names
        assert "drag" not in names

    def test_vision_grid_tools(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        tool = get_gemini_tools("vision_grid")
        names = [d.name for d in tool.function_declarations]
        assert "navigate" in names
        assert "drag" not in names

    def test_vision_drag_tools(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        tool = get_gemini_tools("vision_drag")
        names = [d.name for d in tool.function_declarations]
        assert "drag" in names
        assert "navigate" not in names

    def test_unknown_mode_raises(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        with pytest.raises(ValueError, match="Unknown mode"):
            get_gemini_tools("unknown")

    def test_navigate_has_parameters(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        tool = get_gemini_tools("text_grid")
        nav = tool.function_declarations[0]
        assert nav.name == "navigate"
        assert nav.description is not None

    def test_only_one_tool_per_mode(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        assert len(get_gemini_tools("text_grid").function_declarations) == 1
        assert len(get_gemini_tools("vision_grid").function_declarations) == 1
        assert len(get_gemini_tools("vision_drag").function_declarations) == 1


# ─── Gemini Loop Helpers ─────────────────────────────────────────


class TestGeminiLoopHelpers:
    def test_build_tool_feedback_text_valid(self):
        from mazerunner.agent.gemini_loop import _build_tool_feedback_text
        text = _build_tool_feedback_text(
            "navigate", {"directions": "R"},
            {"valid": True, "finished": False, "position": [0, 1]},
            "text_grid",
        )
        assert "Valid action" in text

    def test_build_tool_feedback_text_finished(self):
        from mazerunner.agent.gemini_loop import _build_tool_feedback_text
        text = _build_tool_feedback_text(
            "navigate", {"directions": "R"},
            {"valid": True, "finished": True, "position": [2, 2]},
            "text_grid",
        )
        assert "Maze complete" in text

    def test_build_tool_feedback_text_invalid(self):
        from mazerunner.agent.gemini_loop import _build_tool_feedback_text
        text = _build_tool_feedback_text(
            "navigate", {"directions": "U"},
            {"valid": False, "finished": False, "position": [0, 0]},
            "text_grid",
        )
        assert "Invalid action" in text
        assert "navigate('U')" in text

    def test_build_tool_feedback_drag_position(self):
        from mazerunner.agent.gemini_loop import _build_tool_feedback_text
        text = _build_tool_feedback_text(
            "drag", {"points": [[10, 20]]},
            {"valid": True, "finished": False, "position": [50.0, 60.0]},
            "vision_drag",
        )
        assert "[50, 60]" in text

    def test_build_tool_feedback_drag_invalid(self):
        from mazerunner.agent.gemini_loop import _build_tool_feedback_text
        text = _build_tool_feedback_text(
            "drag", {"points": [[10, 20]]},
            {"valid": False, "finished": False, "position": [10.0, 20.0]},
            "vision_drag",
        )
        assert "position has not changed" in text


# ─── Runner Factory ──────────────────────────────────────────────


class TestRunnerFactory:
    def test_get_runner_openai(self):
        from mazerunner.agent.runner import OpenAIAgentRunner, get_runner
        config = AgentConfig(model="gpt-5.4", mode="text_grid", provider="openai")
        runner = get_runner(config)
        assert isinstance(runner, OpenAIAgentRunner)

    def test_get_runner_gemini(self):
        from mazerunner.agent.runner import GeminiAgentRunner, get_runner
        config = AgentConfig(model="gemini-2.5-flash", mode="text_grid", provider="gemini")
        runner = get_runner(config)
        assert isinstance(runner, GeminiAgentRunner)
