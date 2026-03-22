"""Tests for Fireworks agent tool defs, config, and context manager."""

import pytest

from mazerunner.agent.types import AgentConfig


# ─── AgentConfig Fireworks Fields ────────────────────────────────


class TestAgentConfigFireworks:
    def test_fireworks_provider(self):
        config = AgentConfig(
            model="accounts/fireworks/models/deepseek-r1",
            mode="text_grid", provider="fireworks",
        )
        assert config.provider == "fireworks"

    def test_thinking_budget_fireworks(self):
        config = AgentConfig(
            model="accounts/fireworks/models/deepseek-r1",
            mode="text_grid", provider="fireworks",
            thinking_budget=4096,
        )
        assert config.thinking_budget == 4096


# ─── Chat Tool Schemas ──────────────────────────────────────────


class TestChatToolSchemas:
    def test_text_grid_wrapped(self):
        from mazerunner.agent.tool_defs import get_chat_tool_schemas
        schemas = get_chat_tool_schemas("text_grid")
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "navigate"

    def test_drag_wrapped(self):
        from mazerunner.agent.tool_defs import get_chat_tool_schemas
        schemas = get_chat_tool_schemas("vision_drag")
        assert schemas[0]["function"]["name"] == "drag"

    def test_single_step_chat(self):
        from mazerunner.agent.tool_defs import get_chat_tool_schemas
        schemas = get_chat_tool_schemas("text_grid", single_step=True)
        params = schemas[0]["function"]["parameters"]
        assert "enum" in params["properties"]["directions"]

    def test_unknown_mode_raises(self):
        from mazerunner.agent.tool_defs import get_chat_tool_schemas
        with pytest.raises(ValueError, match="Unknown mode"):
            get_chat_tool_schemas("unknown")


# ─── Chat Completions Context ────────────────────────────────────


class TestChatCompletionsContext:
    def test_system_message(self):
        from mazerunner.agent.chat_context import ChatCompletionsContext
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_system("System prompt")
        msgs = ctx.messages
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System prompt"

    def test_user_message(self):
        from mazerunner.agent.chat_context import ChatCompletionsContext
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_user_message("Hello")
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[0]["content"] == "Hello"

    def test_assistant_message(self):
        from mazerunner.agent.chat_context import ChatCompletionsContext
        ctx = ChatCompletionsContext("text_grid")
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "navigate", "arguments": '{"directions": "R"}'},
            }],
        }
        ctx.add_assistant_message(msg)
        assert ctx.messages[0]["role"] == "assistant"
        assert ctx.messages[0]["tool_calls"][0]["id"] == "call_1"

    def test_tool_result_text(self):
        from mazerunner.agent.chat_context import ChatCompletionsContext
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_tool_result("call_1", "Valid action.\nASCII maze")
        msg = ctx.messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"


# ─── Runner Factory ──────────────────────────────────────────────


class TestFireworksRunner:
    def test_get_runner_fireworks(self):
        from mazerunner.agent.runner import FireworksAgentRunner, get_runner
        config = AgentConfig(
            model="accounts/fireworks/models/deepseek-r1",
            mode="text_grid", provider="fireworks",
        )
        runner = get_runner(config)
        assert isinstance(runner, FireworksAgentRunner)
