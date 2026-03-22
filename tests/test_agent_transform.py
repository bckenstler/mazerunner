"""Tests for agent tool transform, context manager, and tool defs."""

import json

import pytest

from mazerunner.agent.chat_context import ChatCompletionsContext
from mazerunner.agent.context_manager import SlidingWindowContext
from mazerunner.agent.tool_defs import get_chat_tool_schemas, get_tool_schemas
from mazerunner.agent.tool_transform import transform_tool_output


# ─── Tool Transform ─────────────────────────────────────────────


class TestTransformToolOutput:
    def test_navigate_valid_text_grid(self):
        raw = {
            "valid": True,
            "position": [0, 1],
            "finished": False,
            "steps_applied": 1,
            "rendered": "+--+--+\n|X    |\n+--+--+",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "text_grid")
        assert isinstance(result, str)
        assert "Valid action" in result
        assert "+--+--+" in result
        # Internal fields should not appear
        assert "reward" not in result
        assert "step_count" not in result
        assert "position" not in result

    def test_navigate_valid_vision_grid(self):
        raw = {
            "valid": True,
            "position": [0, 1],
            "finished": False,
            "steps_applied": 1,
            "rendered": "iVBORw0KGgo=",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "vision_grid")
        assert isinstance(result, list)
        assert result[0]["type"] == "input_text"
        assert "Valid action" in result[0]["text"]
        assert result[1]["type"] == "input_image"
        assert "data:image/png;base64," in result[1]["image_url"]
        assert result[1]["detail"] == "auto"

    def test_navigate_invalid(self):
        raw = {
            "valid": False,
            "position": [0, 0],
            "finished": False,
            "steps_applied": 0,
            "rendered": "+--+\n|X |\n+--+",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output("navigate", {"directions": "U"}, raw, "text_grid")
        assert "Invalid action" in result
        assert "navigate('U')" in result

    def test_navigate_finished(self):
        raw = {
            "valid": True,
            "position": [2, 2],
            "finished": True,
            "steps_applied": 1,
            "rendered": "...",
            "reward": 1.0,
            "step_count": 5,
            "done": True,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "text_grid")
        assert result == "Maze complete! You reached the goal."

    def test_drag_valid_vision(self):
        raw = {
            "valid": True,
            "position": [50.0, 50.0],
            "finished": False,
            "steps_applied": 3,
            "rendered": "AAABBB==",
            "reward": 0.1,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output(
            "drag", {"points": [[10, 20], [30, 40]]}, raw, "vision_drag"
        )
        assert isinstance(result, list)
        assert "Valid action" in result[0]["text"]

    def test_drag_invalid_vision(self):
        raw = {
            "valid": False,
            "position": [10.0, 20.0],
            "finished": False,
            "steps_applied": 0,
            "rendered": "AAABBB==",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output(
            "drag", {"points": [[10, 20], [30, 40]]}, raw, "vision_drag"
        )
        assert isinstance(result, list)
        assert "Invalid action" in result[0]["text"]
        assert "drag(" in result[0]["text"]

    def test_get_maze_info(self):
        raw = {
            "grid_rows": 5,
            "grid_cols": 7,
            "start": "0,0",
            "goal": "4,6",
            "mode": "text_grid",
            "maze_id": "maze_001",
        }
        result = transform_tool_output("get_maze_info", {}, raw, "text_grid")
        assert "5x7" in result
        assert "0,0" in result
        assert "4,6" in result

    def test_get_maze_info_error(self):
        raw = {"error": "No maze loaded. Call reset() first."}
        result = transform_tool_output("get_maze_info", {}, raw, "text_grid")
        assert "No maze loaded" in result

    def test_finished_drag_vision(self):
        raw = {
            "valid": True,
            "position": [100.0, 100.0],
            "finished": True,
            "steps_applied": 2,
            "rendered": "...",
            "reward": 1.0,
            "step_count": 3,
            "done": True,
        }
        result = transform_tool_output(
            "drag", {"points": [[50, 50], [100, 100]]}, raw, "vision_drag"
        )
        assert result == "Maze complete! You reached the goal."


# ─── Sliding Window Context ─────────────────────────────────────


class TestSlidingWindowContext:
    def test_text_mode_no_windowing(self):
        ctx = SlidingWindowContext("text_grid")
        ctx.add_system("You are a maze solver.")
        ctx.add_tool_output("call_1", "Valid action.\nASCII maze here")
        ctx.add_tool_output("call_2", "Valid action.\nASCII maze 2")
        # Both outputs should be preserved as-is
        outputs = [i for i in ctx.items if i.get("type") == "function_call_output"]
        assert len(outputs) == 2
        assert "ASCII maze here" in outputs[0]["output"]
        assert "ASCII maze 2" in outputs[1]["output"]

    def test_vision_mode_removes_old_images(self):
        ctx = SlidingWindowContext("vision_grid")
        ctx.add_tool_output("call_1", [
            {"type": "input_text", "text": "Valid action."},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "auto"},
        ])
        ctx.add_tool_output("call_2", [
            {"type": "input_text", "text": "Valid action."},
            {"type": "input_image", "image_url": "data:image/png;base64,BBB", "detail": "auto"},
        ])

        outputs = [i for i in ctx.items if i.get("type") == "function_call_output"]
        # First output should have image replaced
        first_blocks = outputs[0]["output"]
        assert any("[Previous maze image omitted]" in b.get("text", "") for b in first_blocks)
        assert not any(b.get("type") == "input_image" for b in first_blocks)
        # Second should keep image
        second_blocks = outputs[1]["output"]
        assert any(b.get("type") == "input_image" for b in second_blocks)

    def test_vision_mode_single_image_preserved(self):
        ctx = SlidingWindowContext("vision_drag")
        ctx.add_tool_output("call_1", [
            {"type": "input_text", "text": "Valid action."},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "auto"},
        ])
        outputs = [i for i in ctx.items if i.get("type") == "function_call_output"]
        assert any(b.get("type") == "input_image" for b in outputs[0]["output"])

    def test_add_system(self):
        ctx = SlidingWindowContext("text_grid")
        ctx.add_system("System prompt")
        assert ctx.items[0]["role"] == "developer"
        assert ctx.items[0]["content"] == "System prompt"

    def test_add_response_items(self):
        ctx = SlidingWindowContext("text_grid")
        items = [{"type": "message", "content": "thinking..."}]
        ctx.add_response_items(items)
        assert ctx.items == items

    def test_three_images_only_last_kept(self):
        ctx = SlidingWindowContext("vision_grid")
        for i in range(3):
            ctx.add_tool_output(f"call_{i}", [
                {"type": "input_text", "text": f"Action {i}"},
                {"type": "input_image", "image_url": f"data:image/png;base64,IMG{i}", "detail": "auto"},
            ])

        outputs = [item for item in ctx.items if item.get("type") == "function_call_output"]
        # Only the last should have an image
        for out in outputs[:-1]:
            assert not any(b.get("type") == "input_image" for b in out["output"])
            assert any("[Previous maze image omitted]" in b.get("text", "") for b in out["output"])
        last_blocks = outputs[-1]["output"]
        assert any(b.get("type") == "input_image" for b in last_blocks)

    def test_string_output_no_windowing_in_vision(self):
        """String outputs (like get_maze_info) shouldn't trigger windowing."""
        ctx = SlidingWindowContext("vision_grid")
        ctx.add_tool_output("call_1", [
            {"type": "input_text", "text": "Valid action."},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "auto"},
        ])
        ctx.add_tool_output("call_2", "Grid: 5x7\nStart: 0,0")
        # The image in call_1 should still be there since call_2 has no image
        outputs = [item for item in ctx.items if item.get("type") == "function_call_output"]
        first_blocks = outputs[0]["output"]
        assert any(b.get("type") == "input_image" for b in first_blocks)


# ─── Tool Defs ───────────────────────────────────────────────────


class TestToolDefs:
    def test_text_grid_schemas(self):
        schemas = get_tool_schemas("text_grid")
        names = [s["name"] for s in schemas]
        assert "navigate" in names
        assert "get_maze_info" not in names
        assert "drag" not in names

    def test_vision_grid_schemas(self):
        schemas = get_tool_schemas("vision_grid")
        names = [s["name"] for s in schemas]
        assert "navigate" in names
        assert "get_maze_info" not in names
        assert "drag" not in names

    def test_vision_drag_schemas(self):
        schemas = get_tool_schemas("vision_drag")
        names = [s["name"] for s in schemas]
        assert "drag" in names
        assert "get_maze_info" not in names
        assert "navigate" not in names

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_tool_schemas("unknown")

    def test_navigate_schema_structure(self):
        schemas = get_tool_schemas("text_grid")
        nav = next(s for s in schemas if s["name"] == "navigate")
        assert nav["type"] == "function"
        assert "directions" in nav["parameters"]["properties"]
        assert "directions" in nav["parameters"]["required"]

    def test_drag_schema_structure(self):
        schemas = get_tool_schemas("vision_drag")
        drag = next(s for s in schemas if s["name"] == "drag")
        assert drag["type"] == "function"
        assert "points" in drag["parameters"]["properties"]
        assert "points" in drag["parameters"]["required"]

    def test_only_one_tool_per_mode(self):
        assert len(get_tool_schemas("text_grid")) == 1
        assert len(get_tool_schemas("vision_grid")) == 1
        assert len(get_tool_schemas("vision_drag")) == 1


# ─── Single-Step Tool Schemas ────────────────────────────────────


class TestSingleStepSchemas:
    def test_single_step_has_enum(self):
        schemas = get_tool_schemas("text_grid", single_step=True)
        nav = schemas[0]
        assert nav["name"] == "navigate"
        assert "enum" in nav["parameters"]["properties"]["directions"]
        assert set(nav["parameters"]["properties"]["directions"]["enum"]) == {"U", "D", "L", "R"}

    def test_single_step_description_differs(self):
        multi = get_tool_schemas("text_grid", single_step=False)[0]
        single = get_tool_schemas("text_grid", single_step=True)[0]
        assert multi["description"] != single["description"]
        assert "single" in single["description"].lower() or "one step" in single["description"].lower()

    def test_single_step_no_effect_on_drag(self):
        schemas = get_tool_schemas("vision_drag", single_step=True)
        assert schemas[0]["name"] == "drag"

    def test_default_is_multi_step(self):
        schemas = get_tool_schemas("text_grid")
        nav = schemas[0]
        assert "enum" not in nav["parameters"]["properties"]["directions"]

    def test_single_step_chat_schemas(self):
        schemas = get_chat_tool_schemas("text_grid", single_step=True)
        nav = schemas[0]["function"]
        assert "enum" in nav["parameters"]["properties"]["directions"]

    def test_single_step_gemini_tools(self):
        from mazerunner.agent.tool_defs import get_gemini_tools
        tool = get_gemini_tools("text_grid", single_step=True)
        assert tool.function_declarations[0].name == "navigate"


# ─── Chat Tool Schemas ──────────────────────────────────────────


class TestChatToolSchemas:
    def test_wraps_in_function_key(self):
        schemas = get_chat_tool_schemas("text_grid")
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert "function" in s
        assert s["function"]["name"] == "navigate"
        assert "parameters" in s["function"]

    def test_drag_schema_wrapped(self):
        schemas = get_chat_tool_schemas("vision_drag")
        s = schemas[0]
        assert s["function"]["name"] == "drag"
        assert "points" in s["function"]["parameters"]["properties"]

    def test_preserves_descriptions(self):
        responses_schemas = get_tool_schemas("text_grid")
        chat_schemas = get_chat_tool_schemas("text_grid")
        assert chat_schemas[0]["function"]["description"] == responses_schemas[0]["description"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_chat_tool_schemas("unknown")


# ─── Chat Format Transform ──────────────────────────────────────


class TestTransformChatFormat:
    def test_vision_uses_chat_image_format(self):
        raw = {
            "valid": True,
            "position": [0, 1],
            "finished": False,
            "steps_applied": 1,
            "rendered": "iVBORw0KGgo=",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "vision_grid", format="chat")
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "Valid action" in result[0]["text"]
        assert result[1]["type"] == "image_url"
        assert "data:image/png;base64," in result[1]["image_url"]["url"]

    def test_text_grid_unchanged(self):
        raw = {
            "valid": True,
            "position": [0, 1],
            "finished": False,
            "steps_applied": 1,
            "rendered": "+--+\n|X |\n+--+",
            "reward": 0.0,
            "step_count": 1,
            "done": False,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "text_grid", format="chat")
        assert isinstance(result, str)
        assert "Valid action" in result

    def test_finished_unchanged(self):
        raw = {
            "valid": True,
            "position": [2, 2],
            "finished": True,
            "steps_applied": 1,
            "rendered": "...",
            "reward": 1.0,
            "step_count": 5,
            "done": True,
        }
        result = transform_tool_output("navigate", {"directions": "R"}, raw, "vision_grid", format="chat")
        assert result == "Maze complete! You reached the goal."


# ─── Chat Completions Context ───────────────────────────────────


class TestChatCompletionsContext:
    def test_system_message(self):
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_system("System prompt")
        assert ctx.messages[0]["role"] == "system"
        assert ctx.messages[0]["content"] == "System prompt"

    def test_user_message_string(self):
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_user_message("Hello")
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[0]["content"] == "Hello"

    def test_user_message_multimodal(self):
        ctx = ChatCompletionsContext("vision_grid")
        content = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        ctx.add_user_message(content)
        assert ctx.messages[0]["content"] == content

    def test_assistant_message_preserves_tool_calls(self):
        ctx = ChatCompletionsContext("text_grid")
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "navigate", "arguments": '{"directions":"R"}'}}],
        }
        ctx.add_assistant_message(msg)
        assert ctx.messages[0]["tool_calls"][0]["id"] == "tc_1"

    def test_assistant_message_preserves_reasoning_content(self):
        ctx = ChatCompletionsContext("text_grid")
        msg = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I think I should go right.",
            "tool_calls": [],
        }
        ctx.add_assistant_message(msg)
        assert ctx.messages[0]["reasoning_content"] == "I think I should go right."

    def test_tool_result_string(self):
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_tool_result("tc_1", "Valid action.\nMaze here")
        msg = ctx.messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert msg["content"] == "Valid action.\nMaze here"

    def test_tool_result_list_serialized(self):
        ctx = ChatCompletionsContext("vision_grid")
        content = [
            {"type": "text", "text": "Valid action."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        ctx.add_tool_result("tc_1", content)
        msg = ctx.messages[0]
        assert msg["role"] == "tool"
        assert isinstance(msg["content"], str)
        parsed = json.loads(msg["content"])
        assert parsed[1]["type"] == "image_url"

    def test_text_mode_no_windowing(self):
        ctx = ChatCompletionsContext("text_grid")
        ctx.add_tool_result("tc_1", "Maze 1")
        ctx.add_tool_result("tc_2", "Maze 2")
        msgs = [m for m in ctx.messages if m["role"] == "tool"]
        assert "Maze 1" in msgs[0]["content"]
        assert "Maze 2" in msgs[1]["content"]

    def test_vision_mode_removes_old_images(self):
        ctx = ChatCompletionsContext("vision_grid")
        content1 = [
            {"type": "text", "text": "Valid action."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        content2 = [
            {"type": "text", "text": "Valid action."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
        ]
        ctx.add_tool_result("tc_1", content1)
        ctx.add_tool_result("tc_2", content2)

        msgs = [m for m in ctx.messages if m["role"] == "tool"]
        first_blocks = json.loads(msgs[0]["content"])
        assert any("[Previous maze image omitted]" in b.get("text", "") for b in first_blocks)
        assert not any(b.get("type") == "image_url" for b in first_blocks)
        second_blocks = json.loads(msgs[1]["content"])
        assert any(b.get("type") == "image_url" for b in second_blocks)

    def test_three_images_only_last_kept(self):
        ctx = ChatCompletionsContext("vision_grid")
        for i in range(3):
            ctx.add_tool_result(f"tc_{i}", [
                {"type": "text", "text": f"Action {i}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i}"}},
            ])
        msgs = [m for m in ctx.messages if m["role"] == "tool"]
        for m in msgs[:-1]:
            blocks = json.loads(m["content"])
            assert not any(b.get("type") == "image_url" for b in blocks)
        last_blocks = json.loads(msgs[-1]["content"])
        assert any(b.get("type") == "image_url" for b in last_blocks)
