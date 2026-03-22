"""Tests for agent tool transform, context manager, and tool defs."""

import pytest

from mazerunner.agent.context_manager import SlidingWindowContext
from mazerunner.agent.tool_defs import get_tool_schemas
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
