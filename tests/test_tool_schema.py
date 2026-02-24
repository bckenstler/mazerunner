"""Tests for agentic tool schema definitions."""

from mazerunner.evaluator.session import (
    SegmentResult,
    SegmentStatus,
    SessionResult,
    SessionStats,
)
from mazerunner.evaluator.tool_schema import (
    AGENTIC_TOOLS,
    FINISH_TOOL,
    SUBMIT_SEGMENT_TOOL,
    format_finish_result,
    format_tool_result,
)


class TestToolSchemaStructure:
    def test_submit_segment_schema_has_required_fields(self):
        func = SUBMIT_SEGMENT_TOOL["function"]
        assert func["name"] == "submit_segment"
        assert "description" in func
        params = func["parameters"]
        assert "points" in params["properties"]
        assert params["required"] == ["points"]

    def test_finish_schema_has_required_fields(self):
        func = FINISH_TOOL["function"]
        assert func["name"] == "finish"
        assert "description" in func

    def test_agentic_tools_list(self):
        assert len(AGENTIC_TOOLS) == 2
        names = {t["function"]["name"] for t in AGENTIC_TOOLS}
        assert names == {"submit_segment", "finish"}

    def test_tool_type_is_function(self):
        for tool in AGENTIC_TOOLS:
            assert tool["type"] == "function"


class TestFormatToolResult:
    def test_accepted_format(self):
        result = SegmentResult(
            status=SegmentStatus.ACCEPTED,
            segment_index=0,
            reason="Segment accepted (5 points).",
            path_length_so_far=42.0,
            num_points_so_far=5,
        )
        text = format_tool_result(result)
        assert "ACCEPTED" in text
        assert "5 points" in text

    def test_rejected_wall_format(self):
        result = SegmentResult(
            status=SegmentStatus.REJECTED_WALL,
            segment_index=1,
            reason="Path hits a wall at (100.0, 200.0).",
            violation_point=(100.0, 200.0),
            path_length_so_far=0.0,
            num_points_so_far=0,
        )
        text = format_tool_result(result)
        assert "REJECTED_WALL" in text
        assert "100.0" in text
        assert "200.0" in text

    def test_format_has_status_and_reason(self):
        for status in SegmentStatus:
            result = SegmentResult(
                status=status,
                segment_index=0,
                reason=f"Test {status.value}",
            )
            text = format_tool_result(result)
            assert status.value in text
            assert f"Test {status.value}" in text


class TestFormatFinishResult:
    def test_completed_format(self):
        stats = SessionStats(
            total_segments_submitted=3,
            segments_accepted=2,
            segments_rejected=1,
            wall_rejections=1,
            contiguity_rejections=0,
            total_tool_calls=4,
        )
        result = SessionResult(
            eval_result=None,
            stats=stats,
            accepted_path=[(0, 0)],
            finish_reason="completed",
        )
        text = format_finish_result(result)
        assert "completed" in text
        assert "2 accepted" in text
        assert "1 rejected" in text

    def test_empty_path_format(self):
        stats = SessionStats()
        result = SessionResult(
            eval_result=None,
            stats=stats,
            accepted_path=[],
            finish_reason="empty_path",
        )
        text = format_finish_result(result)
        assert "empty_path" in text
        assert "No evaluation" in text
