"""Provider response normalization against mocked SDK payloads."""

import json
from types import SimpleNamespace

from mazerunner.providers import anthropic_provider, gemini_provider, openai_provider

GOOD_ARGS = {"points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]}


def test_openai_parses_function_call():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(
                type="function_call",
                name="submit_drag_path",
                arguments=json.dumps(GOOD_ARGS),
            ),
        ]
    )
    args, error = openai_provider.parse_response(response)
    assert error is None
    assert args == GOOD_ARGS


def test_openai_reports_missing_call():
    response = SimpleNamespace(output=[SimpleNamespace(type="message")])
    args, error = openai_provider.parse_response(response)
    assert args is None
    assert "no submit_drag_path" in error


def test_openai_reports_malformed_arguments():
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="submit_drag_path", arguments="{oops")]
    )
    args, error = openai_provider.parse_response(response)
    assert args is None
    assert "malformed" in error


def test_anthropic_parses_tool_use():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="tracing now"),
            SimpleNamespace(type="tool_use", name="submit_drag_path", input=GOOD_ARGS),
        ]
    )
    args, error = anthropic_provider.parse_response(response)
    assert error is None
    assert args == GOOD_ARGS


def test_anthropic_reports_missing_call():
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="prose only")])
    args, error = anthropic_provider.parse_response(response)
    assert args is None
    assert "no submit_drag_path" in error


def test_gemini_parses_function_call():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(function_call=None),
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="submit_drag_path", args=GOOD_ARGS)
                        ),
                    ]
                )
            )
        ]
    )
    args, error = gemini_provider.parse_response(response)
    assert error is None
    assert args == GOOD_ARGS


def test_openai_compat_parses_tool_call_and_reasoning():
    from mazerunner.providers import openai_compat

    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="submit_drag_path", arguments=json.dumps(GOOD_ARGS)
                            )
                        )
                    ],
                    reasoning_content="traced the corridor north then east",
                    model_extra={},
                )
            )
        ]
    )
    args, error = openai_compat.parse_response(completion)
    assert error is None
    assert args == GOOD_ARGS
    assert "corridor" in openai_compat.extract_reasoning(completion)


def test_openai_compat_parses_inline_xml_call():
    from mazerunner.providers import openai_compat

    content = (
        '<atem:function_calls>\n<atem:invoke name="default.submit_drag_path">\n'
        '<atem:parameter name="points">[{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]'
        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=content, model_extra={}))]
    )
    args, error = openai_compat.parse_response(completion)
    assert error is None
    assert args == GOOD_ARGS


def test_openai_compat_reports_missing_call():
    from mazerunner.providers import openai_compat

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, model_extra={}))]
    )
    args, error = openai_compat.parse_response(completion)
    assert args is None
    assert "no submit_drag_path" in error


def test_gemini_reports_missing_call():
    response = SimpleNamespace(candidates=[])
    args, error = gemini_provider.parse_response(response)
    assert args is None
    assert "no submit_drag_path" in error
