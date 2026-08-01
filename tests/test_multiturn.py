"""Request construction, single-turn and continuation, across all four dialects.

Nothing tested request *shape* before this: the suite only covered response
parsing. These pin what each adapter actually sends, which is what a
continuation can silently get wrong.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mazerunner.contract import TOOL_NAME
from mazerunner.providers.base import ProviderError, ProviderResponse

GOOD_ARGS = json.dumps({"points": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.9}]})
PNG = b"\x89PNG\r\n\x1a\nfake"


# ---------- OpenAI Responses ----------

def _openai(monkeypatch, captured):
    from mazerunner.providers.openai_provider import OpenAIProvider

    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name=TOOL_NAME,
                               arguments=GOOD_ARGS, call_id="call_abc")],
        id="resp_1", model="m", usage=None,
    )

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    provider = OpenAIProvider(model="m", base_url="https://x/v1", env_key="K")
    provider._client = SimpleNamespace(responses=FakeResponses())
    return provider


def test_openai_first_turn_sends_image_and_tool(monkeypatch):
    captured = {}
    provider = _openai(monkeypatch, captured)
    result = provider.run(PNG, "trace it")

    content = captured["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[1]["text"] == "trace it"
    assert captured["tools"][0]["name"] == TOOL_NAME
    assert "previous_response_id" not in captured
    assert result.tool_call_id == "call_abc"
    assert result.response_id == "resp_1"


def test_openai_continuation_uses_previous_response_id(monkeypatch):
    captured = {}
    provider = _openai(monkeypatch, captured)
    first = provider.run(PNG, "trace it")
    provider.continue_run(first, "you hit a wall", PNG)

    assert captured["previous_response_id"] == "resp_1"
    kinds = [item.get("type") or item.get("role") for item in captured["input"]]
    assert "function_call_output" in kinds, "the tool call must be answered"
    assert "user" in kinds


def test_openai_continuation_without_an_id_is_an_error(monkeypatch):
    provider = _openai(monkeypatch, {})
    with pytest.raises(ProviderError):
        provider.continue_run(ProviderResponse(), "feedback", PNG)


# ---------- Anthropic Messages ----------

def _anthropic(captured):
    from mazerunner.providers.anthropic_provider import AnthropicProvider

    thinking = SimpleNamespace(type="thinking", thinking="hmm", signature="sig-xyz")
    tool_use = SimpleNamespace(type="tool_use", name=TOOL_NAME,
                               input={"points": [{"x": 0.1, "y": 0.1}]}, id="toolu_1")
    response = SimpleNamespace(content=[thinking, tool_use], id="msg_1", model="m", usage=None)

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return response

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    provider = AnthropicProvider(model="m")
    provider._client = SimpleNamespace(messages=FakeMessages())
    return provider, thinking


def test_anthropic_first_turn_shape():
    captured = {}
    provider, _ = _anthropic(captured)
    result = provider.run(PNG, "trace it")

    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["text"] == "trace it"
    assert captured["tool_choice"]["type"] == "auto", "forcing suppresses thinking"
    assert result.tool_call_id == "toolu_1"


def test_anthropic_continuation_replays_thinking_blocks_verbatim():
    """Adaptive thinking requires the signature back; flattening to text loses it."""
    captured = {}
    provider, thinking = _anthropic(captured)
    first = provider.run(PNG, "trace it")
    provider.continue_run(first, "you hit a wall", PNG)

    assistant = [m for m in captured["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1
    replayed = assistant[0]["content"]
    assert thinking in replayed, "thinking block must be the SDK object, not a string"
    assert getattr(replayed[0], "signature", None) == "sig-xyz"


def test_anthropic_continuation_answers_the_tool_call_first():
    captured = {}
    provider, _ = _anthropic(captured)
    first = provider.run(PNG, "trace it")
    provider.continue_run(first, "you hit a wall", PNG)

    final_user = [m for m in captured["messages"] if m["role"] == "user"][-1]
    assert final_user["content"][0]["type"] == "tool_result"
    assert final_user["content"][0]["tool_use_id"] == "toolu_1"


# ---------- OpenAI-compatible chat completions ----------

def _compat(captured, with_tool_call=True):
    from mazerunner.providers.openai_compat import OpenAICompatProvider

    function = SimpleNamespace(name=TOOL_NAME, arguments=GOOD_ARGS)
    call = SimpleNamespace(function=function, id="tc_1", type="function")
    message = SimpleNamespace(
        tool_calls=[call] if with_tool_call else None,
        content=None if with_tool_call else "no call here",
        model_dump=lambda **kw: {"role": "assistant", "tool_calls": ["..."]},
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], id="c1", model="m", usage=None
    )

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return completion

    provider = OpenAICompatProvider(model="m", base_url="https://x/v1", env_key="K")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    return provider


def test_compat_first_turn_shape():
    captured = {}
    provider = _compat(captured)
    result = provider.run(PNG, "trace it")

    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert captured["tools"][0]["function"]["name"] == TOOL_NAME
    assert result.tool_call_id == "tc_1"


def test_compat_continuation_puts_the_image_after_the_tool_message():
    """Tool messages are text-only in this dialect; the overlay needs a user turn."""
    captured = {}
    provider = _compat(captured)
    first = provider.run(PNG, "trace it")
    provider.continue_run(first, "you hit a wall", PNG)

    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]
    tool_message = captured["messages"][2]
    assert isinstance(tool_message["content"], str)
    assert captured["messages"][3]["content"][0]["type"] == "image_url"


def test_compat_continuation_skips_tool_message_without_an_id():
    """The inline-XML fallback yields no tool-call id; a tool message would 400."""
    captured = {}
    provider = _compat(captured, with_tool_call=False)
    first = provider.run(PNG, "trace it")
    assert first.tool_call_id is None

    provider.continue_run(first, "you hit a wall", PNG)
    assert "tool" not in [m["role"] for m in captured["messages"]]


# ---------- Gemini ----------

def _gemini(captured):
    from mazerunner.providers.gemini_provider import GeminiProvider

    call = SimpleNamespace(name=TOOL_NAME, args={"points": [{"x": 0.1, "y": 0.1}]})
    part = SimpleNamespace(function_call=call, thought=None, text=None,
                           thought_signature=b"sig")
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    response = SimpleNamespace(candidates=[candidate], response_id="g1",
                               model_version="m", usage_metadata=None)

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return response

    provider = GeminiProvider(model="m")
    provider._client = SimpleNamespace(models=FakeModels())
    return provider, part


def test_gemini_first_turn_uses_role_tagged_content():
    captured = {}
    provider, _ = _gemini(captured)
    provider.run(PNG, "trace it")

    contents = captured["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_gemini_continuation_replays_the_model_turn_with_signatures():
    captured = {}
    provider, part = _gemini(captured)
    first = provider.run(PNG, "trace it")
    provider.continue_run(first, "you hit a wall", PNG)

    roles = [c.role for c in captured["contents"]]
    assert roles == ["user", "model", "user", "user"]
    model_turn = captured["contents"][1]
    # The SDK coerces parts into typed Part objects; what must survive is the
    # thought signature (Gemini 3 thinking models reject a turn without it)
    # and the function call itself.
    assert getattr(model_turn.parts[0], "thought_signature", None) == b"sig"
    assert model_turn.parts[0].function_call.name == TOOL_NAME


def test_gemini_continuation_without_state_is_an_error():
    provider, _ = _gemini({})
    with pytest.raises(ProviderError):
        provider.continue_run(ProviderResponse(), "feedback", PNG)
