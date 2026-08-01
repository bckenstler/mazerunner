"""Client construction: timeouts set, SDK-level retries disabled.

A shard of the main run hung for 106 minutes on a wedged connection because no
adapter set a timeout. Separately, the OpenAI and Anthropic SDKs retry twice by
default underneath the runner's own classified ladder, so recorded transport
history undercounted real API calls. Both are pinned here.
"""

from __future__ import annotations

import pytest

from mazerunner.providers.base import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_S
from mazerunner.providers.anthropic_provider import AnthropicProvider
from mazerunner.providers.gemini_provider import GeminiProvider
from mazerunner.providers.openai_compat import OpenAICompatProvider
from mazerunner.providers.openai_provider import OpenAIProvider


def _adapters():
    return [
        OpenAIProvider(model="m", base_url="https://x/v1", env_key="K"),
        AnthropicProvider(model="m"),
        GeminiProvider(model="m"),
        OpenAICompatProvider(model="m", base_url="https://x/v1", env_key="K"),
    ]


def test_defaults_are_sane():
    assert DEFAULT_MAX_RETRIES == 0, "the runner owns retries; SDKs must not add their own"
    assert 0 < DEFAULT_TIMEOUT_S <= 1800


@pytest.mark.parametrize("adapter", _adapters(), ids=lambda a: type(a).__name__)
def test_every_adapter_carries_timeout_and_retry_settings(adapter):
    assert adapter.timeout == DEFAULT_TIMEOUT_S
    assert adapter.max_retries == DEFAULT_MAX_RETRIES


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: OpenAIProvider(model="m", base_url="https://x/v1", env_key="K", **kw),
        lambda **kw: AnthropicProvider(model="m", **kw),
        lambda **kw: GeminiProvider(model="m", **kw),
        lambda **kw: OpenAICompatProvider(model="m", base_url="https://x/v1", env_key="K", **kw),
    ],
    ids=["openai", "anthropic", "gemini", "compat"],
)
def test_config_can_override_timeout(factory):
    """The xhigh/max legs legitimately think for many minutes."""
    adapter = factory(timeout=900.0)
    assert adapter.timeout == 900.0


def test_openai_client_receives_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("K", "sk-test")
    OpenAIProvider(model="m", base_url="https://x/v1", env_key="K", timeout=42.0)._get_client()

    assert captured["timeout"] == 42.0
    assert captured["max_retries"] == 0


def test_compat_client_receives_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("K", "sk-test")
    OpenAICompatProvider(
        model="m", base_url="https://x/v1", env_key="K", timeout=7.5
    )._get_client()

    assert captured["timeout"] == 7.5
    assert captured["max_retries"] == 0


def test_anthropic_client_receives_timeout(monkeypatch):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    AnthropicProvider(model="m", timeout=61.0)._get_client()

    assert captured["timeout"] == 61.0
    assert captured["max_retries"] == 0


def test_gemini_client_receives_timeout_in_milliseconds(monkeypatch):
    """google-genai takes milliseconds; passing seconds would be a 1000x error."""
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    GeminiProvider(model="m", timeout=30.0)._get_client()

    assert captured["http_options"].timeout == 30_000
