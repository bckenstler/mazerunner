"""Retry classification: what gets retried, how long we wait, what gets recorded."""

from __future__ import annotations

import random

import pytest

from mazerunner.providers.base import ProviderError, RETRYABLE_STATUS, is_retryable, serving_stack
from mazerunner import runner


class FakeResponse:
    """Minimal stand-in for an SDK response carrying a status and headers."""

    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class FakeSDKError(Exception):
    """An SDK exception shaped like the ones retry classification reads."""

    def __init__(self, message, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = FakeResponse(status_code, headers)


class FlakyProvider:
    """Raises a scripted sequence of errors, then succeeds."""

    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def run(self, png_bytes, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_wrap_salvages_status_and_retry_after():
    err = ProviderError.wrap("openai", FakeSDKError("slow down", 429, {"retry-after": "17"}))
    assert err.status == 429
    assert err.retry_after == 17.0
    assert "openai" in str(err)


def test_unknown_status_is_retryable():
    # Connection resets never reach the API and carry no status.
    assert is_retryable(ProviderError("connection reset"))


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_transient_statuses_retry(status):
    assert is_retryable(ProviderError("x", status=status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_terminal(status):
    assert not is_retryable(ProviderError("x", status=status))


def test_terminal_error_is_not_retried(monkeypatch):
    """A bad parameter fails identically forever; retrying it burns money."""
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    bad = ProviderError.wrap("openai", FakeSDKError("unsupported value: 'minimal'", 400))
    provider = FlakyProvider([bad] * 5)

    response, error, history = runner._attempt_with_retries(provider, b"", "")

    assert response is None
    assert provider.calls == 1
    assert len(history) == 1
    assert history[0]["retryable"] is False
    assert history[0]["waited_s"] == 0.0
    assert "minimal" in error


def test_rate_limit_retries_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    errors = [ProviderError.wrap("p", FakeSDKError("rate", 429)) for _ in range(2)]
    provider = FlakyProvider(errors)

    response, error, history = runner._attempt_with_retries(
        provider, b"", "", rng=random.Random(0)
    )

    assert response == "ok"
    assert error is None
    assert provider.calls == 3
    assert len(history) == 2
    assert all(h["status"] == 429 for h in history)
    assert len(slept) == 2


def test_rate_limit_gives_up_after_limit(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    errors = [ProviderError.wrap("p", FakeSDKError("rate", 429)) for _ in range(20)]
    provider = FlakyProvider(errors)

    response, _, history = runner._attempt_with_retries(provider, b"", "", rng=random.Random(0))

    assert response is None
    assert provider.calls == runner.RATE_LIMIT_TRIES
    assert len(history) == runner.RATE_LIMIT_TRIES


def test_retry_after_header_is_honored():
    err = ProviderError("rate", status=429, retry_after=12.0)
    assert runner._backoff_seconds(err, 0, random.Random(0)) == 12.0


def test_retry_after_is_capped():
    err = ProviderError("rate", status=429, retry_after=9999.0)
    assert runner._backoff_seconds(err, 0, random.Random(0)) == runner.RATE_LIMIT_CAP_S


def test_backoff_is_jittered_and_bounded():
    """Full jitter: identical shards must not retry in lockstep."""
    err = ProviderError("rate", status=429)
    waits = [runner._backoff_seconds(err, 2, random.Random(seed)) for seed in range(20)]
    assert len(set(waits)) > 1
    assert all(0.0 <= w <= runner.RATE_LIMIT_CAP_S for w in waits)


def test_backoff_grows_with_attempt():
    err = ProviderError("boom", status=500)
    rng = random.Random(1)
    early = max(runner._backoff_seconds(err, 0, rng) for _ in range(50))
    late = max(runner._backoff_seconds(err, 3, rng) for _ in range(50))
    assert late > early


def test_history_records_every_attempt(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    errors = [
        ProviderError.wrap("p", FakeSDKError("boom", 500)),
        ProviderError.wrap("p", FakeSDKError("rate", 429)),
    ]
    provider = FlakyProvider(errors)

    _, _, history = runner._attempt_with_retries(provider, b"", "", rng=random.Random(0))

    assert [h["status"] for h in history] == [500, 429]
    assert [h["attempt"] for h in history] == [1, 2]


def test_serving_stack_extraction():
    assert serving_stack({"provider": "BaseTen"}) == "BaseTen"
    assert serving_stack({"served_by": "fireworks"}) == "fireworks"
    assert serving_stack({}) is None
    assert serving_stack(None) is None
