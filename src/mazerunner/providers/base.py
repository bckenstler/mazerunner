"""Provider-neutral request/response shapes.

Every provider receives the identical PNG, prompt, and tool schema from
contract.py; adapters only translate that into their API and normalize the
returned tool call. Parsing is separated from transport so it can be tested
against mocked SDK payloads without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderResponse:
    """One provider turn, normalized across vendor dialects.

    `raw` keeps the provider's own serialized response so a run can be audited
    against what the API actually returned — nothing ships without its trace.
    """

    tool_arguments: dict | None = None
    error: str | None = None  # set when no valid submit_drag_path call came back
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    response_id: str | None = None
    model: str | None = None
    raw: dict | None = None  # provider-serialized response for full-trace audit
    reasoning: str | None = None  # summarized reasoning trace, where the API offers one
    # Opaque, adapter-owned state for continuing the conversation: the typed
    # assistant turn and tool-call id that a follow-up must replay. Each dialect
    # needs something different (Anthropic wants its thinking blocks back
    # *with signatures*, OpenAI just wants the response id), so the shape is
    # private to the adapter that produced it.
    conversation: dict | None = None
    tool_call_id: str | None = None


class ProviderError(Exception):
    """Transport or API failure.

    Carries the HTTP status when the SDK exposed one, because retryability is
    not uniform: a 429 should be waited out, a 400 (bad parameter) will fail
    identically forever. Blindly retrying everything cost us 225 wasted calls
    on a leg whose `reasoning_effort` value the model simply did not accept.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        original: Exception | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.original = original

    @classmethod
    def wrap(cls, prefix: str, exc: Exception) -> "ProviderError":
        """Build from an arbitrary SDK exception, salvaging status/Retry-After."""
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        if status is None:
            code = getattr(exc, "code", None)
            status = code if isinstance(code, int) else None

        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = None
            try:
                raw = headers.get("retry-after") or headers.get("Retry-After")
            except AttributeError:
                raw = None
            if raw is not None:
                try:
                    retry_after = float(raw)
                except (TypeError, ValueError):
                    retry_after = None

        return cls(f"{prefix}: {exc}", status=status, retry_after=retry_after, original=exc)


# 408 request timeout, 409 conflict, 429 rate limit, 5xx server-side, 529 overloaded.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def is_retryable(error: ProviderError) -> bool:
    """Unknown-status failures are treated as retryable (they are usually
    connection resets or timeouts that never reached the API)."""
    if error.status is None:
        return True
    return error.status in RETRYABLE_STATUS


# A wedged connection with no timeout stalled one shard of the main run for
# 106 minutes while the rest of the leg sat idle. Every adapter sets one.
DEFAULT_TIMEOUT_S = 600.0

# The SDKs retry internally by default (openai and anthropic: max_retries=2),
# which multiplies against the runner's own classified ladder and makes the
# recorded transport history undercount real API calls. The runner owns retries.
DEFAULT_MAX_RETRIES = 0


def serving_stack(raw: dict | None) -> str | None:
    """Which upstream actually served the request.

    Gateways route the same model to different hosts, and the same open-weight
    checkpoint can reason differently per host — so the leaderboard needs this
    recorded per attempt, not assumed from config.
    """
    if not isinstance(raw, dict):
        return None
    for key in ("provider", "provider_name", "served_by"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None
