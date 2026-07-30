"""Anthropic adapter: Messages API with a forced named client tool."""

from __future__ import annotations

import base64
import time

from ..contract import PROMPT_TEXT, TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA
from .base import ProviderError, ProviderResponse

NAME = "anthropic"
ENV_KEY = "ANTHROPIC_API_KEY"


def parse_response(response) -> tuple[dict | None, str | None]:
    """Normalize a Messages API result to (tool_arguments, error)."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            if isinstance(block.input, dict):
                return block.input, None
            return None, "tool_use input was not an object"
    return None, "response contained no submit_drag_path tool call"


class AnthropicProvider:
    name = NAME
    env_key = ENV_KEY

    def __init__(self, model: str, max_tokens: int = 48000, thinking: str = "adaptive", **_ignored):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        client = self._get_client()
        b64 = base64.standard_b64encode(png_bytes).decode()

        # Forcing tool_choice suppresses thinking entirely (measured: ~66
        # output tokens even with explicit adaptive), so the reasoning
        # configuration uses auto tool choice and relies on the prompt's
        # tool-only instruction; a missing tool call is a scored failure.
        if self.thinking == "disabled":
            request = {
                "thinking": {"type": "disabled"},
                "tool_choice": {
                    "type": "tool",
                    "name": TOOL_NAME,
                    "disable_parallel_tool_use": True,
                },
            }
        else:
            request = {
                # display: summarized returns readable reasoning summaries in
                # thinking blocks (the raw chain of thought is never exposed).
                "thinking": {"type": "adaptive", "display": "summarized"},
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }

        start = time.monotonic()
        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                **request,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tools=[
                    {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "input_schema": TOOL_SCHEMA,
                    }
                ],
            ) as stream:
                response = stream.get_final_message()
        except Exception as exc:
            raise ProviderError.wrap("anthropic", exc) from exc
        latency = time.monotonic() - start

        arguments, error = parse_response(response)
        reasoning = (
            "\n\n".join(
                block.thinking
                for block in response.content
                if getattr(block, "type", None) == "thinking" and getattr(block, "thinking", "")
            )
            or None
        )
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            tool_arguments=arguments,
            error=error,
            latency_s=latency,
            usage=usage.model_dump() if usage is not None else {},
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", self.model),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            reasoning=reasoning,
        )
