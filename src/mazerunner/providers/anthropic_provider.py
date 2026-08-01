"""Anthropic adapter: Messages API with a forced named client tool."""

from __future__ import annotations

import base64
import time

from ..contract import PROMPT_TEXT, TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA
from .base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ProviderError,
    ProviderResponse,
)

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

    def __init__(
        self,
        model: str,
        max_tokens: int = 48000,
        thinking: str = "adaptive",
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **_ignored,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                timeout=self.timeout, max_retries=self.max_retries
            )
        return self._client

    def _image_block(self, png_bytes: bytes) -> dict:
        b64 = base64.standard_b64encode(png_bytes).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }

    def _thinking_config(self) -> dict:
        # Forcing tool_choice suppresses thinking entirely (measured: ~66
        # output tokens even with explicit adaptive), so the reasoning
        # configuration uses auto tool choice and relies on the prompt's
        # tool-only instruction; a missing tool call is a scored failure.
        if self.thinking == "disabled":
            return {
                "thinking": {"type": "disabled"},
                "tool_choice": {
                    "type": "tool",
                    "name": TOOL_NAME,
                    "disable_parallel_tool_use": True,
                },
            }
        return {
            # display: summarized returns readable reasoning summaries in
            # thinking blocks (the raw chain of thought is never exposed).
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }

    def _send(self, messages: list) -> tuple[object, float]:
        client = self._get_client()
        start = time.monotonic()
        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                **self._thinking_config(),
                messages=messages,
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
        return response, time.monotonic() - start

    def continue_run(
        self, prior: ProviderResponse, feedback: str, png_bytes: bytes | None = None
    ) -> ProviderResponse:
        """Second and later turns of a feedback episode.

        The assistant turn is replayed as the SDK's own typed blocks rather than
        reconstructed: under adaptive thinking the thinking blocks carry
        signatures that must come back verbatim, and flattening them to text
        would drop those.
        """
        if not prior.conversation:
            raise ProviderError("anthropic: prior response carries no conversation state")
        messages = list(prior.conversation["messages"])
        messages.append({"role": "assistant", "content": prior.conversation["assistant_content"]})

        content = []
        if prior.tool_call_id:
            # A tool_result must be the first block of the user turn.
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": prior.tool_call_id,
                    "content": [{"type": "text", "text": feedback}],
                }
            )
        if png_bytes is not None:
            content.append(self._image_block(png_bytes))
        content.append({"type": "text", "text": feedback})
        messages.append({"role": "user", "content": content})

        response, latency = self._send(messages)
        return self._to_provider_response(response, latency, messages)

    def _to_provider_response(self, response, latency, messages) -> ProviderResponse:
        arguments, error = parse_response(response)
        reasoning = "\n\n".join(
            block.thinking
            for block in response.content
            if getattr(block, "type", None) == "thinking" and getattr(block, "thinking", "")
        )
        tool_call_id = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                tool_call_id = getattr(block, "id", None)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            tool_arguments=arguments,
            error=error,
            latency_s=latency,
            usage=usage.model_dump() if usage is not None else {},
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", self.model),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            reasoning=reasoning or None,
            conversation={"messages": messages, "assistant_content": response.content},
            tool_call_id=tool_call_id,
        )

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        messages = [
            {
                "role": "user",
                "content": [self._image_block(png_bytes), {"type": "text", "text": prompt}],
            }
        ]
        response, latency = self._send(messages)
        return self._to_provider_response(response, latency, messages)
