"""Generic adapter for OpenAI-compatible chat-completions providers.

Covers Moonshot (Kimi K3), Meta Model API (Muse Spark), OpenRouter /
Together / Fireworks (Inkling and other open-weight models), and Z.ai —
configured per provider in configs/*.json with `type: "openai_compat"`,
a `base_url`, an `env_key`, and a `model` id. The shared contract (image +
prompt + submit_drag_path schema) is unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time

from ..contract import PROMPT_TEXT, TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA
from .base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ProviderError,
    ProviderResponse,
)

TYPE_NAME = "openai_compat"


def parse_response(completion) -> tuple[dict | None, str | None]:
    """Normalize a chat-completions result to (tool_arguments, error).

    Some compat shims (observed: Meta Model API) leave the model's structured
    call in the text content as an XML-ish invoke block instead of populating
    tool_calls; that is still a structured call in the provider's wire format,
    so we normalize it here rather than scoring a shim bug against the model.
    """
    choices = getattr(completion, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        for call in (getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            if function is not None and function.name == TOOL_NAME:
                try:
                    return json.loads(function.arguments), None
                except (json.JSONDecodeError, TypeError) as exc:
                    return None, f"malformed function arguments: {exc}"
        content = getattr(message, "content", None)
        if isinstance(content, str) and TOOL_NAME in content:
            match = re.search(
                rf'<\w+:invoke name="[^"]*{TOOL_NAME}">\s*'
                rf'<\w+:parameter name="points">(.*?)</\w+:parameter>',
                content,
                re.S,
            )
            if match:
                try:
                    return {"points": json.loads(match.group(1))}, None
                except json.JSONDecodeError as exc:
                    return None, f"malformed inline function arguments: {exc}"
    return None, "response contained no submit_drag_path tool call"


def extract_reasoning(completion) -> str | None:
    """Best-effort reasoning capture across compat dialects."""
    choices = getattr(completion, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        for attr in ("reasoning_content", "reasoning"):
            text = getattr(message, attr, None)
            if isinstance(text, str) and text.strip():
                return text
        # Some SDKs stash unknown response fields in model_extra.
        extra = getattr(message, "model_extra", None) or {}
        for attr in ("reasoning_content", "reasoning"):
            text = extra.get(attr)
            if isinstance(text, str) and text.strip():
                return text
    return None


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        base_url: str,
        env_key: str,
        tool_choice: str = "forced",
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        schema_sanitize: bool = False,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **_ignored,
    ):
        self.model = model
        self.base_url = base_url
        self.env_key = env_key
        self.timeout = timeout
        self.max_retries = max_retries
        # "forced" | "auto" | "omit" — omit skips the parameter entirely for
        # gateways that reject it (e.g. LiteLLM's fireworks_ai integration).
        self.tool_choice = tool_choice
        # Strip JSON-Schema keywords Gemini rejects (minItems etc.) when the
        # gateway forwards our schema to a Gemini backend.
        self.schema_sanitize = schema_sanitize
        self.max_tokens = max_tokens
        # Provider-dialect knobs (e.g. Kimi's reasoning_effort, OpenRouter's
        # reasoning object) merged verbatim into the request JSON.
        self.extra_body = extra_body or {}
        self._client = None

    def _schema(self) -> dict:
        if self.schema_sanitize:
            from .gemini_provider import gemini_safe_schema

            return gemini_safe_schema(TOOL_SCHEMA)
        return TOOL_SCHEMA

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=os.environ[self.env_key],
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _user_message(self, prompt: str, png_bytes: bytes | None) -> dict:
        content = []
        if png_bytes is not None:
            b64 = base64.standard_b64encode(png_bytes).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": content}

    def continue_run(
        self, prior: ProviderResponse, feedback: str, png_bytes: bytes | None = None
    ) -> ProviderResponse:
        """Second and later turns of a feedback episode.

        Chat-completions tool messages are text-only, so the overlay cannot ride
        on the tool result and goes in a following user message instead. When
        the model answered through the inline-XML fallback there is no tool-call
        id at all, and emitting a tool message would 400 — those continue as a
        plain user turn.
        """
        if not prior.conversation:
            raise ProviderError(f"{self.base_url}: prior response carries no conversation state")
        messages = list(prior.conversation["messages"])
        assistant = prior.conversation.get("assistant_message")
        if assistant is not None:
            messages.append(assistant)
        if prior.tool_call_id:
            messages.append(
                {"role": "tool", "tool_call_id": prior.tool_call_id, "content": feedback}
            )
        messages.append(self._user_message(feedback, png_bytes))
        return self._send(messages)

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        return self._send([self._user_message(prompt, png_bytes)])

    def _send(self, messages: list) -> ProviderResponse:
        client = self._get_client()
        request = {
            "model": self.model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "parameters": self._schema(),
                    },
                }
            ],
        }
        if self.tool_choice != "omit":
            request["tool_choice"] = (
                {"type": "function", "function": {"name": TOOL_NAME}}
                if self.tool_choice == "forced"
                else "auto"
            )
        if self.max_tokens:
            request["max_tokens"] = self.max_tokens

        start = time.monotonic()
        try:
            completion = client.chat.completions.create(**request, extra_body=self.extra_body)
        except Exception as exc:
            raise ProviderError.wrap(self.base_url, exc) from exc
        latency = time.monotonic() - start

        arguments, error = parse_response(completion)
        tool_call_id = None
        assistant_message = None
        for choice in getattr(completion, "choices", None) or []:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            if hasattr(message, "model_dump"):
                assistant_message = message.model_dump(exclude_none=True)
            for call in getattr(message, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                if function is not None and function.name == TOOL_NAME:
                    tool_call_id = getattr(call, "id", None)
        usage = getattr(completion, "usage", None)
        return ProviderResponse(
            tool_arguments=arguments,
            error=error,
            latency_s=latency,
            usage=usage.model_dump() if usage is not None else {},
            response_id=getattr(completion, "id", None),
            model=getattr(completion, "model", self.model),
            raw=completion.model_dump() if hasattr(completion, "model_dump") else None,
            reasoning=extract_reasoning(completion),
            conversation={"messages": messages, "assistant_message": assistant_message},
            tool_call_id=tool_call_id,
        )
