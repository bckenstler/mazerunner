"""OpenAI adapter: Responses API with a forced function call."""

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

NAME = "openai"
ENV_KEY = "OPENAI_API_KEY"


def parse_response(response) -> tuple[dict | None, str | None]:
    """Normalize a Responses API result to (tool_arguments, error)."""
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call" and item.name == TOOL_NAME:
            try:
                return json.loads(item.arguments), None
            except (json.JSONDecodeError, TypeError) as exc:
                return None, f"malformed function arguments: {exc}"
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and TOOL_NAME in text:
                match = re.search(
                    rf'<\w+:invoke name="[^"]*{TOOL_NAME}">\s*'
                    rf'<\w+:parameter name="points">(.*?)</\w+:parameter>',
                    text,
                    re.S,
                )
                if match:
                    try:
                        return {"points": json.loads(match.group(1))}, None
                    except json.JSONDecodeError as exc:
                        return None, f"malformed inline function arguments: {exc}"
    return None, "response contained no submit_drag_path function call"


class OpenAIProvider:
    """OpenAI Responses API with a forced function call. Reasoning-effort is
    passed through per config, which is what the effort sweep varies."""
    name = NAME
    env_key = ENV_KEY

    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        env_key: str = "OPENAI_API_KEY",
        tool_choice: str = "forced",
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **_ignored,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        # A base_url points this Responses-API adapter at a compatible
        # provider (e.g. Meta Model API for muse-spark reasoning summaries).
        self.base_url = base_url
        self.env_key = env_key
        self.tool_choice = tool_choice
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            options = {"timeout": self.timeout, "max_retries": self.max_retries}
            if self.base_url:
                self._client = OpenAI(
                    base_url=self.base_url, api_key=os.environ[self.env_key], **options
                )
            else:
                self._client = OpenAI(**options)
        return self._client

    def _user_turn(self, prompt: str, png_bytes: bytes | None) -> dict:
        content = []
        if png_bytes is not None:
            b64 = base64.standard_b64encode(png_bytes).decode()
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }
            )
        content.append({"type": "input_text", "text": prompt})
        return {"role": "user", "content": content}

    def continue_run(
        self, prior: ProviderResponse, feedback: str, png_bytes: bytes | None = None
    ) -> ProviderResponse:
        """Second and later turns, via server-side conversation state.

        `previous_response_id` is the documented continuation for the Responses
        API and avoids replaying reasoning items by hand — which matters here,
        because a reasoning model's prior items must come back intact or the
        follow-up loses the thinking that produced the first attempt.
        """
        if not prior.response_id:
            raise ProviderError("openai: prior response carries no id to continue from")
        tool_output = []
        if prior.tool_call_id:
            tool_output.append(
                {
                    "type": "function_call_output",
                    "call_id": prior.tool_call_id,
                    "output": feedback,
                }
            )
        return self._send(
            [*tool_output, self._user_turn(feedback, png_bytes)],
            previous_response_id=prior.response_id,
        )

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        return self._send([self._user_turn(prompt, png_bytes)])

    def _send(self, input_items: list, previous_response_id: str | None = None) -> ProviderResponse:
        client = self._get_client()
        request = {
            "model": self.model,
            "input": input_items,
            "tools": [
                {
                    "type": "function",
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    # strict mode rejects minItems/maxItems, so schema
                    # enforcement stays in the shared evaluator
                    "parameters": TOOL_SCHEMA,
                    "strict": False,
                }
            ],
            "tool_choice": (
                {"type": "function", "name": TOOL_NAME}
                if self.tool_choice == "forced"
                else "auto"
            ),
            "parallel_tool_calls": False,
        }
        reasoning_config = {"summary": "auto"}
        if self.reasoning_effort:
            reasoning_config["effort"] = self.reasoning_effort
        request["reasoning"] = reasoning_config
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        start = time.monotonic()
        try:
            response = client.responses.create(**request)
        except Exception as exc:
            raise ProviderError.wrap("openai", exc) from exc
        latency = time.monotonic() - start

        arguments, error = parse_response(response)
        summaries = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "reasoning":
                for part in getattr(item, "summary", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        summaries.append(text)
        tool_call_id = None
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "function_call" and item.name == TOOL_NAME:
                tool_call_id = getattr(item, "call_id", None)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            tool_arguments=arguments,
            error=error,
            latency_s=latency,
            usage=usage.model_dump() if usage is not None else {},
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", self.model),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            reasoning="\n\n".join(summaries) or None,
            conversation={"previous_response_id": getattr(response, "id", None)},
            tool_call_id=tool_call_id,
        )
