"""Gemini adapter: Google Gen AI SDK with function-calling mode ANY."""

from __future__ import annotations

import time

from ..contract import PROMPT_TEXT, TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA
from .base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ProviderError,
    ProviderResponse,
)

NAME = "gemini"
ENV_KEY = "GEMINI_API_KEY"

# Gemini function schemas 400 on these JSON Schema keywords; the shared
# evaluator still enforces them, so the contract is unchanged.
UNSUPPORTED_SCHEMA_KEYS = {"minItems", "maxItems", "minimum", "maximum", "additionalProperties"}


def gemini_safe_schema(schema):
    if isinstance(schema, dict):
        return {
            key: gemini_safe_schema(value)
            for key, value in schema.items()
            if key not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [gemini_safe_schema(item) for item in schema]
    return schema


def parse_response(response) -> tuple[dict | None, str | None]:
    """Normalize a generate_content result to (tool_arguments, error)."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            call = getattr(part, "function_call", None)
            if call is not None and call.name == TOOL_NAME:
                if isinstance(call.args, dict):
                    return call.args, None
                return dict(call.args), None
    return None, "response contained no submit_drag_path function call"


class GeminiProvider:
    name = NAME
    env_key = ENV_KEY

    def __init__(
        self,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **_ignored,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            # google-genai takes milliseconds here, unlike the httpx-seconds
            # the other three SDKs use.
            self._client = genai.Client(
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000))
            )
        return self._client

    def _user_turn(self, prompt: str, png_bytes: bytes | None):
        from google.genai import types

        parts = []
        if png_bytes is not None:
            parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
        parts.append(types.Part(text=prompt))
        return types.Content(role="user", parts=parts)

    def continue_run(
        self, prior: ProviderResponse, feedback: str, png_bytes: bytes | None = None
    ) -> ProviderResponse:
        """Second and later turns of a feedback episode.

        Unlike the single-shot path, this must use explicit role-tagged
        `types.Content` turns — a bare part list has no role and cannot express
        a conversation. The model turn is replayed as the SDK's own parts so any
        thought signatures on it survive the round trip.
        """
        from google.genai import types

        if not prior.conversation:
            raise ProviderError("gemini: prior response carries no conversation state")
        contents = list(prior.conversation["contents"])
        model_parts = prior.conversation.get("model_parts")
        if model_parts:
            contents.append(types.Content(role="model", parts=model_parts))
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=TOOL_NAME, response={"feedback": feedback}
                    )
                ],
            )
        )
        contents.append(self._user_turn(feedback, png_bytes))
        return self._send(contents)

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        return self._send([self._user_turn(prompt, png_bytes)])

    def _send(self, contents: list) -> ProviderResponse:
        from google.genai import types

        client = self._get_client()
        start = time.monotonic()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    tools=[
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(
                                    name=TOOL_NAME,
                                    description=TOOL_DESCRIPTION,
                                    parameters_json_schema=gemini_safe_schema(TOOL_SCHEMA),
                                )
                            ]
                        )
                    ],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[TOOL_NAME],
                        )
                    ),
                ),
            )
        except Exception as exc:
            raise ProviderError.wrap("gemini", exc) from exc
        latency = time.monotonic() - start

        arguments, error = parse_response(response)
        thoughts = []
        model_parts = None
        for candidate in getattr(response, "candidates", None) or []:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            if parts and model_parts is None:
                model_parts = list(parts)
            for part in parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    thoughts.append(part.text)
        usage = getattr(response, "usage_metadata", None)
        usage_dict = {}
        if usage is not None:
            for key in ("prompt_token_count", "candidates_token_count", "total_token_count", "thoughts_token_count"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_dict[key] = value
        return ProviderResponse(
            tool_arguments=arguments,
            error=error,
            latency_s=latency,
            usage=usage_dict,
            response_id=getattr(response, "response_id", None),
            model=getattr(response, "model_version", self.model),
            raw=response.to_json_dict() if hasattr(response, "to_json_dict") else None,
            reasoning="\n\n".join(thoughts) or None,
            conversation={"contents": contents, "model_parts": model_parts},
        )
