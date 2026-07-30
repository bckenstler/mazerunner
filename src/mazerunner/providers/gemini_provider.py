"""Gemini adapter: Google Gen AI SDK with function-calling mode ANY."""

from __future__ import annotations

import time

from ..contract import PROMPT_TEXT, TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA
from .base import ProviderError, ProviderResponse

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

    def __init__(self, model: str, **_ignored):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def run(self, png_bytes: bytes, prompt: str = PROMPT_TEXT) -> ProviderResponse:
        from google.genai import types

        client = self._get_client()
        start = time.monotonic()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    prompt,
                ],
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
        for candidate in getattr(response, "candidates", None) or []:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
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
        )
