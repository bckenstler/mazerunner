"""Chat Completions context management for the agent loop.

Manages role-based messages for the Chat Completions API (Fireworks, OpenAI Chat).
In vision modes, only the most recent maze image is kept in context.
"""

from __future__ import annotations

import json
from typing import Any


class ChatCompletionsContext:
    """Manages conversation messages in Chat Completions format.

    In vision modes (vision_grid, vision_drag), after each tool result is added,
    older images are replaced with "[Previous maze image omitted]" to keep
    context lean. Text mode is a no-op for windowing.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._messages: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        """Current messages list."""
        return self._messages

    def add_system(self, text: str) -> None:
        """Add a system message."""
        self._messages.append({"role": "system", "content": text})

    def add_user_message(self, content: str | list[dict]) -> None:
        """Add a user message with text or multimodal content."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message_dict: dict) -> None:
        """Add an assistant message preserving tool_calls and reasoning_content."""
        self._messages.append(message_dict)

    def add_tool_result(self, tool_call_id: str, content: str | list[dict]) -> None:
        """Add a tool result message.

        Chat Completions requires tool content to be a string.
        Lists are JSON-serialized automatically.
        """
        if isinstance(content, list):
            content_str = json.dumps(content)
        else:
            content_str = content

        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content_str,
        })

        self._apply_image_window()

    def _apply_image_window(self) -> None:
        """Replace images in all but the most recent tool result message."""
        if self._mode == "text_grid":
            return

        # Find indices of tool messages that contain image_url blocks
        # Tool messages with JSON-serialized content blocks containing images
        image_msg_indices = []
        for i, msg in enumerate(self._messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if '"image_url"' in content and '"type": "image_url"' in content:
                image_msg_indices.append(i)

        if len(image_msg_indices) <= 1:
            return

        # Replace images in all but the last one
        for idx in image_msg_indices[:-1]:
            msg = self._messages[idx]
            try:
                blocks = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            new_blocks = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    new_blocks.append({
                        "type": "text",
                        "text": "[Previous maze image omitted]",
                    })
                else:
                    new_blocks.append(block)
            self._messages[idx] = {**msg, "content": json.dumps(new_blocks)}
