"""Sliding window image context management for the agent loop.

In vision modes, only the most recent maze image is kept in context.
Older images are replaced with a placeholder text.
"""

from __future__ import annotations

import copy
from typing import Any


class SlidingWindowContext:
    """Manages conversation items with sliding window for images.

    In vision modes (vision_grid, vision_drag), after each tool output is added,
    older images are replaced with "[Previous maze image omitted]" to keep
    context lean. Text mode is a no-op for windowing.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._items: list[dict] = []

    @property
    def items(self) -> list[dict]:
        """Current input items with image windowing applied."""
        return self._items

    def add_system(self, text: str) -> None:
        """Add a system message (developer turn) to the context."""
        self._items.append({
            "role": "developer",
            "content": text,
        })

    def add_response_items(self, items: list[dict]) -> None:
        """Add response output items (function_call, message, etc.) to context."""
        self._items.extend(items)

    def add_tool_output(self, call_id: str, output: str | list[dict]) -> None:
        """Add a function_call_output item and apply image windowing.

        Args:
            call_id: The call_id from the function_call item.
            output: Transformed tool output (str or list of content blocks).
        """
        if isinstance(output, str):
            content = output
        else:
            content = output

        self._items.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": content,
        })

        self._apply_image_window()

    def _apply_image_window(self) -> None:
        """Replace images in all but the most recent function_call_output."""
        if self._mode == "text_grid":
            return

        # Find indices of function_call_output items that contain images
        image_output_indices = []
        for i, item in enumerate(self._items):
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and isinstance(item.get("output"), list)
                and any(
                    block.get("type") == "input_image"
                    for block in item["output"]
                    if isinstance(block, dict)
                )
            ):
                image_output_indices.append(i)

        if len(image_output_indices) <= 1:
            return

        # Replace images in all but the last one
        for idx in image_output_indices[:-1]:
            item = self._items[idx]
            new_output = []
            for block in item["output"]:
                if isinstance(block, dict) and block.get("type") == "input_image":
                    new_output.append({
                        "type": "input_text",
                        "text": "[Previous maze image omitted]",
                    })
                else:
                    new_output.append(block)
            self._items[idx] = {**item, "output": new_output}
