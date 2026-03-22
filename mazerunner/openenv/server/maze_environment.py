"""MazeEnvironment — OpenEnv MCPEnvironment for maze navigation."""

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastmcp import FastMCP
from PIL import Image

from openenv.core.env_server import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    MCPEnvironment,
    Observation,
    State,
    Tool,
)
from openenv.core.env_server.mcp_environment import get_server_tools

from mazerunner.navigator.base import MazeNavigator
from mazerunner.navigator.drag_navigator import DragNavigator
from mazerunner.navigator.grid_navigator import GridNavigator
from mazerunner.openenv.models import MazeObservation
from mazerunner.openenv.reward import compute_reward
from mazerunner.renderer.base import load_instance


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_to_string(navigator: MazeNavigator, mode: str) -> str:
    rendered = navigator.render()
    if isinstance(rendered, Image.Image):
        return _image_to_base64(rendered)
    return rendered


class MazeEnvironment(MCPEnvironment):
    """OpenEnv environment for maze navigation benchmarks.

    Supports three modes: text_grid, vision_grid, vision_drag.
    """

    def __init__(
        self,
        mode: str = "text_grid",
        instance_dir: str | None = None,
        instance: Dict[str, Any] | None = None,
        reward_mode: str = "sparse",
        max_steps: int = 100,
        seed: int = 42,
    ) -> None:
        if instance_dir is None and instance is None:
            raise ValueError("Must provide either instance_dir or instance")
        if mode not in ("text_grid", "vision_grid", "vision_drag"):
            raise ValueError(f"Unknown mode: {mode}")

        self._mode = mode
        self._instance_dir = instance_dir
        self._fixed_instance = instance
        self._reward_mode = reward_mode
        self._max_steps = max_steps
        self._seed = seed

        # Load instance list for directory mode
        self._instance_paths: List[str] = []
        self._instance_idx = 0
        if instance_dir is not None:
            instances_dir = os.path.join(instance_dir, "instances")
            if os.path.isdir(instances_dir):
                self._instance_paths = sorted(
                    str(p) for p in Path(instances_dir).glob("*.json")
                )
            else:
                self._instance_paths = sorted(
                    str(p) for p in Path(instance_dir).glob("*.json")
                )
            # Deterministic shuffle via seed
            rng_seed = hashlib.sha256(str(seed).encode()).digest()
            indices = list(range(len(self._instance_paths)))
            # Fisher-Yates shuffle with deterministic seed bytes
            import struct

            seed_ints = struct.unpack(f"<{len(rng_seed) // 4}I", rng_seed)
            s = seed_ints[0] if seed_ints else 0
            for i in range(len(indices) - 1, 0, -1):
                s = (s * 1103515245 + 12345) & 0x7FFFFFFF
                j = s % (i + 1)
                indices[i], indices[j] = indices[j], indices[i]
            self._instance_paths = [self._instance_paths[i] for i in indices]

        # Navigator state (set on reset)
        self._navigator: Optional[MazeNavigator] = None
        self._current_instance: Optional[Dict[str, Any]] = None
        self._step_count = 0
        self._prev_position: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None

        # Build FastMCP server with tools registered directly on the mcp instance
        mcp = FastMCP("maze_env")
        self._register_tools(mcp)
        super().__init__(mcp)

    def _register_tools(self, mcp: FastMCP) -> None:
        env = self

        if self._mode in ("text_grid", "vision_grid"):

            @mcp.tool()
            def navigate(directions: str) -> dict:
                """Move through the maze using direction characters (U/D/L/R).

                Args:
                    directions: String of direction characters, e.g. 'RRDD'.

                Returns:
                    Dict with valid, position, finished, steps_applied,
                    rendered, reward, step_count, done.
                """
                return env._handle_navigate(directions)

        else:

            @mcp.tool()
            def drag(points: list) -> dict:
                """Drag through the maze along a pixel coordinate path.

                Args:
                    points: List of [x, y] coordinate pairs, e.g. [[10,20],[30,40]].

                Returns:
                    Dict with valid, position, finished, steps_applied,
                    rendered, reward, step_count, done.
                """
                return env._handle_drag(points)

        @mcp.tool()
        def get_maze_info() -> dict:
            """Get information about the current maze.

            Returns:
                Dict with grid_rows, grid_cols, start, goal, mode, maze_id.
            """
            if env._current_instance is None:
                return {"error": "No maze loaded. Call reset() first."}
            inst = env._current_instance
            return {
                "grid_rows": inst["grid_rows"],
                "grid_cols": inst["grid_cols"],
                "start": inst["start"],
                "goal": inst["goal"],
                "mode": env._mode,
                "maze_id": inst.get("id", "unknown"),
            }

    def _handle_navigate(self, directions: str) -> dict:
        if self._navigator is None:
            return {"error": "No maze loaded. Call reset() first."}

        self._prev_position = self._navigator.position
        result = self._navigator.interact(directions)
        self._step_count += 1

        reward = compute_reward(
            self._navigator,
            self._current_instance,
            result,
            self._prev_position,
            self._step_count,
            self._reward_mode,
            self._mode,
            self._max_steps,
        )

        done = result.finished or self._step_count >= self._max_steps
        rendered = _render_to_string(self._navigator, self._mode)

        return {
            "valid": result.valid,
            "position": list(result.position),
            "finished": result.finished,
            "steps_applied": result.steps_applied,
            "rendered": rendered,
            "reward": reward,
            "step_count": self._step_count,
            "done": done,
        }

    def _handle_drag(self, points: list) -> dict:
        if self._navigator is None:
            return {"error": "No maze loaded. Call reset() first."}

        self._prev_position = self._navigator.position
        result = self._navigator.interact(points)
        self._step_count += 1

        reward = compute_reward(
            self._navigator,
            self._current_instance,
            result,
            self._prev_position,
            self._step_count,
            self._reward_mode,
            self._mode,
            self._max_steps,
        )

        done = result.finished or self._step_count >= self._max_steps
        rendered = _render_to_string(self._navigator, self._mode)

        return {
            "valid": result.valid,
            "position": list(result.position),
            "finished": result.finished,
            "steps_applied": result.steps_applied,
            "rendered": rendered,
            "reward": reward,
            "step_count": self._step_count,
            "done": done,
        }

    def _load_next_instance(self) -> Dict[str, Any]:
        if self._fixed_instance is not None:
            return self._fixed_instance
        if not self._instance_paths:
            raise RuntimeError("No maze instances found in instance_dir")
        path = self._instance_paths[self._instance_idx % len(self._instance_paths)]
        self._instance_idx += 1
        return load_instance(path)

    def _create_navigator(self, instance: Dict[str, Any]) -> MazeNavigator:
        if self._mode == "vision_drag":
            return DragNavigator(instance)
        elif self._mode == "vision_grid":
            return GridNavigator(instance, render_mode="vision_grid")
        else:
            return GridNavigator(instance, render_mode="text_grid")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._current_instance = self._load_next_instance()
        self._navigator = self._create_navigator(self._current_instance)
        self._step_count = 0
        self._prev_position = self._navigator.position

        rendered = _render_to_string(self._navigator, self._mode)
        maze_id = self._current_instance.get("id", "unknown")

        obs = MazeObservation(
            rendered=rendered,
            mode=self._mode,
            position=list(self._navigator.position),
            valid=True,
            finished=False,
            steps_applied=0,
            reward=0.0,
            maze_id=maze_id,
            step_count=0,
            done=False,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata=obs.model_dump(),
        )

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Route MCP actions directly, bypassing async FastMCP client."""
        if isinstance(action, ListToolsAction):
            tools = []
            for name, tool in get_server_tools(self.mcp_server).items():
                tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.parameters if hasattr(tool, "parameters") else {},
                    )
                )
            return ListToolsObservation(tools=tools)

        if isinstance(action, CallToolAction):
            server_tools = get_server_tools(self.mcp_server)
            tool = server_tools.get(action.tool_name)
            if tool is None:
                from openenv.core.env_server import ToolError, ToolErrorType

                return CallToolObservation(
                    tool_name=action.tool_name,
                    result=None,
                    error=ToolError(
                        error_type=ToolErrorType.TOOL_NOT_FOUND,
                        message=f"Tool '{action.tool_name}' not found",
                    ),
                )
            result = tool.fn(**action.arguments)
            from fastmcp.client.client import CallToolResult
            from mcp.types import TextContent

            return CallToolObservation(
                tool_name=action.tool_name,
                result=CallToolResult(
                    content=[TextContent(type="text", text=str(result))],
                    structured_content=result,
                    meta=None,
                    data=result,
                    is_error=False,
                ),
            )

        return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _step_impl(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": "Use MCP tools (navigate/drag, get_maze_info) to interact."},
        )

    @property
    def state(self) -> State:
        meta = {}
        if self._current_instance:
            meta["maze_id"] = self._current_instance.get("id", "unknown")
            meta["mode"] = self._mode
        if self._navigator:
            meta["position"] = list(self._navigator.position)
            meta["finished"] = self._navigator.finished
        return State(step_count=self._step_count, **meta)
