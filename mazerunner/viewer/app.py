"""FastAPI application for the eval trajectory viewer."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
EVAL_DIR = Path("data/eval_results")


def create_app(eval_dir: str | None = None) -> FastAPI:
    """Create the viewer FastAPI application."""
    app = FastAPI(title="MazeRunner Eval Viewer")
    results_dir = Path(eval_dir) if eval_dir else EVAL_DIR

    # Serve static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text()

    # ─── Replay API ──────────────────────────────────────────────

    @app.get("/api/evals")
    async def list_evals():
        """List available eval result files."""
        if not results_dir.exists():
            return []
        evals = []
        for f in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                evals.append({
                    "filename": f.name,
                    "run_id": data.get("run_id", ""),
                    "mode": data.get("mode", ""),
                    "model": data.get("model", ""),
                    "num_episodes": data.get("num_episodes", 0),
                    "metrics": data.get("metrics", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return evals

    @app.get("/api/evals/{filename}")
    async def get_eval(filename: str):
        """Load a full eval result."""
        path = results_dir / filename
        if not path.exists():
            return {"error": f"File not found: {filename}"}
        return json.loads(path.read_text())

    @app.get("/api/evals/{filename}/episodes/{index}")
    async def get_episode(filename: str, index: int):
        """Load a single episode from an eval result."""
        path = results_dir / filename
        if not path.exists():
            return {"error": f"File not found: {filename}"}
        data = json.loads(path.read_text())
        records = data.get("records", [])
        if index < 0 or index >= len(records):
            return {"error": f"Episode index {index} out of range (0-{len(records)-1})"}
        return records[index]

    # ─── Live Mode API ───────────────────────────────────────────

    # Active live sessions: session_id -> {websocket, thread, ...}
    live_sessions: dict[str, dict] = {}

    @app.get("/api/instances")
    async def list_instances():
        """List available maze instance files."""
        inst_dir = Path("data/dev/instances")
        if not inst_dir.exists():
            inst_dir = Path("data/dev")
        if not inst_dir.exists():
            return []
        return sorted(f.name for f in inst_dir.glob("*.json"))

    @app.websocket("/ws/live")
    async def live_websocket(ws: WebSocket):
        """WebSocket endpoint for live eval streaming."""
        await ws.accept()
        session_id = uuid.uuid4().hex[:12]

        try:
            # Wait for config message from client
            config_msg = await ws.receive_json()
            provider = config_msg.get("provider", "openai")
            model = config_msg.get("model", "gpt-5.4")
            mode = config_msg.get("mode", "text_grid")
            instance_file = config_msg.get("instance", "")
            max_turns = config_msg.get("max_turns", 30)
            single_step = config_msg.get("single_step", False)

            # Resolve instance path
            inst_path = Path("data/dev/instances") / instance_file
            if not inst_path.exists():
                inst_path = Path("data/dev") / instance_file
            if not inst_path.exists():
                await ws.send_json({"type": "error", "data": {"message": f"Instance not found: {instance_file}"}})
                return

            await ws.send_json({"type": "config", "data": {
                "session_id": session_id,
                "provider": provider,
                "model": model,
                "mode": mode,
                "instance": instance_file,
            }})

            # Run the episode in a thread, streaming steps via queue
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def run_episode_thread():
                """Run the agent episode, pushing steps to the async queue."""
                try:
                    from mazerunner.agent.runner import get_runner
                    from mazerunner.agent.types import AgentConfig
                    from mazerunner.openenv.server.maze_environment import MazeEnvironment
                    from mazerunner.renderer.base import load_instance

                    instance = load_instance(str(inst_path))
                    env = MazeEnvironment(
                        mode=mode, instance=instance,
                        max_steps=max_turns, single_step=single_step,
                    )

                    config = AgentConfig(
                        model=model, mode=mode, provider=provider,
                        max_turns=max_turns, single_step=single_step,
                    )

                    # Reset and send initial observation
                    obs = env.reset()
                    meta = obs.metadata
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "initial",
                        "data": meta,
                    })

                    # Use the runner to get an EpisodeRecord
                    runner = get_runner(config, verbose=False)
                    record = runner.run_episode(env, instance.get("id", "unknown"))

                    # Stream each step
                    for i, step in enumerate(record.trajectory):
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "step",
                            "data": {
                                "index": i,
                                "action": step.action,
                                "tool_name": step.tool_name,
                                "reward": step.reward,
                                "valid": step.valid,
                                "reasoning": step.reasoning,
                                "raw_result": step.raw_result,
                            },
                        })

                    # Send done
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "done",
                        "data": {
                            "success": record.success,
                            "total_reward": record.reward,
                            "total_steps": record.steps,
                            "maze_id": record.maze_id,
                        },
                    })
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "error",
                        "data": {"message": str(e)},
                    })

            # Start the episode thread
            thread = threading.Thread(target=run_episode_thread, daemon=True)
            thread.start()

            # Stream queue messages to WebSocket
            while True:
                msg = await queue.get()
                await ws.send_json(msg)
                if msg["type"] in ("done", "error"):
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "data": {"message": str(e)}})
            except Exception:
                pass

    return app


def main(host: str = "0.0.0.0", port: int = 8080, eval_dir: str | None = None):
    """Run the viewer server."""
    import uvicorn
    app = create_app(eval_dir=eval_dir)
    print(f"MazeRunner Eval Viewer: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)
