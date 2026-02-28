"""FastAPI server for MazeRunner Attempt Viewer."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from mazerunner.ui.replay import (
    list_result_mazes,
    list_results,
    list_runs,
    load_result_entry,
    load_run,
)
from mazerunner.ui.run_manager import run_maze_live, save_run_log

# These are set by cli.py before the app starts
IMAGE_DIR = "data/dev/images"
GT_DIR = "data/dev/gt"
RUNS_DIR = "runs"
RESULTS_DIR = "results"

app = FastAPI(title="MazeRunner Viewer")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/mazes")
async def list_mazes():
    """List available mazes (intersection of images and GT)."""
    if not os.path.isdir(IMAGE_DIR) or not os.path.isdir(GT_DIR):
        return {"mazes": [], "error": "Image or GT directory not found"}

    image_files = {Path(f).stem for f in os.listdir(IMAGE_DIR) if f.endswith(".png")}
    gt_files = {Path(f).stem for f in os.listdir(GT_DIR) if f.endswith(".json")}
    available = sorted(image_files & gt_files)

    mazes = []
    for maze_id in available:
        gt_path = os.path.join(GT_DIR, f"{maze_id}.json")
        try:
            with open(gt_path) as f:
                gt = json.load(f)
            mazes.append({
                "id": maze_id,
                "grid": f"{gt['difficulty']['grid_rows']}x{gt['difficulty']['grid_cols']}",
                "tier": gt["difficulty"].get("tier", "?"),
                "image_size": gt.get("image_size", {}),
            })
        except (json.JSONDecodeError, OSError, KeyError):
            mazes.append({"id": maze_id, "grid": "?", "tier": "?"})

    return {"mazes": mazes}


@app.get("/api/mazes/{maze_id}/image")
async def get_maze_image(maze_id: str):
    """Serve a maze PNG image."""
    image_path = os.path.join(IMAGE_DIR, f"{maze_id}.png")
    if not os.path.exists(image_path):
        return {"error": "Image not found"}
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/mazes/{maze_id}/gt")
async def get_maze_gt(maze_id: str):
    """Serve GT solution polyline for overlay."""
    gt_path = os.path.join(GT_DIR, f"{maze_id}.json")
    if not os.path.exists(gt_path):
        return {"error": "GT not found"}

    with open(gt_path) as f:
        gt = json.load(f)

    return {
        "maze_id": maze_id,
        "solution_polyline": gt.get("gt", {}).get("solution_polyline", []),
        "image_size": gt.get("image_size", {}),
        "difficulty": gt.get("difficulty", {}),
    }


@app.get("/api/runs")
async def api_list_runs():
    """List saved run logs."""
    runs = list_runs(RUNS_DIR)
    return {"runs": runs}


@app.get("/api/runs/{run_id:path}")
async def api_get_run(run_id: str):
    """Load a specific run log."""
    # run_id is model_dir/filename_stem
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return {"error": "Run not found"}
    return load_run(path)


@app.get("/api/results")
async def api_list_results():
    """List existing JSONL result files."""
    results = list_results(RESULTS_DIR)
    return {"results": results}


@app.get("/api/results/{filename}/mazes")
async def api_list_result_mazes(filename: str):
    """List maze IDs in a result file."""
    maze_ids = list_result_mazes(RESULTS_DIR, filename)
    return {"maze_ids": maze_ids}


@app.get("/api/results/{filename}/{maze_id}")
async def api_get_result_entry(filename: str, maze_id: str):
    """Get single-shot result for a maze from a JSONL file."""
    entry = load_result_entry(RESULTS_DIR, filename, maze_id)
    if entry is None:
        return {"error": "Entry not found"}
    return entry


@app.websocket("/ws/run")
async def websocket_run(ws: WebSocket):
    """Live run WebSocket endpoint.

    Client sends: {model, maze_id, max_turns?, temperature?, max_tokens?, api_base?}
    Server streams turn events as JSON messages.
    """
    await ws.accept()

    try:
        # Wait for config message
        config = await ws.receive_json()
        model = config.get("model", "")
        maze_id = config.get("maze_id", "")

        if not model or not maze_id:
            await ws.send_json({"type": "error", "message": "model and maze_id required"})
            await ws.close()
            return

        max_turns = config.get("max_turns", 30)
        temperature = config.get("temperature", 0.0)
        max_tokens = config.get("max_tokens", 8192)
        api_base = config.get("api_base")
        reasoning_effort = config.get("reasoning_effort")

        await ws.send_json({"type": "started", "maze_id": maze_id, "model": model})

        async def on_turn(turn_data: dict):
            await ws.send_json({"type": "turn", "data": turn_data})

        run_log = await run_maze_live(
            maze_id=maze_id,
            model=model,
            image_dir=IMAGE_DIR,
            gt_dir=GT_DIR,
            on_turn=on_turn,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            reasoning_effort=reasoning_effort,
        )

        # Save run log
        saved_path = save_run_log(run_log, RUNS_DIR)

        await ws.send_json({
            "type": "completed",
            "final_result": run_log.get("final_result"),
            "saved_path": saved_path,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
