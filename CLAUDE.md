# CLAUDE.md

## Project Overview

MazeRunner is a benchmark for evaluating vision-based GUI agents on maze navigation. The maze generator produces canonical maze graph instances (JSON) across 3 complexity tiers. The renderer module produces 3 benchmark modes (vision drag, vision grid, text grid) and is decoupled from the CLI for reuse at eval time.

## Commands

```bash
# Install dependencies (creates .venv and installs everything)
uv sync

# Generate mazes
uv run python -m mazerunner generate --output-dir data/dev --num-mazes 1000 --master-seed 42 --tier-distribution 300,400,300

# Render mazes (modes: vision_drag, vision_grid, text_grid, all)
uv run python -m mazerunner visualize --input-dir data/dev --output-dir data/dev/renderings --mode all

# Run tests
uv run pytest tests/ -v

# Serve maze environment (OpenEnv)
uv run python -m mazerunner serve
# Configure via env vars: MAZE_MODE, MAZE_INSTANCE_DIR, MAZE_REWARD_MODE, MAZE_MAX_STEPS, MAZE_SEED

# Run agent on mazes (requires OPENAI_API_KEY)
uv run python -m mazerunner agent --mode text_grid --instance-dir data/dev --model gpt-5.4 --num-episodes 1

# Run agent with Gemini (requires GEMINI_API_KEY)
uv run python -m mazerunner agent --provider gemini --mode text_grid --instance-dir data/dev --model gemini-2.5-flash --num-episodes 1

# Single-step mode (one direction per tool call, harder eval)
uv run python -m mazerunner agent --mode text_grid --instance-dir data/dev --model gpt-5.4 --num-episodes 1 --single-step

# Run eval harness (requires OPENAI_API_KEY)
uv run python -m mazerunner eval --mode text_grid --instance-dir data/dev --model gpt-5.4 --num-episodes 10 --output eval_results.json

# Run e2e agent eval across all modes (requires OPENAI_API_KEY in .env)
uv run python scripts/e2e_agent_eval.py                    # 3 episodes per mode
uv run python scripts/e2e_agent_eval.py -v --num-episodes 1 # verbose single episode

# Launch eval viewer (replay saved evals + live mode)
uv run python -m mazerunner viewer                         # http://localhost:8080
uv run python -m mazerunner viewer --port 3000 --eval-dir data/eval_results
```

## Architecture

```
mazerunner/
  common/types.py        ← Cell, DifficultyConfig, MazeGrid, MazeInstance dataclasses
  generator/
    seed_utils.py        ← SHA-256 seed derivation + numpy RNG factory
    maze_graph.py        ← Iterative randomized DFS generation + BFS solving
    placement.py         ← Start/goal placement (4 endpoint types)
    difficulty.py        ← 3-tier parameter configs + sampling
    serialization.py     ← MazeGrid → canonical JSON
  renderer/
    base.py              ← Shared utilities: parse_cell, hex_to_rgb, has_wall, config dataclasses
    vision_drag.py       ← Corridor maze image + cell_to_pixel_center/rect (eval API)
    vision_grid.py       ← Cell-and-wall grid image
    text_grid.py         ← ASCII text grid rendering
  navigator/
    base.py              ← InteractionResult, HistoryEntry dataclasses + MazeNavigator ABC
    grid_navigator.py    ← GridNavigator — L/R/U/D cell movement with wall validation
    drag_navigator.py    ← DragNavigator — pixel path movement with collision mask
    rendering.py         ← State overlay rendering (X marker, dotted breadcrumbs)
  generate.py            ← CLI entry point + dataset orchestration
  visualize.py           ← Batch-render CLI (vision_drag, vision_grid, text_grid)
  openenv/
    models.py            ← MazeObservation Pydantic model
    reward.py            ← compute_reward() — sparse / shaped / efficiency modes
    client.py            ← MazeEnvClient(MCPToolClient) thin wrapper
    server/
      maze_environment.py ← MazeEnvironment(MCPEnvironment) — core OpenEnv environment
      app.py             ← create_app() FastAPI entry point + uvicorn main
  agent/
    types.py             ← AgentConfig, TurnRecord, EpisodeResult dataclasses
    tool_transform.py    ← Raw tool result → model-facing content (strips internals)
    context_manager.py   ← SlidingWindowContext — image windowing for vision modes
    chat_context.py      ← ChatCompletionsContext — Chat Completions API context manager
    tool_defs.py         ← Tool schemas (Responses API, Anthropic, Chat Completions, Gemini formats)
    openai_loop.py       ← Main agent loop using OpenAI Responses API
    anthropic_loop.py    ← Main agent loop using Anthropic Messages API
    gemini_loop.py       ← Agent loop using Google Gemini API
    fireworks_loop.py    ← Agent loop using Fireworks Chat Completions API
    runner.py            ← OpenAI/Anthropic/Gemini/Fireworks runners + get_runner() factory
  eval/
    protocol.py          ← EpisodeRunner Protocol, StepRecord, EpisodeRecord, EvalResult
    metrics.py           ← compute_metrics() — success rate, efficiency, etc.
    harness.py           ← run_eval() — orchestrates episodes across instances
    io.py                ← JSON save/load for EvalResult
  viewer/
    app.py               ← FastAPI viewer app (replay + live WebSocket)
    static/              ← HTML/CSS/JS single-page viewer UI
  __main__.py            ← CLI dispatcher (generate, visualize, serve, agent, eval, viewer)
```

### Generator Pipeline

`seed_utils.derive_seed()` → `difficulty.sample_difficulty_params()` → `maze_graph.generate_maze()` → `placement.place_endpoints()` → `maze_graph.solve_bfs()` → `serialization.maze_grid_to_instance()` → JSON file

## Key Invariants

- Perfect mazes (spanning trees): passage count == rows * cols - 1, all cells reachable
- Passages stored as `Set[FrozenSet[Cell]]` for O(1) edge lookup
- Interior endpoints must be dead-end cells (degree 1) for natural placement
- Edge-edge placement requires start and goal on different borders
- Seeds derived via SHA-256 (`derive_seed`), not Python's random module, for platform independence
- Adjacency dicts in JSON output are sorted (keys and neighbor lists) for determinism

## Difficulty Tiers

| Parameter | Tier 1 (Easy) | Tier 2 (Medium) | Tier 3 (Hard) |
|-----------|---------------|-----------------|----------------|
| Grid rows | 5-8 | 10-16 | 18-28 |
| Grid cols | 7-12 | 14-22 | 25-40 |
| Min solution length | 8 | 20 | 40 |

Difficulty score (1-9): `min(9, 1 + int(8 * path_len / (rows * cols)))`

## Git Workflow

Always checkout a new branch for every new feature before committing changes. Never commit directly to main.

## Code Style

- Python 3.11+, type hints on function signatures
- Dataclasses for structured data (`types.py`)
- No external dependencies beyond numpy, Pillow, pytest, openenv-core, fastmcp, uvicorn, openai, anthropic, google-genai, fireworks-ai
- Tests use pytest with class-based grouping and parametrize for grid sizes

## Testing

280+ unit tests covering: seed determinism/uniqueness, maze graph properties (perfect maze invariant, reachability, BFS correctness), all 4 endpoint types with dead-end constraints, difficulty tier parameter ranges, serialization symmetry/sorting/schema, end-to-end pipeline determinism + file I/O, renderer utilities (parse_cell, hex_to_rgb, has_wall), text/vision_drag/vision_grid rendering (dimensions, pixel colors, markers, antialias), batch CLI integration, navigator module (grid movement/wall rejection, drag pixel paths/collision mask, history tracking, X marker overlays, breadcrumb rendering), OpenEnv integration (reward functions, environment reset/step/tools, maze loading, image encoding), agent tool transform (output stripping, vision content blocks, sliding window context, tool schema validation), and eval harness (metrics computation, IO roundtrip, protocol conformance, harness orchestration).

5 e2e test scripts (`scripts/e2e_*.py`) covering: text_grid full episode with BFS solve, vision_grid PNG rendering, vision_drag pixel-path navigation, all 3 reward modes, and max_steps cutoff. Run with `python scripts/e2e_all.py`.

`scripts/e2e_agent_eval.py` runs agent eval across all 3 modes (loads keys from `.env`). Supports `-v` for verbose trajectory streaming, `--modes`, `--num-episodes`, `--model`, `--reasoning-effort`, `--provider` (openai/anthropic/gemini/fireworks), `--thinking-budget`, `--thinking-level`.

## Generated Data

Output goes to `data/` (gitignored). Each maze is `instances/maze_{i:06d}.json` containing id, grid dimensions, start/goal, sorted adjacency dict, shortest path, and metadata (tier, endpoint_type, difficulty_score, branching_factor).
