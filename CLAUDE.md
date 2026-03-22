# CLAUDE.md

## Project Overview

MazeRunner is a benchmark for evaluating vision-based GUI agents on maze navigation. The maze generator produces canonical maze graph instances (JSON) across 3 complexity tiers. The renderer module produces 3 benchmark modes (vision drag, vision grid, text grid) and is decoupled from the CLI for reuse at eval time.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate mazes
python -m mazerunner generate --output-dir data/dev --num-mazes 1000 --master-seed 42 --tier-distribution 300,400,300

# Render mazes (modes: vision_drag, vision_grid, text_grid, all)
python -m mazerunner visualize --input-dir data/dev --output-dir data/dev/renderings --mode all

# Run tests
pytest tests/ -v

# Serve maze environment (OpenEnv)
python -m mazerunner serve
# Configure via env vars: MAZE_MODE, MAZE_INSTANCE_DIR, MAZE_REWARD_MODE, MAZE_MAX_STEPS, MAZE_SEED
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
  __main__.py            ← CLI dispatcher (generate, visualize, serve)
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
- No external dependencies beyond numpy, Pillow, pytest, openenv-core, fastmcp, uvicorn
- Tests use pytest with class-based grouping and parametrize for grid sizes

## Testing

280 unit tests covering: seed determinism/uniqueness, maze graph properties (perfect maze invariant, reachability, BFS correctness), all 4 endpoint types with dead-end constraints, difficulty tier parameter ranges, serialization symmetry/sorting/schema, end-to-end pipeline determinism + file I/O, renderer utilities (parse_cell, hex_to_rgb, has_wall), text/vision_drag/vision_grid rendering (dimensions, pixel colors, markers, antialias), batch CLI integration, navigator module (grid movement/wall rejection, drag pixel paths/collision mask, history tracking, X marker overlays, breadcrumb rendering), and OpenEnv integration (reward functions, environment reset/step/tools, maze loading, image encoding).

5 e2e test scripts (`scripts/e2e_*.py`) covering: text_grid full episode with BFS solve, vision_grid PNG rendering, vision_drag pixel-path navigation, all 3 reward modes, and max_steps cutoff. Run with `python scripts/e2e_all.py`.

## Generated Data

Output goes to `data/` (gitignored). Each maze is `instances/maze_{i:06d}.json` containing id, grid dimensions, start/goal, sorted adjacency dict, shortest path, and metadata (tier, endpoint_type, difficulty_score, branching_factor).
