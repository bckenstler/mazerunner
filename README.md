# MazeRunner

A benchmark for evaluating vision-based GUI agents on maze navigation. MazeRunner generates canonical maze instances across 3 complexity tiers and renders them in 3 benchmark modes for different agent interaction paradigms.

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Dependencies: numpy, Pillow, pytest (for testing).

## Quick Start

Generate a dataset of 1000 mazes:

```bash
python -m mazerunner generate --output-dir data/dev --num-mazes 1000 --master-seed 42 --tier-distribution 300,400,300
```

Render all benchmark modes:

```bash
python -m mazerunner visualize --input-dir data/dev --output-dir data/dev/renderings --mode all
```

Output structure:

```
data/dev/
  instances/
    maze_000000.json
    maze_000001.json
    ...
  renderings/
    vision_drag/   ← corridor-style PNG images
    vision_grid/   ← cell-and-wall grid PNG images
    text_grid/     ← ASCII .txt files
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
    color_schemas.py     ← 10 predefined color schemas
    serialization.py     ← MazeGrid → canonical JSON
  renderer/
    base.py              ← Shared utilities and config dataclasses
    vision_drag.py       ← Corridor maze image + eval API (cell_to_pixel_center/rect)
    vision_grid.py       ← Cell-and-wall grid image
    text_grid.py         ← ASCII text grid rendering
  navigator/
    base.py              ← InteractionResult, HistoryEntry + MazeNavigator ABC
    grid_navigator.py    ← L/R/U/D cell movement with wall validation
    drag_navigator.py    ← Pixel path movement with collision mask
    rendering.py         ← State overlay rendering (X marker, dotted breadcrumbs)
  generate.py            ← CLI entry point + dataset orchestration
  visualize.py           ← Batch-render CLI
  __main__.py            ← CLI dispatcher (generate, visualize)
```

### Generator Pipeline

Seeds are derived via SHA-256 for platform independence. Each maze follows this pipeline:

`derive_seed` → `sample_difficulty_params` → `generate_maze` → `place_endpoints` → `solve_bfs` → `maze_grid_to_instance` → JSON

All generated mazes are perfect mazes (spanning trees): every cell is reachable, and there is exactly one path between any two cells.

## Benchmark Modes

| Mode | Format | Agent Type | Interaction |
|------|--------|------------|-------------|
| `vision_drag` | PNG image | Pixel-drag agents | Drag paths as pixel coordinate sequences |
| `vision_grid` | PNG image | Directional agents | L/R/U/D cell steps |
| `text_grid` | ASCII text | Text-based agents | L/R/U/D cell steps |

## Difficulty Tiers

| Parameter | Tier 1 (Easy) | Tier 2 (Medium) | Tier 3 (Hard) |
|-----------|---------------|-----------------|----------------|
| Grid rows | 5-8 | 10-16 | 18-28 |
| Grid cols | 7-12 | 14-22 | 25-40 |
| Min solution length | 8 | 20 | 40 |

Difficulty score (1-9): `min(9, 1 + int(8 * path_len / (rows * cols)))`

## Output Format

Each maze JSON file contains:

```json
{
  "id": "maze_000000",
  "grid_rows": 8,
  "grid_cols": 10,
  "start": "0,3",
  "goal": "7,9",
  "adjacency": {"0,0": ["0,1", "1,0"], ...},
  "shortest_path_cells": ["0,3", "1,3", ...],
  "metadata": {
    "tier": 1,
    "endpoint_type": "edge-edge",
    "difficulty_score": 3,
    "path_length": 15,
    "branching_factor": 2.1,
    "color_schema": {"wall": "#1a1a2e", ...}
  }
}
```

Cell coordinates use `"row,col"` string keys (0-indexed). Adjacency keys and neighbor lists are sorted for determinism.

## Navigator API

Navigators provide a programmatic interaction interface for agent evaluation:

```python
from mazerunner.renderer.base import load_instance
from mazerunner.navigator.grid_navigator import GridNavigator
from mazerunner.navigator.drag_navigator import DragNavigator

instance = load_instance("data/dev/instances/maze_000000.json")

# Grid navigation (L/R/U/D cell steps)
nav = GridNavigator(instance, render_mode="text_grid")
result = nav.interact("RRD")  # move right, right, down
print(result.valid, result.position, result.finished)
state = nav.render()  # returns text grid with X marker

# Drag navigation (pixel coordinate paths)
nav = DragNavigator(instance)
result = nav.interact([[10.0, 15.0], [30.0, 15.0], [30.0, 40.0]])
state = nav.render()  # returns PIL Image with breadcrumbs and X marker
```

## Development

```bash
pytest tests/ -v
```
