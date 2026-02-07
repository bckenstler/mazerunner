# MazeRunner Benchmark

A benchmark for evaluating whether vision-based GUI agents can plan and execute valid drag paths through mazes. Given an image of a maze with marked start (green) and goal (red) positions, agents must produce a pixel-coordinate path that navigates from start to goal without crossing walls.

## Quick Start

```bash
pip install -r requirements.txt

# Generate a 50-maze dataset
python -m mazerunner generate --output-dir data/dev --num-mazes 50 --master-seed 42 --tier-distribution 15,20,15

# Evaluate a submission
python -m mazerunner evaluate --submission submission.jsonl --gt-dir data/dev/gt
```

## Maze Generation

Generates orthogonal perfect mazes with randomized DFS, renders them as PNG images with macOS-style UI chrome, and produces ground truth JSON with binary masks and solution paths.

```bash
python -m mazerunner generate \
  --output-dir data/dev \
  --num-mazes 50 \
  --master-seed 42 \
  --tier-distribution 15,20,15
```

**Arguments:**
- `--output-dir`: Where to write `images/` and `gt/` subdirectories
- `--num-mazes`: Total number of mazes (default: 50)
- `--master-seed`: Seed for deterministic generation (default: 42)
- `--tier-distribution`: Comma-separated count per difficulty tier, must sum to `--num-mazes`

### Difficulty Tiers

| Tier | Grid Size | Corridor Width | Min Solution Length |
|------|-----------|----------------|---------------------|
| 1 (easy) | 5-8 rows, 7-12 cols | 28-40px | 8 cells |
| 2 (medium) | 10-16 rows, 14-22 cols | 16-26px | 20 cells |
| 3 (hard) | 18-28 rows, 25-40 cols | 8-16px | 40 cells |

### Visual Themes

Four color themes are randomly applied: `light_classic`, `dark_modern`, `blueprint`, `parchment`.

### Output Format

Each maze produces:
- `images/{id}.png` — Rendered maze image with UI chrome
- `gt/{id}.json` — Ground truth with RLE-encoded masks and solution polyline

Ground truth JSON schema:
```json
{
  "id": "000001",
  "seed": 12345,
  "image_size": {"w": 640, "h": 400},
  "maze_family": "orthogonal",
  "difficulty": {"tier": 2, "corridor_width_px": 20, "grid_rows": 12, "grid_cols": 18, "wall_thickness_px": 4},
  "regions": {
    "start_mask_rle": {"counts": [...], "shape": [H, W]},
    "goal_mask_rle": {"counts": [...], "shape": [H, W]},
    "free_space_mask_rle": {"counts": [...], "shape": [H, W]},
    "wall_mask_rle": {"counts": [...], "shape": [H, W]}
  },
  "gt": {
    "solution_polyline": [[x, y], ...],
    "solution_length": 1234.5
  },
  "render_config": {
    "corridor_width": 20,
    "wall_thickness": 4,
    "chrome_height_top": 38,
    "chrome_width_left": 0,
    "theme_name": "blueprint"
  }
}
```

## Submission Format

Submissions are JSON or JSONL files. Each entry has an `id` matching a maze and a `prediction` with an encoding type and data.

### Supported Encodings

**polyline** — Direct pixel coordinates:
```json
{
  "id": "000001",
  "prediction": {
    "encoding": "polyline",
    "data": {"points": [[100, 200], [102, 200], [102, 220], ...]}
  }
}
```

**delta** — Cumulative deltas from a start point:
```json
{
  "id": "000001",
  "prediction": {
    "encoding": "delta",
    "data": {"start": [100, 200], "deltas": [[2, 0], [0, 20], ...]}
  }
}
```

**cell_route** — Maze cell coordinates (converted to pixel centers using render_config):
```json
{
  "id": "000001",
  "prediction": {
    "encoding": "cell_route",
    "data": {"cells": [[0, 0], [0, 1], [1, 1], ...]}
  }
}
```

**multi_segment** — Concatenation of multiple segments with different encodings:
```json
{
  "id": "000001",
  "prediction": {
    "encoding": "multi_segment",
    "data": {
      "segments": [
        {"encoding": "polyline", "data": {"points": [[100, 200], [150, 200]]}},
        {"encoding": "delta", "data": {"start": [150, 200], "deltas": [[0, 30]]}}
      ]
    }
  }
}
```

## Evaluation

```bash
python -m mazerunner evaluate \
  --submission submission.jsonl \
  --gt-dir data/dev/gt \
  --buffer-radii 0,1,2,4,8 \
  --verbose \
  --output results.json
```

All submitted paths are canonicalized to dense polylines (points interpolated so no gap exceeds 2px) before scoring.

### Metrics

**Primary — Success@r**: Binary pass/fail at buffer radius r pixels. A path succeeds if:
1. First point is within the start region (4px tolerance)
2. Last point is within the goal region (4px tolerance)
3. Every point has wall clearance >= r pixels

Reported at r = 0, 1, 2, 4, 8.

**Secondary metrics:**
| Metric | Description |
|--------|-------------|
| `valid_frac@r` | Fraction of path points with clearance >= r (continuous, even when success is 0) |
| `min_clearance` | Minimum distance-to-wall along the entire path |
| `goal_distance` | Euclidean distance from last path point to nearest goal pixel |
| `path_length` | Total path length in pixels |
| `length_regret` | `(predicted_length - gt_length) / gt_length` |
| `mono_score` | Monotonic progress score; penalizes backtracking along the GT path direction. Range [0, 1]. |

## Reproducibility

All generation is deterministic given a master seed. Each maze's seed is derived via SHA-256: `SHA256(f"{master_seed}:{index}")[:4]`. Running generation twice with the same arguments produces bit-identical output.

## Project Structure

```
mazerunner/
  requirements.txt
  mazerunner/
    __init__.py
    __main__.py              # CLI dispatcher
    generate.py              # Generation CLI
    evaluate_cli.py          # Evaluation CLI
    common/
      types.py               # Shared dataclasses
      rle.py                 # RLE encode/decode for binary masks
    generator/
      seed_utils.py          # Deterministic seed derivation
      difficulty.py          # Tier configs
      maze_graph.py          # DFS generation, BFS solving
      masks.py               # Binary mask generation + pixel geometry
      renderer.py            # PIL-based image rendering
      themes.py              # Color themes
    evaluator/
      schemas.py             # Submission/GT validation
      canonicalize.py        # Path encoding → dense polyline
      metrics.py             # All metric computations
      evaluate.py            # Evaluation pipeline
  scripts/
    validate_evaluator.py    # Synthetic path validation tests
  tests/
    test_rle.py
    test_maze_graph.py
    test_canonicalize.py
    test_metrics.py
    test_seed_determinism.py
    test_end_to_end.py
  data/                      # Generated output (gitignored)
```

## Testing

```bash
# Unit and integration tests
pytest tests/ -v

# Evaluator validation with synthetic paths
python scripts/validate_evaluator.py
```

## Design Notes

- **Masks are pixel-exact**: Generated from logical maze structure, not from rendered images. Anti-aliasing only affects the visual PNG.
- **Densification prevents cheating**: Submitted paths are interpolated to max 2px gaps before scoring, so agents can't jump over walls with sparse points.
- **Perfect mazes**: Every generated maze has exactly one path between any two cells (DFS spanning tree), guaranteeing a unique shortest solution.
- **Structured for extension**: The `maze_family` field and modular architecture support future polar/curvy maze types and multi-step evaluation modes.
