# CLAUDE.md

## Project Overview

MazeRunner is a benchmark for evaluating vision-based GUI agents on maze navigation. It has two main pipelines: **generation** (create maze images + ground truth) and **evaluation** (score submitted paths against ground truth).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate mazes
python -m mazerunner generate --output-dir data/dev --num-mazes 50 --master-seed 42 --tier-distribution 15,20,15

# Evaluate a submission
python -m mazerunner evaluate --submission sub.jsonl --gt-dir data/dev/gt --verbose

# Run tests
pytest tests/ -v

# Run evaluator validation script
PYTHONPATH=. python scripts/validate_evaluator.py
```

## Architecture

Two independent pipelines sharing `common/`:

```
common/types.py, rle.py  ← shared dataclasses and RLE encoding
       ↓                        ↓
generator/               evaluator/
  maze_graph.py            canonicalize.py  (parse submissions → dense polyline)
  masks.py                 metrics.py       (score paths using distance transforms)
  renderer.py              evaluate.py      (orchestrate per-maze evaluation)
  themes.py                schemas.py       (validate JSON structure)
```

### Generator Pipeline
`seed_utils.derive_seed()` → `difficulty.sample_difficulty_params()` → `maze_graph.build_maze()` → `masks.generate_wall_mask()` → `renderer.MazeRenderer.render()` → save PNG + GT JSON

### Evaluator Pipeline
`schemas.load_submission()` → `canonicalize.canonicalize()` (parse + clamp + densify) → `metrics.*` (clearance map, success@r, valid_frac, mono_score, etc.) → `EvalResult`

## Key Invariants

- Masks are generated from the logical maze structure, never from rendered images. The renderer and mask generator share the same cell-to-pixel geometry defined in `masks.py`.
- Cell-to-pixel mapping: `cell_size = corridor_width + wall_thickness`, `maze_origin = chrome_offset + wall_thickness`, cell interior at `origin + col * cell_size`.
- Densification (max 2px gap) is applied to all submitted paths before scoring to prevent wall-jumping.
- `check_endpoint` dilates region masks by 4px tolerance using distance transform.
- Seeds are derived via SHA-256 (`derive_seed`), not Python's random module, for platform independence.

## Git Workflow

Always checkout a new branch for every new feature before committing changes. Never commit directly to main.

## Code Style

- Python 3.11+, type hints on function signatures
- Dataclasses for structured data (`types.py`)
- No external dependencies beyond numpy, Pillow, scipy, pytest
- Tests use pytest with class-based grouping and parametrize for grid sizes

## Testing

97 tests covering: RLE round-trips, maze graph properties (perfect maze, reachability, BFS correctness), seed determinism, canonicalization encodings, all metrics, and end-to-end generation→evaluation. `scripts/validate_evaluator.py` runs 8 synthetic path scenarios (GT path, noisy, reversed, wall-cutting, half path, delta-encoded, cell-route, empty).

## Generated Data

Output goes to `data/` (gitignored). Each maze produces `images/{id}.png` and `gt/{id}.json`. The GT JSON contains RLE-encoded boolean masks (start, goal, free_space, wall), the solution polyline in pixel coordinates, and render_config needed for cell_route decoding.
