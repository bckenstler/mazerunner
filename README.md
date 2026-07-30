# MazeRunner Smoke Benchmark

A benchmark for multimodal models: given only a stylized maze PNG, submit one
continuous drag trajectory from a cyan start flag to an amber goal chest via a
single forced `submit_drag_path` tool call. Ground truth is derived
automatically at generation time — no human annotation anywhere.

Python port of the MazeRunner smoke prototype (see
`docs/` handoff for the full design). Managed with [uv](https://docs.astral.sh/uv/).

## Architecture

Four strictly separated layers (one canonical world per task; everything
derives from it):

1. **Structural** — eight seeded topology generators
   ([src/mazerunner/generators/](src/mazerunner/generators/)): rectilinear,
   braided, rooms, organic, cave (raster/cellular automata), radial, island,
   pipes. Pixel-weighted Dijkstra certifies every retained route.
2. **Rendering** — [render.py](src/mazerunner/render.py) paints eight distinct
   visual styles *through the scored mask as a stencil*, so the visibly open
   region always equals the traversable region.
3. **Agent contract** — [contract.py](src/mazerunner/contract.py): shared
   prompt, normalized `[0,1]` coordinates, one forced tool call, 2–512 points.
4. **Evaluator** — [evaluator.py](src/mazerunner/evaluator.py): every segment
   sampled at 0.75px with a swept 3px pointer disk against the hidden mask;
   endpoint radii; efficiency with an uncapped-ratio canary; min clearance.

### Hardening checks (fail-closed at generation)

- Mask ⊇ render by construction (mask is the render stencil) + test.
- Non-adjacent corridors must keep ≥ (w₁+w₂)/2 + 4px separation, so the mask
  can never contain shortcuts the graph doesn't model.
- `efficiency_raw` > 1.05 raises a loud warning (mask more permissive than
  graph).
- Endpoint acceptance radii shrink automatically until unambiguous; fail if
  below 12px.
- Masks are binary, antialiasing-free, and byte-identical across rebuilds
  (SHA-256 verified against `mazes/manifest.json` on every validate).
- Reference clearance ≥ pointer radius + 0.5px along the whole route.

## Validate before spending money

```bash
uv sync
```

```bash
uv run mazerunner validate
```

`validate` rebuilds all eight tasks twice (determinism check), runs every
fail-closed generation check, verifies mask hashes against the committed
manifest, scores each saved reference route through the exact evaluator used
for model output, rebuilds `mazes/contact-sheet.png` and
`mazes/reference-sheet.png`, and runs the 33-test suite. **It makes zero API
calls.**

Then eyeball the samples:

```bash
open mazes/contact-sheet.png mazes/reference-sheet.png
```

## Live smoke run

Keys are read from the environment only — never from files, never written to
results:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

Cheap canary first (one maze per provider, one call each):

```bash
uv run mazerunner run --providers anthropic --mazes rectilinear
```

```bash
uv run mazerunner run --providers openai,gemini --mazes rectilinear
```

Full smoke — 8 mazes × all keyed providers × 1 trial:

```bash
uv run mazerunner run
```

Options: `--providers a,b`, `--mazes x,y`, `--trials N`,
`--config smoke.config.json` (models/params are config, not benchmark
semantics — edit [smoke.config.json](smoke.config.json)). Providers whose env
key is missing are skipped with a message. Transport errors retry with
backoff; a returned-but-invalid path is never retried (one-shot planning, not
best-of-N).

Each run writes `results/<timestamp>/`:

- `attempts.jsonl` — one normalized row per attempt (latency, usage,
  response id, full evaluation);
- `summary.json` — per-provider/model aggregates (pass rate, mean efficiency
  on successes, mean latency);
- `overlays/` — the submitted trajectory drawn over each maze (green =
  success, red = failure with a collision marker).

## Other commands

```bash
uv run mazerunner generate
```

Rebuilds all task artifacts (`mazes/<family>/{input.png, mask.png, task.json,
ground-truth.json, reference-overlay.png}`), sheets, and manifest.

```bash
uv run pytest
```

Contract rejection cases, evaluator adversarial fixtures (wall-crossing
shortcut, legal-waypoints-illegal-segment, under-sampled bend, tangent
contact, thin wall, permissive-mask canary), generator fail-closed +
determinism + real-maze shortcut rejection, and provider response
normalization against mocked SDK payloads.

## Repository map

```
smoke.config.json          providers, model IDs, trials
src/mazerunner/
  contract.py              tool schema + prompt + submission validation
  geometry.py  solver.py   densification; BFS + weighted Dijkstra certification
  world.py                 canonical world, mask rasterization, fail-closed checks
  render.py  overlay.py    8 styles (mask-stencil), evaluator overlays
  evaluator.py             continuous swept-disk scorer
  io.py  build.py          artifacts, manifest hashing, validate pipeline
  runner.py  cli.py        live smoke runner, CLI
  generators/              8 topology families
  providers/               openai (Responses), anthropic (Messages), gemini adapters
mazes/                     generated tasks + contact/reference sheets + manifest
tests/                     33 offline tests
results/                   per-run outputs
```

## Dataset engine (v1)

Beyond the smoke set, the dataset pipeline builds stratified splits with
style decoupled from topology:

- **20 parameterized style archetypes** ([src/mazerunner/styles/](src/mazerunner/styles/)) —
  any archetype renders any topology; palettes/textures/decor resample per
  task from curated ranges.
- **Pixel-level fairness certification** ([certify.py](src/mazerunner/certify.py)) —
  fail-closed checks that walls are visible everywhere, nothing outside the
  mask reads as corridor at the boundary, interior decor never reads as a
  wall, and markers stay legible. Failing style samples are rejected and
  resampled, with every rejection logged.
- **Mask-certified geometric optimum** ([geodesic.py](src/mazerunner/geodesic.py)) —
  8-connected Dijkstra through the pointer-eroded mask plus swept-disk
  string-pulling; the efficiency denominator is the true shortest legal
  route, and the canary tightens to 1.02.
- **Geometry augmentation** ([augment.py](src/mazerunner/augment.py)) — flips,
  quarter turns, small free rotations (curved families), canvas presets,
  corridor-width jitter; applied pre-raster so ground truth can never desync.
- **Measured difficulty tiers** ([measure.py](src/mazerunner/measure.py)) —
  easy/medium/hard from certified geodesic length, turns, and branchiness.
- **Full reproducibility** — every task's `task.json` carries a provenance
  block: all seeds, the master-seed derivation rule, resolved sampled params,
  certification thresholds/metrics, and rejection history. `dataset verify`
  rebuilds tasks from provenance and demands byte-identical masks.

```bash
uv run mazerunner dataset build            # all splits from dataset.config.json
uv run mazerunner dataset verify           # sampled reproducibility + integrity check
uv run mazerunner dataset stats            # tiers, coverage matrix, length histograms
uv run mazerunner dataset sheet --split dev
```

Splits: `dev` (200, public), `test-public` (300, disjoint seeds), and
`test-hidden` (500, gitignored) holding out six archetypes and eight
family×archetype cells never seen publicly. Cross-split semantic leakage
(Weisfeiler-Lehman graph hash) is a build failure.

Evaluate providers against a split:

```bash
uv run mazerunner run --dataset datasets/v1/dev --dry-run
uv run mazerunner run --dataset datasets/v1/dev --providers anthropic
```

## Known limitations

Single-trial runs carry no statistical weight — use repeated trials and
bootstrap CIs for real comparisons. The tier mix is measured (relaxed slots
are flagged in provenance) — check `dataset stats` rather than assuming the
configured mix.
