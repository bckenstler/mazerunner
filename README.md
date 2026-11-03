# MazeRunner

**Can a multimodal model draw its way out of a maze?**

The model sees one rendered maze image and must submit one continuous drag
path — normalized coordinates, a single tool call — from the cyan start badge
to the amber goal. Scoring sweeps a 3px pointer disk along the path against
the exact mask that stenciled the render. No grid, no move tokens, no partial
credit for a plan that reads well: the drag either stays in the corridor or it
doesn't.

192 mazes in eight layouts and four sizes, plus bent versions of the grid
layouts. The best model solves 72% on the first try; on the largest bent
mazes, none passes 30%.

📊 **[Full study](STUDY.md)** · 🌐 **[Site + leaderboard](https://bckenstler.github.io/mazerunner/)** ·
🔬 **[Trace viewer / replayer](https://bckenstler.github.io/mazerunner/viewer/)** ·
🎬 [promo clip](mazerunner-promo.mp4) · 📦 traces in [Releases](../../releases)

## Results

Each model tried all 192 mazes four times. The score is the share of tries
solved on the first attempt, with each of the 48 groups of mazes (8 layouts ×
4 sizes, plus bent versions of the 4 grid layouts) counting equally. Ranges
are 95% confidence intervals.

| Model | Overall | 95% range | Small mazes | Large mazes | Large grids, straight → bent | Within 4 tries |
|---|---|---|---|---|---|---|
| GPT-6 Astra · medium | **71.9%** | 68.5–75.1 | 86% | 66% | 86% → 38% | 86% |
| GPT-6.1 Terra · medium | **69.7%** | 66.1–73.2 | 88% | 59% | 85% → 35% | 81% |
| GPT-6.1 Sol · medium | **63.9%** | 60.4–67.6 | 82% | 58% | 76% → 26% | 80% |
| Gemini 4 Argon · default | **59.1%** | 56.4–62.0 | 81% | 49% | 68% → 14% | 71% |
| Claude Opus 5.5 · default | **34.2%** | 31.2–37.4 | 63% | 26% | 45% → 4% | 42% |
| Kimi K3.1 Pro · high | **20.7%** | 17.8–23.7 | 41% | 9% | 16% → 4% | 33% |
| Muse Wonder 1.0 · medium | **0.9%** | 0.1–1.8 | 3% | 0% | 0% → 0% | 2% |
| Nemotron 3.5 Super VL · default | **0.0%** | 0.0–0.0 | 0% | 0% | 0% → 0% | 0% |

Small is sizes 1–2 and large is sizes 3–4, over all eight layouts. The large
grids column compares the four grid layouts at sizes 3–4, straight and bent.

Thinking longer, on the large mazes, each model compared with itself maze by
maze:

| Model | Setting | Large mazes solved | Change (95% range) |
|---|---|---|---|
| Claude Opus 5.5 | default → max (1 try per maze) | 19% → 32% | +13.5 (7.8 to 19.3) |
| GPT-6 Astra | medium → xhigh | 57% → 70% | +12.8 (8.1 to 17.4) |
| GPT-6.1 Sol | medium → xhigh | 47% → 60% | +12.5 (8.3 to 16.7) |
| GPT-6.1 Terra | medium → xhigh | 51% → 59% | +7.6 (2.9 to 12.2) |
| Gemini 4 Argon | thinking off → default | 36% → 38% | +1.3 (-2.3 to 5.2) |

What the mazes revealed (details and caveats in [STUDY.md](STUDY.md)):

- **Bigger is harder; bent is much harder.** A bent maze has the same route,
  start, goal and look as its straight twin. GPT-6 Astra loses 9 points to the
  bend on the smallest mazes and 64 on the largest; Claude Opus 5.5 solves
  every small straight grid and 42% of the same mazes bent.
- **The best models know the way; their hand isn't steady enough.** Nine in
  ten of GPT-6 Astra's failures follow the right route and touch a wall
  anyway, and corridor width predicts its success better than any other
  feature. Claude cuts through walls; Kimi and Nemotron read walls as
  corridors.
- **Mistakes come early.** Even for the leader, 72% of failures first touch a
  wall in the first half of the route.
- **Thinking longer helps GPT and Claude by 8–14 points, and not Gemini.** No
  setting gets any model past 28% on the largest bent mazes.
- **The 3-pixel pointer doesn't decide the order.** Re-scored with pointers
  from 1 to 4 pixels, the order holds; only pairs already within each other's
  ranges swap at 5 pixels and up.

## Quickstart

```bash
uv sync
uv run pytest -q                       # offline, no API keys
uv run mazerunner validate             # rebuild + certify the smoke set
```

Run a model on a few mazes (keys via environment only):

```bash
export OPENAI_API_KEY=...
uv run mazerunner run --config configs/providers.example.json --providers openai \
    --dataset datasets/v2/test --mazes braided-L1-s0016,pipes-L2w-s0116 --trials 1
```

Reproduce the protocol for one configuration (the v2 settings for every
model are in `configs/providers.v2.json`):

```bash
uv run mazerunner run --config configs/providers.v2.json --providers gpt6-astra \
    --dataset datasets/v2/test --trials 4 --run-id myrun --order-seed 20261028
uv run mazerunner merge --runs 'results/myrun-*' --out results/myrun/merged
uv run mazerunner board --attempts results/myrun/merged/attempts.jsonl
```

Explore any recorded attempt locally:

```bash
uv run python scripts/make_viewer_data.py     # builds docs/viewer/data
uv run mazerunner serve                       # http://localhost:8639/viewer/
```

Operator detail (hardening checks, dataset engine, adding styles):
[docs/USAGE.md](docs/USAGE.md).

## Dataset

`datasets/v2/test` — 192 tasks: 8 layouts × 4 sizes (task ids say L1–L4) plus
bent ("warped", `w`) twins of the four grid layouts (rectilinear, braided,
pipes, island), 4 tasks per group, 14 visual styles. A size-4 route has 3.4×
the turns of a size-1 route, in narrower corridors. A bent task is built from the same
generation attempt as its straight twin, so only the geometry differs.

Every task carries full provenance (seeds, resolved parameters, certification
metrics, rejection history); `mazerunner dataset verify --full` rebuilds every
task from provenance and requires byte-identical masks. The rendered image can
never disagree with the scored mask: corridors are painted *through* the mask
as a stencil, then every render passes pixel-level fairness certification
(fail-closed) before it may ship. Build it yourself with
`mazerunner dataset build --config dataset.v2.config.json`.

v2 has no hidden split. Please don't run it through training pipelines.

## Repository map

| | |
|---|---|
| `src/mazerunner/` | generators, renderer, certifier, evaluator, providers, analysis |
| `datasets/v2/` | the benchmark |
| `configs/` | provider configs (env-key indirection only) and board metadata |
| `results/analysis/` | the board, failure verdicts, tolerance re-scoring, figure findings |
| `results/figures/` | the study's charts |
| `results/failure-modes.md` | the failure taxonomy and how it was applied to v2 |
| `docs/` | landing page + trace viewer (GitHub Pages) |

## Citation

```bibtex
@software{kenstler2026mazerunner,
  author  = {Kenstler, Bradley},
  title   = {MazeRunner: Can an AI Draw Its Way Out of a Maze?},
  year    = {2026},
  version = {2.0.0},
  license = {MIT}
}
```

MIT — see [LICENSE](LICENSE).
