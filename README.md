# MazeRunner

**Can a multimodal model draw its way out of a maze?**

The model sees one rendered maze image and must submit one continuous drag
path — normalized coordinates, a single tool call — from the cyan start badge
to the amber goal. Scoring sweeps a 3px pointer disk along the path against
the exact mask that stenciled the render. No grid, no move tokens, no partial
credit for a plan that reads well: the drag either stays in the corridor or it
doesn't.

v2 is a difficulty ladder: eight maze families at four sizes, plus warped
twins of the four grid families. The best model solves 72% of it; on the
largest warped mazes, none passes 30%.

📊 **[Full study](STUDY.md)** · 🌐 **[Site + leaderboard](https://bckenstler.github.io/mazerunner/)** ·
🔬 **[Trace viewer / replayer](https://bckenstler.github.io/mazerunner/viewer/)** ·
🎬 [promo clip](mazerunner-promo.mp4) · 📦 traces in [Releases](../../releases)

## Results

192 mazes × 4 attempts. pass@1 is the mean over 48 equally weighted cells
(12 maze types × 4 sizes); 95% CIs resample tasks within cells.

| Model | pass@1 | 95% CI | pass@4 | L1 | L4 | L4 warped |
|---|---|---|---|---|---|---|
| GPT-6 Astra · medium | **71.9%** | [68.5, 75.1] | 86% | 93 | 59 | 19 |
| GPT-6.1 Terra · medium | **69.7%** | [66.1, 73.2] | 81% | 92 | 47 | 22 |
| GPT-6.1 Sol · medium | **63.9%** | [60.4, 67.6] | 80% | 87 | 51 | 12 |
| Gemini 4 Argon · default | **59.1%** | [56.4, 62.0] | 71% | 89 | 37 | 6 |
| Claude Opus 5.5 · default | **34.2%** | [31.2, 37.4] | 42% | 70 | 20 | 2 |
| Kimi K3.1 Pro · high | **20.7%** | [17.8, 23.7] | 33% | 57 | 4 | 0 |
| Muse Wonder 1.0 · medium | **0.9%** | [0.1, 1.8] | 2% | 5 | 0 | 0 |
| Nemotron 3.5 Super VL · default | **0.0%** | [0.0, 0.0] | 0% | 0 | 0 | 0 |

More effort, on the 24 L3/L4 cells, paired by task:

| Model | Setting | L3/L4 pass@1 | Gain [95% CI] |
|---|---|---|---|
| Claude Opus 5.5 | default → max (1 attempt per maze) | 18.8% → 32.3% | +13.5 [+7.8, +19.3] |
| GPT-6 Astra | medium → xhigh | 56.8% → 69.5% | +12.8 [+8.1, +17.4] |
| GPT-6.1 Sol | medium → xhigh | 47.1% → 59.6% | +12.5 [+8.3, +16.7] |
| GPT-6.1 Terra | medium → xhigh | 51.0% → 58.6% | +7.6 [+2.9, +12.2] |
| Gemini 4 Argon | thinking off → default | 36.2% → 37.5% | +1.3 [−2.3, +5.2] |

Findings (details and caveats in [STUDY.md](STUDY.md)):

- **Bending a maze is harder than enlarging it.** Each warped maze has the
  same route, endpoints and style as a straight twin. GPT-6 Astra loses 9
  points to the warp at L1 and 64 at L4; Claude Opus 5.5 solves every straight
  L1 grid and 42% of their warped twins.
- **The leaders' failures are drawing failures.** Nine in ten of GPT-6
  Astra's failed attempts follow a route that stays in the corridor and clip a
  wall; corridor width is its strongest predictor of success. Claude loses the
  route itself, and Kimi and Nemotron read the walls as corridors.
- **Thinking longer helps GPT and Claude, by 8–14 points, and not Gemini.**
  Switching Gemini 4 Argon's thinking off costs it 1.3 points, inside the
  noise. No setting gets any model past 28% on the largest warped mazes.
- **The scoring constant doesn't decide the order.** Re-scored at pointer
  radii 1–4px the order holds; only pairs that already overlap at 3px swap at
  5px and above.

v1 (1,000 tasks, July 2026) is retired; it remains available at the
[v1.0.0 release](../../releases/tag/v1.0.0).

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

`datasets/v2/test` — 192 tasks: 8 families × 4 size levels (L1–L4) plus
warped twins of the four grid families (rectilinear, braided, pipes, island),
4 tasks per cell, 14 style archetypes. L1 is v1's medium setting; an L4 route
has 3.4× the turns in narrower corridors. A warped task is built from the same
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
  title   = {MazeRunner: A Continuous-Control Benchmark for Multimodal Models},
  year    = {2026},
  version = {2.0.0},
  license = {MIT}
}
```

MIT — see [LICENSE](LICENSE).
