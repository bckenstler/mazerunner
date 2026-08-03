# MazeRunner

**Can a multimodal model draw its way out of a maze?**

The model sees one rendered maze image and must submit one continuous drag
path — normalized coordinates, a single tool call — from the cyan start badge
to the amber goal. Scoring sweeps a 3px pointer disk along the path against
the exact mask that stenciled the render. No grid, no move tokens, no partial
credit for a plan that reads well: the drag either stays in the corridor or it
doesn't.

A person does this without thinking. The best model configuration tested
fails 39% of the time.

📊 **[Full study](STUDY.md)** · 🔬 **[Trace viewer / replayer](docs/viewer/)** ·
🎬 [promo clip](mazerunner-promo.mp4) · 📦 traces in
[Releases](../../releases)

## Results

100 mazes × 8 attempts, frozen protocol, task-clustered 95% CIs. (The dataset
is 1,000 tasks; every evaluated number here is from the frozen 100-task dev
subset — the other 800 tasks ship unevaluated as headroom for future runs and
contamination auditing.)

| Model | pass@1 | 95% CI | pass@8 | Route progress |
|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | **60.6%** | [53.4, 67.4] | 86% | 0.719 |
| GPT-5.6 Sol @ medium | **48.4%** | [40.6, 55.8] | 74% | 0.612 |
| Gemini 3.6 Flash | **29.8%** | [22.2, 37.5] | 46% | 0.432 |
| Kimi K3 | **20.6%** | [14.6, 27.0] | 38% | 0.329 |
| Claude Opus 5 | **16.4%** | [10.2, 23.1] | 26% | 0.320 |
| Muse Spark 1.1 | **6.0%** | [2.8, 9.8] | 14% | 0.177 |
| Inkling | **0.0%** | [0.0, 0.0] | 0% | 0.029 |

Findings that survived the ablations (details and caveats in
[STUDY.md](STUDY.md)):

- **Test-time compute is not a portable knob.** The same effort ladder scales
  GPT (+37pp, unsaturated), steps once for Gemini (then exactly flat across a
  6× token spread), and does nothing for Claude or Kimi.
- **Showing a model its own mistake makes it worse.** After a failed attempt,
  seeing its own path with the collision marked beats a blind retry for *no*
  model; for mid-tier models it costs up to 24 points.
- **Style is a per-maze difficulty axis.** Identical mazes re-rendered in five
  styles: style's average effect is ~0, but the per-maze swing is 24pp.
- **Each model fails differently** (per-trace classification of all 4,140
  failures): GPT runs out of corridor *width*, Gemini and Claude drive through
  walls, Kimi announces its path is approximate and submits anyway, Inkling
  perceives the layout only coarsely — its reported positions are quantized to
  a 0.01 grid with ~25px of slop, fatal at this task's tolerances.
- Control: with a blank or wrong image, 0 passes in 700 attempts — the scores
  come from reading the specific picture.

## Quickstart

```bash
uv sync
uv run pytest -q                       # 255 tests, no network
uv run mazerunner validate             # rebuild + certify the smoke set
```

Run models on a few mazes (keys via environment only):

```bash
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=...
uv run mazerunner run --config configs/providers.example.json \
    --dataset datasets/v1/dev --mazes braided-easy-s0024,rooms-easy-s0039 --trials 1
```

Reproduce the full protocol:

```bash
uv run mazerunner run --config configs/providers.example.json \
    --dataset datasets/v1/dev --mazes-file evals/dev-eval-100.txt \
    --trials 8 --run-id myrun --order-seed 20260730
uv run mazerunner merge --runs 'results/myrun-*' --out results/myrun/merged
uv run mazerunner failuremodes --attempts results/myrun/merged/attempts.jsonl
```

Explore any recorded attempt locally:

```bash
uv run python scripts/make_viewer_data.py     # builds docs/viewer/data
uv run mazerunner serve                       # http://localhost:8639/viewer/
```

Operator detail (hardening checks, dataset engine, adding styles):
[docs/USAGE.md](docs/USAGE.md).

## Dataset

`datasets/v1/` — 1,000 tasks, 8 topology families × 20 certified style
archetypes, three splits:

| Split | Tasks | In repo | Notes |
|---|---|---|---|
| dev | 200 | ✅ | all public archetypes |
| test-public | 300 | ✅ | same cells, disjoint seeds |
| test-hidden | 500 | ❌ encrypted release asset | 6 unseen archetypes + held-out family×style cells |

Every task carries full provenance (seeds, resolved parameters, certification
metrics, rejection history); `mazerunner dataset verify` rebuilds tasks from
provenance and requires byte-identical masks. The rendered image can never
disagree with the scored mask: corridors are painted *through* the mask as a
stencil, then every render passes pixel-level fairness certification
(fail-closed) before it may ship.

The hidden split ships encrypted so future results on it are verifiable
without exposing the tasks. Please don't run the public splits through
training pipelines; the hidden split is how we'll know.

## Repository map

| | |
|---|---|
| `src/mazerunner/` | generators, renderer, certifier, evaluator, providers, analysis |
| `datasets/v1/` | the benchmark |
| `evals/` | frozen task subsets with selection manifests |
| `configs/` | provider configs (env-key indirection only) |
| `results/*.md` | findings: leaderboard, sweeps, ablations, failure modes |
| `results/figures/` | the study's charts |
| `docs/` | landing page + trace viewer (GitHub Pages) |
| `ANALYSIS_PLAN.md` | the analysis plan frozen before the main run, with dated amendments |

## Citation

```bibtex
@software{kenstler2026mazerunner,
  author  = {Kenstler, Bradley},
  title   = {MazeRunner: A Continuous-Control Benchmark for Multimodal Models},
  year    = {2026},
  version = {1.0.0},
  license = {MIT}
}
```

MIT — see [LICENSE](LICENSE).
