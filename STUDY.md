# MazeRunner v2

**A continuous-control benchmark for multimodal models, rebuilt as a difficulty
ladder, and what it found.**

A model is shown a rendered maze and must submit one continuous drag path from
a cyan start badge to an amber goal badge, as normalized coordinates, in a
single tool call. Scoring sweeps a 3-pixel pointer disk along the submitted
polyline against the same mask that stenciled the image. There is no grid, no
move vocabulary, no partial credit for a plan that reads well.

Version 2 replaces the v1 set with a ladder: eight maze families at four sizes,
plus warped twins of the four grid families, 192 tasks in 48 equally weighted
cells. The best configuration, GPT-6 Astra at medium effort, solves 71.9% on a
single attempt. On the largest warped mazes no model passes 30%, whatever its
effort setting.

![Leaderboard](results/figures/01-leaderboard.png)

---

## 1. Why a v2

v1 (July 2026, 1,000 tasks, released as v1.0.0) separated the models of its
day: the best configuration solved 61% and four of seven were below 21%. By
October the newer models had caught up with it. In a refresh run on the same
100-task protocol, seven configurations scored between 88% and 94% with
overlapping intervals, so the top of the board no longer meant anything.

Rather than hand-pick harder mazes, v2 changes the generator settings that make
a maze hard and exposes them as axes:

- **Size.** Four levels per family. L1 is v1's medium setting; each level up
  adds rows and columns (or rings, rooms, islands) and narrows the corridors.
  An L4 route has 3.4× the turns of an L1 route and 2.3× the branch points.
- **Warp.** The four grid families (rectilinear, braided, pipes, island) also
  come warped at every level: the same maze drawn on a smoothly bent grid.
  A warped task is built from the same generation attempt as its straight
  twin, so the topology, start, goal and visual style are identical and only
  the geometry changes. 64 warped/straight pairs in all.

Every cell holds four tasks and carries equal weight in the headline, so no
family or size dominates the score.

---

## 2. Construction, and why you can trust the scores

The generator produces four artifacts per task from one seed: a latent world, a
deterministic solver route, an open-space mask, and a rendered PNG. The
**renderer paints corridors through the mask as a stencil**, so anything that
looks open is open by construction: a model can never be penalized for
trusting the picture.

Every task must also pass pixel-level **fairness certification** before it
ships: corridor boundaries legible along the whole mask outline, no
corridor-coloured region outside the mask, decoration inside a corridor never
wider than a third of it, and enough contrast at both badges. Certification
fails closed: a style is resampled up to six times, and a task that still
fails is dropped, with every rejection logged.

Four fixes went into the v2 build, each off by default so the v1 generators
still rebuild byte-identically:

- **Paired warp.** Grid generators apply the warp after picking the endpoints,
  so a warped twin keeps its straight task's route.
- **Pinhole fill.** Rasterization left isolated wall specks of 1–6 px at some
  corridor joins. v2 fills enclosed specks of 12 px or less (the smallest real
  wall is 116 px).
- **Seamless caves.** Cave cells were drawn as pixel-inclusive rectangles, and
  augmentation scaling opened 1-px seams between them that rendered as dashed
  lines and acted as walls. v2 caves are seamless. (About a quarter of v1's
  caves are affected; v1 is retired rather than patched.)
- **Drawable references.** The stored reference route is the fewest-point
  simplification that still passes the scorer, under the 512-point submission
  cap. The longest needs 349 points, so every task can be solved within the
  protocol.

The endpoint pair is drawn from the same far-endpoint band at every level, and
each straight topology is deduplicated against all 1,000 v1 tasks
(`datasets/v2/dedup-v1-graph-hashes.txt`). `mazerunner dataset verify --full`
rebuilds every task from its recorded provenance and demands byte-identical
masks.

**Dataset:** `datasets/v2/test`, 192 tasks, 14 style archetypes, all public.

---

## 3. Protocol

| | |
|---|---|
| Task set | all 192 tasks of `datasets/v2/test` |
| Design | every default-effort configuration × every task × 4 trials; the extra-effort configurations ran the 96 L3/L4 tasks |
| Prompt | the v1 prompt, unchanged, dimension-free |
| Order | shuffled per configuration, order seed 20261028 |
| Settings | `configs/providers.v2.json` |
| Time budget | 600 s per call; 1,800 s for GPT xhigh and Claude max |
| Run | 2026-10-29 to 2026-11-02, 7,776 attempts |

**Headline.** Macro pass@1: the mean over the 48 cells of the mean over a
cell's tasks of each task's pass rate. The extra-effort configurations are
scored on their 24 L3/L4 cells and reported in a separate table, never ranked
against full-ladder scores. Route progress, the fraction of the reference
route completed before failure, is reported beside pass@1 and never blended
into it.

**Intervals.** 95% intervals resample tasks within each cell. Four attempts on
one maze are not four independent observations; resampling attempts would
shrink the intervals by about half and manufacture significance. Comparisons
between two settings of one model are paired by task: both are resampled with
the same task draw, so task difficulty cancels.

**Attempts that never came back.** A call can fail in transit (a dropped
connection) or reach the model and not finish in time. Connection errors never
reached the model and are replayed until they get an answer. An attempt where
the model was asked twice and did not answer either time (a client timeout, or
the gateway resetting the long-running request) counts as a **fail**. This
applies to 32 of 7,776 attempts: 17 GPT-6 Astra xhigh, 12 GPT-6.1 Sol xhigh,
3 GPT-6 Astra medium, all on the longest routes. No attempt is missing from
the final set.

---

## 4. Main result

| | Model | pass@1 [95% CI] | pass@4 | L1 | L4 | L4 warped | Route progress |
|---|---|---|---|---|---|---|---|
| 1 | GPT-6 Astra · medium | **71.9%** [68.5, 75.1] | 86% | 93 | 59 | 19 | 81.8% |
| 2 | GPT-6.1 Terra · medium | **69.7%** [66.1, 73.2] | 81% | 92 | 47 | 22 | 78.3% |
| 3 | GPT-6.1 Sol · medium | **63.9%** [60.4, 67.6] | 80% | 87 | 51 | 12 | 75.7% |
| 4 | Gemini 4 Argon · default | **59.1%** [56.4, 62.0] | 71% | 89 | 37 | 6 | 69.2% |
| 5 | Claude Opus 5.5 · default | **34.2%** [31.2, 37.4] | 42% | 70 | 20 | 2 | 46.1% |
| 6 | Kimi K3.1 Pro · high | **20.7%** [17.8, 23.7] | 33% | 57 | 4 | 0 | 32.1% |
| 7 | Muse Wonder 1.0 · medium | **0.9%** [0.1, 1.8] | 2% | 5 | 0 | 0 | 8.0% |
| 8 | Nemotron 3.5 Super VL · default | **0.0%** [0.0, 0.0] | 0% | 0 | 0 | 0 | 4.1% |

L1 and L4 average the straight cells of all eight families; L4 warped averages
the four warped families. Astra and Terra are within each other's intervals,
as are Sol and Argon.

![Size ladder](results/figures/02-ladder.png)

Every model falls at every step of the ladder. By L4 the GPT-6 models have
lost 37–49% of their L1 score and Gemini 4 Argon 58%; Claude Opus 5.5 and Kimi
K3.1 Pro lose 71% and 93%.

---

## 5. Bending a maze is harder than enlarging it

![Warp](results/figures/03-warp.png)

The paired design answers a question v1 could only raise: are irregular mazes
hard because of their geometry, or because irregular families also happen to
have different topologies? Here the topology is held fixed and only the
geometry bends.

- For the leaders the warp cost grows with size: GPT-6 Astra loses 9 points at
  L1 and 64 at L4 (83% → 19%). A small bent maze is still easy; a large one is
  where they lose the route.
- For Claude Opus 5.5 the bending itself is the problem: it solves every
  straight L1 grid and 42% of their warped twins, and from L2 up it solves
  almost no warped maze.

![Cells](results/figures/05-cells.png)

Families are not equally hard, and not hard for the same models. Organic mazes
(curved, irregular corridors) defeat Claude from L1 and Gemini from L2;
island and pipes stay solvable for all three models shown. The leader's
weakest cells are the warped L4 grids and the irregular families (organic,
radial, cave) at L4.

### What predicts failure

![Regression](results/figures/06-regression.png)

A logistic regression of success on task features, per model, with a
task-clustered bootstrap. Route length and turn count are correlated; with
length held fixed, more turns means a denser maze, not a longer one, which is
why turns can come out positive.

- For GPT-6 Astra and Gemini 4 Argon, **corridor width** is the largest effect
  (+1.5 and +1.2 log-odds per SD): their failures are drawing failures, a line
  run too close to an edge.
- For Claude Opus 5.5, **warping** (−1.0) and **branching** (−1.0) dominate:
  it loses the route itself.
- Warping hurts all three after length, turns, branches and width are held
  fixed.

---

## 6. Thinking longer

![Effort](results/figures/04-effort.png)

Each model was run at a second effort setting on the L3/L4 cells and compared
task by task with its default.

| Model | Setting | L3/L4 pass@1 | Gain [95% CI] | Median output |
|---|---|---|---|---|
| Claude Opus 5.5 | default → max | 18.8% → 32.3% | **+13.5** [+7.8, +19.3] | 1.5K → 46K tokens |
| GPT-6 Astra | medium → xhigh | 56.8% → 69.5% | **+12.8** [+8.1, +17.4] | 2.0K → 6.0K |
| GPT-6.1 Sol | medium → xhigh | 47.1% → 59.6% | **+12.5** [+8.3, +16.7] | 2.0K → 6.8K |
| GPT-6.1 Terra | medium → xhigh | 51.0% → 58.6% | **+7.6** [+2.9, +12.2] | 1.6K → 5.6K |
| Gemini 4 Argon | thinking off → default | 36.2% → 37.5% | +1.3 [−2.3, +5.2] | 0.5K → 1.2K |

- GPT-6 at xhigh and Claude at max effort gain 8–14 points. The gain is on the
  straight mazes: warped L4 stays at or below 28% for every configuration.
- Claude Opus 5.5 at its default (adaptive thinking) writes under 350 tokens
  of reasoning per call. Its max setting writes 30× more output and costs 26×
  as much per attempt, for less than twice the score. It ran one attempt per
  maze (96 calls), so its interval is wider.
- Gemini 4 Argon thinks by default, deciding per call whether to (it thought
  on 55% of calls). Its higher settings had no effect through the serving
  gateway, so the comparison runs the other way: switching thinking off costs
  1.3 points, inside the noise, at half the price.

![Cost](results/figures/10-cost.png)

At list prices, GPT-6.1 Terra at medium is the cheapest strong configuration
($0.011 per L3/L4 attempt for 51%), and GPT-6 Astra at xhigh the most accurate
($0.36 for 70%). Muse Wonder 1.0 and Nemotron 3.5 Super VL had no published
price on the route used and are left off.

---

## 7. How models fail

![Failure modes](results/figures/07-failure-modes.png)

Every failed attempt was classified from its path geometry and, where the
model returned one, its reasoning trace; a trace-based label is assigned only
when the path agrees with it. Full taxonomy, per-model shares, and the two
classifier rules changed for v2: [results/failure-modes.md](results/failure-modes.md).

- Nine in ten of GPT-6 Astra's failures follow a route that stays inside the
  corridor and clip a wall with the 3px pointer. Most GPT-6 and Gemini
  attempts return no reasoning summary, so for them these labels are
  geometric.
- Claude Opus 5.5 cuts across walls (44% of its failures), and in 14% says it
  will commit to a best-guess path it could not verify.
- Kimi K3.1 Pro and Nemotron 3.5 Super VL most often read the walls as the
  corridors, and their traces say so.

![Progress](results/figures/08-progress.png)

Failures come early. Even for the leader, 72% of failed attempts first touch a
wall in the first half of the route; for Claude and Kimi it is 90% and 91%.

---

## 8. The scoring constant

![Tolerance](results/figures/09-tolerance.png)

Every score depends on one constant, the 3px pointer radius. Re-scoring every
stored submission at radii from 1 to 8 px (no new API calls) changes every
score, and the order of the six models with non-trivial scores holds from 1 to
4 px. At 5 px and above, two pairs swap: Astra and Terra, and Sol and Argon.
Both pairs already overlap in their intervals at 3 px, so the board does not
claim an order between them.

---

## 9. Decisions made during the run, and other caveats

The v2 design (cells, levels, warp, equal weights, extra effort on L3/L4 only)
was fixed before the main run. Decisions made after it started, in order:

- **2026-10-27: calibrating the levels.** The generator settings for each
  level were chosen by running GPT-6.1 Terra at medium and Gemini 4 Argon on
  159 candidate mazes, then a pilot with those two and GPT-6 Astra at xhigh,
  aiming for a ladder that falls steadily rather than for a target score.
  Settings were chosen per level, never task by task. Even so, three of the
  configurations on the board informed the design.
- **2026-10-29 to 10-31: outages.** Overnight network loss turned about 750
  calls into connection errors, and Kimi K3.1 Pro's only host returned rate
  limits for hours. Every affected call was replayed; none reached the model,
  so the replays are first attempts.
- **2026-10-31: the unanswered-attempt rule** (§3), decided while the
  extra-effort configurations were still running and before any final score
  was computed.
- **2026-11-02: gateway resets count as attempts.** Three units had one client
  timeout and one gateway reset of a long request. Both reached the model, so
  both count, and the three score as fails.
- **2026-11-02: the extra effort settings** for Claude Opus 5.5 (max, one
  trial) and Gemini 4 Argon (thinking off) were added after the GPT effort
  results were in, so that more than one vendor has an effort comparison.
- **2026-11-02: two failure-classifier rules** were changed after spot checks
  of v2 verdicts ([details](results/failure-modes.md#changes-from-the-v1-classifier)).

Other caveats:

- Serving routes: the OpenAI, Gemini, Anthropic and Meta models ran through an
  OpenAI-compatible LiteLLM gateway; Kimi K3.1 Pro (Moonshot) and Nemotron
  3.5 Super VL (Crusoe, bf16) ran through OpenRouter, pinned to one host each.
  A different host or serving stack may score differently.
- Four tasks per cell is a small sample. Cell-level numbers (§5) carry wide
  intervals; the headline averages 48 of them.
- v2 has no hidden split. Please don't train on it.

---

## 10. Reproducibility

Every number in this document, on the site, and in the figures is computed
from the merged run and the offline analyses; nothing is hand-typed.

| | |
|---|---|
| Full traces | release asset `mazerunner-v2-traces.tar.gz`: all 7,776 attempts with reasoning, raw payloads, usage and provenance, plus the board and analyses |
| Board | `results/analysis/board.json` (`mazerunner board --out`) |
| Analyses | `results/analysis/`: failure verdicts, tolerance re-scoring, figure findings |
| Settings | `configs/providers.v2.json`, `configs/board.v2.json` |

### Commands

```bash
uv run mazerunner dataset verify --full                # rebuild all 192 tasks from provenance
uv run mazerunner run --config configs/providers.v2.json --providers gpt6-astra \
    --dataset datasets/v2/test --trials 4 --run-id v2-main-gpt6-astra \
    --order-seed 20261028 --resume --resume-max-timeouts 2
uv run mazerunner merge --runs 'results/v2-main-*/shard-*' --out results/v2/final
uv run mazerunner board --out results/analysis/board.json
uv run mazerunner failuremodes                         # results/analysis/failure-modes.jsonl
uv run python scripts/make_figures.py                  # figures + findings
uv run python scripts/build_site.py                    # landing page data
```

To score from the release traces instead of re-running, extract the tarball
to `results/v2/final/` and start from `mazerunner board`.
