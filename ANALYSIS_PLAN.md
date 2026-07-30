# MazeRunner v1 — Pre-registered Analysis & Research Plan

Status: frozen before main-run launch. Changes after launch require a dated
amendment note in this file.

## 1. Research questions

RQ1. Can frontier multimodal models execute continuous, collision-free drag
trajectories from rendered mazes, and what separates the models that can from
those that cannot?
RQ2. Is the binding constraint visual grounding precision (sub-corridor-width
coordinate accuracy) rather than planning or reasoning budget?
RQ3. How much of task difficulty is topology vs. rendering style, when the two
are factorized?
RQ4. Do measured scores actually derive from perception of the specific image
(vs. task priors)?
RQ5. How do reliability (pass@1), capability (pass@8), and closed-loop
correction (feedback mode) relate per model?

Positioning: prior maze work is discrete/grid-token based (MazeBench,
SpatialEval Maze-Nav, MazeEval); grounding work scores points or boxes
(Point-Bench, Point-It-Out, DragOn). MazeRunner's contributions: continuous
trajectories with swept-disk collision physics, mask-certified geometric
optima, and factorized topology×style with pixel-level fairness certification.

## 2. Main run protocol (frozen)

- Task set: `dev-eval-100` — 100 stratified tasks from `datasets/v1/dev`
  (~12–13/family, tier mix 30/40/30 measured, all 14 public archetypes ≥6),
  committed as `evals/dev-eval-100.txt` with selection seed. test-public and
  test-hidden are never sent to provider APIs in this phase.
- Design: uniform flat — every model × every task × exactly 8 trials.
  No tiering, no early stopping, no adaptivity.
- Prompt: frozen dimension-free variant (`prompt_variant: "frozen"`).
- Task order randomized per model leg.
- Roster (8): gpt-5.6-sol, claude-opus-5, gemini-3.6-flash, kimi-k3,
  qwen3-vl-235b-a22b-thinking, muse-spark-1.1, inkling, (+ claude-fable-5 if
  budget approved). Reasoning policy: each model at its vendor default or the
  midpoint of its effort ladder; effective spend verified from usage telemetry
  and reported.
- Serving: provider-pinned where routing exists; serving stack recorded per
  attempt. Batch APIs permitted (cost-only change).

## 3. Scoring (frozen)

Headline (two numbers, never blended):
1. **pass@1** — mean success over 8 trials, macro-averaged over tasks;
   bootstrap 95% CI over tasks. No partial credit.
2. **route progress** — fraction of certified geodesic completed before
   failure (pass = 1.0; missed start = 0; else first-collision point projected
   onto the geodesic). Mean over attempts, macro over tasks.

Also reported: pass@k for k ≤ 8 (unbiased estimator); the pass@1↔pass@8 gap as
the reliability measure.

Threshold-free robustness (offline re-scoring of stored submissions; no API
calls): success-vs-pointer-radius curves (r ∈ {1,2,3,5,8}px) and
success-vs-acceptance-radius curves; report rank stability across tolerances.

Diagnostic panel (reported, unranked): start/goal localization error CDFs;
efficiency on passes vs geometric optimum; minimum clearance on passes;
coordinate-quantization fingerprint (decimal-grid distribution of submitted
points); flakiness (share of tasks with 1–7/8 successes); tokens, latency, and
cost-per-solve (economics table, separate from capability ranking).

Statistics: all model-vs-model claims use paired per-task tests (paired
bootstrap on per-task success counts; McNemar for any-success). Slices
(tier/family/archetype) always reported alongside global means.

Scoring rules (predeclared):
- Transport failures: excluded from denominators, re-queued once; unresolved
  ones listed in the run manifest.
- Schema violations, missing tool calls, prose answers: count as failures.
  A structured call in the provider's own wire format normalized by the
  adapter counts as a call; free text does not.
- Efficiency-canary firing quarantines the task for manual inspection before
  its attempts are scored.

## 4. Hypotheses

- **H1 (difficulty validity):** success declines monotonically in measured
  difficulty. Test: logistic regression of success on normalized geodesic
  length, turns, min clearance, branchiness; report per-tier pass rates.
- **H2 (grounding is binding):** clearance-at-collision and coordinate
  quantization predict success better than reasoning-token spend. Test:
  per-attempt logistic model comparison; localization CDFs; trace audit.
- **H3 (style vs topology):** rendering style is a measurable difficulty
  covariate, separable from topology. Test: paired style-swap ablation (§5.2),
  mixed-effects variance decomposition.
- **H4 (vision validity):** without the true image, performance collapses.
  Test: blind ablation (§5.1). Any residual defines the prior-guessable
  floor; weak-model scores are reported relative to it.
- **H5 (reasoning dose-response):** within-model effort sweeps show
  saturating returns for measurers and ~zero slope for gist-perceivers.
- **H6 (feedback asymmetry):** overlay-feedback retries improve models whose
  traces measure pixels, not models that confabulate geometry (per
  See-Point-Refine-style closed-loop grounding).
- **H7 (dimension disclosure redistributes):** disclosing canvas size helps
  pixel-planners, hurts proportional reasoners, nets ~zero (pilot: +7/0/−12pp).

## 5. Ablations (priority order, budgets ≈)

1. **Blind ablation** (~$30): 25 tasks × k=2 × all models × {blank image,
   mismatched image}. Tests H4; wrong-image variant separates "uses vision"
   from "uses this image."
2. **Style-swap pairs** (~$150–250): ~20 topologies × ~5 archetypes, identical
   topo/augmentation seeds; 3–4 models × k=3. Tests H3. Unique to this
   benchmark's factorized engine.
3. **Effort sweeps** (~$100): Kimi (low/high/max) and GPT (minimal→xhigh) on
   25 tasks × k=3. Tests H5.
4. **Feedback mode** (~$150–250, separate leaderboard): ≤4 attempts, stop at
   first success; feedback = failure category + the model's own failed-attempt
   overlay with ⊗ at the stop point (what a real drag shows; no oracle
   geometry). Metrics: success@≤4, attempts-to-success. Tests H6.
5. **Dimension disclosure confirmation** (~$60): GPT + Gemini only (pilot's
   significant movers), 100 tasks × k=2 with `--include-dimensions`.
6. **Input-resolution ablation** (cheap): send-time PNG upscaling (mask and
   ground truth untouched) — isolates encoder resolution from task physics.
7. Documented findings requiring no new runs: forced tool_choice silently
   disables thinking (Claude Opus 5, Kimi K3, rejected outright by Meta);
   serving-stack variance (Inkling: same weights, three hosts, reasoning from
   0 to 12K tokens).

## 6. Deliverables

- Leaderboard (pass@1 + route progress, CIs, slices) + economics table.
- Tolerance-curve rank-stability figure.
- Failure-taxonomy breakdown per model; trace-grounded case studies
  (measurement vs confabulation).
- Ablation reports per §5.
- Full release: per-attempt records (submissions, traces, raw responses),
  task provenance, this plan, and the run manifests.

## Amendment 2026-07-29

§5.3 Effort sweeps extended to include Claude Opus 5:
`output_config.effort` ∈ {low, medium, high, xhigh} (adaptive thinking on;
`max` excluded for cost), via the LiteLLM route with extra_body passthrough
(verified: effort modulates output 276→1,770 tokens low→xhigh on the canary).
Same design as other sweeps: 25-task pilot set × 3 trials per level.

## Amendment 2026-07-30

§5.3 Effort sweeps completed for four models (25 pilot tasks × 3 trials per
level; full table and interpretation in `results/effort-sweeps.md`):
GPT-5.6 Sol {low, medium, high, xhigh}, Claude Opus 5 {low, medium, high,
xhigh}, Gemini 3.6 Flash {none, minimal, low, medium, high}, Kimi K3
{low, high, max}. Verified effort-value ladders per provider are recorded in
that file. GPT's `minimal` leg was voided (invalid value for this model — 75
transport 400s, no attempts reached the model) and replaced by the `none`
floor being available on Gemini only.

H5 outcome: three regimes rather than a single saturating curve —
effort-scaling (GPT, +37pp, unsaturated), threshold (Gemini, +12pp for any
thinking then exactly flat across a 6× token spread), and flat/
grounding-limited (Opus within noise at 6× latency; Kimi flat low→high,
+7pp at max for 5× latency).

Declared main-run settings are unchanged (frozen policy, §2). Not swept:
Muse Spark and Inkling, which sit at the performance floor where a flat
curve is uninformative.

## Amendment 2026-07-30 (b)

Floor checks completed for the two remaining roster models, so all six are now
swept. Muse Spark 1.1 at `xhigh` (top of its verified ladder): 5/75 = 6.7% vs
4% at declared `medium`, +2.7pp — inside noise at n=75, with median 8,221
reasoning tokens. Inkling at `max` via BaseTen: 0/75 with median 12,925
reasoning tokens (>2× its default spend). Both confirm grounded failure rather
than budget starvation.

Pre-declared main-run rule (stated before any main-run data is collected):
a model receives a second "ceiling arm" at its best swept setting only where
its sweep showed a gap from the declared setting exceeding 2 standard errors.
Applying it to the completed sweeps, GPT-5.6 Sol is the only qualifying model
(medium 56% vs xhigh 73%, >3 SE). For Opus 5, Gemini 3.6 Flash, Kimi K3, Muse
Spark, and Inkling the declared leg is the ceiling arm. Results will be
reported as a two-panel leaderboard (as-declared / at-ceiling).
