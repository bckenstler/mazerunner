# Effort sweeps (H5) — 25 pilot tasks × 3 trials per level

Run 2026-07-29/30 on the frozen pilot set (`results/pilot-tasks.txt`), frozen
dimension-free prompt, all routes via the LiteLLM proxy. `*` marks the setting
declared for the main run. pass@1 = mean success per attempt (n=75); pass@3 =
any-of-3 per task (n=25). Reasoning = median reasoning tokens where the
provider reports them.

| Model | Effort | pass@1 | pass@3 | Reasoning | Latency |
|---|---|---|---|---|---|
| GPT-5.6 Sol | low | 36% | 48% | — | 65s |
| | medium* | 56% | 80% | — | 85s |
| | high | 63% | 80% | — | 121s |
| | xhigh | **73%** | 84% | — | 146s |
| Claude Opus 5 | low | 8% | 8% | — | 32s |
| | medium | 7% | 8% | — | 67s |
| | high* | 13% | 24% | — | 122s |
| | xhigh | 12% | 16% | — | 205s |
| Gemini 3.6 Flash | none | 19% | 32% | 0 | 4s |
| | minimal | 20% | 28% | 0 | 4s |
| | low | 32% | 40% | 2,385 | 18s |
| | medium* | 32% | 48% | 8,135 | 45s |
| | high | 32% | 44% | 14,607 | 77s |
| Kimi K3 | low | 16% | 24% | — | 111s |
| | high* | 16% | 24% | — | 239s |
| | max | 28%† | 28% | — | 560s |

† **Kimi `max` is the one leg with a data-quality caveat.** 15 of its 75
attempts were request timeouts, excluded from the denominator per §3, so the
rate is 17/60 rather than 17/75 (an earlier draft reported 22.7%, dividing by
all 75). Worse, the losses are uneven: 11 of 25 tasks have fewer than 3 scored
attempts, so this point is both noisier than the others and not strictly paired
with them. Every other leg in this file lost at most one attempt to transport
and is unaffected. Treat the low→max difference as suggestive, not established;
if the Kimi effort curve becomes load-bearing, re-run `max` with the longer
timeout now in the runner.

## Three regimes, not one

1. **Effort-scaling (GPT-5.6 Sol).** Monotone across the whole ladder,
   +37pp low→xhigh, no saturation at the top. Deliberation is the binding
   constraint.
2. **Threshold (Gemini 3.6 Flash).** Any thinking is worth +12pp; more
   thinking is worth nothing — low/medium/high are identical (24/75 each)
   across a 6× spread in reasoning tokens. A short deliberation unlocks
   whatever this model can do; the rest is spent without effect.
3. **Flat / grounding-limited (Opus 5, Kimi K3).** Opus is inside noise at
   every level while latency grows 6×. Kimi is flat low→high (16% both) and
   may gain at max, at 5× the latency of low (9+ min/attempt) — see the caveat
   on that leg above before leaning on it.

The split supports H2: where accuracy is bounded by visual grounding
precision rather than planning, reasoning budget cannot buy accuracy. It also
means "reasoning effort" is not a single comparable axis across vendors — the
same nominal knob produces a scaling curve, a step function, and a flat line
on the identical task set.

## Economics

Gemini at `none` scores 19% in 4s; GPT at `xhigh` scores 73% in 146s — a
~37× latency ratio for ~4× the accuracy. Cost-per-solve belongs in the
economics table, separate from the capability ranking (per ANALYSIS_PLAN §3).

## Protocol note

The declared settings (vendor default / mid-rung) are unchanged by these
results, per the frozen policy in ANALYSIS_PLAN §2. GPT's declared `medium`
(56%) sits 17pp below its `xhigh` (73%); Opus, Gemini, and Kimi are at or
statistically indistinguishable from their best level. Re-declaring GPT alone
after observing its scores would be post-hoc tuning of one competitor, so
the sweep is reported as the H5 finding rather than used to re-tune the
leaderboard. A max-effort-for-everyone protocol remains a defensible
alternative, but requires sweeping every roster model (incl. Muse, Inkling)
and accepting Kimi at ~9 min/attempt.

## Provider knobs (verified by canary probe, 2026-07-30)

| Model | Accepted effort values | Notes |
|---|---|---|
| GPT-5.6 Sol | none, low, medium, high, xhigh, max | `minimal` rejected (older GPT-5 value) |
| Claude Opus 5 | low, medium, high, xhigh (+max) | via `extra_body.output_config.effort` |
| Gemini 3.6 Flash | none, minimal, low, medium, high | xhigh/max rejected; none+minimal = 0 reasoning tokens |
| Kimi K3 | low, high, max | no zero-reasoning setting (always-on) |
| Muse Spark 1.1 | minimal, low, medium, high, xhigh | none/max rejected; 258→905 reasoning tok minimal→xhigh |
| Inkling | none…max (OpenRouter) | effort **is** honored on BaseTen (90/237/181 tok low/high/max); Scale's Fireworks deployment lacks the thinking template |

## Floor checks (2026-07-30) — all six roster models now swept

Muse Spark and Inkling sit at the performance floor, so instead of full
ladders they got a single top-effort leg each (same 25 tasks × 3 trials) to
rule out "the score was a starved-budget artifact."

| Model | Setting | pass@1 | pass@3 | Median reasoning | Latency |
|---|---|---|---|---|---|
| Muse Spark 1.1 | medium* (declared) | 4% (3/75) | — | — | — |
| | xhigh (top of ladder) | 6.7% (5/75) | 3/25 | 8,221 | 67s |
| Inkling | default* (declared) | 0% (0/75) | 0/25 | 5,900 | — |
| | max | **0% (0/75)** | 0/25 | 12,925 | 74s |

Neither moves. Muse gains +2.7pp — inside the ±3pp noise band at n=75 — for
roughly an order of magnitude more reasoning. Inkling scores zero at maximum
effort while emitting a median 12,925 reasoning tokens, more than twice its
default spend: the failure is grounded in perception, not in reasoning
budget, and cannot be attributed to underfeeding the model.

With these two legs, **every model in the roster has been swept**, so the
ceiling-arm rule (below) applies uniformly rather than skipping the floor
models.
