# MazeRunner v1 — analysis

Offline over stored attempts; no API calls. Resampling is clustered over
tasks (8 attempts on one maze are not 8 independent observations), and
model comparisons are paired per task.

## Leaderboard

| Model | pass@1 | 95% CI | pass@8 | Route progress |
|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | **60.6%** | [53.4, 67.4] | 86% | 0.719 |
| GPT-5.6 Sol @ medium | **48.4%** | [40.6, 55.8] | 74% | 0.612 |
| Gemini 3.6 Flash | **29.8%** | [22.2, 37.5] | 46% | 0.432 |
| Kimi K3 | **20.6%** | [14.6, 27.0] | 38% | 0.329 |
| Claude Opus 5 | **16.4%** | [10.2, 23.1] | 26% | 0.320 |
| Muse Spark 1.1 | **6.0%** | [2.8, 9.8] | 14% | 0.177 |
| Inkling | **0.0%** | [0.0, 0.0] | 0% | 0.029 |

## Paired comparisons (adjacent ranks)

| A vs B | Difference | 95% CI | p | McNemar p |
|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh vs GPT-5.6 Sol @ medium | +12.2pp | [7.1, 17.5] | 0.0000 | 0.0074 |
| GPT-5.6 Sol @ medium vs Gemini 3.6 Flash | +18.6pp | [10.2, 26.9] | 0.0000 | 0.0000 |
| Gemini 3.6 Flash vs Kimi K3 | +9.1pp | [3.8, 14.4] | 0.0007 | 0.1153 |
| Kimi K3 vs Claude Opus 5 | +4.2pp | [-2.8, 11.1] | 0.2484 | 0.0169 |
| Claude Opus 5 vs Muse Spark 1.1 | +10.4pp | [4.6, 16.8] | 0.0006 | 0.0075 |
| Muse Spark 1.1 vs Inkling | +6.0pp | [2.8, 9.8] | 0.0000 | 0.0001 |

## Coordinate fingerprints and localization

A model that measures the image emits irregular coordinates and lands on
the start badge. One that confabulates snaps to round numbers and misses it.

| Model | On 0.01 grid | Start error p50 | p90 |
|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 8.7% | 1.3px | 15px |
| GPT-5.6 Sol @ medium | 11.5% | 1.3px | 11px |
| Gemini 3.6 Flash | 38.9% | 2.1px | 4px |
| Kimi K3 | 19.7% | 1.8px | 3px |
| Claude Opus 5 | 27.3% | 2.1px | 7px |
| Muse Spark 1.1 | 19.2% | 8.2px | 30px |
| Inkling | 67.9% | 24.7px | 77px |

## Tolerance-curve rank stability

Every score depends on one arbitrary constant, the 3px pointer disk.
Re-scoring the stored submissions at other radii costs nothing and tests
whether the ranking is a property of the models or of that constant.

| Model | r=1 | r=2 | r=3 | r=5 | r=8 |
|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 68.4% | 65.5% | 60.6% | 48.9% | 31.4% |
| GPT-5.6 Sol @ medium | 55.9% | 53.4% | 48.4% | 38.8% | 26.8% |
| Gemini 3.6 Flash | 33.2% | 31.1% | 29.8% | 22.0% | 12.9% |
| Kimi K3 | 22.6% | 22.1% | 20.6% | 15.5% | 8.2% |
| Claude Opus 5 | 20.4% | 17.8% | 16.4% | 12.1% | 6.0% |
| Muse Spark 1.1 | 7.1% | 6.6% | 6.0% | 5.1% | 3.6% |
| Inkling | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**The ranking is identical at every tolerance**, with no inversions anywhere
between 1px and 8px. The leaderboard is not an artifact of the chosen radius.

## What the numbers say

**The field is not saturated.** The best configuration solves 61% of tasks on a
single attempt and 86% given eight; the hard tier drops it to 48%. Four of
seven models sit below 21%.

**Adjacent ranks are mostly separated, with one honest exception.** Kimi K3 vs
Claude Opus 5 is +4.2pp with a CI spanning zero (p=0.25) — those two should be
reported as tied on pass@1. (McNemar on any-of-8 does separate them, p=0.017,
so they differ in *reliability* while matching in per-attempt rate.) Every
other adjacent pair separates on both tests.

**Route progress tracks pass@1 but compresses the bottom.** Inkling scores
0.029 — it barely leaves the start badge — while Muse at 6% pass@1 reaches
0.177. The metric distinguishes "cannot start" from "starts and fails," which
pass@1 alone cannot.

**Fingerprints support H2 directly.** Inkling puts 67.9% of its coordinates on
an exact 0.01 grid and misses the start badge by 25px at the median: it is
emitting round numbers from a mental sketch rather than measuring pixels. The
GPT configurations are the mirror image — irregular coordinates, 1.3px badge
accuracy.

A subtler separation: **Gemini localizes better than GPT** (2.1px median, 4px
at p90, a tighter tail than GPT's 15px) yet scores half as well. Finding where
things are and tracing a corridor between them are different abilities, and
this benchmark measures the second one.

## Economics

Reported beside capability, never blended into it. Routes the proxy does not
publish a price for are marked unpriced rather than estimated.

| Model | Passes | Output tok/attempt | Mean latency | Total cost | Cost per solve |
|---|---|---|---|---|---|
| GPT-5.6 Sol @ medium | 387 | 2,542 | 54s | $66 | **$0.17** |
| GPT-5.6 Sol @ xhigh | 485 | 5,746 | 131s | $142 | $0.29 |
| Gemini 3.6 Flash | 238 | 8,831 | 45s | $72 | $0.30 |
| Claude Opus 5 | 131 | 5,522 | 93s | $117 | $0.89 |
| Kimi K3 | 165 | 9,662 | 198s | unpriced | — |
| Muse Spark 1.1 | 48 | 5,863 | 38s | unpriced | — |
| Inkling | 0 | 1,042 | 7s | unpriced | — |

**GPT at medium is the efficiency winner at $0.17 per solved maze**; the
ceiling arm buys 12 more points for 1.7× the cost per solve. Opus costs 5×
GPT-medium per solve. Gemini is cheap per token but its low pass rate makes it
no cheaper per *result* than GPT at xhigh.

Inkling's 1,042 output tokens and 7-second latency are themselves diagnostic:
it is not failing after long deliberation, it is answering almost immediately.

## Failure taxonomy across conditions

| Condition | pass | collision | wrong start | stopped short | no tool call |
|---|---|---|---|---|---|
| Main run | 26% | 64% | 9% | 0% | 0% |
| Blind (blank) | 0% | **0%** | **97%** | 0% | 3% |
| Resolution 0.5× | 22% | 64% | 13% | 0% | 0% |
| Resolution 2.0× | 35% | 56% | 9% | 0% | 0% |
| Dimensions disclosed | 39% | 60% | **1%** | 0% | 0% |

Two mechanisms show up here that the pass rates alone conceal.

**Blind failure is total, not degraded.** With no image, 97% of attempts do not
even begin on the start badge and *zero* reach a wall — models never get far
enough to collide. This is the cleanest possible statement of H4: without the
image there is no attempt, only a guess.

**Dimension disclosure works by fixing start localization.** It cuts
wrong-start errors from 9% to 1% while barely moving collisions (64%→60%).
Knowing the canvas size lets a model convert its perception of the badge into
correct normalized coordinates; it does nothing for tracing the corridor. That
explains H7's redistribution: the models it helps are those whose errors were
in the normalization step, and once that is fixed the remaining failure mode is
untouched.

Raising resolution moves the other lever: collisions fall 64%→56% while
wrong-start is unchanged. The two interventions repair different halves of the
task.
