# MazeRunner v1 — main run

100 tasks (`evals/dev-eval-100.txt`) x 8 trials x 7 legs = 5,600 attempts.
Frozen dimension-free prompt, per-leg randomized task order (seed 20260730).
pass@1 is macro-averaged over tasks; CI is a task-clustered bootstrap.

| Model | pass@1 | 95% CI | pass@8 | Route progress | Scored |
|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh (ceiling) | **60.6%** | [53.4, 67.5] | 86% | 0.714 | 794 |
| GPT-5.6 Sol · medium | **48.4%** | [40.6, 55.8] | 74% | 0.612 | 800 |
| Gemini 3.6 Flash · medium | **29.8%** | [22.4, 37.5] | 46% | 0.432 | 800 |
| Kimi K3 · high | **20.6%** | [14.8, 27.0] | 38% | 0.329 | 800 |
| Claude Opus 5 · high | **16.4%** | [10.2, 23.0] | 26% | 0.320 | 800 |
| Muse Spark 1.1 · medium | **6.0%** | [2.8, 9.8] | 14% | 0.177 | 800 |
| Inkling · default | **0.0%** | [0.0, 0.0] | 0% | 0.029 | 800 |

## By measured difficulty tier (pass@1)

| Model | easy | medium | hard |
|---|---|---|---|
| GPT-5.6 Sol @ xhigh (ceiling) | 67% | 67% | 48% |
| GPT-5.6 Sol · medium | 64% | 51% | 29% |
| Gemini 3.6 Flash · medium | 41% | 36% | 10% |
| Kimi K3 · high | 36% | 23% | 2% |
| Claude Opus 5 · high | 26% | 20% | 1% |
| Muse Spark 1.1 · medium | 13% | 4% | 1% |
| Inkling · default | 0% | 0% | 0% |

## Failure taxonomy (share of scored attempts)

| Model | collision | stopped short | wrong start | no tool call | schema |
|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh (ceiling) | 38% | 0% | 1% | 0% | 0% |
| GPT-5.6 Sol · medium | 48% | 0% | 3% | 0% | 0% |
| Gemini 3.6 Flash · medium | 70% | 0% | 0% | 0% | 0% |
| Kimi K3 · high | 78% | 0% | 1% | 0% | 0% |
| Claude Opus 5 · high | 83% | 0% | 0% | 0% | 0% |
| Muse Spark 1.1 · medium | 76% | 1% | 16% | 0% | 1% |
| Inkling · default | 56% | 0% | 42% | 1% | 1% |

## Notes

- **Two panels, one rule.** Per the pre-declared ceiling-arm rule, a model
  gets a second arm only where its effort sweep showed a gap over 2 SE from
  its declared setting. GPT-5.6 Sol was the only qualifier; for every other
  model the declared leg *is* the ceiling, demonstrated rather than assumed.
- **Inkling scores a true zero at n=800** with full reasoning traces, and its
  route progress of 0.029 shows it barely leaves the start badge. Combined
  with the max-effort floor check (0/75 at 12,925 median reasoning tokens),
  this is grounded failure, not starved budget.
- **6 attempts excluded**: gpt-xhigh on `organic-hard-s0187`, requeued once
  per protocol and still timing out. No model solved that task in 56
  attempts; it is the hardest item in the set (29 turns, 7 route branches).
