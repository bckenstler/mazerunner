# MazeRunner v1 — ablations

Every ablation draws from `dev-eval-100`, so each attempt is paired with the
same task's main-run attempts. Baselines below are each model's main-run rate
**restricted to the ablation's own task subset**, not its full-set headline.

---

## 1. Blind (H4) — does the score come from seeing *this* image?

`evals/ablation-25.txt`, k=2, 7 legs, 700 attempts. Two variants: a neutral
canvas at the task's true dimensions, and another task's render assigned by a
tier-matched seeded derangement.

| Model | Sighted | Blank | Mismatched |
|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 68% | **0%** (rp 0.000) | **0%** (rp 0.000) |
| GPT-5.6 Sol @ medium | 54% | **0%** | **0%** |
| Gemini 3.6 Flash | 26% | **0%** | **0%** (rp 0.002) |
| Kimi K3 | 20% | **0%** | **0%** (rp 0.006) |
| Claude Opus 5 | 18% | **0%** | **0%** |
| Muse Spark 1.1 | 4% | **0%** | **0%** |
| Inkling | 0% | **0%** | **0%** |

**Zero passes in 700 scored attempts** (full coverage after 50 attempts lost to
a network outage were requeued). Route progress is ~0.000 throughout: a blind
model does not merely fail to finish, it does not get anywhere. The
prior-guessable floor is nil, so every number in the main leaderboard is
attributable to reading the specific image, and the weak models' scores
(Inkling 0.0%, Muse 6.0%) are real measurements rather than artifacts of a
guessable task.

The wrong-image variant matters as much as the blank one: it rules out models
succeeding from generic maze priors plus the prompt. Including the ceiling arm
was the point — a strong configuration scoring above floor while blind would
have been the single most damaging possible result, and it did not happen.

---

## 2. Input resolution — is perception resolution-bound?

`evals/ablation-25.txt`, k=2, 6 legs, 600 attempts. Send-time rescale only;
mask and ground truth untouched, so scoring is identical.

| Model | 0.5× | 1.0× (main) | 2.0× |
|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 51% | 68% | 65% |
| GPT-5.6 Sol @ medium | 32% | 54% | 58% |
| Gemini 3.6 Flash | 30% | 26% | 28% |
| **Kimi K3** | 17% | 20% | **42%** |
| Claude Opus 5 | 6% | 18% | 22% |
| Inkling | 0% | 0% | 0% |

**Kimi more than doubles at 2× (20% → 42%).** Read against its effort sweep —
flat at ~16% across low, high, and max — this is a clean dissociation: extra
reasoning bought Kimi nothing, extra pixels bought it a great deal. Its
bottleneck is perceptual acuity, not deliberation, which is H2 stated about as
sharply as this benchmark can state it.

Halving resolution hurts nearly everyone (Opus 18→6%, GPT-medium 54→32%), so
corridor-scale detail is genuinely load-bearing rather than incidental. Two
exceptions carry information: Gemini is flat across all three scales, and
**Inkling is 0% at every resolution**, which rules out perception resolution as
the explanation for its 42% start-badge miss.

---

## 3. Dimension disclosure (H7) — does telling the model the canvas size help?

Full `dev-eval-100`, k=2, 4 legs, 800 attempts, paired per task against each
model's main-run attempts under the frozen dimension-free prompt.

| Model | Frozen | With dimensions | Delta | Pilot (n=25) |
|---|---|---|---|---|
| GPT-5.6 Sol @ medium | 48% | 56% | **+8pp** | +7pp |
| Claude Opus 5 | 16% | 18% | +2pp | 0pp |
| GPT-5.6 Sol @ xhigh | 61% | 61% | **0pp** | — |
| Gemini 3.6 Flash | 30% | 22% | **−8pp** | −12pp |

Replicates the pilot within 1–4 points at four times the sample. Disclosure
**redistributes rather than lifts**: GPT gains roughly what Gemini loses and the
field nets near zero, which is why the frozen prompt stays dimension-free.

The ceiling arm adds something the original two-model design could not see: at
`xhigh` the effect vanishes entirely. Enough reasoning budget lets GPT establish
scale from the image alone, so the disclosure becomes redundant — the benefit at
`medium` is a workaround for a deficit that more deliberation also fixes.

---

## 4. Feedback (H6) — can a model correct its own failed drag?

`evals/ablation-50.txt`, ≤4 turns, stop at first success, 5 legs, true
multi-turn conversations. Feedback is the failure category plus the model's own
path drawn over the same maze with a ⊗ at the stop point. No oracle geometry.

Surface view — ≤4 informed attempts roughly matches what 8 independent
attempts already buy:

| Model | Solved ≤4 | Any-of-8 baseline | Mean turns to solve |
|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 85% | 82% | 1.3 |
| GPT-5.6 Sol @ medium | 70% | 72% | 1.2 |
| Gemini 3.6 Flash | 32% | 38% | 1.2 |
| Kimi K3 | 31% | 32% | 1.9 |
| Claude Opus 5 | 24% | 22% | 1.7 |

That comparison flatters feedback, because most episodes are solved on turn 1
and never use it. The actual test restricts to episodes whose first turn
failed, and asks whether seeing your own error beats simply trying again blind
(the model's own pass@1 on these tasks):

| Model | Failed turn 1 | Rescued | Rescue rate | Blind retry | Delta |
|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 15 | 10 | **67%** | 63% | +3pp |
| GPT-5.6 Sol @ medium | 20 | 5 | 25% | 48% | **−24pp** |
| Gemini 3.6 Flash | 37 | 3 | 8% | 24% | **−16pp** |
| Kimi K3 | 31 | 5 | 16% | 18% | −2pp |
| Claude Opus 5 | 34 | 4 | 12% | 14% | −2pp |

**Seeing its own failed path makes most models worse than starting fresh.**
Only the strongest configuration beats a blind retry, and only barely.

The route-progress column explains the mechanism: every model advances further
along the certified route after feedback (+0.15 to +0.58) while converting
almost none of that into passes. They correct *locally* toward the marked
collision point and then fail somewhere else — repairing the reported error
rather than re-perceiving the corridor. The overlay anchors them to the failed
route instead of prompting a fresh measurement.

This inverts the naive reading of H6. Closed-loop correction is not a general
capability that weaker models merely lack; for mid-tier models the loop is
actively harmful, and independent resampling is the better use of the same
budget.

⚠️ GPT-xhigh's 67% rests on only 15 failed-first-turn episodes, because it
rarely fails first. That interval is wide and the number should not be
published without more data behind it.

---

## 5. Style-swap (H3)

Running; results appended on completion.

---

## Method notes

- Blind's `mismatched` mapping is fixed-point-free and tier-matched, so no task
  sees its own image and the substitute carries comparable difficulty priors.
- Rescaling needs no evaluator change: submissions are normalized coordinates
  and `evaluate_task` never reads the image bytes. Each row records both the
  sent and the true dimensions.
- The frozen dimension-free prompt is retained for the resolution ablation;
  disclosing true dimensions alongside a rescaled image would confound it.
