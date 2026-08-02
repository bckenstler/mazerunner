# MazeRunner v1 — ablations

Every ablation draws from `dev-eval-100`, so each attempt is paired with the
same task's main-run attempts. Baselines below are each model's main-run rate
**restricted to the ablation's own task subset**, not its full-set headline.

---

## 1. Blind (H4) — does the score come from seeing *this* image?

**The intuition:** a model could score above zero without doing the task —
memorized maze layouts, priors about where goals usually sit, or geometry that
"usually works" on benchmark mazes. Sending a blank canvas measures that
prior-only floor directly; sending *another maze's* image is the stricter
version, catching a model that uses generic vision but not this picture. If
either scores above zero, the leaderboard partly measures guessing.

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

**The intuition:** vision encoders see a downsampled image, so a corridor a
few pixels wide may simply not survive into the model's internal
representation — in which case no amount of reasoning can recover it. Resending
the identical maze at 2× and 0.5× isolates that channel: if a model improves
with pixels but not with thinking (Kimi), its ceiling was perceptual acuity all
along. And a model stuck at zero *regardless of resolution* (Inkling) cannot
blame the encoder — its failure is upstream of pixel budget, which is exactly
what its coordinate fingerprints (0.01-grid snapping, 25px badge misses)
independently suggest: it is not reading the image at any resolution.

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

**The intuition:** submissions are normalized coordinates, so a model must
implicitly know the canvas size to convert "the badge is at pixel (55, 527)"
into (0.069, 0.712). Telling it the true dimensions removes that conversion
step. If disclosure helps, the model was measuring pixels correctly and
fumbling the normalization; if it hurts, the model was reasoning
proportionally and the numbers distracted it. Either way it localizes *where*
in the pipeline errors live — which is why the effect redistributes instead of
lifting.

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

**The intuition:** the agentic-loop assumption is that showing a model its
error makes the retry better than a fresh guess. This tests the cheapest
version of that loop on a task where the error is *visually exact*: here is
your path, here is the pixel where it left the corridor. If closed-loop
correction works anywhere, it should work here — the fix requires no
diagnosis, only re-perception of one marked location. The baseline that makes
the test fair is the model's own pass@1: an informed retry must beat an
uninformed one, or the loop is worthless.

**Message history:** true multi-turn — each turn's context contains the full
prior conversation. At turn 2 the model sees 2 images (the original maze and
the overlay of its first attempt); at turn 3, 3 images, and so on. Verified
from stored usage: input tokens grow monotonically per turn (e.g. 1.4K → 19K →
25K → 30K). The original image is never dropped.

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

All 250 episodes, bootstrap CI over episodes:

| Model | Failed turn 1 | Rescued | Rescue rate | Blind retry | Delta | 95% CI |
|---|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 20 | 12 | 60% | 63% | −3pp | [40, 80] |
| GPT-5.6 Sol @ medium | 20 | 5 | 25% | 48% | **−24pp** | [5, 45] |
| Gemini 3.6 Flash | 37 | 3 | 8% | 24% | **−16pp** | [0, 19] |
| Kimi K3 | 42 | 5 | 12% | 18% | −6pp | [2, 21] |
| Claude Opus 5 | 44 | 4 | 9% | 14% | −5pp | [2, 18] |

**No model benefits from seeing its own failed path.** The strongest
configuration is statistically indistinguishable from a blind retry — its
interval [40, 80] contains the 63% baseline — and every other model is worse,
GPT-medium and Gemini clearly so.

(An earlier partial reading of this ablation, at 15 failed-first-turn episodes,
put GPT-xhigh at 67% and appeared to show a small benefit. The complete set of
20 puts it at 60%, below its own baseline. The apparent advantage was noise.)

The route-progress column explains the mechanism: every model advances further
along the certified route after feedback (+0.15 to +0.58) while converting
almost none of that into passes. They correct *locally* toward the marked
collision point and then fail somewhere else — repairing the reported error
rather than re-perceiving the corridor. The overlay anchors them to the failed
route instead of prompting a fresh measurement.

This inverts the naive reading of H6. Closed-loop correction is not a capability
the strong models have and the weak ones lack — **none of them have it here.**
For mid-tier models the loop is actively harmful, and independent resampling is
a strictly better use of the same budget.

⚠️ Interpretation limits. Failed-first-turn counts are small for the strongest
legs (20 for both GPT configurations, because they rarely fail first), so their
intervals are wide; the safe claim is "no detectable benefit," not "harmful."
And this tests *one* feedback design — failure category plus the model's own
overlaid path. A richer signal (an explicit corridor-boundary trace, or several
prior attempts at once) might do better. What these data rule out is the
cheapest and most natural version working at all.

---

## 5. Style-swap (H3) — is rendering style a difficulty axis of its own?

**The intuition:** every rendered benchmark quietly entangles two questions —
is the *structure* hard, or is the *picture* hard? Because MazeRunner generates
topology and rendering from independent seeds, the same maze can be re-painted
with a byte-identical scored mask, making the picture the only variable. If
pass rates move across renderings of one fixed maze, "style difficulty" is
real and a fixed-style benchmark would be silently mis-attributing it to
topology.

20 topologies × 5 archetypes = 100 variants, k=3, 5 legs, 1,494 scored
attempts. Every variant in a pair-group is asserted to share its source's
`mask_sha256` and to still pass the certified reference route, so within a
group the *only* thing that changes is how the maze is painted.

**Marginal style effect is small; topology dominates.**

| Archetype (topology held fixed) | pass@1 |
|---|---|
| desert-canyon | 38.3% |
| pencil-sketch | 36.5% |
| glow-cavern | 36.3% |
| forest-path | 35.5% |
| candy-pastel | 32.1% |

Two-way decomposition over the (topology × style) grid:

| Source | Share of variance | Effect range |
|---|---|---|
| **Topology** | **85.7%** | 92pp |
| Style (main effect) | 0.6% | 6pp |
| Topology × style interaction | 13.7% | — |

Averaged over mazes, style barely matters: the best and worst archetypes differ
by 6 points, against a 92-point spread across topologies. But the marginal view
understates it. **Within a single fixed maze, the best and worst styles differ
by 24 points on average and up to 51 points**, and the interaction term is
13.7% — more than twenty times the style main effect.

So H3 resolves in a specific way: style is not a global difficulty dial, it is
a *per-maze* one. No archetype is uniformly harder; particular renderings are
hard for particular topologies. That is exactly the effect the factorized
engine was built to expose and that a benchmark pairing one style to one maze
would report as topology difficulty.

It also has a practical consequence for the benchmark: because style effects
are interaction-dominated, sampling many topologies with one style each would
give an unbiased mean but understate per-task variance — the dataset's
independent style sampling per task is the right construction.

---

## Method notes

- Blind's `mismatched` mapping is fixed-point-free and tier-matched, so no task
  sees its own image and the substitute carries comparable difficulty priors.
- Rescaling needs no evaluator change: submissions are normalized coordinates
  and `evaluate_task` never reads the image bytes. Each row records both the
  sent and the true dimensions.
- The frozen dimension-free prompt is retained for the resolution ablation;
  disclosing true dimensions alongside a rescaled image would confound it.
