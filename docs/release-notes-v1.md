## MazeRunner v1.0.0

**Can a multimodal model draw its way out of a maze?** One image, one
continuous drag path, swept-disk scoring against the exact mask that stenciled
the render. The best configuration tested fails 39% of the time.

**In this release**

- 1,000-task dataset (dev 200 + test-public 300 in-repo; test-hidden 500 as an
  encrypted asset below), every task with full seed/parameter provenance and
  pixel-level fairness certification
- The complete study: 5,600-attempt main run over 7 model configurations,
  effort sweeps, five ablations, per-trace failure-mode classification —
  [STUDY.md](https://github.com/bckenstler/mazerunner/blob/main/STUDY.md)
- [Interactive trace viewer](https://bckenstler.github.io/mazerunner/viewer/) —
  replay any of the 5,594 recorded attempts, with the model's reasoning beside
  the maze
- Full raw traces for every attempt, attached below

**Headline results**

| Model | pass@1 | pass@8 |
|---|---|---|
| GPT-5.6 Sol · xhigh | 60.6% | 86% |
| GPT-5.6 Sol · medium | 48.4% | 74% |
| Gemini 3.6 Flash · medium | 29.8% | 46% |
| Kimi K3 · high | 20.6% | 38% |
| Claude Opus 5 · high | 16.4% | 26% |
| Muse Spark 1.1 · medium | 6.0% | 14% |
| Inkling · default | 0.0% | 0% |

- Test-time compute is not a portable knob: the same effort ladder scales GPT
  (+37pp), steps once for Gemini, and does nothing for Claude or Kimi.
- Showing a model its own mistake makes its retry *worse* than a fresh
  attempt, for every model tested.
- Style is a per-maze difficulty axis: ~0 average effect, 24pp per-maze swing.
- Each model fails in its own way — GPT runs out of corridor width, Gemini
  and Claude drive through walls, Kimi submits paths it says are approximate,
  Inkling reports positions from a coarse quantized sketch.

**Assets**

| File | Contents |
|---|---|
| `mazerunner-v1-traces-main.tar.gz` | main run: 5,600 attempts with full provider payloads + failure-mode verdicts |
| `mazerunner-v1-traces-ablations.tar.gz` | blind, resolution, style-swap, dimensions, feedback episodes |
| `mazerunner-v1-traces-sweeps.tar.gz` | all effort-sweep legs |
| `mazerunner-v1-test-hidden.tar.gz.enc` | the hidden split, AES-256 encrypted; key withheld so future results are verifiable |
| `SHA-256SUMS` | checksums for everything above |

Serving-route prefixes from the gateway used during the study were rewritten
to public model names; the mapping is documented inside each tarball. MIT.
