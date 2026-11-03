# MazeRunner v2.0.0

v2 replaces the v1 task set with a difficulty ladder and re-runs the
leaderboard on current models.

**The set.** 192 tasks in 48 equally weighted cells: 8 maze families at 4 size
levels (L1 is v1's medium setting; an L4 route has 3.4× the turns in narrower
corridors), plus warped twins of the 4 grid families at every level. A warped
task shares its straight twin's route, endpoints and style; only the geometry
bends. Same task contract, prompt and frozen scorer as v1.

**The board.** 8 models, 4 attempts per task, plus a second effort setting for
four models on the L3/L4 cells: 7,776 attempts.

| Model | pass@1 [95% CI] |
|---|---|
| GPT-6 Astra · medium | 71.9% [68.5, 75.1] |
| GPT-6.1 Terra · medium | 69.7% [66.1, 73.2] |
| GPT-6.1 Sol · medium | 63.9% [60.4, 67.6] |
| Gemini 4 Argon · default | 59.1% [56.4, 62.0] |
| Claude Opus 5.5 · default | 34.2% [31.2, 37.4] |
| Kimi K3.1 Pro · high | 20.7% [17.8, 23.7] |
| Muse Wonder 1.0 · medium | 0.9% [0.1, 1.8] |
| Nemotron 3.5 Super VL · default | 0.0% [0.0, 0.0] |

**Findings.** Bending a maze costs more the larger it is (GPT-6 Astra: −9
points at L1, −64 at L4). Thinking longer gains 8–14 points on L3/L4 for GPT-6
and Claude Opus 5.5, and nothing for Gemini 4 Argon. The leaders' failures are
mostly a correct route drawn too close to a wall. Full study: `STUDY.md`.

**Assets.**
- `mazerunner-v2-traces.tar.gz`: every attempt (submission, evaluation,
  reasoning, raw payload, usage, provenance), the board, failure verdicts,
  tolerance re-scoring, and the exact provider settings.
- `SHA-256SUMS`.

**Also in this release.**
- `mazerunner board`: the v2 leaderboard and paired effort table from a
  merged run.
- `mazerunner run --resume-max-timeouts N`: stop replaying a unit the model
  has failed to answer N times; it scores as a fail.
- The merge records `unanswered_attempts` on units left with only transport
  failures, so the scoring rule applies from the merged file alone.

v1 (1,000 tasks) is retired from the benchmark. Its data, traces and study
remain at the v1.0.0 release.
