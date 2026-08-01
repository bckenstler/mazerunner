# Contributing

## Setup

```bash
uv sync
uv run pytest -q          # 255 tests, fully offline
uv run mazerunner validate --skip-tests
```

## Ground rules

- **Nothing ships without its trace.** A result submitted to the leaderboard
  must come with the full per-attempt records (`attempts.jsonl` including
  `submission`, `evaluation`, and the provider's reasoning where offered).
  Scores without traces are not accepted.
- **The scorer is frozen for v1.** Changes to `evaluator.py`, the pointer
  radius, or acceptance radii invalidate cross-run comparability and belong in
  a v2 branch, not a patch.
- **Fairness certification is fail-closed.** New style archetypes must pass
  `certify.certified_render` on every family they claim to support; see
  `tests/test_certify.py` for the adversarial fixtures a new style must survive.
- **Run the release gate.** `uv run python scripts/sanitize_release.py` must be
  green; CI enforces it.
- The hidden split (`datasets/v1/test-hidden`) is never sent to a provider API
  in public work and never committed. Don't ask; that's the point of it.

## Adding a model

Add an entry to `configs/providers.example.json` (env_key indirection only —
no keys in files), validate on 2–3 mazes, then run the frozen protocol:
`evals/dev-eval-100.txt`, 8 trials, randomized order with a recorded seed.
