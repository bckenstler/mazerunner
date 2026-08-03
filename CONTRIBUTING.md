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

## Documentation conventions

The docstrings here are narrative rationale, not API reference. Types live in
signatures; prose explains decisions and their consequences. When adding or
editing code:

- First line: a noun phrase naming what you get, or an imperative naming what
  it does — never "This function…". Nested dict shapes use the
  `provider -> task -> [values]` idiom.
- **No `Args:`/`Returns:`/`Raises:` sections.** The signature is the
  parameter documentation. A parameter worth explaining gets a sentence that
  also says why it exists.
- Every multi-line docstring must answer: *what breaks if this were done the
  obvious way?* If you can't name a consequence, write one line, not five.
- Cite evidence as bare repo paths (`results/failure-modes.md`,
  `tests/test_certify.py`) or real recorded numbers. Never invent an incident.
- A contract shared by many functions is stated once, in the module or package
  docstring; each function says only what differs.
- Trivial two-line helpers stay bare. A docstring that restates the signature
  is worse than none — it dilutes the docstrings that carry real invariants.
- Ceiling: ~8 lines per function docstring. Longer explanations belong in the
  module docstring or `docs/USAGE.md`.
- Tests: long assertion-style names (`test_completed_keys_survives_a_truncated_
  final_line`) substitute for docstrings. A test docstring is reserved for the
  fact you can't derive from the name — why the test exists, what real mistake
  it pins ("google-genai takes milliseconds; passing seconds would be a 1000x
  error"). Do not add docstrings that restate test names.
