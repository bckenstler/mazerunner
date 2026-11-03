# Releasing

The v2.0.0 sequence, and the checks that gate it.

## Pre-flight (all must be green)

```bash
uv run pytest -q                                  # offline
uv run ruff check . && uv run python scripts/check_docstrings.py
uv run python scripts/sanitize_release.py         # no internal references
uv run mazerunner validate --skip-tests           # generation determinism
uv run mazerunner dataset verify --full           # every v2 task rebuilds from provenance
uv run mazerunner board --out results/analysis/board.json
uv run python scripts/make_figures.py             # figures + results/analysis/findings.json
uv run python scripts/make_viewer_data.py         # viewer payload < 100MB
uv run python scripts/build_site.py               # leaderboard, effort table, captions
uv run python scripts/make_promo.py               # promo frames + mazerunner-promo.mp4
```

Sanity numbers: largest tracked blob < 5MB
(`git rev-list --objects --all | git cat-file --batch-check | sort -k3 -n | tail -3`).

## Release

1. Build the trace tarball (scrubbed of gateway routes and hostnames):
   ```bash
   uv run python scripts/build_release_assets.py
   ```
2. Merge to `main`, push, then tag and release:
   ```bash
   git tag v2.0.0 && git push origin v2.0.0
   gh release create v2.0.0 release-assets/mazerunner-v2-traces.tar.gz \
       release-assets/SHA-256SUMS \
       --title "MazeRunner v2.0.0" --notes-file docs/release-notes-v2.md
   ```
   The `pages.yml` workflow deploys `docs/` on the tag.
3. Confirm Pages is live: the landing page loads its leaderboard and effort
   table, the viewer replays an attempt, and a deep link
   (`viewer/#gpt6-astra/rectilinear-L4w-s0012/2`) restores state.

v1.0.0 stays published as its own release, with its traces and the encrypted
v1 hidden split.

## What never ships

- `.env`, any key material (CI greps for it)
- `litellm.config.json` (names the private gateway)
- `archive/` (local provenance store; public traces are the release assets)
- anything matching the markers in `scripts/sanitize_release.py`
