# Releasing

The one-time v1.0.0 sequence, and the checks that gate it.

## Pre-flight (all must be green)

```bash
uv run pytest -q                                  # 255 tests, offline
uv run python scripts/sanitize_release.py         # no internal references
uv run mazerunner validate --skip-tests           # generation determinism
uv run mazerunner dataset verify                  # provenance rebuild
uv run mazerunner evalset verify                  # frozen sets re-solve
uv run python scripts/make_viewer_data.py         # viewer payload < 100MB
uv run python scripts/build_site.py               # leaderboard from results
```

Sanity numbers: fresh clone < 150MB, largest tracked blob < 5MB
(`git rev-list --objects --all | git cat-file --batch-check | sort -k3 -n | tail -3`).

## Release

1. Create the GitHub repo, `git remote add origin …`, `git push -u origin main`.
2. Enable Pages: Settings → Pages → GitHub Actions (the `pages.yml` workflow
   deploys `docs/` on tag).
3. Build assets, including the encrypted hidden split:
   ```bash
   export MAZERUNNER_HIDDEN_KEY="$(openssl rand -hex 32)"
   uv run python scripts/build_release_assets.py --hidden
   # store MAZERUNNER_HIDDEN_KEY somewhere durable and PRIVATE (password
   # manager). Publishing it later is how hidden-split results get verified.
   ```
4. Tag and release:
   ```bash
   git tag v1.0.0 && git push origin v1.0.0
   gh release create v1.0.0 release-assets/*.tar.gz \
       release-assets/*.enc release-assets/SHA-256SUMS \
       --title "MazeRunner v1.0.0" --notes-file docs/release-notes-v1.md
   ```
5. Confirm Pages is live: landing page loads, viewer replays an attempt, a
   deep link (`viewer/#gemini/braided-easy-s0104/6`) restores state.

## Decrypting the hidden split (maintainer only)

```bash
openssl enc -d -aes-256-cbc -pbkdf2 \
    -in mazerunner-v1-test-hidden.tar.gz.enc -out test-hidden.tar.gz \
    -pass env:MAZERUNNER_HIDDEN_KEY
```

## What never ships

- `.env`, any key material (CI greps for it)
- `datasets/v1/test-hidden/` unencrypted, or the hidden QC sheet, or the
  hidden build log (all gitignored; the log names task seeds)
- `archive/` (local provenance store; public traces are the release assets)
- anything matching the markers in `scripts/sanitize_release.py`
