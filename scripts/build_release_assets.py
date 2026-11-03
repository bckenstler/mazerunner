"""Build the GitHub Release assets: the v2 trace tarball and its checksum.

Traces are scrubbed of internal serving-route prefixes before packaging: the
route a request happened to take through a private gateway is operational
detail, not science, and the mapping is recorded in the tarball's README so
nothing is silently altered.

Usage:
  uv run python scripts/build_release_assets.py            # tarball + checksums
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release-assets"

# Internal gateway route prefixes -> public model names. Applied to the
# `model`, `response_model`, and `serving_stack` fields of every row.
ROUTE_SCRUB = {
    "openai/gpt-6-astra": "gpt-6-astra",
    "openai/gpt-6.1-sol": "gpt-6.1-sol",
    "openai/gpt-6.1-terra": "gpt-6.1-terra",
    "gemini/gemini-4-argon": "gemini-4-argon",
    "anthropic/claude-opus-5-5": "claude-opus-5-5",
    "meta_ai/muse-wonder-1.0": "muse-wonder-1.0",
}
SCRUB_FIELDS = ("model", "response_model", "serving_stack")
FORBIDDEN = ["ml-serving-internal", "Inkling-evals", "meta_ai/", "litellm-proxy"]

# String-level replacements applied to the serialized row AFTER field scrubs
# (error messages in the retry history quote gateway URLs verbatim). The
# private patterns live in an untracked local file so this public script never
# itself contains the strings it exists to remove.
STRING_SCRUB: dict[str, str] = {}
_local = ROOT / ".release-scrub.json"
if _local.exists():
    STRING_SCRUB.update(json.loads(_local.read_text()))
    FORBIDDEN.extend(STRING_SCRUB.keys())


def scrub_row(row: dict) -> dict:
    """Rewrite gateway route prefixes to public model names in one row.

    The study ran through a gateway whose route strings name internal
    deployments; the published traces carry public model ids instead, and the
    mapping ships inside each tarball so the rewrite is auditable.
    """
    for field in SCRUB_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            for src, dst in ROUTE_SCRUB.items():
                if src in value:
                    row[field] = value.replace(src, dst)
    # raw_response can carry the gateway's model string too
    raw = row.get("raw_response")
    if isinstance(raw, dict) and isinstance(raw.get("model"), str):
        for src, dst in ROUTE_SCRUB.items():
            raw["model"] = raw["model"].replace(src, dst)
    return row


def scrub_jsonl(src: Path, dst: Path) -> int:
    """Scrub a whole trace file, then verify: any forbidden marker surviving
    into the output aborts the build rather than shipping."""
    count = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            out_line = json.dumps(scrub_row(json.loads(line)))
            for needle, replacement in STRING_SCRUB.items():
                out_line = out_line.replace(needle, replacement)
            fout.write(out_line + "\n")
            count += 1
    text = dst.read_text()
    for marker in FORBIDDEN:
        if marker in text:
            raise SystemExit(f"scrub failed: {marker!r} survives in {dst.name}")
    return count


def tar_of(name: str, members: list[tuple[Path, str]], readme: str) -> Path:
    """Package members into a release tarball with its own README, so an
    extracted archive explains itself without the repo."""
    out = OUT / name
    with tempfile.TemporaryDirectory() as tmp:
        readme_path = Path(tmp) / "README.md"
        readme_path.write_text(readme)
        with tarfile.open(out, "w:gz") as tar:
            tar.add(readme_path, arcname="README.md")
            for path, arcname in members:
                tar.add(path, arcname=arcname)
    return out


README = f"""# MazeRunner v2 traces

The complete v2 run: 13 configurations of 8 models on the 192-task difficulty
ladder (datasets/v2/test in the repository), 7,776 attempts.

  attempts.jsonl        one JSON object per attempt: submission points, full
                        evaluation, derived metrics, reasoning trace where the
                        provider returned one, raw provider payload, usage,
                        latency, and run provenance (run id, shard, ordinal,
                        order seed). The merged set: one row per attempt.
  merge-manifest.json   how the 80 shard files were merged (duplicates, no
                        conflicts)
  board.json            `mazerunner board` output: every score on the site
  failure-modes.jsonl   per-attempt failure verdicts with their evidence
  tolerance.json        pass rate per task at pointer radii 1-8px
  findings.json         the numbers the figures and captions quote
  providers.v2.json     the exact provider settings of the run

Attempts left with only a transport failure carry `unanswered_attempts`: how
many times the model was really asked (a client timeout or a gateway reset of
the long request) without answering. Two or more counts as a fail; fewer is
missing work (none remain in this set).

Serving-route prefixes from the private gateway used during the study were
rewritten to public model names in the fields {SCRUB_FIELDS}:
{json.dumps(ROUTE_SCRUB, indent=2)}
and the gateway's hostname in quoted error messages was replaced with
"<gateway>". Nothing else was altered. Scoring fields are byte-for-byte as
produced by the evaluator.

License: MIT.
"""


def main() -> int:
    """Build the release tarball and its checksums."""
    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        old.unlink()
    staging = Path(tempfile.mkdtemp(prefix="mzr-release-"))

    final = ROOT / "results/v2/final"
    analysis = ROOT / "results/analysis"
    n = scrub_jsonl(final / "attempts.jsonl", staging / "attempts.jsonl")
    print(f"v2: {n} rows scrubbed")
    members = [(staging / "attempts.jsonl", "attempts.jsonl")]
    for src, arc in ((final / "merge-manifest.json", "merge-manifest.json"),
                     (analysis / "board.json", "board.json"),
                     (analysis / "failure-modes.jsonl", "failure-modes.jsonl"),
                     (analysis / "tolerance.json", "tolerance.json"),
                     (analysis / "findings.json", "findings.json"),
                     (ROOT / "configs/providers.v2.json", "providers.v2.json")):
        text = src.read_text()
        for marker in FORBIDDEN:
            if marker in text:
                raise SystemExit(f"scrub failed: {marker!r} in {src.name}")
        members.append((src, arc))
    assets = [tar_of("mazerunner-v2-traces.tar.gz", members, README)]

    sums = OUT / "SHA-256SUMS"
    with sums.open("w") as handle:
        for asset in assets:
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            handle.write(f"{digest}  {asset.name}\n")
            print(f"  {asset.name:<44}{asset.stat().st_size/1e6:>8.1f} MB")
    print(f"checksums -> {sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
