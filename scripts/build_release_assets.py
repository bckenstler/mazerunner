"""Build the GitHub Release assets: trace tarballs + encrypted hidden split.

Traces are scrubbed of internal serving-route prefixes before packaging: the
route a request happened to take through a private gateway is operational
detail, not science, and the mapping is recorded in the tarball's README so
nothing is silently altered. The hidden split ships encrypted so later results
on it are verifiable without exposing the tasks.

Usage:
  uv run python scripts/build_release_assets.py            # tarballs + checksums
  MAZERUNNER_HIDDEN_KEY=... uv run python ... --hidden     # + encrypted hidden split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release-assets"

# Internal gateway route prefixes -> public model names. Applied to the
# `model`, `response_model`, and `serving_stack` fields of every row.
ROUTE_SCRUB = {
    "fireworks_ai/kimi-k3": "kimi-k3",
    "fireworks_ai/inkling": "inkling",
    "openai/gpt-5.6-sol": "gpt-5.6-sol",
    "anthropic/claude-opus-5": "claude-opus-5",
    "gemini/gemini-3.6-flash": "gemini-3.6-flash",
}
SCRUB_FIELDS = ("model", "response_model", "serving_stack")
FORBIDDEN = ("REDACTED-DOMAIN", "ml-serving-internal", "Inkling-evals", "meta_ai/", "litellm-proxy")

# String-level replacements applied to the serialized row AFTER field scrubs:
# error messages in the transport-retry history quote the gateway URL verbatim.
STRING_SCRUB = {
    "https://REDACTED-GATEWAY/v1": "<gateway>",
    "REDACTED-GATEWAY": "<gateway>",
    "litellm.RateLimitError": "RateLimitError",
    "litellm.BadRequestError": "BadRequestError",
    "litellm.APIConnectionError": "APIConnectionError",
}


def scrub_row(row: dict) -> dict:
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
    out = OUT / name
    with tempfile.TemporaryDirectory() as tmp:
        readme_path = Path(tmp) / "README.md"
        readme_path.write_text(readme)
        with tarfile.open(out, "w:gz") as tar:
            tar.add(readme_path, arcname="README.md")
            for path, arcname in members:
                tar.add(path, arcname=arcname)
    return out


README_COMMON = f"""# MazeRunner v1 traces

One JSON object per attempt: submission points, full evaluation, derived
metrics, reasoning trace where the provider returned one, raw provider
payload, usage, latency, and run provenance (run id, shard, ordinal, order
seed).

Serving-route prefixes from the private gateway used during the study were
rewritten to public model names in the fields {SCRUB_FIELDS}:
{json.dumps(ROUTE_SCRUB, indent=2)}
Nothing else was altered. Scoring fields are byte-for-byte as produced by the
evaluator.

License: MIT.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", action="store_true", help="also package the encrypted hidden split")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="mzr-release-"))

    # main run
    n = scrub_jsonl(ROOT / "results/main/merged/attempts.jsonl", staging / "main-attempts.jsonl")
    print(f"main: {n} rows scrubbed")
    assets = [tar_of(
        "mazerunner-v1-traces-main.tar.gz",
        [(staging / "main-attempts.jsonl", "main/attempts.jsonl"),
         (ROOT / "results/failure-modes.jsonl", "main/failure-modes.jsonl"),
         (ROOT / "evals/dev-eval-100.txt", "main/task-set.txt")],
        README_COMMON,
    )]

    # ablations
    ablation_members = []
    for merged in sorted((ROOT / "results/abl/merged").glob("*/attempts.jsonl")):
        name = merged.parent.name
        dst = staging / f"abl-{name}.jsonl"
        n = scrub_jsonl(merged, dst)
        print(f"ablation {name}: {n} rows")
        ablation_members.append((dst, f"ablations/{name}/attempts.jsonl"))
    for episodes in sorted((ROOT / "results/abl/feedback").glob("*/episodes.jsonl")):
        leg = episodes.parent.name
        dst = staging / f"fb-{leg}.jsonl"
        scrub_jsonl(episodes, dst)
        ablation_members.append((dst, f"feedback/{leg}/episodes.jsonl"))
    assets.append(tar_of("mazerunner-v1-traces-ablations.tar.gz", ablation_members, README_COMMON))

    # sweeps (per-leg merged files live under results/sweep-*/)
    sweep_members = []
    for sweep in sorted(ROOT.glob("results/sweep-*/*/attempts.jsonl")):
        leg = sweep.parts[-3].replace("sweep-", "")
        dst = staging / f"sweep-{leg}-{sweep.parts[-2]}.jsonl"
        scrub_jsonl(sweep, dst)
        sweep_members.append((dst, f"sweeps/{leg}/attempts.jsonl"))
    assets.append(tar_of("mazerunner-v1-traces-sweeps.tar.gz", sweep_members, README_COMMON))

    if args.hidden:
        key = os.environ.get("MAZERUNNER_HIDDEN_KEY")
        if not key:
            raise SystemExit("--hidden requires MAZERUNNER_HIDDEN_KEY in the environment")
        plain = OUT / "hidden-split.tar.gz"
        with tarfile.open(plain, "w:gz") as tar:
            tar.add(ROOT / "datasets/v1/test-hidden", arcname="test-hidden")
        enc = OUT / "mazerunner-v1-test-hidden.tar.gz.enc"
        subprocess.run(
            ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt",
             "-in", str(plain), "-out", str(enc), "-pass", "env:MAZERUNNER_HIDDEN_KEY"],
            check=True,
        )
        plain.unlink()
        assets.append(enc)
        print("hidden split encrypted (openssl aes-256-cbc, pbkdf2)")

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
