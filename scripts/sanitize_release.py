#!/usr/bin/env python3
"""Release gate: no internal reference may exist in any tracked file.

Runs in CI. Greps every git-tracked file for markers that must never ship —
internal hostnames, internal model routes, and machine-local absolute paths.
Exits non-zero with file:line evidence on any hit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARKERS = (
    r"scale\.com",
    r"ml-serving-internal",
    r"Inkling-evals",
    r"meta_ai/",
    r"llmengine",
    r"llama_experimental",
    r"litellm-proxy",
    r"/Users/[a-z]",
)

# Files allowed to *name* the markers (this script, and docs that discuss the
# sanitization itself).
EXEMPT = {"scripts/sanitize_release.py"}


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    pattern = re.compile("|".join(f"({m})" for m in MARKERS))
    hits = []
    for name in tracked:
        if name in EXEMPT:
            continue
        path = Path(name)
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = pattern.search(line)
            if m:
                hits.append(f"{name}:{i}: {m.group(0)!r} in: {line.strip()[:100]}")
    if hits:
        print(f"RELEASE BLOCKED — {len(hits)} internal reference(s) in tracked files:")
        for h in hits[:40]:
            print(" ", h)
        return 1
    print(f"sanitize green — {len(tracked)} tracked files, no internal references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
