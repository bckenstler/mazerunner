"""Documentation gate: the two rules ruff cannot express.

1. A public function or method whose body exceeds MAX_BARE_STATEMENTS logical
   statements must have a docstring. This encodes the actual house policy —
   trivial helpers stay bare, complex logic does not — where pydocstyle's
   D103 would force noise onto every two-line wrapper.
2. Every repo path cited in a docstring must exist on disk. Doc references
   rot silently; this is the check that would have caught the stale
   smoke.config.json citation that shipped in v1.0.0's USAGE.md.

Conventions themselves: CONTRIBUTING.md, "Documentation conventions".
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BARE_STATEMENTS = 20

# Grandfathered or deliberately bare despite length. Additions need a reason.
ALLOWLIST: set[str] = set()

# Candidate repo paths inside docstrings: token with a slash or a known
# extension, not a URL, not a normalized-coordinate example.
PATH_RE = re.compile(r"(?<![\w:/])((?:[\w.-]+/)+[\w.-]+\.\w{1,5}|[\w-]+\.(?:py|md|json|jsonl|png|txt|cff|toml|yml))\b")
IGNORE_PATHS = {"task.json", "mask.png", "input.png", "ground-truth.json",
                "attempts.jsonl", "index.jsonl", "episodes.jsonl", "manifest.json",
                "index.json", "pricing.json", "reference-overlay.png",
                "episode-summaries.jsonl", "SHA-256SUMS.txt",
                "summary.json", "config.json", "SHA-256SUMS",
                # untracked by design: holds site-specific scrub markers
                "release-scrub.json"}


def statements(node: ast.AST) -> int:
    return sum(isinstance(n, ast.stmt) for n in ast.walk(node))


def resolves(cited: str, from_dir: Path) -> bool:
    """Whether a cited path names a real file.

    Docstrings cite siblings by bare name (`io.py`) as readily as they cite
    from the repo root (`results/failure-modes.md`), so a citation counts as
    resolved if it exists at the root, beside the citing file, or anywhere in
    the tree under that name. Loose by design: the target is dead references
    like smoke.config.json, not a strict path grammar.
    """
    if (ROOT / cited).exists() or (from_dir / cited).exists():
        return True
    name = Path(cited).name
    return any(
        p.name == name
        for sub in ("src", "scripts", "tests", "docs", "results", "configs", "evals")
        for p in (ROOT / sub).rglob(name)
        if "__pycache__" not in p.parts
    )


def check_file(path: Path) -> list[str]:
    """Undocumented long functions and dead path citations in one file."""
    tree = ast.parse(path.read_text())
    problems: list[str] = []
    rel = path.relative_to(ROOT)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or f"{rel}:{node.name}" in ALLOWLIST:
                continue
            if statements(node) > MAX_BARE_STATEMENTS and not ast.get_docstring(node):
                problems.append(
                    f"{rel}:{node.lineno} {node.name} has {statements(node)} "
                    f"statements and no docstring"
                )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if not doc:
                continue
            for cited in PATH_RE.findall(doc):
                if cited in IGNORE_PATHS or "*" in cited:
                    continue
                if not resolves(cited, path.parent):
                    problems.append(f"{rel}: docstring cites {cited}, which does not exist")
    return problems


def main() -> int:
    problems: list[str] = []
    for sub in ("src", "scripts", "tests"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            problems.extend(check_file(path))
    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print("docstring gate green")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
