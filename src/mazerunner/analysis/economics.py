"""Cost and latency per solve — reported beside capability, never blended into it.

Vendors disagree on usage field names (`prompt_tokens`, `input_tokens`,
`prompt_token_count`), so normalization happens here rather than being assumed
at the call site. Routes with no published price are reported as unpriced; a
guessed number in a cost table is worse than an admitted gap.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

INPUT_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
OUTPUT_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")


def normalize_usage(usage: dict | None) -> tuple[int, int]:
    """(input, output) tokens from any of the vendor dialects."""
    if not isinstance(usage, dict):
        return (0, 0)
    def pick(keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0
    return (pick(INPUT_KEYS), pick(OUTPUT_KEYS))


def load_pricing(path: Path = Path("pricing.json")) -> dict:
    """model -> per-token prices, or {} when no pricing file is present.

    Missing pricing degrades to token and latency reporting rather than
    failing: prices go stale and are not part of the benchmark.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("models", {})


def economics(rows: list[dict], pricing: dict) -> dict[str, dict]:
    """provider -> tokens, latency, cost, and cost per solved task."""
    agg: dict[str, dict] = defaultdict(
        lambda: {"attempts": 0, "passes": 0, "input": 0, "output": 0, "latency": 0.0, "model": None}
    )
    for row in rows:
        if row.get("error"):
            continue
        a = agg[row["provider"]]
        a["attempts"] += 1
        a["model"] = a["model"] or row.get("model")
        if (row.get("evaluation") or {}).get("success"):
            a["passes"] += 1
        i, o = normalize_usage(row.get("usage"))
        a["input"] += i
        a["output"] += o
        a["latency"] += row.get("latency_s") or 0.0

    out = {}
    for provider, a in agg.items():
        price = pricing.get(a["model"] or "", {})
        priced = price.get("input") is not None
        cost = (
            a["input"] / 1e6 * price["input"] + a["output"] / 1e6 * price["output"]
            if priced else None
        )
        out[provider] = {
            "model": a["model"],
            "attempts": a["attempts"],
            "passes": a["passes"],
            "input_tokens": a["input"],
            "output_tokens": a["output"],
            "mean_latency_s": a["latency"] / a["attempts"] if a["attempts"] else 0.0,
            "cost_usd": cost,
            "cost_per_solve_usd": (cost / a["passes"]) if cost and a["passes"] else None,
            "priced": priced,
        }
    return out
