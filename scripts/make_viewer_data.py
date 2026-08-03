"""Build the static data tree the trace viewer loads.

The viewer is serverless: a small manifest is fetched eagerly, and each
attempt's full payload (submission points, evaluation, reasoning trace) is a
separate JSON fetched on selection. `raw_response` never ships — it is 78% of
a row's bytes and the viewer has no use for the wire format.

Output: docs/viewer/data/
  index.json                              manifest: models, tasks, attempt summaries
  attempts/{provider}--{maze}--{trial}.json
  tasks/{maze}/input.png                  the exact image the model saw
  tasks/{maze}/task.json                  slim geometry: badges, radii, reference route
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "docs" / "viewer" / "data"

NAMES = {
    "gpt-xhigh": "GPT-5.6 Sol · xhigh", "openai": "GPT-5.6 Sol · medium",
    "gemini": "Gemini 3.6 Flash · medium", "kimi": "Kimi K3 · high",
    "anthropic": "Claude Opus 5 · high", "muse-spark": "Muse Spark 1.1 · medium",
    "inkling": "Inkling · default",
}


def main() -> int:
    attempts_path = ROOT / "results/main/merged/attempts.jsonl"
    modes_path = ROOT / "results/failure-modes.jsonl"
    index_path = ROOT / "datasets/v1/dev/index.jsonl"

    dataset_index = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in index_path.read_text().splitlines() if l.strip()
    }
    verdicts = {}
    if modes_path.exists():
        for l in modes_path.read_text().splitlines():
            if l.strip():
                v = json.loads(l)
                verdicts[(v["provider"], v["maze"], v["trial"])] = v

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "attempts").mkdir(parents=True)
    (OUT / "tasks").mkdir(parents=True)

    tasks_seen: dict[str, dict] = {}
    summaries = []
    rows = 0

    with attempts_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            rows += 1
            provider, maze, trial = r["provider"], r["maze"], r["trial"]
            ev = r.get("evaluation") or {}
            d = r.get("derived") or {}
            verdict = verdicts.get((provider, maze, trial))

            key = f"{provider}--{maze}--{trial}"
            payload = {
                "provider": provider,
                "model_name": NAMES.get(provider, provider),
                "maze": maze,
                "trial": trial,
                "submission": r.get("submission"),
                "evaluation": ev,
                "derived": d,
                "reasoning": r.get("reasoning"),
                "latency_s": r.get("latency_s"),
                "usage": {
                    k: v for k, v in (r.get("usage") or {}).items()
                    if isinstance(v, (int, float))
                },
                "failure_mode": (
                    {"primary": verdict["primary"],
                     "why": verdict["evidence"].get("why"),
                     "quote": verdict["evidence"].get("quote")}
                    if verdict else None
                ),
            }
            (OUT / "attempts" / f"{key}.json").write_text(
                json.dumps(payload, separators=(",", ":"))
            )

            summaries.append({
                "k": key,
                "p": provider,
                "m": maze,
                "t": trial,
                "ok": 1 if ev.get("success") else 0,
                "rp": round(d.get("route_progress", 0), 3),
                "fm": verdict["primary"] if verdict else ("pass" if ev.get("success") else None),
                "lat": r.get("latency_s"),
                "np": len((r.get("submission") or {}).get("points") or []),
                "tr": 1 if r.get("reasoning") else 0,
            })

            if maze not in tasks_seen:
                meta = dataset_index[maze]
                task_dir = ROOT / meta["dir"]
                task = json.loads((task_dir / "task.json").read_text())
                slim = {
                    "id": maze,
                    "width": task["width"], "height": task["height"],
                    "start": task["start"], "goal": task["goal"],
                    "pointer_radius_px": task["pointer_radius_px"],
                    "start_radius_px": task["start_radius_px"],
                    "goal_radius_px": task["goal_radius_px"],
                    "reference_path": task["reference"]["optimal_path"],
                    "family": meta["family"], "tier": meta["tier"],
                    "archetype": meta["archetype"],
                    "measures": meta.get("measures"),
                }
                dest = OUT / "tasks" / maze
                dest.mkdir()
                shutil.copy2(task_dir / "input.png", dest / "input.png")
                shutil.copy2(task_dir / task["mask_file"], dest / "mask.png")
                (dest / "task.json").write_text(json.dumps(slim, separators=(",", ":")))
                tasks_seen[maze] = slim

    manifest = {
        "generated_from": "results/main/merged/attempts.jsonl",
        "models": [
            {"id": p, "name": NAMES.get(p, p)}
            for p in sorted({s["p"] for s in summaries})
        ],
        "tasks": {
            maze: {"family": t["family"], "tier": t["tier"], "archetype": t["archetype"]}
            for maze, t in tasks_seen.items()
        },
        "attempts": summaries,
    }
    (OUT / "index.json").write_text(json.dumps(manifest, separators=(",", ":")))

    # every attempt JSON must parse back
    for f in (OUT / "attempts").iterdir():
        json.loads(f.read_text())

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    per = defaultdict(int)
    for f in OUT.rglob("*"):
        if f.is_file():
            per[f.relative_to(OUT).parts[0]] += f.stat().st_size
    print(f"viewer data: {rows} attempts, {len(tasks_seen)} tasks, {size/1e6:.1f} MB total")
    for part, bytes_ in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {part:<12}{bytes_/1e6:>8.1f} MB")
    assert size < 100e6, "viewer payload exceeds the Pages budget"
    return 0


if __name__ == "__main__":
    sys.exit(main())
