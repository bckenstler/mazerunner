"""mazerunner CLI: generate | validate | run | dataset | archive."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_generate(args: argparse.Namespace) -> int:
    from .build import build_all

    manifest = build_all(Path(args.mazes_dir))
    for task in manifest["tasks"]:
        print(
            f"  {task['id']:<12} {task['style']:<24} nodes={task['nodes']:<4} "
            f"edges={task['edges']:<4} ref={task['optimal_length_px']}px"
        )
    print(f"wrote {len(manifest['tasks'])} tasks to {args.mazes_dir}/")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .build import validate_all

    print("validating: rebuild determinism, fail-closed checks, reference scoring")
    failures = validate_all(Path(args.mazes_dir))
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("all generation and scoring checks passed")
    if args.skip_tests:
        return 0
    print("running test suite")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"])
    if proc.returncode != 0:
        return proc.returncode
    print("validate green: no API calls were made")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import dataset as ds

    out_dir = Path(json.loads(Path(args.config).read_text())["out_dir"]) if args.command_ds != "build" else None
    if args.command_ds == "build":
        manifest = ds.build_all(
            Path(args.config),
            workers=args.workers,
            splits=args.split.split(",") if args.split else None,
        )
        for split, summary in manifest["summaries"].items():
            print(f"{split}: {summary['size']} tasks, {summary['total_rejections']} rejections, tiers={summary['tiers']}")
        return 0
    if args.command_ds == "verify":
        failures = []
        for split in (args.split.split(",") if args.split else _built_splits(out_dir)):
            failures += [f"{split}: {f}" for f in ds.verify_split(out_dir, split, None if args.full else 25)]
        for failure in failures:
            print(f"FAIL {failure}")
        print("verify green" if not failures else f"{len(failures)} failures")
        return 1 if failures else 0
    if args.command_ds == "stats":
        for split in (args.split.split(",") if args.split else _built_splits(out_dir)):
            stats = ds.split_stats(out_dir, split)
            print(f"\n== {split} ==")
            print(json.dumps(stats, indent=2))
        return 0
    if args.command_ds == "sheet":
        out = ds.qc_sheet(out_dir, args.split or "dev", args.count, args.sheet_seed)
        print(out)
        return 0
    return 1


def _built_splits(out_dir) -> list[str]:
    return sorted(p.parent.name for p in out_dir.glob("*/index.jsonl"))


def cmd_feedback(args: argparse.Namespace) -> int:
    import os

    from .evalset import read_task_list
    from .feedback import episode_summary, run_episode, write_episodes
    from .io import load_task
    from .providers import ENV_KEYS, PROVIDERS

    config = json.loads(Path(args.config).read_text())
    names = args.providers.split(",") if args.providers else None
    task_ids = read_task_list(Path(args.tasks))
    index = {
        json.loads(line)["task_id"]: json.loads(line)
        for line in (Path(args.dataset) / "index.jsonl").read_text().splitlines()
        if line.strip()
    }
    selected = [t for t in task_ids if t in index]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        selected = selected[i::n]

    out_root = Path(args.results_dir)
    total = 0
    for name, settings in config.get("providers", {}).items():
        if names is not None and name not in names:
            continue
        env_key = settings.get("env_key") or ENV_KEYS.get(name)
        if not env_key or not os.environ.get(env_key):
            print(f"skipping {name}: {env_key or 'API key'} not set")
            continue
        adapter = PROVIDERS[settings.get("type", name)](
            **{k: v for k, v in settings.items() if k not in ("enabled", "type")}
        )
        out_dir = out_root / name
        for task_id in selected:
            task_dir = Path(index[task_id]["dir"])
            task, mask = load_task(task_dir)
            rows = run_episode(
                adapter, name, task_id, task_dir, task, mask, max_attempts=args.max_attempts
            )
            summary = episode_summary(rows)
            write_episodes(rows, [summary], out_dir)
            total += 1
            mark = "SOLVED" if summary["solved"] else summary["stop_reason"]
            print(f"{name} · {task_id} · {summary['turns_used']} turns … {mark}")
    print(f"\n{total} episodes -> {out_root}/")
    return 0


def cmd_styleswap(args: argparse.Namespace) -> int:
    from .evalset import read_task_list
    from .styleswap import build_style_swap_set

    task_ids = read_task_list(Path(args.tasks))
    manifest = build_style_swap_set(
        Path(args.dataset), task_ids, Path(args.out), seed=args.seed
    )
    print(f"built {manifest['variants']} variants in {manifest['complete_groups']}"
          f"/{manifest['requested_groups']} complete pair-groups -> {args.out}/")
    print(f"  archetypes: {', '.join(manifest['archetypes'])}")
    for skip in manifest["skipped"]:
        print(f"  SKIPPED {skip['task_id']}: {skip['reason']}")
    return 0 if manifest["complete_groups"] else 1


def cmd_merge(args: argparse.Namespace) -> int:
    from .evalset import read_task_list
    from .merge import merge_runs, missing_units, write_missing_task_list

    paths = sorted(Path().glob(args.runs)) if any(c in args.runs for c in "*?[") else [Path(args.runs)]
    paths = [p / "attempts.jsonl" if p.is_dir() else p for p in paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"no attempts.jsonl matched {args.runs!r}")
        return 1

    out_dir = Path(args.out)
    manifest = merge_runs(paths, out_dir, expected_units=args.expected or None)

    print(f"merged {len(paths)} files -> {out_dir}/attempts.jsonl")
    print(f"  {manifest['rows_in']} rows in, {manifest['rows_out']} out, "
          f"{manifest['duplicates_collapsed']} duplicates collapsed")
    for leg, stats in sorted(manifest["per_leg"].items()):
        print(f"  {leg:<16}{stats['attempts']:>5} attempts  {stats['successes']:>4} pass  "
              f"{stats['transport_failures']:>3} transport fail")

    reexec = manifest.get("re_execution_conflicts", [])
    if reexec:
        print(f"  note: {len(reexec)} unit(s) ran twice (a resume racing a live shard) and "
              f"disagreed; dedup keeps one, which is unbiased w.r.t. success")
    cross = manifest.get("cross_shard_conflicts", [])
    if cross:
        for conflict in cross[:5]:
            print(f"FAIL two shards scored the same unit differently: {conflict['key']} "
                  f"success={conflict['a']['success']} vs {conflict['b']['success']}")
        print(f"{len(cross)} cross-shard conflicts — the work was split wrong")
        return 1

    if args.mazes_file and args.providers:
        task_ids = read_task_list(Path(args.mazes_file))
        missing = missing_units(
            out_dir / "attempts.jsonl", task_ids, args.providers.split(","), args.trials
        )
        if missing:
            count = write_missing_task_list(missing, out_dir / "missing.txt")
            print(f"  {len(missing)} attempts missing across {count} tasks "
                  f"-> {out_dir}/missing.txt (rerun with --resume)")
        else:
            print("  complete: every planned attempt is present")

    canaries = manifest["efficiency_canary_tasks"]
    if canaries and not args.allow_canary:
        print(f"FAIL efficiency canary fired on {len(canaries)} task(s): "
              f"{', '.join(canaries[:5])} — quarantined pending inspection")
        return 1
    return 0


def cmd_evalset(args: argparse.Namespace) -> int:
    from .evalset import SelectionSpec, build_eval_set, verify_eval_set

    out_path = Path(args.out)
    if args.command_es == "build":
        tiers = None
        if args.tiers:
            try:
                easy, medium, hard = (int(n) for n in args.tiers.split(","))
            except ValueError:
                print(f"--tiers expects easy,medium,hard (got {args.tiers!r})")
                return 1
            tiers = {"easy": easy, "medium": medium, "hard": hard}
        spec = SelectionSpec(
            size=args.size,
            per_family_min=args.per_family_min,
            per_family_max=args.per_family_max,
            archetype_floor=args.archetype_floor,
            **({"tier_targets": tiers} if tiers else {}),
        )
        manifest = build_eval_set(
            Path(args.dataset),
            out_path,
            seed=args.seed,
            spec=spec,
            pool_path=Path(args.pool) if args.pool else None,
        )
        print(f"selected {manifest['selected']} tasks -> {out_path} (seed {manifest['seed']})"
              + (f", drawn from {manifest['pool_size']} in {manifest['pool_path']}"
                 if args.pool else ""))
        for key, counts in manifest["achieved"].items():
            print(f"  {key}: {counts}")
        return 0

    if args.command_es == "verify":
        failures = verify_eval_set(out_path)
        for failure in failures:
            print(f"FAIL {failure}")
        print("evalset verify green" if not failures else f"{len(failures)} failures")
        return 1 if failures else 0
    return 1


def cmd_archive(args: argparse.Namespace) -> int:
    from . import archive as arch

    results_root = Path(args.results_dir)
    out_dir = Path(args.archive_dir)

    if args.command_ar == "inventory":
        records = arch.inventory(results_root)
        for record in records:
            note = f"  [{record.note}]" if record.note else ""
            print(
                f"  {record.leg + '/' + record.stamp:<52}{record.rows:>5} rows  "
                f"{record.rows_with_raw:>5} traced  {record.bytes_total / 1e6:>7.1f} MB{note}"
            )
        coverage = arch._coverage_table(records)
        print(f"\n{len(records)} runs, {coverage['rows_total']} rows, "
              f"{coverage['rows_with_raw_response']} with raw traces, "
              f"{coverage['malformed_lines']} malformed")
        return 0

    if args.command_ar == "build":
        records = arch.inventory(results_root)
        manifest = arch.archive_runs(records, out_dir, results_root=results_root)
        coverage = manifest["coverage"]
        print(f"archived {manifest['runs']} runs -> {out_dir}/")
        print(f"  {coverage['rows_total']} rows, "
              f"{coverage['rows_with_raw_response']} with raw traces, "
              f"{coverage['rows_transport_error']} transport errors")
        for label in ("empty_runs", "runs_without_traces", "voided_runs"):
            if coverage[label]:
                print(f"  {label}: {len(coverage[label])} — {', '.join(coverage[label][:4])}"
                      + (" ..." if len(coverage[label]) > 4 else ""))
        return 0

    if args.command_ar == "verify":
        report = arch.verify_archive(out_dir)
        if not report.get("ok"):
            for label in ("mismatched", "unreadable_archives"):
                for item in report.get(label, []):
                    print(f"FAIL {label}: {item}")
            if report.get("error"):
                print(f"FAIL {report['error']}")
            return 1
        missing = report.get("missing_source", [])
        stale = report.get("superseded_snapshots", [])
        print(f"archive verify green — {report['files_checked']} files re-hashed and matching")
        if stale:
            print(f"  note: {len(stale)} file(s) archived mid-run have since grown; "
                  f"re-run `archive build` once the run is quiesced")
        if missing:
            print(f"  note: {len(missing)} source files no longer present (archived copy retained)")
        return 0
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    from .evalset import read_task_list
    from .imagesrc import spec_from_args
    from .runner import run_smoke

    mazes = args.mazes.split(",") if args.mazes else None
    if getattr(args, "mazes_file", None):
        from_file = read_task_list(Path(args.mazes_file))
        mazes = sorted(set(mazes) & set(from_file)) if mazes else from_file

    shard_index, shard_count = 0, 1
    if args.shard:
        try:
            raw_index, raw_count = args.shard.split("/")
            shard_index, shard_count = int(raw_index), int(raw_count)
        except ValueError:
            print(f"--shard expects i/N (got {args.shard!r})")
            return 1

    return run_smoke(
        config_path=Path(args.config),
        providers=args.providers.split(",") if args.providers else None,
        mazes=mazes,
        trials=args.trials,
        mazes_dir=Path(args.mazes_dir),
        results_dir=Path(args.results_dir),
        dataset_dir=Path(args.dataset) if args.dataset else None,
        dry_run=args.dry_run,
        include_dimensions=args.include_dimensions,
        run_id=args.run_id,
        shard_index=shard_index,
        shard_count=shard_count,
        resume=args.resume,
        order_seed=args.order_seed,
        image_spec=spec_from_args(args.image_variant, args.image_scale, args.image_seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="mazerunner")
    parser.add_argument("--mazes-dir", default="mazes")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="rebuild all task artifacts")

    validate = sub.add_parser("validate", help="offline validation; no API calls")
    validate.add_argument("--skip-tests", action="store_true")

    run = sub.add_parser("run", help="live provider smoke run (needs API keys)")
    run.add_argument("--config", default="smoke.config.json")
    run.add_argument("--providers", default=None, help="comma-separated subset")
    run.add_argument("--mazes", default=None, help="comma-separated subset")
    run.add_argument(
        "--mazes-file",
        default=None,
        help="file of task ids (newline or comma separated); avoids huge command lines",
    )
    run.add_argument("--trials", type=int, default=None, help="override config trials")
    run.add_argument("--results-dir", default="results")
    run.add_argument("--dataset", default=None, help="run against a dataset split dir (e.g. datasets/v1/dev)")
    run.add_argument("--dry-run", action="store_true", help="print planned API call count and exit")
    run.add_argument(
        "--image-variant",
        default="real",
        choices=["real", "blank", "mismatched", "rescale"],
        help="ablation: substitute the image sent to the model (scoring is unaffected)",
    )
    run.add_argument("--image-scale", type=float, default=1.0, help="rescale factor, e.g. 2.0")
    run.add_argument("--image-seed", type=int, default=None, help="seed for the mismatched pairing")
    run.add_argument("--run-id", default=None, help="group shards under results/<run-id>/")
    run.add_argument("--shard", default=None, help="run only shard i of N, as i/N")
    run.add_argument("--resume", action="store_true", help="skip attempts already recorded")
    run.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="seed the per-leg task-order shuffle (required for reproducible runs)",
    )
    run.add_argument(
        "--include-dimensions",
        action="store_true",
        help="ablation: disclose the image's pixel dimensions in the prompt",
    )

    dataset = sub.add_parser("dataset", help="dataset build/verify/stats/sheet")
    dataset.add_argument("command_ds", choices=["build", "verify", "stats", "sheet"])
    dataset.add_argument("--config", default="dataset.config.json")
    dataset.add_argument("--split", default=None, help="comma-separated subset of splits")
    dataset.add_argument("--workers", type=int, default=8)
    dataset.add_argument("--full", action="store_true", help="verify every task, not a sample")
    dataset.add_argument("--count", type=int, default=24, help="sheet: tasks per sheet")
    dataset.add_argument("--sheet-seed", type=int, default=0)

    feedback = sub.add_parser("feedback", help="closed-loop retry episodes (separate leaderboard)")
    feedback.add_argument("--config", default="litellm.config.json")
    feedback.add_argument("--providers", default=None)
    feedback.add_argument("--dataset", default="datasets/v1/dev")
    feedback.add_argument("--tasks", default="evals/ablation-50.txt")
    feedback.add_argument("--results-dir", default="results/abl/feedback")
    feedback.add_argument("--max-attempts", type=int, default=4)
    feedback.add_argument("--shard", default=None, help="run only shard i of N, as i/N")

    styleswap = sub.add_parser("styleswap", help="build same-topology/different-style variants")
    styleswap.add_argument("--dataset", default="datasets/v1/dev")
    styleswap.add_argument("--tasks", default="evals/styleswap-20.txt")
    styleswap.add_argument("--out", default="datasets/styleswap-v1")
    styleswap.add_argument("--seed", type=int, default=20260731)

    merge = sub.add_parser("merge", help="consolidate sharded runs into one result set")
    merge.add_argument("--runs", required=True, help="glob of shard dirs or attempts.jsonl files")
    merge.add_argument("--out", required=True, help="output directory")
    merge.add_argument("--expected", type=int, default=0, help="planned attempt count")
    merge.add_argument("--mazes-file", default=None, help="task list, to compute what is missing")
    merge.add_argument("--providers", default=None, help="comma-separated, to compute what is missing")
    merge.add_argument("--trials", type=int, default=8)
    merge.add_argument("--allow-canary", action="store_true",
                       help="do not fail when the efficiency canary fired")

    evalset = sub.add_parser("evalset", help="build/verify a frozen stratified task subset")
    evalset.add_argument("command_es", choices=["build", "verify"])
    evalset.add_argument("--dataset", default="datasets/v1/dev")
    evalset.add_argument("--out", default="evals/dev-eval-100.txt")
    evalset.add_argument("--seed", type=int, default=20260730)
    evalset.add_argument("--size", type=int, default=100)
    evalset.add_argument("--archetype-floor", type=int, default=6)
    evalset.add_argument("--per-family-min", type=int, default=12)
    evalset.add_argument("--per-family-max", type=int, default=13)
    evalset.add_argument("--tiers", default=None, help="easy,medium,hard counts e.g. 7,11,7")
    evalset.add_argument(
        "--pool",
        default=None,
        help="restrict candidates to an existing frozen list, nesting the subset inside it",
    )

    archive = sub.add_parser("archive", help="durably archive run artifacts with checksums")
    archive.add_argument("command_ar", choices=["inventory", "build", "verify"])
    archive.add_argument("--results-dir", default="results")
    archive.add_argument("--archive-dir", default="archive")

    args = parser.parse_args()
    handler = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "run": cmd_run,
        "dataset": cmd_dataset,
        "archive": cmd_archive,
        "evalset": cmd_evalset,
        "merge": cmd_merge,
        "styleswap": cmd_styleswap,
        "feedback": cmd_feedback,
    }
    sys.exit(handler[args.command](args))


if __name__ == "__main__":
    main()
