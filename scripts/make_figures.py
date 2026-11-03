"""Figures for the v2 study writeup, in the promo reel's visual language.

Dark ground, cyan for the subject under discussion, amber for the thing the
reader should notice, everything else desaturated. Monospace throughout so
numbers align and the charts read as instrument output rather than slideware.

Every number is computed here from the merged run (results/v2/final, shipped
as a release asset) and the offline analyses tracked in results/analysis —
nothing is hand-typed. The two slow analyses are rebuilt when missing:
results/analysis/failure-modes.jsonl (`mazerunner failuremodes` on the merged
run) and results/analysis/tolerance.json (stored submissions re-scored at other
pointer radii). The computed findings go to results/analysis/findings.json.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mazerunner.analysis.board import (  # noqa: E402
    LEVELS, UPPER, build_board, by_cell, macro, paired, parse_task,
)
from mazerunner.analysis.economics import normalize_usage  # noqa: E402

FIG = ROOT / "results" / "figures"
ANALYSIS = ROOT / "results" / "analysis"
META = json.loads((ROOT / "configs/board.v2.json").read_text())
ATTEMPTS = ROOT / META["attempts"]
DATASET = ROOT / META["dataset"]
LEGS = META["legs"]
UPPER_ONLY = frozenset(k for k, v in LEGS.items() if v.get("upper_only"))
MAIN = [k for k in LEGS if k not in UPPER_ONLY]
LEAD = "gpt6-astra"
# Models with enough passes to have a shape; the two floor models stay on the
# leaderboard and in the failure analysis but would be flat lines at zero here.
CAPABLE = ["gpt6-astra", "gpt61-terra", "gpt61-sol", "argon", "opus55", "kimi31"]

BG = "#090B12"
PANEL = "#0E121C"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#34D399"
RED = "#F84454"
WHITE = "#EDF2F7"
DIM = "#6E7A8C"
GRID = "#1A2030"
# Desaturated greys for the non-lead series, lightest first, so rank order
# still reads when the lines cross.
GREYS = ["#C3CCD8", "#A8B2C4", "#8C97AA", "#768094", "#5E687B", "#4B5568", "#3E4757"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "monospace", "font.monospace": ["Menlo", "DejaVu Sans Mono"],
    "text.color": WHITE, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": WHITE,
    "figure.dpi": 160,
})


def name(leg):
    return LEGS[leg]["name"]


def color(leg, i=0):
    return CYAN if leg == LEAD else GREYS[min(i, len(GREYS) - 1)]


def _frame(ax, title=None, sub=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", pad=34 if sub else 10)
    if sub:
        lines = sub.count("\n") + 1
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=34 + (lines - 1) * 14)
        ax.text(0, 1.055, sub, transform=ax.transAxes, color=DIM, fontsize=9,
                va="bottom", linespacing=1.5)


def _suptitle(fig, title, sub):
    fig.suptitle(title, x=0.02, y=1.0, ha="left", fontsize=13, fontweight="bold", color=WHITE)
    fig.text(0.02, 0.945, sub, color=DIM, fontsize=9, va="top")


def save(fig, name_, tight=True):
    FIG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / name_, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name_}")


def spread_labels(values, gap):
    """Nudge end-of-line label positions apart by at least `gap`, keeping each
    as close to its own line as the gap allows."""
    order = sorted(values, key=values.get)
    out, last = {}, None
    for key in order:
        y = values[key] if last is None else max(values[key], last + gap)
        out[key] = last = y
    return out


# ---------------------------------------------------------------- loading

def load_rows():
    """Slim merged rows (no raw payloads) for the per-attempt analyses."""
    rows = []
    with ATTEMPTS.open() as handle:
        for line in handle:
            r = json.loads(line)
            r.pop("raw_response", None)
            rows.append(r)
    return rows


def ensure_analyses():
    """Rebuild the cached offline analyses if they are missing."""
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    modes = ANALYSIS / "failure-modes.jsonl"
    if not modes.exists():
        subprocess.run(["uv", "run", "mazerunner", "failuremodes", "--attempts", str(ATTEMPTS),
                        "--dataset", str(DATASET), "--out", str(modes)], check=True, cwd=ROOT)
    tol = ANALYSIS / "tolerance.json"
    if not tol.exists():
        from mazerunner.analysis.load import load_attempts, load_index
        from mazerunner.analysis.tolerance import rescore
        rows = load_attempts([ATTEMPTS], DATASET, backfill=False)
        curves = rescore(rows, (1, 2, 3, 4, 5, 8), index=load_index(DATASET))
        tol.write_text(json.dumps({str(r): p for r, p in curves.items()}))


# ------------------------------------------------------------------ figures

def fig_leaderboard(board):
    """Macro pass@1 bars with task-clustered CIs, pass@4 as the ghost bar."""
    legs = sorted(MAIN, key=lambda k: board[k]["headline"]["score"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, leg in enumerate(legs):
        r = board[leg]
        m, lo, hi = (100 * r["headline"][k] for k in ("score", "lo", "hi"))
        pk = 100 * r["pass_at_k"]["score"]
        lead = leg == LEAD
        ax.barh(i, pk, height=0.62, color=CYAN, alpha=0.16)
        ax.barh(i, m, height=0.62, color=CYAN if lead else "#2B7A8C")
        ax.plot([lo, hi], [i, i], color=WHITE, lw=1.6, alpha=0.75)
        label_x = hi + 1.6
        ax.text(label_x, i, f"{m:.1f}%", va="center", color=WHITE if lead else DIM,
                fontsize=10, fontweight="bold" if lead else "normal")
        ax.text(max(pk, label_x + 11) + 1.6, i, f"pass@4 {pk:.0f}%", va="center",
                color=DIM, fontsize=8)
    ax.set_yticks(range(len(legs)), [name(k) for k in legs])
    ax.set_xlim(0, 100)
    ax.set_xlabel("pass@1, mean over 48 equally weighted cells  (%)")
    _frame(ax, "Leaderboard",
           "192 mazes × 4 attempts · ghost bars pass@4 · whiskers 95% CI (tasks resampled within cells)")
    save(fig, "01-leaderboard.png")


def fig_ladder(tasks):
    """Pass rate by size level on the straight cells of all eight families."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    finals = {}
    for i, leg in enumerate(CAPABLE):
        cells = by_cell(tasks[leg])
        vals = [100 * np.mean([v for (f, lv), v in cells.items() if lv == level]) for level in LEVELS]
        ax.plot(LEVELS, vals, marker="o", lw=2.6 if leg == LEAD else 1.6,
                color=color(leg, i), alpha=1 if leg == LEAD else 0.85, zorder=3 if leg == LEAD else 2)
        finals[leg] = vals[-1]
    for i, leg in enumerate(CAPABLE):
        y = spread_labels(finals, 4.2)[leg]
        ax.text(3.1, y, f"{name(leg).split(' · ')[0]} {finals[leg]:.0f}%",
                color=CYAN if leg == LEAD else DIM, fontsize=8, va="center")
    ax.set_ylabel("pass@1  (%)")
    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.2, 4.3)
    _frame(ax, "Every model falls as the maze grows",
           "straight mazes, all 8 families · L1 is v1's medium size; L4 has 3.4× the turns in narrower corridors")
    save(fig, "02-ladder.png")


def fig_warp(board):
    """Straight vs warped twins per level, one panel per capable model."""
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), sharex=True, sharey=True)
    for ax, leg in zip(axes.flat, CAPABLE):
        w = board[leg]["warp"]
        s = [100 * w[lv]["straight"] for lv in LEVELS]
        b = [100 * w[lv]["warped"] for lv in LEVELS]
        ax.fill_between(LEVELS, b, s, color=AMBER, alpha=0.14, lw=0)
        ax.plot(LEVELS, s, marker="o", color=GREYS[0], lw=1.8, label="straight")
        ax.plot(LEVELS, b, marker="o", color=CYAN, lw=2.2, label="warped twin")
        gap = s[-1] - b[-1]
        ax.text(3.08, (s[-1] + b[-1]) / 2, f"−{gap:.0f}", color=AMBER, fontsize=9,
                va="center", fontweight="bold")
        ax.set_title(name(leg).split(" · ")[0], loc="left", fontsize=10.5, pad=6)
        ax.set_ylim(-4, 106)
        ax.set_xlim(-0.25, 3.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("pass@1  (%)")
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=DIM, loc="lower left")
    _suptitle(fig, "Bending the same maze costs more the bigger it gets",
              "rectilinear, braided, pipes, island · each warped maze vs the straight maze it was "
              "built from (same route, start, goal, style)")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "03-warp.png", tight=False)


def fig_effort(board, effort, rows):
    """Dumbbells: L3/L4 score at less vs more effort, with the paired delta."""
    tokens = collections.defaultdict(list)
    for r in rows:
        if not r.get("error") and parse_task(r["maze"])["level"] in UPPER:
            _i, o = normalize_usage(r.get("usage"))
            if o:
                tokens[r["provider"]].append(o)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ordered = sorted(effort, key=lambda e: e["delta"]["score"])
    for i, e in enumerate(ordered):
        lo_s = 100 * e["less_score"]["score"]
        hi_s = 100 * e["more_score"]["score"]
        d = e["delta"]
        significant = d["lo"] > 0
        ax.plot([lo_s, hi_s], [i, i], color=AMBER if significant else DIM, lw=2.4, alpha=0.8)
        ax.scatter([lo_s], [i], s=70, color=GREYS[2], zorder=3)
        ax.scatter([hi_s], [i], s=90, color=CYAN, zorder=3)
        t_lo = np.median(tokens[e["less"]]) / 1000
        t_hi = np.median(tokens[e["more"]]) / 1000
        ax.text(max(lo_s, hi_s) + 2, i + 0.13,
                f"{100*d['score']:+.1f} pts  [{100*d['lo']:+.0f}, {100*d['hi']:+.0f}]",
                color=WHITE if significant else DIM, fontsize=9, va="center",
                fontweight="bold" if significant else "normal")
        ax.text(max(lo_s, hi_s) + 2, i - 0.22, f"median output {t_lo:.1f}K → {t_hi:.1f}K tokens",
                color=DIM, fontsize=7.5, va="center")
    labels = [f"{LEGS[e['more']]['model']}  {LEGS[e['less']]['effort']} → {LEGS[e['more']]['effort']}"
              for e in ordered]
    ax.set_yticks(range(len(ordered)), labels)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlabel("pass@1 on the 24 L3/L4 cells  (%)    grey = less effort · cyan = more")
    _frame(ax, "More thinking buys 8–14 points on big mazes — except for Argon",
           "same tasks at both settings · delta and 95% CI paired by task · amber = interval above zero\n"
           "4 attempts per maze, except Opus max (1)")
    save(fig, "04-effort.png")


def fig_families(tasks):
    """Cell heatmaps for the lead model and Claude, the sharpest contrast."""
    types = sorted({parse_task(t)["family"] for t in tasks[LEAD]})
    warped = sorted({parse_task(t)["family"] for t in tasks[LEAD] if parse_task(t)["warped"]})
    row_keys = [(f, "") for f in types] + [(f, "w") for f in warped]
    cmap = LinearSegmentedColormap.from_list("cy", [PANEL, "#155E6E", CYAN])
    pick = ["gpt6-astra", "argon", "opus55"]
    fig, axes = plt.subplots(1, len(pick), figsize=(11.2, 5.6), sharey=True)
    for ax, leg in zip(axes, pick):
        cells = by_cell(tasks[leg])
        M = np.array([[cells.get((f, lv + w), np.nan) for lv in LEVELS] for f, w in row_keys])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (yi, xi), v in np.ndenumerate(M):
            ax.text(xi, yi, f"{100*v:.0f}", ha="center", va="center", fontsize=8,
                    color=BG if v > 0.62 else WHITE)
        ax.set_xticks(range(4), LEVELS)
        ax.set_title(name(leg).split(" · ")[0], loc="left", fontsize=10.5, pad=6)
        ax.axhline(len(types) - 0.5, color=AMBER, lw=1.2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(row_keys)), [f + (" · warped" if w else "") for f, w in row_keys])
    _suptitle(fig, "Where each model breaks",
              "pass@1 per cell (%) · 4 mazes × 4 attempts each · below the amber line: warped twins")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "05-cells.png", tight=False)


def fig_regression(rows):
    """Forest plot: which task features predict failure, with warp held as
    its own covariate. Computed live with a task-clustered bootstrap."""
    from mazerunner.analysis.difficulty import logistic_with_cluster_ci
    from mazerunner.analysis.load import load_index

    index = load_index(DATASET)
    task_json = {tid: json.loads((ROOT / meta["dir"] / "task.json").read_text())
                 for tid, meta in index.items()}
    for tid, meta in index.items():
        task_json[tid]["warped"] = meta.get("warped")
    for r in rows:
        r["measures"] = index[r["maze"]]["measures"]
    groups = [
        ("GPT-6 Astra · medium", "gpt6-astra", CYAN),
        ("Gemini 4 Argon · default", "argon", AMBER),
        ("Claude Opus 5.5 · default", "opus55", GREYS[0]),
    ]
    feats = [("normalized_length", "route length"), ("turns", "turns"),
             ("route_branches", "branches"), ("min_clearance_px", "corridor width"),
             ("warped", "warped")]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    offsets = {0: 0.26, 1: 0.0, 2: -0.26}
    fits = {}
    for gi, (_label, leg, colr) in enumerate(groups):
        fit = logistic_with_cluster_ci([r for r in rows if r["provider"] == leg], task_json,
                                       resamples=300, with_warp=True)
        fits[leg] = fit
        for fi, (key, _f) in enumerate(feats):
            c = fit["coefficients"][key]
            y = fi + offsets[gi]
            lo, hi = c["ci"]
            significant = not (lo < 0 < hi)
            ax.plot([lo, hi], [y, y], color=colr, lw=2, alpha=0.85)
            ax.scatter([c["beta"]], [y], s=70, color=colr if significant else BG,
                       edgecolor=colr, linewidth=1.6, zorder=3)
    ax.axvline(0, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.set_yticks(range(len(feats)), [label for _k, label in feats])
    ax.invert_yaxis()
    ax.set_xlabel("effect on success (log-odds per standard deviation)")
    ax.text(0.985, 0.03, "filled = interval excludes zero", transform=ax.transAxes,
            color=DIM, fontsize=8, ha="right", va="bottom")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=n) for n, _l, c in groups]
    ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=DIM, loc="upper right")
    _frame(ax, "Corridor width matters most to the leaders; bending, to Claude",
           "logistic regression per model on 192 tasks · route length and turns are correlated\n"
           "(more turns at a fixed length is a denser, not a longer, maze) · task-clustered bootstrap CIs")
    save(fig, "06-regression.png")
    return fits


MODES = [
    ("corridor_departure", "cut through a wall", "#2B7A8C"),
    ("clearance_failure", "right route, no margin", CYAN),
    ("satisficing", "submitted unverified", GREEN),
    ("figure_ground_inversion", "walls↔corridors", RED),
    ("procedural_template", "never looked", "#8B5CF6"),
    ("graph_abstraction", "graph, not pixels", "#4B5568"),
    ("endpoint_misidentification", "wrong start/goal", AMBER),
    ("other", "other", "#2A3140"),
]


def failure_counts():
    per = collections.defaultdict(collections.Counter)
    known = {k for k, _l, _c in MODES}
    for line in (ANALYSIS / "failure-modes.jsonl").read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            per[v["provider"]][v["primary"] if v["primary"] in known else "other"] += 1
    return per


def fig_failure_modes():
    """Stacked share of each model's failures by mode."""
    per = failure_counts()
    legs = MAIN
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    y = np.arange(len(legs))
    left = np.zeros(len(legs))
    for key, label, colr in MODES:
        vals = np.array([100 * per[k][key] / max(1, sum(per[k].values())) for k in legs])
        ax.barh(y, vals, left=left, height=0.66, color=colr, label=label,
                edgecolor=BG, linewidth=1)
        for i, v in enumerate(vals):
            if v >= 9:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=BG if colr not in ("#4B5568", "#2A3140", "#2B7A8C", "#8B5CF6") else WHITE,
                        fontweight="bold")
        left += vals
    ax.set_yticks(y, [name(k) for k in legs])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that model's failures  (%)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), labelcolor=DIM)
    ax.invert_yaxis()
    total = sum(sum(per[k].values()) for k in legs)
    _frame(ax, "The strong models find the route and clip a wall; the rest never had it",
           f"{total:,} failed attempts at default effort, classified from trace + path geometry")
    save(fig, "07-failure-modes.png")
    return per


def fig_progress(rows):
    """How far along the reference route failed attempts got."""
    buckets = [(0, 0.25, "<25%"), (0.25, 0.5, "25–50%"), (0.5, 0.75, "50–75%"), (0.75, 1.01, "≥75%")]
    shades = ["#4B5568", "#5E7A8C", "#2B9AB0", CYAN]
    fails = collections.defaultdict(list)
    for r in rows:
        if r["provider"] in CAPABLE and not r.get("error") and not (r.get("evaluation") or {}).get("success"):
            fails[r["provider"]].append((r.get("derived") or {}).get("route_progress") or 0.0)
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    y = np.arange(len(CAPABLE))
    left = np.zeros(len(CAPABLE))
    for (lo, hi, label), shade in zip(buckets, shades):
        vals = np.array([100 * np.mean([lo <= p < hi for p in fails[k]]) for k in CAPABLE])
        ax.barh(y, vals, left=left, height=0.62, color=shade, label=f"failed {label} of the way",
                edgecolor=BG, linewidth=1)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=BG if shade == CYAN else WHITE, fontweight="bold")
        left += vals
    ax.set_yticks(y, [name(k) for k in CAPABLE])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("share of that model's failed attempts  (%)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), labelcolor=DIM)
    _frame(ax, "Most failures happen in the first half of the route",
           "how far along the reference route each failed attempt got before its first wall contact")
    save(fig, "08-progress.png")
    return {k: float(np.mean([p < 0.5 for p in fails[k]])) for k in CAPABLE}


def fig_tolerance(tasks_by_leg):
    """Macro score at each pointer radius from re-scored submissions."""
    curves = json.loads((ANALYSIS / "tolerance.json").read_text())
    radii = sorted(int(r) for r in curves)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    finals, rankings = {}, {}
    for i, leg in enumerate(CAPABLE):
        vals = []
        for r in radii:
            per_task = curves[str(r)].get(leg, {})
            vals.append(100 * macro({t: [v] for t, v in per_task.items()})["score"])
        ax.plot(radii, vals, marker="o", lw=2.4 if leg == LEAD else 1.5, color=color(leg, i),
                alpha=1 if leg == LEAD else 0.85)
        finals[leg] = vals[-1]
        rankings[leg] = vals
    for leg, y in spread_labels(finals, 3.8).items():
        ax.text(radii[-1] + 0.25, y, name(leg).split(" · ")[0], color=CYAN if leg == LEAD else DIM,
                fontsize=8, va="center")
    ax.axvline(3, color=AMBER, lw=1, ls="--", alpha=0.7)
    ax.text(3.1, 97, "scored radius", color=AMBER, fontsize=8)
    ax.set_xlabel("pointer radius (px)")
    ax.set_ylabel("pass@1  (%)")
    ax.set_xlim(0.5, radii[-1] + 3.2)
    ax.set_ylim(-3, 103)
    base = sorted(CAPABLE, key=lambda k: -rankings[k][radii.index(3)])
    swaps = sorted({tuple(sorted((a, b), key=base.index))
                    for j in range(len(radii)) for a in CAPABLE for b in CAPABLE
                    if base.index(a) < base.index(b) and rankings[a][j] < rankings[b][j]})
    stable = not swaps
    if stable:
        title = "The ranking holds at every tolerance"
    else:
        first = min(r for j, r in enumerate(radii) for a, b in swaps if rankings[a][j] < rankings[b][j])
        held = [r for r in radii if r < first]
        title = (f"The order holds from {held[0]} to {held[-1]}px; stricter, "
                 f"{len(swaps)} close pair{'s' if len(swaps) > 1 else ''} swap")
    _frame(ax, title,
           "every stored submission re-scored with pointer radii 1–8px · no new API calls")
    save(fig, "09-tolerance.png")
    return {"radii": radii, "scores": rankings, "stable": stable, "swaps": swaps}


# Label placement for the cost scatter: (x multiplier, y offset, alignment),
# chosen so neighbouring configurations do not overprint each other.
COST_LABELS = {
    "gpt61-terra-xhigh": (0.88, 2.4, "right"),
    "gpt61-sol-xhigh": (1.12, 2.2, "left"),
    "gpt6-astra": (1.12, -2.0, "left"),
    "argon": (1.12, 2.0, "left"),
    "argon-nothink": (1.12, -2.4, "left"),
    "gpt61-sol": (1.12, -1.6, "left"),
}


def fig_cost(board, rows):
    """L3/L4 score against list-price cost per attempt, effort pairs joined."""
    prices = json.loads((ROOT / "pricing.json").read_text())["models"]
    spend = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r.get("error") or parse_task(r["maze"])["level"] not in UPPER:
            continue
        price = prices.get((r.get("model") or "").split("/")[-1]) or {}
        if price.get("input") is None:
            continue
        i, o = normalize_usage(r.get("usage"))
        s = spend[r["provider"]]
        s[0] += (i * price["input"] + o * price["output"]) / 1e6
        s[1] += 1
    cost = {k: v[0] / v[1] for k, v in spend.items() if v[1]}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for e in META["effort_pairs"]:
        a, b = e["less"], e["more"]
        if a in cost and b in cost:
            ax.plot([cost[a], cost[b]], [100 * board[a]["upper"]["score"], 100 * board[b]["upper"]["score"]],
                    color=DIM, lw=1, alpha=0.6, zorder=1)
    for leg, c in cost.items():
        s = 100 * board[leg]["upper"]["score"]
        lead = leg in (LEAD, "gpt6-astra-xhigh")
        ax.scatter([c], [s], s=60, color=CYAN if lead else (GREYS[2] if leg in UPPER_ONLY else GREYS[0]),
                   zorder=3, edgecolor=BG, linewidth=1.5)
        dx, dy, ha = COST_LABELS.get(leg, (1.12, 0, "left"))
        ax.text(c * dx, s + dy, name(leg), fontsize=7.5, va="center", ha=ha,
                color=WHITE if lead else DIM)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"${v:g}"))
    ax.set_xlabel("list-price cost per attempt on L3/L4 (USD, log scale)")
    ax.set_ylabel("pass@1 on L3/L4  (%)")
    ax.set_xlim(min(cost.values()) / 1.6, max(cost.values()) * 9)
    ax.set_ylim(-3, 85)
    _frame(ax, "What a solve costs",
           "lines join the two effort settings of one model · Muse Wonder and Nemotron unpriced")
    save(fig, "10-cost.png")
    return cost


def main():
    """Rebuild every figure; write the computed findings beside them so the
    prose can be checked against the numbers it quotes."""
    print("figures ->", FIG)
    ensure_analyses()
    board, tasks = build_board(ATTEMPTS, UPPER_ONLY)
    effort = []
    for e in META["effort_pairs"]:
        effort.append({**e, "delta": paired(tasks[e["more"]], tasks[e["less"]], UPPER),
                       "more_score": board[e["more"]]["upper"], "less_score": board[e["less"]]["upper"]})
    rows = load_rows()
    for old in FIG.glob("*.png"):
        old.unlink()
    fig_leaderboard(board)
    fig_ladder(tasks)
    fig_warp(board)
    fig_effort(board, effort, rows)
    fig_families(tasks)
    fits = fig_regression([r for r in rows if not r.get("error")])
    modes = fig_failure_modes()
    early = fig_progress(rows)
    tolerance = fig_tolerance(tasks)
    cost = fig_cost(board, rows)
    findings = {
        "effort": effort,
        "regression": {leg: fit["coefficients"] for leg, fit in fits.items()},
        "failure_modes": {k: dict(v) for k, v in modes.items()},
        "failed_before_halfway": early,
        "tolerance": tolerance,
        "cost_per_attempt_upper": cost,
    }
    (ANALYSIS / "findings.json").write_text(json.dumps(findings, indent=2, default=float) + "\n")
    (ANALYSIS / "board.json").write_text(json.dumps(
        {"legs": board, "effort": effort, "meta": LEGS}, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
