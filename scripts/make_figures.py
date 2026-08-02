"""Figures for the study writeup, in the promo reel's visual language.

Dark ground, cyan for the subject under discussion, amber for the thing the
reader should notice, everything else desaturated. Monospace throughout so
numbers align and the charts read as instrument output rather than slideware.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mazerunner.analysis.load import load_attempts, scored, task_means
from mazerunner.analysis.stats import bootstrap_ci, pass_at_k

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"

BG = "#090B12"
PANEL = "#0E121C"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#34D399"
RED = "#F84454"
WHITE = "#EDF2F7"
DIM = "#6E7A8C"
GRID = "#1A2030"

NAMES = {
    "gpt-xhigh": "GPT-5.6 Sol · xhigh", "openai": "GPT-5.6 Sol",
    "gemini": "Gemini 3.6 Flash", "kimi": "Kimi K3",
    "anthropic": "Claude Opus 5", "muse-spark": "Muse Spark 1.1", "inkling": "Inkling",
}
ORDER = ["gpt-xhigh", "openai", "gemini", "kimi", "anthropic", "muse-spark", "inkling"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "monospace", "font.monospace": ["Menlo", "DejaVu Sans Mono"],
    "text.color": WHITE, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": WHITE,
    "figure.dpi": 160,
})


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


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- leaderboard

def fig_leaderboard(rows):
    means = task_means(rows)
    succ = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in scored(rows):
        succ[r["provider"]][r["maze"]].append(bool((r.get("evaluation") or {}).get("success")))

    data = []
    for p in means:
        v = list(means[p].values())
        m = sum(v) / len(v)
        lo, hi = bootstrap_ci(v)
        full = [a for a in succ[p].values() if len(a) >= 8]
        p8 = sum(pass_at_k(len(a), sum(a), 8) for a in full) / len(full)
        data.append((m, lo, hi, p8, p))
    data.sort()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(data))
    for i, (m, lo, hi, p8, p) in enumerate(data):
        lead = p == "gpt-xhigh"
        ax.barh(i, 100 * p8, height=0.62, color=CYAN, alpha=0.16)
        ax.barh(i, 100 * m, height=0.62, color=CYAN if lead else "#2B7A8C")
        ax.plot([100 * lo, 100 * hi], [i, i], color=WHITE, lw=1.6, alpha=0.75)
        label_x = 100 * hi + 1.6              # past the whisker, never under it
        ax.text(label_x, i, f"{100*m:.1f}%", va="center", color=WHITE if lead else DIM,
                fontsize=10, fontweight="bold" if lead else "normal")
        p8_x = max(100 * p8, label_x + 11) + 1.6   # clears the pass@1 label too
        ax.text(p8_x, i, f"pass@8 {100*p8:.0f}%", va="center", color=DIM, fontsize=8)
    ax.set_yticks(y, [NAMES[p] for *_, p in data])
    ax.set_xlim(0, 100)
    ax.set_xlabel("pass@1  (%)")
    _frame(ax, "Leaderboard", "100 mazes × 8 attempts · bars are pass@1, ghost bars pass@8, "
                              "whiskers 95% CI clustered over tasks")
    save(fig, "01-leaderboard.png")


# ------------------------------------------------------------------ difficulty

def fig_tiers(rows):
    tiers = ["easy", "medium", "hard"]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in scored(rows):
        if r.get("tier"):
            per[r["provider"]][r["tier"]].append(bool((r.get("evaluation") or {}).get("success")))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for p in ORDER:
        vals = [100 * sum(per[p][t]) / len(per[p][t]) if per[p][t] else np.nan for t in tiers]
        lead = p.startswith("gpt")
        ax.plot(tiers, vals, marker="o", lw=2.4 if lead else 1.6,
                color=CYAN if p == "gpt-xhigh" else (WHITE if p == "openai" else DIM),
                alpha=1.0 if lead else 0.75, label=NAMES[p])
        ax.text(2.04, vals[-1], NAMES[p], color=CYAN if p == "gpt-xhigh" else DIM,
                fontsize=8, va="center")
    ax.set_ylabel("pass@1  (%)")
    ax.set_xlim(-0.15, 2.9)
    _frame(ax, "Difficulty is monotone for every model",
           "measured tiers, not declared")
    save(fig, "02-tiers.png")


# --------------------------------------------------------------- effort sweeps

def fig_effort():
    series = [
        ("GPT-5.6 Sol", ["low", "medium", "high", "xhigh"], [36, 56, 63, 73], CYAN, 2.8),
        ("Gemini 3.6 Flash", ["low", "medium", "high"], [32, 32, 32], "#A8B2C4", 1.8),
        ("Claude Opus 5", ["low", "medium", "high", "xhigh"], [8, 7, 13, 12], "#768094", 1.8),
        ("Kimi K3", ["low", "high", "max"], [16, 16, 28], "#4B5568", 1.8),
    ]
    ticks = ["low", "medium", "high", "xhigh/max"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name, levels, values, colr, lw in series:
        xs = [ticks.index(l) if l in ticks else 3 for l in levels]
        ax.plot(xs, values, marker="o", color=colr, lw=lw)
        ax.text(xs[-1] + 0.06, values[-1], f" {name} · {values[-1]}%", color=colr,
                fontsize=8.5, va="center")
    ax.set_xticks(range(4), ticks)
    ax.set_ylabel("pass@1  (%)")
    ax.set_xlim(-0.1, 4.4)
    ax.set_ylim(0, 80)
    _frame(ax, "Only GPT converts test-time compute into accuracy",
           "25-task sweep set × 3 trials. Gemini returns 32% three times across a "
           "6× increase in thinking tokens")
    save(fig, "03-effort.png")


# ------------------------------------------------------------------- tolerance

def fig_tolerance():
    radii = [1, 2, 3, 5, 8]
    curves = {
        "gpt-xhigh": [68.4, 65.5, 60.6, 48.9, 31.4], "openai": [55.9, 53.4, 48.4, 38.8, 26.8],
        "gemini": [33.2, 31.1, 29.8, 22.0, 12.9], "kimi": [22.6, 22.1, 20.6, 15.5, 8.2],
        "anthropic": [20.4, 17.8, 16.4, 12.1, 6.0], "muse-spark": [7.1, 6.6, 6.0, 5.1, 3.6],
        "inkling": [0, 0, 0, 0, 0],
    }
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for p in ORDER:
        colr = CYAN if p == "gpt-xhigh" else (WHITE if p == "openai" else DIM)
        ax.plot(radii, curves[p], marker="o", lw=2.2 if p == "gpt-xhigh" else 1.5,
                color=colr, alpha=1 if p.startswith("gpt") else 0.8)
        ax.text(8.2, curves[p][-1], NAMES[p], color=colr, fontsize=8, va="center")
    ax.axvline(3, color=AMBER, lw=1, ls="--", alpha=0.7)
    ax.text(3.1, 72, "scored radius", color=AMBER, fontsize=8)
    ax.set_xlabel("pointer radius (px)")
    ax.set_ylabel("pass@1  (%)")
    ax.set_xlim(0.5, 11.5)
    _frame(ax, "The ranking never inverts", "re-scoring stored submissions at other tolerances — "
                                            "no order change anywhere from 1px to 8px")
    save(fig, "04-tolerance.png")


# --------------------------------------------------------------- failure modes

def fig_failure_modes():
    per = collections.defaultdict(collections.Counter)
    for line in (ROOT / "results/failure-modes.jsonl").read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            per[v["provider"]][v["primary"]] += 1
    modes = [
        ("corridor_departure", "cut through a wall", "#2B7A8C"),
        ("clearance_failure", "no margin", CYAN),
        ("endpoint_misidentification", "wrong start", AMBER),
        ("figure_ground_inversion", "walls↔corridors", RED),
        ("procedural_template", "never looked", "#8B5CF6"),
        ("satisficing", "gave up verifying", GREEN),
        ("analytic_parameterisation", "drew from formula", "#94A3B8"),
        ("graph_abstraction", "graph, not pixels", "#4B5568"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    y = np.arange(len(ORDER))
    left = np.zeros(len(ORDER))
    for key, label, colr in modes:
        vals = np.array([100 * per[p][key] / max(1, sum(per[p].values())) for p in ORDER])
        ax.barh(y, vals, left=left, height=0.66, color=colr, label=label)
        for i, v in enumerate(vals):
            if v >= 9:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=BG, fontweight="bold")
        left += vals
    ax.set_yticks(y, [NAMES[p] for p in ORDER])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that model's failures  (%)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), labelcolor=DIM)
    ax.invert_yaxis()
    _frame(ax, "Every model fails in its own way",
           "4,140 failed attempts classified from trace + geometry")
    save(fig, "05-failure-modes.png")


# ---------------------------------------------------------------- ablations

def fig_resolution():
    data = {
        "gpt-xhigh": (51, 68, 65), "openai": (32, 54, 58), "gemini": (30, 26, 28),
        "kimi": (17, 20, 42), "anthropic": (6, 18, 22), "inkling": (0, 0, 0),
    }
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = [0, 1, 2]
    for p, vals in data.items():
        hero = p == "kimi"
        ax.plot(xs, vals, marker="o", lw=2.8 if hero else 1.5,
                color=AMBER if hero else DIM, alpha=1 if hero else 0.7)
        ax.text(2.06, vals[-1], NAMES[p], color=AMBER if hero else DIM, fontsize=8, va="center")
    ax.set_xticks(xs, ["0.5×", "1.0× (main)", "2.0×"])
    ax.set_ylabel("pass@1  (%)")
    ax.set_xlim(-0.1, 2.9)
    _frame(ax, "Kimi doubles on pixels alone",
           "same mazes, resent at other resolutions — and its effort ladder was flat at ~16%")
    save(fig, "06-resolution.png")


def fig_feedback():
    data = [("gpt-xhigh", 60, 63), ("openai", 25, 48), ("gemini", 8, 24),
            ("kimi", 12, 18), ("anthropic", 9, 14)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(data))
    ax.barh(y - 0.19, [d[2] for d in data], height=0.36, color="#3B4658", label="blind retry")
    ax.barh(y + 0.19, [d[1] for d in data], height=0.36, color=RED,
            label="retry after seeing its own error")
    for i, (p, resc, blind) in enumerate(data):
        ax.text(max(resc, blind) + 1.2, i, f"{resc-blind:+d}pp", va="center",
                color=RED, fontsize=9, fontweight="bold")
    ax.set_yticks(y, [NAMES[p] for p, *_ in data])
    ax.set_xlabel("recovery rate after a failed first attempt  (%)")
    ax.legend(fontsize=8, frameon=False, labelcolor=DIM, loc="lower right")
    ax.invert_yaxis()
    _frame(ax, "Showing a model its own mistake makes it worse",
           "episodes whose first turn failed. No model beats simply trying again")
    save(fig, "07-feedback.png")


def fig_variance():
    comps = [("Topology", 85.4, (69.7, 91.8), CYAN),
             ("Topology × style", 14.2, (7.4, 28.5), AMBER),
             ("Style (main effect)", 0.4, (0.2, 3.8), DIM)]
    fig, ax = plt.subplots(figsize=(8, 3.1))
    y = np.arange(len(comps))
    for i, (name, share, ci, colr) in enumerate(comps):
        ax.barh(i, share, height=0.55, color=colr)
        ax.plot(ci, [i, i], color=WHITE, lw=1.6, alpha=0.8)
        ax.text(share + 1.6, i, f"{share:.1f}%  [{ci[0]:.1f}, {ci[1]:.1f}]", va="center",
                color=WHITE if i < 2 else DIM, fontsize=9)
    ax.set_yticks(y, [c[0] for c in comps])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of variance  (%)")
    ax.invert_yaxis()
    _frame(ax, "Style is a per-maze axis, not a global one",
           "20 topologies × 5 archetypes, identical masks")
    save(fig, "08-variance.png")


def fig_dimensions():
    """Paired delta chart: what disclosing the canvas size does per model."""
    data = [("GPT-5.6 Sol", 48, 56, 7), ("Claude Opus 5", 16, 18, 0),
            ("GPT-5.6 Sol · xhigh", 61, 61, None), ("Gemini 3.6 Flash", 30, 22, -12)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(data))
    for i, (name, frozen, disclosed, pilot) in enumerate(data):
        delta = disclosed - frozen
        colr = GREEN if delta > 2 else (RED if delta < -2 else DIM)
        ax.plot([frozen, disclosed], [i, i], color=colr, lw=3, solid_capstyle="round")
        ax.scatter([frozen], [i], s=70, color=DIM, zorder=3, label="_")
        ax.scatter([disclosed], [i], s=90, color=colr, zorder=3, marker="D")
        ax.annotate("", xy=(disclosed, i), xytext=(frozen, i),
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=2))
        ax.text(max(frozen, disclosed) + 2, i, f"{delta:+d}pp", va="center",
                color=colr, fontsize=10, fontweight="bold")
        if pilot is not None:
            ax.text(max(frozen, disclosed) + 11, i, f"(pilot {pilot:+d})", va="center",
                    color=DIM, fontsize=8)
    ax.set_yticks(y, [d[0] for d in data])
    ax.set_xlabel("pass@1  (%)   ○ frozen prompt → ◆ canvas size disclosed")
    ax.set_xlim(0, 80)
    ax.invert_yaxis()
    _frame(ax, "Telling the model the canvas size redistributes, nets ~zero",
           "100 mazes × 2 trials, paired per task against the main run")
    save(fig, "10-dimensions.png")


def fig_blind():
    """The blind control as a picture: sighted vs blank vs wrong image."""
    sighted = {"gpt-xhigh": 68, "openai": 54, "gemini": 26, "kimi": 20,
               "anthropic": 18, "muse-spark": 4, "inkling": 0}
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(ORDER))
    ax.barh(y - 0.2, [sighted[p] for p in ORDER], height=0.38,
            color="#2B7A8C", label="real image")
    ax.barh(y + 0.2, [0] * len(ORDER), height=0.38, color=RED)
    for i, p in enumerate(ORDER):
        ax.text(sighted[p] + 1.2, i - 0.2, f"{sighted[p]}%", va="center",
                color=DIM, fontsize=9)
        ax.text(1.2, i + 0.2, "0%  blank · 0%  wrong image", va="center",
                color=RED, fontsize=8)
    ax.set_yticks(y, [NAMES[p] for p in ORDER])
    ax.set_xlim(0, 80)
    ax.set_xlabel("pass@1  (%)  ·  25 tasks × 2 trials per condition")
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False, labelcolor=DIM, loc="lower right")
    _frame(ax, "Take the image away and every model drops to zero",
           "same tasks, same prompt — only the pixels change. 0 passes in 700 attempts")
    save(fig, "11-blind.png")


def fig_fingerprints():
    """Two perception signatures, one scatter. Labels hand-placed to avoid
    collisions; the subtitle wraps so the canvas keeps its aspect."""
    snap = {"gpt-xhigh": 8.7, "openai": 11.5, "kimi": 19.7, "gemini": 38.9,
            "anthropic": 27.3, "muse-spark": 19.2, "inkling": 67.9}
    loc = {"gpt-xhigh": 1.3, "openai": 1.3, "kimi": 1.8, "gemini": 2.1,
           "anthropic": 2.1, "muse-spark": 8.2, "inkling": 24.7}
    # (dx, dy multiplier, alignment) per label, to keep them off each other
    nudge = {"gpt-xhigh": (1.5, 0.88, "left"), "openai": (1.5, 1.12, "left"),
             "kimi": (1.5, 1.0, "left"), "gemini": (1.5, 1.0, "left"),
             "anthropic": (-1.5, 1.0, "right"), "muse-spark": (1.5, 1.0, "left"),
             "inkling": (-1.5, 1.0, "right")}
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for p in ORDER:
        hero = p == "inkling"
        colr = RED if hero else (CYAN if p.startswith("gpt") else DIM)
        ax.scatter(snap[p], loc[p], s=190 if hero else 90, color=colr,
                   zorder=3, edgecolor=BG, linewidth=1.2)
        dx, dy, ha = nudge[p]
        ax.text(snap[p] + dx, loc[p] * dy, NAMES[p], fontsize=8, color=colr,
                va="center", ha=ha)
    ax.annotate("reads the image:\nmessy decimals, starts on the badge",
                xy=(9, 1.3), xytext=(24, 0.72), color=CYAN, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=CYAN, lw=0.8, alpha=0.6))
    ax.annotate("invents the answer:\nround numbers, misses the badge",
                xy=(66, 22), xytext=(38, 12), color=RED, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, alpha=0.6))
    ax.set_xlabel("share of coordinates that are round numbers (exact 0.01 grid)")
    ax.set_ylabel("how far the path starts from\nthe start badge (median px, log)")
    ax.set_yscale("log")
    ax.set_xlim(0, 88)
    ax.set_ylim(0.55, 45)
    _frame(ax, "Is the model measuring, or making numbers up?",
           "every submitted coordinate, two questions: are the numbers round, and does\n"
           "the path begin where the start badge actually is")
    save(fig, "09-fingerprints.png")


def main():
    print("figures ->", FIG)
    rows = load_attempts([ROOT / "results/main/merged/attempts.jsonl"], ROOT / "datasets/v1/dev")
    fig_leaderboard(rows)
    fig_tiers(rows)
    fig_effort()
    fig_tolerance()
    fig_failure_modes()
    fig_resolution()
    fig_feedback()
    fig_variance()
    fig_dimensions()
    fig_blind()
    fig_fingerprints()


if __name__ == "__main__":
    main()
