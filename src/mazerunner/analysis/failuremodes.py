"""Per-attempt failure-mode classification.

The scorer says where an attempt broke; this says why. Each attempt is judged
on two independent kinds of evidence — what the model said in its trace, and
what its submitted path actually did against the mask — and a mode is only
assigned when both agree. Wording alone is not enough: models narrate
uncertainty they do not act on, and act on confusions they never narrate.

Modes are tested in precedence order so each attempt gets one primary label,
the most upstream cause. An attempt that inverted figure and ground also
collides, but the collision is the symptom.

Taxonomy and worked examples: results/failure-modes.md
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..geometry import densify_polyline

# Ordered: earliest match wins. Causes named by the trace precede the purely
# geometric descriptions, which act as fallbacks for attempts whose trace says
# nothing diagnostic (or, for some providers, says nothing at all).
PRECEDENCE = (
    "procedural_template",
    "figure_ground_inversion",
    "endpoint_misidentification",
    "analytic_parameterisation",
    "graph_abstraction",
    "exploration_leakage",
    "topology_fabrication",
    "fabricated_verification",
    "satisficing",
    # geometric fallbacks
    "clearance_failure",
    "corridor_departure",
    "near_miss_precision",
    "wrong_branch",
)

LEXICAL = {
    "figure_ground_inversion": (
        r"\bwalls?\b[^.]{0,40}\b(are|as)\b[^.]{0,20}\b(paths?|corridors?|routes?|traversable)",
        r"\b(corridors?|paths?)\b[^.]{0,40}\b(are|as)\b[^.]{0,20}\bwalls?\b",
        r"\b(actually|instead)\b[^.]{0,50}\b(traversable|walkable|open)\b",
        r"\binvert(ed|ing)?\b[^.]{0,30}\b(wall|corridor|figure|ground)",
        r"\b(background|bands?|regions?)\b[^.]{0,30}\brepresents?\b[^.]{0,20}\bwalls?\b",
    ),
    "topology_fabrication": (
        r"\b(hidden|secret|implicit)\b[^.]{0,30}\bconnect(ion|ed|s)?\b",
        r"\bteleport",
        r"\b(likely|probably|must be|could be)\b[^.]{0,40}\b(interconnected|connected)\b",
        r"\bassume\b[^.]{0,40}\b(connect|passage|opening|gap)\b",
        r"\bthere (must|should) be\b[^.]{0,30}\b(a )?(gap|opening|passage|corridor)\b",
    ),
    "analytic_parameterisation": (
        r"\bradius\b[^.]{0,30}\b(about|approximately|~|of)\b",
        r"\b(stepping|step) through angles?\b",
        r"\b(polar|angular)\b[^.]{0,20}\b(coordinates?|sweep|steps?)\b",
        r"\bsmooth arc\b",
        r"\barc path\b",
        r"\bcounterclockwise|clockwise\b[^.]{0,40}\b(spiral|ring|around)\b",
        r"\bat angle\b\s*\d",
    ),
    "graph_abstraction": (
        r"\bnodes?\b[^.]{0,30}\b(connected|edges?|graph)\b",
        r"\b[A-J]\s*\((start|goal)\)",
        r"\blet'?s identify (the )?nodes\b",
        r"\bnode\s+[A-J]\b",
        r"\badjacency\b",
    ),
    "exploration_leakage": (
        r"\bdead[- ]end\b[^.]{0,60}\b(avoid|not the correct|wrong)\b",
        r"\bthis is not the correct path\b",
        r"\bbacktrack",
        r"\bgo back to the (start|beginning)\b",
        r"\bfrom the starting point,? (go|head)\b",
    ),
    "fabricated_verification": (
        r"\b(checked|verified|confirmed)\b[^.]{0,30}\b(all|entire|every|whole)\b",
        r"\bverified the (entire|whole|full)\b",
        r"\bno collisions?\b",
        r"\bstays? (well )?(within|inside) the corridor\b",
        r"\bthis path is (valid|clear|safe)\b",
        r"\bi'?m confident in this (path|route)\b",
        r"\bcheck(ed|ing)? its integrity\b",
        r"\bno (blockages?|obstructions?|dead ends?) (on the way|along)\b",
        r"\beach node connects correctly\b",
    ),
    "satisficing": (
        r"\b(can be|is) approximate\b",
        r"\bgood enough\b",
        r"\bmaybe\b[^.]{0,20}\.\s*(fine|ok|okay)\b",
        r"\bfine\.\s",
        r"\bnot (fully |entirely )?sure\b[^.]{0,40}\b(submit|proceed|go with)\b",
        r"\bi'?ll (just )?(commit|go with|trust)\b",
        r"\bapproximate(ly)? (route|path)\b",
    ),
}

# Any of these implies the trace actually looked at something specific.
PERCEPTUAL = (
    r"\(\s*\d+\s*,\s*\d+\s*\)",          # pixel pairs
    r"\b0\.\d{2,}\b",                     # normalized coords
    r"\bx\s*[=~]\s*\d",
    r"\by\s*[=~]\s*\d",
    r"\b(cyan|amber|teal|orange)\b",
    r"\b(top|bottom)[- ](left|right)\b",
)


@dataclass
class Verdict:
    """One attempt's classification: the mode, plus what justified it.

    `evidence` carries the trace quote and the reason, and `measures` the
    geometry that corroborated it, so every verdict in
    results/failure-modes.jsonl can be audited without re-running the
    classifier.
    """

    primary: str = "unclassified"
    secondary: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)


def _points_px(task: dict, submission: dict | None) -> list[tuple[float, float]]:
    if not isinstance(submission, dict):
        return []
    out = []
    w, h = task["width"], task["height"]
    for p in submission.get("points") or []:
        if isinstance(p, dict) and "x" in p and "y" in p:
            try:
                out.append((float(p["x"]) * (w - 1), float(p["y"]) * (h - 1)))
            except (TypeError, ValueError):
                continue
    return out


def geometry_measures(task: dict, mask: np.ndarray, points: list[tuple[float, float]]) -> dict:
    """What the submitted path actually did, independent of what was said."""
    if len(points) < 2:
        return {"points": len(points), "outside_fraction": 1.0 if points else 0.0}

    dense = densify_polyline(points, 2.0)
    h, w = mask.shape
    inside = 0
    first_outside_at = None
    for i, (x, y) in enumerate(dense):
        xi, yi = int(round(x)), int(round(y))
        ok = 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi])
        inside += ok
        if not ok and first_outside_at is None:
            first_outside_at = i / len(dense)
    outside_fraction = 1.0 - inside / len(dense)

    segs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    mean_seg = sum(segs) / len(segs)
    spacing_cv = (
        (sum((s - mean_seg) ** 2 for s in segs) / len(segs)) ** 0.5 / mean_seg
        if mean_seg > 0 else 0.0
    )

    angles = []
    for i in range(1, len(points) - 1):
        a = math.atan2(points[i][1] - points[i - 1][1], points[i][0] - points[i - 1][0])
        b = math.atan2(points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0])
        d = (b - a + math.pi) % (2 * math.pi) - math.pi
        angles.append(abs(d))
    mean_turn = sum(angles) / len(angles) if angles else 0.0
    turn_cv = (
        (sum((a - mean_turn) ** 2 for a in angles) / len(angles)) ** 0.5 / mean_turn
        if angles and mean_turn > 1e-6 else 0.0
    )
    reversals = sum(1 for a in angles if a > 2.0)  # >115 degrees: a doubling back

    # Does the path revisit somewhere it already was?
    revisit = 0.0
    if len(dense) > 40:
        step = max(1, len(dense) // 200)
        sampled = dense[::step]
        for i, p in enumerate(sampled):
            for q in sampled[i + 8:]:
                if math.dist(p, q) < 12:
                    revisit += 1
                    break
        revisit /= max(1, len(sampled))

    start_px = (task["start"]["x"] * (w - 1), task["start"]["y"] * (h - 1))
    return {
        "points": len(points),
        "outside_fraction": outside_fraction,
        "first_outside_at": first_outside_at,
        "spacing_cv": spacing_cv,
        "turn_cv": turn_cv,
        "reversals": reversals,
        "revisit": revisit,
        "mean_segment_px": mean_seg,
        "start_error_px": math.dist(points[0], start_px),
        "start_radius_px": task["start_radius_px"],
    }


def lexical_hits(trace: str | None) -> dict[str, list[str]]:
    """mode -> matched phrases in the model's own reasoning.

    Lexical evidence alone never decides a verdict: a model claiming it
    verified the route is a claim, not a fact, so `classify` requires geometry
    to agree before it assigns the mode.
    """
    if not trace:
        return {}
    low = trace.lower()
    hits: dict[str, list[str]] = {}
    for mode, patterns in LEXICAL.items():
        for pat in patterns:
            m = re.search(pat, low)
            if m:
                s = max(0, m.start() - 40)
                hits.setdefault(mode, []).append(trace[s:m.end() + 40].replace("\n", " ").strip())
    return hits


def has_perceptual_content(trace: str | None) -> bool:
    """Whether the trace mentions anything it could only have seen.

    Separates a trace that is uninformative from one that is absent — a model
    that reasons at length without ever referring to the image is a finding,
    not missing data.
    """
    if not trace:
        return False
    return any(re.search(p, trace.lower()) for p in PERCEPTUAL)


def classify(row: dict, task: dict, mask: np.ndarray) -> Verdict:
    """Primary failure mode for one attempt, with the evidence behind it."""
    verdict = Verdict()
    evaluation = row.get("evaluation") or {}
    if evaluation.get("success"):
        verdict.primary = "pass"
        return verdict
    if row.get("error"):
        verdict.primary = "transport"
        return verdict

    trace = row.get("reasoning")
    points = _points_px(task, row.get("submission"))
    g = geometry_measures(task, mask, points)
    lex = lexical_hits(trace)
    progress = (row.get("derived") or {}).get("route_progress", 0.0)
    verdict.measures = g

    if not points:
        verdict.primary = "no_usable_path"
        return verdict

    # --- candidate tests: each needs lexical AND geometric agreement ---
    candidates: dict[str, dict] = {}

    # "Nothing was looked at" requires the trace to be uninformative in *both*
    # senses: no perceptual detail and no diagnostic cue. A trace that says
    # nothing about coordinates but plainly states a confusion is evidence of
    # that confusion, not of an empty template.
    if trace is not None and not has_perceptual_content(trace) and not lex and len(trace) < 1200:
        candidates["procedural_template"] = {
            "why": "trace names no coordinate, colour, or position, and no recognisable cause",
            "trace_chars": len(trace),
        }

    first_out = g["first_outside_at"] if g["first_outside_at"] is not None else 1.0
    if g["outside_fraction"] > 0.45 and first_out < 0.25:
        # Geometry alone can support this only when the path *started* in the
        # right place: a wrong start with everything offset is more
        # parsimoniously an endpoint failure, unless the trace says otherwise.
        started_right = evaluation.get("starts_correctly") is not False
        if "figure_ground_inversion" in lex or (g["outside_fraction"] > 0.7 and started_right):
            candidates["figure_ground_inversion"] = {
                "why": f"{g['outside_fraction']:.0%} of the path lies outside the open mask, "
                       f"leaving it at {first_out:.0%} of the way in",
                "quote": (lex.get("figure_ground_inversion") or [None])[0],
            }

    # The scorer already decided this one; no need to re-derive it heuristically.
    if evaluation.get("starts_correctly") is False:
        candidates["endpoint_misidentification"] = {
            "why": f"path begins {g['start_error_px']:.0f}px from the start badge "
                   f"(acceptance radius {g['start_radius_px']:.0f}px)",
        }

    if "analytic_parameterisation" in lex and g["spacing_cv"] < 0.55 and g["points"] >= 12:
        candidates["analytic_parameterisation"] = {
            "why": f"evenly spaced points (spacing CV {g['spacing_cv']:.2f}) generated from a "
                   f"geometric construction",
            "quote": lex["analytic_parameterisation"][0],
        }

    if "graph_abstraction" in lex and g["mean_segment_px"] > 55:
        candidates["graph_abstraction"] = {
            "why": f"long straight hops between inferred nodes "
                   f"(mean segment {g['mean_segment_px']:.0f}px)",
            "quote": lex["graph_abstraction"][0],
        }

    if "exploration_leakage" in lex and (g["reversals"] >= 1 or g["revisit"] > 0.12):
        candidates["exploration_leakage"] = {
            "why": f"{g['reversals']} sharp reversal(s), {g['revisit']:.0%} of the path revisits "
                   f"ground it already covered",
            "quote": lex["exploration_leakage"][0],
        }

    if "topology_fabrication" in lex and progress > 0.25:
        candidates["topology_fabrication"] = {
            "why": f"coherent route to {progress:.0%} of the way, through a connection "
                   f"the trace invents",
            "quote": lex["topology_fabrication"][0],
        }

    if "fabricated_verification" in lex:
        candidates["fabricated_verification"] = {
            "why": "claims the route was checked; it collides",
            "quote": lex["fabricated_verification"][0],
        }

    if "satisficing" in lex:
        candidates["satisficing"] = {
            "why": "states the path is approximate or unverified, then submits",
            "quote": lex["satisficing"][0],
        }

    # --- geometric fallbacks: what the path did, when the trace does not say why ---

    legal_centreline = g["outside_fraction"] < 0.02
    collided = evaluation.get("collision_free") is False

    if legal_centreline and collided:
        # Every sampled centre sits on open pixels, yet the swept 3px pointer
        # still clipped a wall: the route is right and the margin is not.
        candidates["clearance_failure"] = {
            "why": "centreline stays inside the corridor the whole way, but the swept pointer "
                   "clips a wall — the route is correct, the margin is not",
        }

    if not legal_centreline and collided:
        candidates["corridor_departure"] = {
            "why": f"traces the corridor, then cuts across a wall: "
                   f"{g['outside_fraction']:.0%} of the path is outside the open region, "
                   f"first leaving at {(g['first_outside_at'] or 0):.0%} of the way",
        }

    if (
        progress > 0.6
        and g["outside_fraction"] < 0.08
        and g["start_error_px"] <= g["start_radius_px"]
    ):
        candidates["near_miss_precision"] = {
            "why": f"correct route to {progress:.0%}, only {g['outside_fraction']:.1%} of the "
                   f"path outside the corridor",
        }

    if legal_centreline and not collided:
        # Never touched a wall, never reached the goal: a legal route to the
        # wrong place. A planning error, not a perception one.
        candidates["wrong_branch"] = {
            "why": f"stays legal the whole way but ends elsewhere, reaching {progress:.0%} "
                   f"of the certified route",
        }

    for mode in PRECEDENCE:
        if mode in candidates:
            verdict.primary = mode
            verdict.evidence = candidates[mode]
            break
    verdict.secondary = [m for m in PRECEDENCE if m in candidates and m != verdict.primary]
    # Whether a cause could be named at all depends on the provider returning a
    # trace; 29% of the first pass's residual had none. Record it so a
    # geometry-only label is never mistaken for a trace-supported one.
    verdict.evidence["trace"] = (
        "absent" if not trace else ("uninformative" if not lex else "diagnostic")
    )
    return verdict


def classify_run(attempts_path, dataset_dir, out_path=None):
    """Classify every failed attempt in a merged run, one trace at a time.

    Writes one verdict per attempt (JSONL) so any label can be traced back to
    the evidence that produced it, and returns the summary.
    """
    import json
    from collections import Counter, defaultdict
    from pathlib import Path as _P

    from PIL import Image

    index = {json.loads(l)["task_id"]: json.loads(l)
             for l in (_P(dataset_dir) / "index.jsonl").read_text().splitlines() if l.strip()}
    cache = {}

    def load(task_id):
        if task_id not in cache:
            d = _P(index[task_id]["dir"])
            task = json.loads((d / "task.json").read_text())
            mask = np.asarray(Image.open(d / task["mask_file"]).convert("L")) > 127
            cache[task_id] = (task, mask)
        return cache[task_id]

    counts = Counter()
    per_model = defaultdict(Counter)
    trace_quality = Counter()
    verdicts = []

    with _P(attempts_path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or (row.get("evaluation") or {}).get("success"):
                continue
            if row.get("maze") not in index:
                continue
            task, mask = load(row["maze"])
            v = classify(row, task, mask)
            counts[v.primary] += 1
            per_model[row["provider"]][v.primary] += 1
            trace_quality[v.evidence.get("trace")] += 1
            verdicts.append({
                "provider": row["provider"],
                "maze": row["maze"],
                "trial": row.get("trial"),
                "primary": v.primary,
                "secondary": v.secondary,
                "evidence": v.evidence,
                "measures": {k: (round(x, 4) if isinstance(x, float) else x)
                             for k, x in v.measures.items()},
            })

    if out_path:
        with _P(out_path).open("w") as out:
            for v in verdicts:
                out.write(json.dumps(v) + "\n")

    total = sum(counts.values())
    return {
        "failures": total,
        "counts": dict(counts.most_common()),
        "per_model": {m: dict(c.most_common()) for m, c in per_model.items()},
        "trace_quality": dict(trace_quality),
        "coverage": 1.0 - counts["unclassified"] / total if total else 0.0,
    }
