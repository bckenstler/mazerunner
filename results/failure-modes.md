# Failure modes

Every failed try in the run — 4,392 across 13 model settings, 3,680 at
default effort — was classified by `mazerunner failuremodes`
(`src/mazerunner/analysis/failuremodes.py`). Per-attempt verdicts, each with
the evidence that produced it, are in `results/analysis/failure-modes.jsonl`
and in the trace viewer.

## How an attempt is labelled

Two independent kinds of evidence are read for each attempt: the model's
reasoning trace, when the provider returned one, and what the submitted path
did against the scoring mask. A trace-based mode is assigned only when the path
agrees with what the trace says. Wording alone is not enough: models narrate
uncertainty they do not act on, and act on confusions they never narrate. When
the trace says nothing diagnostic, or there is no trace, a geometric fallback
describes what the path did.

Modes are tested in precedence order and each attempt gets the most upstream
cause as its primary label. An attempt that read the walls as corridors also
collides, but the collision is the symptom.

| Mode | Label on the site | Evidence |
|---|---|---|
| `procedural_template` | never looked | trace names no coordinate, colour or position, *and* the path ignores the corridors |
| `figure_ground_inversion` | walls↔corridors | trace says walls are paths (or vice versa), and most of the path lies on walls |
| `endpoint_misidentification` | wrong start/goal | the scorer finds the path does not start on the start badge |
| `analytic_parameterisation` | drew from a formula | trace describes an arc or formula, and the points are evenly spaced along it |
| `graph_abstraction` | graph, not pixels | trace reasons over nodes and edges, and the path hops between them |
| `exploration_leakage` | exploration left in | trace explores dead ends, and the path retraces them |
| `topology_fabrication` | invented a passage | trace asserts a connection that the mask does not have |
| `fabricated_verification` | claimed a check it didn't do | trace claims the path was verified, and the path leaves the corridor |
| `satisficing` | submitted unverified | trace says the path is approximate or a best guess, then submits it |
| `clearance_failure` | right route, no margin | geometric: the centreline stays in the corridor the whole way, but the swept 3px pointer clips a wall |
| `corridor_departure` | cut through a wall | geometric: the path follows corridors, then crosses a wall |
| `near_miss_precision` | near miss | geometric: correct route to more than 60%, almost entirely inside |
| `wrong_branch` | wrong branch | geometric: never touches a wall, ends somewhere other than the goal |
| `no_usable_path` | no usable path | no submission (no tool call or malformed points) |

## Shares at default effort

Percent of each model's failures (· = none). Trace coverage matters for
reading this: GPT-6 and Gemini often return no reasoning summary, so their
labels are mostly geometric.

| mode | GPT-6 Astra | GPT-6.1 Terra | GPT-6.1 Sol | Gemini 4 Argon | Claude Opus 5.5 | Kimi K3.1 Pro | Muse Wonder 1.0 | Nemotron 3.5 Super VL |
|---|---|---|---|---|---|---|---|---|
| `clearance_failure` | 92 | 70 | 80 | 72 | 35 | 16 | 2 | 0 |
| `corridor_departure` | 4 | 24 | 14 | 12 | 44 | 20 | 39 | 17 |
| `figure_ground_inversion` | 0 | 2 | 1 | 6 | 0 | 29 | 3 | 30 |
| `satisficing` | · | · | 1 | 1 | 14 | 4 | 0 | 27 |
| `graph_abstraction` | · | · | · | 0 | 1 | 14 | 13 | 13 |
| `procedural_template` | 2 | 2 | 3 | · | 5 | · | 27 | · |
| `endpoint_misidentification` | · | · | · | 0 | · | 0 | 13 | · |
| `no_usable_path` | · | · | · | 2 | · | 8 | · | 5 |
| other modes | 2 | 3 | 1 | 6 | 1 | 9 | 2 | 8 |
| failures | 213 | 233 | 277 | 314 | 505 | 609 | 761 | 768 |
| trace diagnostic | 2% | 6% | 2% | 7% | 17% | 75% | 27% | 81% |
| trace absent | 72% | 31% | 66% | 49% | 0% | 0% | 0% | 0% |

What this says, with that caveat:

- The leaders' failures are drawing failures. Nine in ten of GPT-6 Astra's
  failed attempts follow a route that stays inside the corridor and still clip
  a wall with the 3px pointer. That matches the difficulty regression, where
  corridor width is Astra's largest effect.
- Claude Opus 5.5 mostly cuts across walls, and in 14% of its failures says
  it will commit to a best-guess path it could not verify.
- Kimi K3.1 Pro and Nemotron 3.5 Super VL most often read the walls as the
  corridors, and their traces say so.
- Muse Wonder 1.0's failures include the largest share of paths that ignore
  the maze altogether, and of paths that do not start on the start badge.

## Two rules adjusted after spot checks

The classifier was first checked by hand on an earlier set of traces. Spot
checks of these labels found two rules that misfired, and both were changed
before the figures were made:

1. **`procedural_template` now needs the path to agree.** A short trace with
   no perceptual content used to be enough. GPT-6 returns one-line reasoning
   summaries ("Planning the drag path."), so that rule labelled paths that
   stayed inside the corridor the whole way as "never looked". The label now
   also requires the path to leave the corridor; a legal centreline shows the
   image was read. This moved 173 verdicts, almost all GPT-6 and Claude, to
   `clearance_failure`.
2. **A bare "Fine." no longer counts as satisficing.** Kimi K3.1 Pro writes it
   after checking each waypoint against the image, which is verification, not
   its absence. The explicit phrases ("good enough", "approximate path", "I'll
   commit to") remain. This moved 245 verdicts, 207 of them Kimi's, to other
   labels.

These labels have not been re-checked by hand beyond these spot checks.
Treat the trace-based modes as indicative and the geometric ones
(`clearance_failure`, `corridor_departure`) as measurements.
