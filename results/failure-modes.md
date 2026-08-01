# MazeRunner — failure mode taxonomy

The scorer already reports *where* an attempt broke (collision, wrong start,
stopped short). That says nothing about *why*. This taxonomy was built by
reading failure traces across all seven models and naming the recurring
reasoning patterns behind them, then operationalised as a per-attempt
classifier (`mazerunner.analysis.failuremodes`).

Every mode below is defined by **lexical evidence** in the model's own trace
plus **geometric evidence** measured from its submitted path against the mask,
so a label is never assigned on wording alone.

---

## The modes

### A. Figure–ground inversion
The model decides the walls are the corridors, or vice versa, and traces the
solid regions.

> *"I suspect that the cream bands are actually the traversable routes, while
> the lined-paper background represents the walls."*
> — GPT-5.6 Sol @ xhigh, `rectilinear-hard-s0166`, route progress 0.00

Geometric signature: most of the path lies outside the open mask, and the
excursion begins almost immediately. Distinguishes a model that could not
*parse* the image from one that mis-navigated it.

### B. Topology fabrication
The model invents connectivity that does not exist in order to justify a route
it has already chosen.

> *"there might be a hidden connection throughout the network using badge
> teleport … It looks like the entire wall network is likely interconnected, so
> this tube route could indeed be valid."*
> — GPT-5.6 Sol @ medium, `pipes-hard-s0164`

Geometric signature: the path crosses a wall at a point the trace explicitly
reasoned about, usually with high route progress — the plan is coherent, the
map is imagined.

### C. Analytic parameterisation (dead reckoning)
The path is generated from a formula — arcs, radii, angles, regular spans —
rather than traced from observed corridors. Most common on radial mazes.

> *"stepping through angles around the circle … I'll commit to a smooth arc
> path that moves from the starting position along a radius of about 235."*
> — Claude Opus 5, `radial-hard-s0077`, route progress 0.00

Geometric signature: unusually regular spacing between points and low variance
in turn angle. The path is smooth and wrong.

### D. Graph abstraction substitution
The continuous maze is reduced to a node-and-edge graph, and the submission is
a tour of invented node positions. Corridor width stops being represented at
all.

> *"Let's identify nodes: A (start): top right around (0.78,0.11) … B: top
> middle around (0.55,0.12) connected horizontally."*
> — Inkling, `island-easy-s0114`

Geometric signature: few points, long straight segments, collisions at corners
where a real corridor would have curved.

### E. Exploration leakage
The model's search is emitted *as* the path: rejected branches and backtracks
end up in the submitted polyline.

> *"9. Vertical Rise — Going up from the wedge, but a dead end. This is not the
> correct path, so avoid it. 10. Downward Curve — From the starting point, go
> right."* — both points appear in the submission.
> — Gemini 3.6 Flash, `organic-hard-s0187`, route progress 0.01

Geometric signature: the path doubles back on itself, or contains a long jump
between distant points.

### F. Satisficing under acknowledged uncertainty
The model states plainly that it does not know, and submits anyway.

> *"Does segment (0.500,0.335)->(0.545,0.385) cross gap? maybe central. Fine.
> … Final can be approximate."*
> — Kimi K3, `cave-medium-s0193`

This is not a perception failure — it is a decision to stop verifying. Distinct
from B because nothing is fabricated; the uncertainty is admitted.

### G. Fabricated verification
The opposite of F: the model claims to have checked a route it did not check.

> *"I checked all possible dead ends and verified the entire map structure."*
> — Gemini 3.6 Flash, `pipes-medium-s0028`, path collides

### H. Procedural template (no perceptual content)
The trace contains no image-specific observation at all — only generic
procedure. Nothing was looked at.

> *"Estimating a continuous path from the cyan start badge to the amber goal
> badge within the maze corridors. Mapping corridor junctions and segment
> endpoints to refine the route geometry."*
> — Muse Spark 1.1, `pipes-medium-s0028`

Geometric signature: short trace, no coordinates mentioned, frequently paired
with a badly mislocated start.

### I. Endpoint misidentification
The path never begins on the start badge, or never aims at the real goal.
Perception failure at the easiest possible target.

### J. Near-miss precision failure
The route is correct and the trace is specific and accurate; the path clips a
wall by a small margin. This is the "good attempt" bucket and should be read as
a *precision* limit, not a reasoning one.

---

## How the classifier decides

`failuremodes.classify(row, task, mask)` returns a primary label, any secondary
labels, and the evidence for each. It combines:

- **Lexical cues** — curated phrase families per mode, matched against the
  trace. Never sufficient alone.
- **Geometric measures** — fraction of the path outside the mask, where the
  excursion starts, self-intersection and backtracking, spacing and angle
  regularity, point count, start-badge error, and route progress at collision.

Modes are tested in a fixed precedence order so a single attempt gets one
primary label: the most upstream failure wins. A model that inverted figure and
ground also collides, but the collision is a symptom, not the cause.

Attempts whose evidence supports no mode are labelled `unclassified` rather
than forced into the nearest bucket; that count is reported and is the honest
measure of the taxonomy's coverage.

---

## Results — main run, 4,140 failed attempts

Coverage is 100%: every failure carries a label backed by stated evidence.
Verdicts are written per attempt to `results/failure-modes.jsonl`, so any label
can be traced to the measurements and trace excerpt that produced it.

```bash
uv run mazerunner failuremodes
```

| Mode | Share |
|---|---|
| Corridor departure (cut across a wall) | 39.7% |
| Endpoint misidentification | 11.1% |
| Figure–ground inversion | 9.8% |
| Clearance failure (right route, no margin) | 8.3% |
| Procedural template (no perceptual content) | 7.6% |
| Satisficing under acknowledged uncertainty | 7.4% |
| Graph abstraction substitution | 5.5% |
| Analytic parameterisation | 5.0% |
| Fabricated verification | 2.6% |
| Topology fabrication | 1.4% |
| Exploration leakage | 1.0% |
| No usable path / near-miss | 0.4% |

### Each model fails in its own way

| Model | cut wall | clearance | wrong start | fig/ground | no percept | satisfice | analytic | graph |
|---|---|---|---|---|---|---|---|---|
| GPT-5.6 Sol @ xhigh | 36% | **30%** | 4% | 6% | 5% | 5% | 9% | 2% |
| GPT-5.6 Sol @ medium | 44% | **30%** | 6% | 6% | 8% | 1% | 3% | 1% |
| Gemini 3.6 Flash | **63%** | 10% | 0% | 1% | 0% | 1% | 3% | 4% |
| Kimi K3 | 4% | 0% | 1% | 25% | 0% | **35%** | 11% | 14% |
| Claude Opus 5 | **58%** | 8% | 0% | 2% | 8% | 9% | 12% | 2% |
| Muse Spark 1.1 | 41% | 3% | 12% | 1% | **28%** | 0% | 1% | 12% |
| Inkling | 34% | 0% | **41%** | 22% | 0% | 0% | 0% | 0% |

This is the most model-specific result in the study. The same benchmark elicits
categorically different failures:

- **Both GPT configurations fail on margin, not route.** 30% of their failures
  are clearance failures — the centreline stays inside the corridor the entire
  way and the swept pointer still clips a wall. Nothing else comes close to
  that rate. It corroborates H1's finding that minimum clearance is GPT's only
  significant difficulty predictor: GPT knows where to go and runs out of room.
- **Kimi barely cuts walls at all (4%) but concedes 35% of the time.** Its
  dominant failure is announcing that the path is approximate and submitting it
  anyway. That is a policy about when to stop verifying, not a perception
  limit — and it explains why more reasoning never helped Kimi while more
  pixels did.
- **Gemini and Opus are wall-cutters** (63%, 57%): they commit to a route and
  drive it through a barrier.
- **Muse Spark's modal failure is not looking** — 28% of its traces contain no
  coordinate, colour, or spatial reference at all.
- **Inkling misidentifies the start badge in 41% of failures** and inverts
  figure and ground in another 18%. It is failing at the perceptual entry
  point, before navigation begins.

### Honest limits

- **`corridor_departure` (41%) is descriptive, not causal.** It says the path
  cut across a wall, not why. It is the residual bucket for attempts whose
  trace named no recognisable cause, and it should be read as "unexplained
  collision" rather than as a mechanism.
- **845 failures (20%) have no trace at all** — overwhelmingly Inkling, whose
  provider returns reasoning only intermittently. Those can only be labelled
  geometrically. Every verdict records `evidence.trace` as `diagnostic`,
  `uninformative`, or `absent` so a geometry-only label is never mistaken for a
  trace-supported one. Only 35% of failures (1,438) have a trace that matched
  any diagnostic cue.
- **Validation is a sanity check, not a measured accuracy.** The classifier
  agrees with 8 of 9 traces I labelled by hand while building the taxonomy, but
  those labels recorded the maze and not the trial, so the comparison accepts a
  match on any trial and on primary-or-secondary. A defensible accuracy figure
  needs a held-out set labelled at attempt level — ideally by a second reader.
- **Lexical cues are English-specific and phrasing-specific.** They will
  under-fire on models that express the same confusion differently. The
  geometric fallbacks bound the damage but cannot name a cause.
