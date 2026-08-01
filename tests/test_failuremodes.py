"""Failure-mode classification: each mode needs both kinds of evidence."""

from __future__ import annotations

import numpy as np
import pytest

from mazerunner.analysis.failuremodes import (
    PRECEDENCE,
    classify,
    geometry_measures,
    has_perceptual_content,
    lexical_hits,
)

TASK = {
    "width": 201, "height": 201,
    "start": {"x": 0.05, "y": 0.05}, "goal": {"x": 0.95, "y": 0.95},
    "start_radius_px": 14.0, "goal_radius_px": 14.0,
}


def _mask(open_band=True):
    """A 201x201 mask with an open L-shaped corridor."""
    m = np.zeros((201, 201), dtype=bool)
    if open_band:
        m[0:40, :] = True     # top corridor
        m[:, 160:201] = True  # right corridor
    return m


def _row(points, success=False, trace=None, **ev):
    evaluation = {"success": success, "collision_free": False, "starts_correctly": True}
    evaluation.update(ev)
    return {
        "submission": {"points": [{"x": x, "y": y} for x, y in points]},
        "evaluation": evaluation,
        "reasoning": trace,
        "derived": {"route_progress": ev.pop("route_progress", 0.5)},
    }


# ---------- geometry ----------

def test_outside_fraction_detects_a_path_through_walls():
    pts = [(0.5, 0.5), (0.5, 0.9)]   # straight through the closed middle
    g = geometry_measures(TASK, _mask(), [(x * 200, y * 200) for x, y in pts])
    assert g["outside_fraction"] > 0.9


def test_outside_fraction_is_zero_inside_the_corridor():
    pts = [(0.05, 0.05), (0.9, 0.05)]  # along the open top band
    g = geometry_measures(TASK, _mask(), [(x * 200, y * 200) for x, y in pts])
    assert g["outside_fraction"] < 0.05


def test_spacing_cv_is_low_for_a_generated_path():
    """Analytic parameterisation produces suspiciously regular spacing."""
    pts = [(10 + 10 * i, 10) for i in range(15)]
    assert geometry_measures(TASK, _mask(), pts)["spacing_cv"] < 0.05


def test_reversal_is_counted_when_the_path_doubles_back():
    pts = [(10, 10), (100, 10), (20, 10), (100, 12)]
    assert geometry_measures(TASK, _mask(), pts)["reversals"] >= 1


# ---------- lexical ----------

def test_lexical_finds_figure_ground_language():
    hits = lexical_hits("I suspect the cream bands are actually the traversable routes")
    assert "figure_ground_inversion" in hits


def test_lexical_finds_satisficing_language():
    assert "satisficing" in lexical_hits("Final can be approximate. Let's answer.")


def test_lexical_ignores_ordinary_narration():
    assert lexical_hits("I trace from the start badge along the corridor to the goal.") == {}


def test_perceptual_content_requires_something_specific():
    assert has_perceptual_content("the cyan badge sits at (120, 240)")
    assert not has_perceptual_content("Mapping corridor junctions to refine route geometry.")


# ---------- classification ----------

def test_pass_is_not_a_failure_mode():
    assert classify(_row([(0.05, 0.05)], success=True), TASK, _mask()).primary == "pass"


def test_figure_ground_needs_the_path_to_actually_leave_the_mask():
    """The phrase alone must not be enough."""
    trace = "the walls are actually the paths here"
    inside = [(0.05, 0.05), (0.85, 0.05)]
    v = classify(_row(inside, trace=trace), TASK, _mask())
    assert v.primary != "figure_ground_inversion"


def test_figure_ground_fires_when_both_agree():
    trace = "I suspect the bands are actually the traversable routes"
    through_walls = [(0.5, 0.3), (0.5, 0.95)]
    v = classify(_row(through_walls, trace=trace), TASK, _mask())
    assert v.primary == "figure_ground_inversion"


def test_clearance_failure_when_centreline_is_legal_but_it_collided():
    legal = [(0.05, 0.05), (0.85, 0.05)]
    v = classify(_row(legal, collision_free=False), TASK, _mask())
    assert v.primary == "clearance_failure"


def test_wrong_branch_when_legal_and_never_collided():
    legal = [(0.05, 0.05), (0.85, 0.05)]
    v = classify(_row(legal, collision_free=True, ends_correctly=False), TASK, _mask())
    assert v.primary == "wrong_branch"


def test_endpoint_misidentification_uses_the_scorer_verdict():
    v = classify(_row([(0.5, 0.5), (0.9, 0.9)], starts_correctly=False), TASK, _mask())
    assert v.primary == "endpoint_misidentification"


def test_procedural_template_needs_an_uninformative_trace():
    v = classify(_row([(0.5, 0.3), (0.5, 0.95)],
                      trace="Mapping corridor junctions. Extending the polyline."),
                 TASK, _mask())
    assert v.primary == "procedural_template"


def test_trace_availability_is_recorded():
    """A geometry-only label must never look trace-supported."""
    v = classify(_row([(0.5, 0.3), (0.5, 0.95)], trace=None), TASK, _mask())
    assert v.evidence["trace"] == "absent"


def test_every_failure_gets_a_label():
    v = classify(_row([(0.5, 0.3), (0.5, 0.95)], trace="some narration about the maze"),
                 TASK, _mask())
    assert v.primary in PRECEDENCE


def test_precedence_prefers_the_upstream_cause():
    """A trace-named cause outranks the generic geometric description."""
    trace = "I suspect the bands are actually the traversable corridors"
    v = classify(_row([(0.5, 0.3), (0.5, 0.95)], trace=trace), TASK, _mask())
    assert v.primary == "figure_ground_inversion"
    assert "corridor_departure" in v.secondary
