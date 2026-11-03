"""v2 ladder scoring: cell macro, the unanswered-attempt rule, warp pairs."""

from __future__ import annotations

import json

import pytest

from mazerunner.analysis.board import (
    build_board, macro, paired, parse_task, score_row, twin_of, warp_pairs,
)
from mazerunner.merge import is_real_attempt, merge_runs


def test_parse_task_and_twin():
    t = parse_task("braided-L3w-s0040")
    assert t == {"family": "braided", "level": "L3", "warped": True, "slot": "0040",
                 "cell": ("braided", "L3w")}
    assert twin_of("braided-L3w-s0040") == "braided-L3-s0040"
    with pytest.raises(ValueError):
        parse_task("braided-easy-s0040")


def test_macro_weights_cells_equally():
    """A cell with more tasks must not outweigh one with fewer."""
    tasks = {"cave-L1-s0001": [True], "cave-L1-s0002": [True], "cave-L1-s0003": [True],
             "rooms-L1-s0004": [False]}
    assert macro(tasks)["score"] == pytest.approx(0.5)


def test_macro_level_filter():
    tasks = {"cave-L1-s0001": [True], "cave-L3-s0002": [False, True]}
    assert macro(tasks, ("L3", "L4"))["score"] == pytest.approx(0.5)


def test_paired_difference_is_over_shared_tasks():
    a = {"cave-L3-s0001": [True, True], "cave-L3-s0002": [True, False]}
    b = {"cave-L3-s0001": [False, False], "cave-L4-s0009": [True]}
    d = paired(a, b, ("L3", "L4"))
    assert d["score"] == pytest.approx(1.0) and d["cells"] == 1


def test_warp_pairs_compare_twins():
    tasks = {"pipes-L2-s0001": [True, True], "pipes-L2w-s0001": [False, True],
             "cave-L2-s0002": [False]}
    assert warp_pairs(tasks) == {"L2": {"straight": 1.0, "warped": 0.5, "pairs": 1}}


@pytest.mark.parametrize("row, expected", [
    ({"evaluation": {"success": True}}, (True, 1.0)),
    ({"evaluation": {"success": False}, "derived": {"route_progress": 0.4}}, (False, 0.4)),
    ({"provider_error": "no tool call", "evaluation": {"success": False}}, (False, 0.0)),
    ({"error": "transport failure: Connection error.", "unanswered_attempts": 0}, (None, None)),
    ({"error": "transport failure: Request timed out.", "unanswered_attempts": 1}, (None, None)),
    ({"error": "transport failure: Request timed out.", "unanswered_attempts": 2}, (False, 0.0)),
])
def test_score_row(row, expected):
    assert score_row(row) == expected


def test_real_attempt_markers():
    assert is_real_attempt("transport failure: openai: Request timed out.")
    assert is_real_attempt("upstream connect error or disconnect/reset before headers")
    assert not is_real_attempt("transport failure: openai: Connection error.")
    assert not is_real_attempt(None)


def test_merge_counts_unanswered_attempts_and_board_fails_them(tmp_path):
    """Merge keeps one row per unit; the count of real unanswered attempts
    must survive the dedup so the board can apply the rule from the merge."""
    timeout = "transport failure: openai: Request timed out."
    dropped = "transport failure: openai: Connection error."
    rows = [
        ("cave-L3-s0001", 0, timeout, "1"), ("cave-L3-s0001", 0, timeout, "2"),
        ("cave-L3-s0002", 0, dropped, "1"), ("cave-L3-s0002", 0, timeout, "2"),
        ("cave-L3-s0003", 0, None, "1"),
    ]
    src = tmp_path / "attempts.jsonl"
    src.write_text("".join(
        json.dumps({"provider": "a", "maze": m, "trial": t, "timestamp": ts,
                    **({"error": e} if e else {"evaluation": {"success": True}})}) + "\n"
        for m, t, e, ts in rows
    ))
    merge_runs([src], tmp_path / "merged")
    merged = [json.loads(line) for line in (tmp_path / "merged/attempts.jsonl").read_text().splitlines()]
    counts = {r["maze"]: r.get("unanswered_attempts") for r in merged}
    assert counts == {"cave-L3-s0001": 2, "cave-L3-s0002": 1, "cave-L3-s0003": None}

    board, tasks = build_board(tmp_path / "merged/attempts.jsonl")
    assert board["a"]["missing"] == 1 and board["a"]["unanswered"] == 1
    assert tasks["a"] == {"cave-L3-s0001": [False], "cave-L3-s0003": [True]}
