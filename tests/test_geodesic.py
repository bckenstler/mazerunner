"""The mask-certified optimum: shortest, legal, and never beatable."""

import math

import numpy as np

from mazerunner.evaluator import evaluate
from mazerunner.geodesic import geodesic_optimum, legal_region

SIZE = 200
POINTER = 3


def test_open_room_geodesic_is_straight():
    mask = np.ones((SIZE, SIZE), dtype=bool)
    length, points = geodesic_optimum(mask, (20, 20), (180, 160), POINTER)
    straight = math.hypot(160, 140)
    assert straight <= length <= straight * 1.02
    assert len(points) <= 6  # string-pulling collapses the grid staircase


def test_tube_maze_geodesic_matches_centerline():
    # U corridor from the evaluator tests: centerline route is ~480px and the
    # corridor is narrow, so the geodesic can only shave corners slightly.
    half = 7
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[13:188, 20 - half : 20 + half + 1] = True
    mask[180 - half : 180 + half + 1, 13:188] = True
    mask[13:188, 180 - half : 180 + half + 1] = True
    length, _points = geodesic_optimum(mask, (20, 20), (180, 20), POINTER)
    assert 440 <= length <= 481


def test_corner_cut_is_captured_and_no_canary():
    # A wide L-shaped corridor: the centerline route is 160+160 = 320px, but a
    # legal path can cut the corner diagonally. The geodesic must capture that
    # (be meaningfully shorter), and a submission that takes the shortcut must
    # NOT fire the efficiency canary.
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[80:121, 20:181] = True  # horizontal bar, 40px wide, y-center 100
    mask[20:121, 140:181] = True  # vertical bar up from its right end, x-center 160
    start, goal = (30.0, 100.0), (160.0, 30.0)
    length, points = geodesic_optimum(mask, start, goal, POINTER)
    centerline = 130 + 70
    assert length < centerline - 10  # the corner cut is real

    submission = {"points": [{"x": p[0] / (SIZE - 1), "y": p[1] / (SIZE - 1)} for p in points]}
    result = evaluate(
        submission,
        mask,
        width=SIZE,
        height=SIZE,
        start_px=start,
        goal_px=goal,
        start_radius_px=12,
        goal_radius_px=12,
        pointer_radius_px=POINTER,
        reference_length_px=length,
    )
    assert result.success
    assert not result.efficiency_canary
    assert 0.99 <= result.efficiency_raw <= 1.01


def test_legal_region_respects_pointer_radius():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100 - 4 : 100 + 5, :] = True  # 9px-wide corridor, pointer diameter 7
    legal = legal_region(mask, POINTER)
    assert legal[100, 100]
    assert not legal[97, 100] and not legal[103, 100]


def test_no_route_fails_closed():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[10:50, 10:50] = True
    mask[150:190, 150:190] = True  # disconnected pockets
    try:
        geodesic_optimum(mask, (30, 30), (170, 170), POINTER)
    except ValueError as exc:
        assert "no legal route" in str(exc)
    else:
        raise AssertionError("expected ValueError")
