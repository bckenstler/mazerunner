import numpy as np

from mazerunner.evaluator import evaluate

SIZE = 200
HALF = 7  # corridor half-width in px; pointer radius 3 leaves 4px of slack


def norm(x_px, y_px):
    return {"x": x_px / (SIZE - 1), "y": y_px / (SIZE - 1)}


def u_corridor_mask():
    """U-shaped corridor: down the left arm, across the bottom, up the right arm.

    Start (20,20) and goal (180,20) are close in space but far along the route.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[13:188, 20 - HALF : 20 + HALF + 1] = True  # left arm
    mask[180 - HALF : 180 + HALF + 1, 13:188] = True  # bottom
    mask[13:188, 180 - HALF : 180 + HALF + 1] = True  # right arm
    return mask


U_REFERENCE_LEN = 480.0  # 160 + 160 + 160 along the corridor centerline


def run(mask, points, reference_length=U_REFERENCE_LEN, start=(20, 20), goal=(180, 20)):
    return evaluate(
        {"points": points},
        mask,
        width=SIZE,
        height=SIZE,
        start_px=start,
        goal_px=goal,
        start_radius_px=10,
        goal_radius_px=10,
        pointer_radius_px=3,
        reference_length_px=reference_length,
    )


def test_reference_route_succeeds():
    ev = run(u_corridor_mask(), [norm(20, 20), norm(20, 180), norm(180, 180), norm(180, 20)])
    assert ev.success
    assert ev.collision_free
    assert ev.starts_correctly and ev.ends_correctly
    assert ev.min_clearance_px is not None and ev.min_clearance_px > 3
    assert 0.9 < ev.efficiency <= 1.0
    assert not ev.efficiency_canary


def test_straight_wall_crossing_shortcut_rejected():
    ev = run(u_corridor_mask(), [norm(20, 20), norm(180, 20)])
    assert not ev.success
    assert not ev.collision_free
    assert ev.first_collision is not None
    # It failed for collision, not endpoint localization.
    assert ev.starts_correctly and ev.ends_correctly


def test_legal_waypoints_illegal_segment_rejected():
    # Every listed waypoint is inside the corridor, but the final segment cuts
    # across the interior wall between the arms.
    ev = run(u_corridor_mask(), [norm(20, 20), norm(20, 180), norm(180, 20)])
    assert not ev.collision_free
    assert ev.first_collision["segment_index"] == 1


def test_under_sampled_bend_rejected():
    # Cutting the bottom-left corner diagonally leaves the corridor.
    ev = run(u_corridor_mask(), [norm(20, 20), norm(20, 160), norm(40, 180), norm(180, 180), norm(180, 20)])
    assert not ev.collision_free


def test_start_localization_failure():
    ev = run(u_corridor_mask(), [norm(20, 60), norm(20, 180), norm(180, 180), norm(180, 20)])
    assert not ev.starts_correctly
    assert not ev.success


def test_goal_localization_failure():
    ev = run(u_corridor_mask(), [norm(20, 20), norm(20, 180), norm(180, 180), norm(180, 60)])
    assert not ev.ends_correctly
    assert not ev.success


def test_efficiency_canary_fires_on_permissive_mask():
    # A fully open mask lets the straight shortcut pass; the declared reference
    # length is the long corridor route, so raw efficiency is ~3.
    open_mask = np.ones((SIZE, SIZE), dtype=bool)
    ev = run(open_mask, [norm(20, 20), norm(180, 20)])
    assert ev.success
    assert ev.efficiency_raw > 1.05
    assert ev.efficiency == 1.0
    assert ev.efficiency_canary
    assert ev.warnings


def test_tangent_contact_against_straight_wall():
    # Horizontal corridor with walls above and below; the pointer disk has
    # radius 3, so a centerline offset of 2px still clears but 5px collides.
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100 - HALF : 100 + HALF + 1, :] = True
    clear = run(mask, [norm(20, 98), norm(180, 98)], reference_length=160, start=(20, 100), goal=(180, 100))
    assert clear.collision_free
    grazing = run(mask, [norm(20, 95), norm(180, 95)], reference_length=160, start=(20, 100), goal=(180, 100))
    assert not grazing.collision_free


def test_thin_wall_cannot_be_stepped_over():
    # Two corridors separated by a 2px wall: even a dense submitted path that
    # hops across must be caught by sub-pixel segment sampling.
    mask = np.ones((SIZE, SIZE), dtype=bool)
    mask[99:101, :] = False
    points = [norm(100, 80), norm(100, 120)]
    ev = run(mask, points, reference_length=40, start=(100, 80), goal=(100, 120))
    assert not ev.collision_free


def test_schema_failure_reports_zero_efficiency():
    ev = run(u_corridor_mask(), [norm(20, 20)])
    assert not ev.schema_valid
    assert ev.schema_error is not None
    assert ev.efficiency == 0.0
    assert not ev.success


def test_out_of_bounds_disk_at_image_edge_collides():
    # Corridor hugging the top edge: a pointer at y=1 pokes above the canvas.
    mask = np.ones((SIZE, SIZE), dtype=bool)
    ev = run(mask, [norm(20, 1), norm(180, 1)], reference_length=160, start=(20, 1), goal=(180, 1))
    assert not ev.collision_free
