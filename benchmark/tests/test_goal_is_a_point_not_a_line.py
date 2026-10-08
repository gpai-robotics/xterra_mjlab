"""Reaching the goal means arriving AT the goal, not crossing its x-coordinate.

The finish check was

    if travel_direction * (position[0] - goal.x) >= -0.25 * body_length:
        status, reason = "completed", "goal"

which is a vertical plane at x = goal.x. Fine for a straight world, where the
goal is at the far end of the only axis. Wrong for any world whose route turns
back: on the L3 grid world L3-grid-M-g025-b000 the goal is at (7.80, 7.80) and
the route runs east along the bottom edge then north; the robot walked the
bottom edge to (7.71, 0.92) -- the first corner, waypoint 10 of 21, 6.9 m from
the goal -- and was scored `goal`. Every grid, ring and tree world was passable
by walking one edge.

The goal is now a finite gate normal to the final route segment. Straight
worlds keep their historical along-track tolerance; curved and composed worlds
must reach the actual end of their route rather than merely the same x value.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODY = 0.376


def _reached(position, goal, spawn=(0.0, 0.0, 0.0), approach=None, lateral=0.27):
    from robustness_lab.backends.mujoco import reached_goal
    return reached_goal(position, goal, spawn, BODY, approach, lateral)


def test_the_first_corner_of_a_grid_is_not_the_goal():
    """The exact case from the run."""
    assert not _reached(
        (7.71, 0.92, 0.27),
        (7.80, 7.80, 0.27),
        (0.84, 0.84, 0.27),
    )


def test_arriving_at_the_goal_is_the_goal():
    assert _reached((7.75, 7.78, 0.27), (7.80, 7.80, 0.27), (0.84, 0.84, 0.27))


def test_the_goal_gate_uses_separate_along_and_lateral_tolerances():
    goal = (5.0, 0.0, 0.27)
    assert _reached((5.0 - 0.24 * BODY, 0.0, 0.27), goal)
    assert not _reached((5.0 - 0.30 * BODY, 0.0, 0.27), goal)
    assert _reached((5.0, 0.26, 0.27), goal)
    assert not _reached((5.0, 0.28, 0.27), goal), "lateral miss must not count"


def test_a_straight_world_still_passes_at_its_end():
    """Same behaviour as before on the worlds where the old check was right."""
    assert _reached((2.80, 0.02, 0.27), (2.82, 0.0, 0.27), (0.23, 0.0, 0.27))


def test_overshooting_a_straight_goal_still_counts():
    """A robot that walks past the end of a straight course has reached it."""
    assert _reached((3.10, 0.05, 0.27), (2.82, 0.0, 0.27), (0.23, 0.0, 0.27))


def test_overshooting_a_point_goal_on_a_bent_route_does_not():
    """On a grid the robot can be 'past' goal.x while 7 m away from the goal."""
    assert not _reached(
        (8.4, 0.9, 0.27),
        (7.80, 7.80, 0.27),
        (0.84, 0.84, 0.27),
        (0.0, 1.0),
    )


def test_x_proximity_does_not_end_a_curved_course_early():
    # Recorded left-curved course: old x-only check fired here, despite the
    # robot still being 0.28 m from the goal along the final diagonal leg.
    position = (2.382182973143517, 1.1924554604011768, 0.31)
    goal = (2.4694771878059236, 1.4597604090491032, 0.27)
    final_leg = (
        2.5545988020448838 - 2.149911159922981,
        1.6274589250772435 - 0.9958100922657341,
    )
    assert not _reached(position, goal, (0.225, 0.009, 0.27), final_leg)


def test_a_reversed_straight_world_still_passes_at_its_end():
    assert _reached((0.25, 0.0, 0.27), (0.19, 0.0, 0.27), (2.60, 0.0, 0.27))
    assert _reached(
        (0.10, 0.0, 0.27),
        (0.19, 0.0, 0.27),
        (2.60, 0.0, 0.27),
    ), "overshoot on a reversed course"


def test_the_recorded_grid_trial_would_no_longer_be_a_pass():
    """Replay the check over the actual trajectory of the trial that exposed this."""
    import json
    log_path = ROOT / "runs/submissions/go2_blind_v2/trials/L3-grid-M-g025-b000-a10/episode_log.json"
    world_path = ROOT / "runs/submissions/go2_blind_v2/worlds/L3-grid-M-g025-b000.json"
    if not (log_path.is_file() and world_path.is_file()):
        pytest.skip("that run is not on this machine")
    log = json.load(open(log_path))
    world = json.load(open(world_path))
    goal = tuple(world["goal_pose"]["position"])
    spawn = tuple(world["spawn_pose"]["position"])
    hits = [
        sample["time_s"]
        for sample in log["samples"]
        if _reached(tuple(sample["base_position_m"]), goal, spawn, (0.0, 1.0))
    ]
    assert not hits, "the trajectory that ended 6.9 m from the goal still registers goal at t=%s" % hits[:1]


def test_the_mujoco_loop_uses_the_route_aligned_goal_gate():
    src = (ROOT / "src/robustness_lab/backends/mujoco.py").read_text()
    assert "travel_direction * (position[0] - world.goal_pose.position[0]) >= -0.25" not in src
    assert "reached_goal(" in src
    assert "route_complete and reached_goal(" in src
