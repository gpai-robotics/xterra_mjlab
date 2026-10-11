"""`route_follower` must actually steer, not just be a label.

The skill contract declares `assistance: route_follower` for velocity-command
policies, but the backend supplied a CONSTANT
`episode.controller.get("command", [0.6, 0.0, 0.0])` for the whole episode.

On the straight RS corridors that is harmless -- forward is the right answer.
In connected worlds whose routes turn, the robot is told to go forward while the
path bends, so a velocity-command policy fails for reasons unrelated to the
policy. Declared-but-unimplemented, and it silently produces policy_fail.

A follower converts the route into a body-frame command each step: forward speed
plus a yaw rate from the heading error to the next waypoint.
"""

import math

import pytest


def _follower(waypoints, speed=0.6):
    from robustness_lab.navigation import RouteFollower
    return RouteFollower(waypoints=waypoints, forward_speed=speed)


def test_straight_route_ahead_gives_pure_forward():
    follower = _follower([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    command = follower.command(position=[0.0, 0.0, 0.0], yaw=0.0)
    assert command[0] == pytest.approx(0.6)
    assert abs(command[2]) < 1e-6, "no yaw rate needed when the path is ahead"


def test_waypoint_to_the_left_commands_a_positive_yaw_rate():
    follower = _follower([[0.0, 1.0, 0.0]])
    command = follower.command(position=[0.0, 0.0, 0.0], yaw=0.0)
    assert command[2] > 0.0, "target at +y should turn left"
    assert command[0] == pytest.approx(0.0, abs=1e-9), "turn 90 degrees before driving"


def test_waypoint_to_the_right_commands_a_negative_yaw_rate():
    follower = _follower([[0.0, -1.0, 0.0]])
    command = follower.command(position=[0.0, 0.0, 0.0], yaw=0.0)
    assert command[2] < 0.0
    assert command[0] == pytest.approx(0.0, abs=1e-9)


def test_command_follows_a_corner_as_the_robot_advances():
    """The case the constant command cannot handle."""
    follower = _follower([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    before = follower.command(position=[0.0, 0.0, 0.0], yaw=0.0)
    after = follower.command(position=[1.98, 0.0, 0.0], yaw=0.0)
    assert abs(before[2]) < 1e-6, "still straight on the approach"
    assert after[2] > 0.0, "after reaching the corner it must turn toward +y"


def test_yaw_rate_is_bounded():
    follower = _follower([[-1.0, 0.0, 0.0]])          # directly behind
    command = follower.command(position=[0.0, 0.0, 0.0], yaw=0.0)
    assert abs(command[2]) <= follower.max_yaw_rate + 1e-9


def test_reaching_a_waypoint_advances_to_the_next():
    follower = _follower([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    follower.command(position=[0.99, 0.0, 0.0], yaw=0.0)
    assert follower.index == 1, "should have advanced past the reached waypoint"


def test_passing_a_waypoint_along_its_segment_advances_without_orbiting_it():
    follower = _follower([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    follower.command(position=[0.0, 0.0, 0.0], yaw=math.pi / 2.0)
    follower.command(position=[0.05, 1.25, 0.0], yaw=math.pi / 2.0)
    assert follower.index == 2


def test_passing_far_outside_the_route_does_not_skip_a_waypoint():
    follower = _follower([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    follower.command(position=[0.0, 0.0, 0.0], yaw=math.pi / 2.0)
    follower.command(position=[1.0, 1.25, 0.0], yaw=math.pi / 2.0)
    assert follower.index == 1


def test_after_the_last_waypoint_it_holds_the_final_heading():
    follower = _follower([[1.0, 0.0, 0.0]])
    follower.command(position=[1.0, 0.0, 0.0], yaw=0.0)
    command = follower.command(position=[2.0, 0.0, 0.0], yaw=0.0)
    assert len(command) == 3
    assert all(abs(v) < 1e6 for v in command)
    assert follower.completed is True


def test_route_is_not_complete_until_every_waypoint_was_reached_in_order():
    follower = _follower([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert follower.completed is False
    follower.command(position=[1.0, 0.0, 0.0], yaw=0.0)
    assert follower.completed is False
    follower.command(position=[1.0, 1.0, 0.0], yaw=math.pi / 2.0)
    assert follower.completed is True


def test_heading_accounts_for_the_robot_orientation():
    """A target at +x while the robot faces +y is a right turn."""
    follower = _follower([[1.0, 0.0, 0.0]])
    command = follower.command(position=[0.0, 0.0, 0.0], yaw=math.pi / 2.0)
    assert command[2] < 0.0
