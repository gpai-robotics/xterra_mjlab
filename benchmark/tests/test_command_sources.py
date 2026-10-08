"""Command and route sources come from the benchmark, not from a sensor.

The world already carries a spawn, a goal and a route; the episode carries the
operator command. Nothing was feeding those into the canonical state, so a
waypoint-driven policy -- which is most parkour work, including FoReSight --
could not run at all.
"""

import pytest

TERRAIN = "rs1"


def _state():
    pytest.importorskip("mujoco")
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene, MujocoBackend
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    robot = "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    profile = inspect_mjcf(robot)
    world = generate_terrain(TERRAIN, profile, 0.2, 5)
    scene = os.path.join(tempfile.mkdtemp(), "s.xml")
    build_mjcf_scene(robot, world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoSensorSuite(model, data, profile), world


def test_backend_declares_the_route_and_command_sources():
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    for name in ("goal_vector_b", "goal_distance", "goal_heading", "waypoints_b",
                 "heading_command", "body_height_command", "pitch_command",
                 "roll_command", "gait_command", "footswing_height_command",
                 "stance_width_command", "terrain_style_command"):
        assert name in declared, name


def test_goal_vector_points_toward_the_world_goal():
    """The goal lies ahead on rs1, so the body-frame vector must have +x."""
    suite, world = _state()
    vector = suite.goal_vector_b({"goal": list(world.goal_pose.position)})
    assert len(vector) == 3
    assert vector[0] > 0.0, "goal is ahead; body-frame x should be positive: %r" % vector


def test_goal_distance_matches_the_geometry():
    import math
    suite, world = _state()
    params = {"goal": list(world.goal_pose.position)}
    vector = suite.goal_vector_b(params)
    distance = suite.goal_distance(params)
    assert len(distance) == 1
    assert distance[0] == pytest.approx(
        math.sqrt(sum(v * v for v in vector)), abs=1e-6)


def test_goal_heading_is_zero_when_the_goal_is_straight_ahead():
    suite, world = _state()
    goal = list(world.goal_pose.position)
    goal[1] = 0.0
    heading = suite.goal_heading({"goal": goal})
    assert len(heading) == 1
    assert abs(heading[0]) < 1e-3, "straight-ahead goal should give ~0 heading error"


def test_waypoints_are_returned_in_body_frame_with_declared_lookahead():
    suite, world = _state()
    points = suite.waypoints_b({"waypoints": [[1.0, 0.0], [2.0, 0.5]], "lookahead": 2})
    assert len(points) == 4          # 2 waypoints x (x, y)


def test_operator_commands_echo_their_declared_values():
    suite, _ = _state()
    assert suite.heading_command({"heading": 0.4}) == pytest.approx([0.4])
    assert suite.body_height_command({"height": 0.32}) == pytest.approx([0.32])
    assert suite.pitch_command({"pitch": -0.1}) == pytest.approx([-0.1])
    assert suite.roll_command({"roll": 0.05}) == pytest.approx([0.05])


def test_gait_command_returns_its_declared_parameters():
    suite, _ = _state()
    values = suite.gait_command({"frequency": 2.5, "duty": 0.6, "offsets": [0.0, 0.5]})
    assert len(values) >= 3
    assert values[0] == pytest.approx(2.5)


def test_terrain_style_command_is_one_hot():
    suite, _ = _state()
    values = suite.terrain_style_command({"style": 2, "classes": 5})
    assert len(values) == 5
    assert values[2] == pytest.approx(1.0)
    assert sum(values) == pytest.approx(1.0)
