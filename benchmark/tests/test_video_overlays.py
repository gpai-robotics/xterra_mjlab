import math
from types import SimpleNamespace

import pytest


def test_overlay_uses_declared_world_waypoints_and_surface_heights():
    from robustness_lab.backends.mujoco import video_overlay_primitives

    route = SimpleNamespace(waypoints=[[1.0, 2.0, 0.1], [3.0, 4.0, 0.5]])
    items = video_overlay_primitives(
        route,
        [3.0, 4.0, 0.8],
        [1.0, 2.0, 0.3],
        0.0,
        [0.6, 0.0, 0.0],
        1,
    )

    line = next(item for item in items if item["kind"] == "line")
    assert line["start"] == pytest.approx([1.0, 2.0, 0.24])
    assert line["end"] == pytest.approx([3.0, 4.0, 0.64])
    target = next(item for item in items if item["kind"] == "target")
    assert target["position"] == pytest.approx([3.0, 4.0, 0.64])
    goal = next(item for item in items if item["kind"] == "goal")
    assert goal["position"] == pytest.approx([3.0, 4.0, 0.64])


def test_velocity_arrow_rotates_body_command_into_world_coordinates():
    from robustness_lab.backends.mujoco import video_overlay_primitives

    items = video_overlay_primitives(
        None,
        [0.0, 0.0, 0.0],
        [2.0, 3.0, 0.4],
        math.pi / 2.0,
        [0.6, 0.2, 0.0],
    )
    arrow = next(item for item in items if item["kind"] == "velocity")
    assert arrow["start"] == pytest.approx([2.0, 3.0, 0.68])
    assert arrow["end"] == pytest.approx([1.8, 3.6, 0.68])


def test_overlay_primitives_do_not_mutate_the_route_or_simulation_inputs():
    from robustness_lab.backends.mujoco import video_overlay_primitives

    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    route = SimpleNamespace(waypoints=waypoints)
    position = [0.2, 0.0, 0.3]
    command = [0.5, 0.0, 0.0]
    video_overlay_primitives(route, [1.0, 0.0, 0.3], position, 0.0, command, 1)
    assert route.waypoints == waypoints
    assert position == [0.2, 0.0, 0.3]
    assert command == [0.5, 0.0, 0.0]


def test_velocity_only_guidance_has_no_fake_waypoints_and_shows_actual_motion():
    from robustness_lab.backends.mujoco import video_overlay_primitives

    items = video_overlay_primitives(
        None,
        [5.0, 0.0, 0.0],
        [1.0, 2.0, 0.4],
        0.0,
        [0.6, 0.0, 0.0],
        actual_velocity_world=[0.25, -0.1, 0.0],
    )

    assert not any(item["kind"] in {"line", "waypoint", "target"} for item in items)
    commanded = next(item for item in items if item["kind"] == "velocity")
    actual = next(item for item in items if item["kind"] == "actual_velocity")
    assert commanded["end"][:2] == pytest.approx([1.6, 2.0])
    assert actual["end"][:2] == pytest.approx([1.25, 1.9])


def test_zero_velocity_does_not_render_as_a_misleading_dot():
    from robustness_lab.backends.mujoco import video_overlay_primitives

    items = video_overlay_primitives(
        None,
        [5.0, 0.0, 0.0],
        [1.0, 2.0, 0.4],
        0.0,
        [0.0, 0.0, 0.0],
        actual_velocity_world=[0.0, 0.0, 0.0],
    )

    assert not any(
        item["kind"] in {"velocity", "actual_velocity"} for item in items
    )
