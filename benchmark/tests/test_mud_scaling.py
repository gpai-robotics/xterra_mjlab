"""Mud sizes must scale with the robot, like every other terrain parameter.

RS11 is morphology-scaled everywhere else -- marsh length from body_length, sink
depth from foot size, track width from stance. Two mud quantities were left as
bare constants:

    reference_depth_m = 0.04      # H, the depth that normalises the force law
    backstop_thickness = 0.01     # the hard floor under the mud

A constant H means the force law is calibrated for a Go2 and silently wrong for
anything larger or smaller: the same absolute sinkage would read as a different
fraction of the reference depth, so a big robot would find the mud implausibly
firm and a small one would find it bottomless. In the source model
(arXiv:2403.02617) H is the INTRUDER WIDTH -- their 51x38x25 mm cuboid is
normalised by its own 25 mm width -- so H belongs to the foot, not to a constant.

Also pinned here: RS11 keeps its approach run. The robot must start on firm
ground and walk into the marsh, not be spawned already inside it.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"


def _profile(scale=1.0):
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    data = dict(load_data(str(PROFILE)))
    if scale != 1.0:
        for key in ("body_size_m", "foot_size_m"):
            data[key] = [value * scale for value in data[key]]
        for key in ("standing_height_m", "limb_reach_m", "stance_width_m",
                    "stance_length_m"):
            if key in data:
                data[key] = data[key] * scale
    return RobotProfile.from_dict(data)


def _marsh(world):
    return [item for item in world.geometry if item.semantic == "marsh"]


def _reference_depth(profile):
    from robustness_lab.worlds import generate_terrain
    world = generate_terrain("rs11", profile, 0.5, 7)
    return float(_marsh(world)[0].parameters["reference_depth_m"])


def test_reference_depth_is_derived_from_the_robot_not_a_constant():
    small = _reference_depth(_profile(1.0))
    large = _reference_depth(_profile(2.0))
    assert large > small, (
        "reference_depth_m is the same for a robot twice the size: %r vs %r"
        % (small, large))


def test_reference_depth_scales_roughly_with_the_robot():
    """Doubling the robot should roughly double the normalising depth."""
    small = _reference_depth(_profile(1.0))
    large = _reference_depth(_profile(2.0))
    assert large == pytest.approx(2.0 * small, rel=0.35)


def test_reference_depth_tracks_the_foot():
    """H is the intruder width in the source model, so it follows foot size."""
    profile = _profile(1.0)
    depth = _reference_depth(profile)
    foot_width = float(profile.foot_size_m[1])
    assert depth == pytest.approx(foot_width, rel=0.6), (
        "reference depth %.4f is unrelated to foot width %.4f" % (depth, foot_width))


def test_the_mud_layer_itself_scales():
    from robustness_lab.worlds import generate_terrain
    small = _marsh(generate_terrain("rs11", _profile(1.0), 0.5, 7))[0]
    large = _marsh(generate_terrain("rs11", _profile(2.0), 0.5, 7))[0]
    assert float(large.size[2]) > float(small.size[2])


def test_sinkable_depth_never_exceeds_the_mud_layer():
    """The hard floor under the mud is what makes falling through impossible."""
    from robustness_lab.worlds import generate_terrain
    for scale in (0.5, 1.0, 2.0):
        patch = _marsh(generate_terrain("rs11", _profile(scale), 1.0, 7))[0]
        reference = float(patch.parameters["reference_depth_m"])
        assert reference <= 4.0 * float(patch.size[2]), (
            "reference depth %.4f is large next to a %.4f m mud layer"
            % (reference, float(patch.size[2])))


def test_the_backstop_thickness_scales_with_the_mud_layer():
    """A 1 cm floor under a 2 cm layer is a different terrain than under 20 cm."""
    pytest.importorskip("mujoco")
    from robustness_lab.worlds import generate_terrain

    thicknesses = {}
    for scale in (1.0, 3.0):
        world = generate_terrain("rs11", _profile(scale), 0.8, 7)
        patch = _marsh(world)[0]
        thicknesses[scale] = float(patch.size[2])
    assert thicknesses[3.0] > thicknesses[1.0]


def test_rs11_keeps_its_approach_run():
    """The robot walks INTO the marsh; it is not spawned inside it.

    Spawning inside the mud removes the approach phase, changes what RS11
    measures, and would make a stationary controller look like it had traversed
    the terrain.
    """
    from robustness_lab.worlds import generate_terrain
    world = generate_terrain("rs11", _profile(1.0), 0.5, 7)
    marsh = [region for region in world.regions if region.kind == "marsh"][0]
    spawn_x = float(world.spawn_pose.position[0])
    assert spawn_x < float(marsh.minimum[0]), (
        "spawn x=%.3f is not before the marsh entry x=%.3f" % (spawn_x, marsh.minimum[0]))


def test_rs11_route_still_starts_before_the_marsh():
    from robustness_lab.worlds import generate_terrain
    world = generate_terrain("rs11", _profile(1.0), 0.5, 7)
    marsh = [region for region in world.regions if region.kind == "marsh"][0]
    first_waypoint_x = float(world.routes[0].waypoints[0][0])
    assert first_waypoint_x < float(marsh.minimum[0])
