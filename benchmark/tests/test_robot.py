from pathlib import Path

import pytest

from robustness_lab.robot import compare_profiles, inspect_mjcf, inspect_urdf


FIXTURES = Path(__file__).parent / "fixtures"


def test_urdf_extracts_collision_morphology_and_kinematics():
    profile = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    assert profile.name == "mini_quad"
    assert profile.base_link == "base"
    assert sorted(profile.foot_links) == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert profile.total_mass_kg == 10.0
    assert profile.body_size_m == [0.5, 0.2, 0.16]
    assert profile.collision_bounds_m == {"minimum": [-0.25, -0.1, -0.08], "maximum": [0.25, 0.1, 0.08]}
    assert profile.link_inertias["base"]["matrix_kg_m2"][2][2] == 0.25
    assert profile.stance_length_m == 0.4
    assert profile.stance_width_m == 0.24
    assert profile.limb_reach_m == pytest.approx(0.5832380758)


def test_mjcf_extracts_robot_and_actuator():
    profile = inspect_mjcf(str(FIXTURES / "mini_robot.xml"))
    assert profile.base_link == "base"
    assert profile.total_mass_kg == 0.0  # explicit geom mass is intentionally reported as unresolved
    assert profile.actuators[0].joint == "leg_joint"
    assert profile.actuators[0].effort_limit == 20.0


def test_conformance_catches_dimension_drift():
    left = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    right = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    right.body_size_m[0] *= 1.1
    result = compare_profiles([left, right])
    assert not result["comparable"]
    assert any(issue["field"] == "body_size_x" for issue in result["comparisons"][0]["issues"])


def test_conformance_catches_joint_sign_drift():
    left = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    right = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    moving_joint = next(joint for joint in right.joints if joint.joint_type == "revolute")
    moving_joint.axis = [-value for value in moving_joint.axis]
    result = compare_profiles([left, right])
    assert not result["comparable"]
    assert any(issue["field"] == "joint_axis" for issue in result["comparisons"][0]["issues"])
