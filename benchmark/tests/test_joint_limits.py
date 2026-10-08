"""Robot profiles must carry JOINT POSITION limits and a default pose inside them.

Isaac Lab validates that every default joint position lies within its limits and
refuses to initialise otherwise. MuJoCo does not. That difference is why the
Isaac adapter never completed an episode while MuJoCo worked:

    ValueError: The following joints have default positions out of the limits:
        - 'FL_calf_joint': 0.000 not in [-2.723, -0.838]

inspect_urdf left default_joint_positions empty and never parsed <limit>, so
every joint defaulted to 0.0 -- impossible for a Go2 calf.

Note these are POSITION limits (rad), distinct from an actuator's force or
control range. Conflating the two is its own defect: the MJCF parser stores a
+/-23.7 N.m control range that must never be mistaken for a joint angle bound.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/cache/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
MJCF = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"


def _profile(inspector, path):
    if not path.is_file():
        pytest.skip("asset not present: %s" % path)
    return inspector(str(path))


def test_urdf_captures_joint_position_limits():
    from robustness_lab.robot import inspect_urdf
    profile = _profile(inspect_urdf, URDF)
    limits = profile.joint_limits
    assert "FL_calf_joint" in limits, "joint <limit> was not parsed"
    low, high = limits["FL_calf_joint"]
    assert low < high
    assert low < -0.5, "Go2 calf lower bound should be strongly negative, got %r" % low


def test_mjcf_captures_joint_position_limits():
    from robustness_lab.robot import inspect_mjcf
    profile = _profile(inspect_mjcf, MJCF)
    assert "FL_calf_joint" in profile.joint_limits


def test_position_limits_are_not_the_actuator_force_range():
    """A +/-23.7 N.m control range must never be read as a joint angle bound."""
    from robustness_lab.robot import inspect_mjcf
    profile = _profile(inspect_mjcf, MJCF)
    low, high = profile.joint_limits["FL_calf_joint"]
    assert abs(low) < 10.0 and abs(high) < 10.0, (
        "limits look like a force range, not radians: (%r, %r)" % (low, high))


@pytest.mark.parametrize("inspector_name,path", [("inspect_urdf", URDF), ("inspect_mjcf", MJCF)])
def test_default_pose_is_populated_and_within_limits(inspector_name, path):
    """The exact condition Isaac Lab enforces before it will initialise."""
    import robustness_lab.robot as robot_module
    profile = _profile(getattr(robot_module, inspector_name), path)
    bad = []
    for joint in profile.controller_joint_order:
        value = profile.default_joint_positions.get(joint)
        if value is None:
            bad.append((joint, "no default"))
            continue
        bounds = profile.joint_limits.get(joint)
        if bounds and not bounds[0] <= value <= bounds[1]:
            bad.append((joint, value, bounds))
    assert not bad, "unusable defaults: %s" % bad


def test_profile_exposes_a_limit_validator():
    """A caller must be able to ask, rather than discover it inside a simulator."""
    from robustness_lab.robot import inspect_urdf, validate_default_pose
    profile = _profile(inspect_urdf, URDF)
    report = validate_default_pose(profile)
    assert report["status"] == "ready", report


def test_validator_reports_a_pose_outside_limits():
    from robustness_lab.robot import inspect_urdf, validate_default_pose
    profile = _profile(inspect_urdf, URDF)
    profile.default_joint_positions["FL_calf_joint"] = 0.0     # impossible for a Go2
    report = validate_default_pose(profile)
    assert report["status"] == "blocked"
    assert "FL_calf_joint" in str(report["violations"])
