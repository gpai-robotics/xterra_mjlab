"""Foot links must be detected from MJCF as well as URDF.

inspect_urdf finds ['FL_foot','FR_foot','RL_foot','RR_foot'] but inspect_mjcf
returned an EMPTY list, so on MJCF every foot-derived source -- foot_contact,
foot_contact_force, foot_position_b, foot_velocity_b, foot_slip -- was declared
produced yet returned nothing. Every MuJoCo run had no foot-contact data.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
URDF = ROOT / "assets/cache/unitree_ros/robots/go2_description/urdf/go2_description.urdf"


def _profile(inspector_name, path):
    if not path.is_file():
        pytest.skip("asset not present: %s" % path)
    import robustness_lab.robot as robot_module
    return getattr(robot_module, inspector_name)(str(path))


def test_mjcf_detects_four_feet():
    profile = _profile("inspect_mjcf", MJCF)
    assert len(profile.foot_links) == 4, profile.foot_links


def test_urdf_detects_four_feet():
    profile = _profile("inspect_urdf", URDF)
    assert len(profile.foot_links) == 4, profile.foot_links


def test_both_formats_agree_on_one_foot_per_leg():
    """Whatever the naming, there must be one foot per quadrant."""
    mjcf = _profile("inspect_mjcf", MJCF)
    urdf = _profile("inspect_urdf", URDF)
    for profile in (mjcf, urdf):
        names = " ".join(profile.foot_links).upper()
        for quadrant in ("FL", "FR", "RL", "RR"):
            assert quadrant in names, "%s missing from %s" % (quadrant, profile.foot_links)


def test_foot_sources_return_one_reading_per_foot_on_mjcf():
    """The behaviour the empty list broke."""
    pytest.importorskip("mujoco")
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.worlds import generate_terrain

    profile = _profile("inspect_mjcf", MJCF)
    world = generate_terrain("rs1", profile, 0.2, 5)
    scene = os.path.join(tempfile.mkdtemp(), "scene.xml")
    build_mjcf_scene(str(MJCF), world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    suite = MujocoSensorSuite(model, data, profile)

    assert len(suite.foot_slip()) == 4
    assert len(suite.foot_velocity_b()) == 12          # 4 feet x 3 axes
    assert len(suite.foot_position_b()) == 12


def test_magnetometer_returns_a_unit_field(tmp_path):
    """A magnetometer reading [0,0,-0] is not a measurement."""
    import math
    pytest.importorskip("mujoco")
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.worlds import generate_terrain

    profile = _profile("inspect_mjcf", MJCF)
    world = generate_terrain("rs1", profile, 0.2, 5)
    scene = os.path.join(tempfile.mkdtemp(), "scene.xml")
    build_mjcf_scene(str(MJCF), world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    suite = MujocoSensorSuite(model, data, profile)
    field = suite.magnetometer()
    assert len(field) == 3
    assert math.sqrt(sum(v * v for v in field)) == pytest.approx(1.0, abs=1e-6)
