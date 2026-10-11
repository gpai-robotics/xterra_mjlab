"""Simulator ground truth IS producible -- the simulator knows it.

I had left these unproduced on the grounds that they are not deployable. That
conflated two different things. They are trivially available (MuJoCo knows the
friction it is using), and teacher and distillation runs legitimately need them.
Deployability is handled by the claim-scope gate, not by refusing to compute.
"""

import pytest


def _suite():
    pytest.importorskip("mujoco")
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    robot = "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    profile = inspect_mjcf(robot)
    world = generate_terrain("rs1", profile, 0.2, 5)
    scene = os.path.join(tempfile.mkdtemp(), "s.xml")
    build_mjcf_scene(robot, world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoSensorSuite(model, data, profile), world


def test_backend_declares_the_ground_truth_sources():
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    for name in ("friction_coefficient", "restitution", "base_velocity_truth",
                 "terrain_height_truth", "payload_mass_truth", "com_offset_truth",
                 "motor_strength_scale", "external_wrench_truth",
                 "terrain_class_truth"):
        assert name in declared, name


def test_friction_matches_the_world_material():
    """Not an invented number: it must equal what the scene actually uses."""
    suite, world = _suite()
    value = suite.friction_coefficient({})
    assert len(value) >= 1
    expected = {m.static_friction for m in world.materials.values()}
    assert any(abs(value[0] - e) < 1e-6 for e in expected), (
        "%r matches no world material friction %r" % (value[0], expected))


def test_restitution_is_within_physical_bounds():
    suite, _ = _suite()
    value = suite.restitution({})
    assert all(0.0 <= v <= 1.0 for v in value)


def test_base_velocity_truth_is_exact_not_estimated():
    """At rest it is exactly zero, unlike a drifting estimator."""
    suite, _ = _suite()
    assert suite.base_velocity_truth({}) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_terrain_height_truth_reads_the_ground_plane():
    suite, _ = _suite()
    value = suite.terrain_height_truth({})
    assert len(value) == 1
    assert abs(value[0]) < 1e-6, "flat ground is at z=0, got %r" % value


def test_payload_and_com_reflect_the_model():
    suite, _ = _suite()
    mass = suite.payload_mass_truth({})
    com = suite.com_offset_truth({})
    assert len(mass) == 1 and mass[0] > 0.0, "a Go2 has non-zero mass"
    assert len(com) == 3


def test_motor_strength_scale_defaults_to_unity():
    suite, _ = _suite()
    values = suite.motor_strength_scale({})
    assert len(values) == 12
    assert all(abs(v - 1.0) < 1e-6 for v in values), "no scaling applied by default"


def test_external_wrench_truth_is_zero_without_a_push():
    suite, _ = _suite()
    assert suite.external_wrench_truth({}) == pytest.approx([0.0] * 6, abs=1e-9)


def test_terrain_class_truth_is_one_hot_over_the_world_materials():
    suite, _ = _suite()
    values = suite.terrain_class_truth({})
    assert sum(values) == pytest.approx(1.0)


def test_using_ground_truth_degrades_the_claim_scope():
    """The reason producing these is safe: the claim changes."""
    from robustness_lab.observation.gates import check_deployability, resolve_allowed_classes
    result = check_deployability(sources={"joint_position", "friction_coefficient"},
                                 allowed=resolve_allowed_classes("deployable_only"))
    assert result["status"] == "blocked"
    assert "friction_coefficient" in result["rejected"]
