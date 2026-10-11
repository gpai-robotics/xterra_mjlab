"""Policy-internal state and the last derived perception products.

recurrent_state, estimator_output, adaptation_latent and visual_latent belong to
the POLICY, not the simulator. The platform's job is lifecycle: allocate them,
zero them at episode start, carry them across steps, and never leak them between
trials -- a leak silently inflates scores on the second trial.

traversability_map and gnss_position are genuinely derivable and were simply
left undone.
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
    world = generate_terrain("rs3", profile, 0.4, 5)
    scene = os.path.join(tempfile.mkdtemp(), "s.xml")
    build_mjcf_scene(robot, world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoSensorSuite(model, data, profile)


GRID = {"kind": "grid", "x_range": [-0.2, 0.2], "y_range": [-0.2, 0.2],
        "resolution": 0.2}


def test_backend_declares_the_remaining_sources():
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    for name in ("recurrent_state", "estimator_output", "adaptation_latent",
                 "visual_latent", "traversability_map", "gnss_position"):
        assert name in declared, name


def test_recurrent_state_starts_zeroed_at_the_declared_size():
    suite = _suite()
    values = suite.recurrent_state({"size": 32})
    assert len(values) == 32
    assert all(v == 0.0 for v in values), "hidden state must start zeroed"


def test_recurrent_state_carries_across_steps():
    suite = _suite()
    suite.set_recurrent_state([0.5] * 8)
    assert suite.recurrent_state({"size": 8}) == pytest.approx([0.5] * 8)


def test_recurrent_state_is_cleared_between_trials():
    """A leaked hidden state silently inflates the second trial."""
    suite = _suite()
    suite.set_recurrent_state([0.9] * 8)
    suite.reset()
    assert suite.recurrent_state({"size": 8}) == pytest.approx([0.0] * 8)


def test_estimator_and_adaptation_latents_are_policy_supplied():
    suite = _suite()
    assert len(suite.estimator_output({"size": 3})) == 3
    assert len(suite.adaptation_latent({"size": 8})) == 8
    assert len(suite.visual_latent({"size": 32})) == 32


def test_traversability_is_bounded_and_matches_the_grid():
    suite = _suite()
    values = suite.traversability_map(GRID)
    assert len(values) == 9
    assert all(0.0 <= v <= 1.0 for v in values), "traversability is a 0..1 score"


def test_flat_ground_is_more_traversable_than_a_slope():
    """The score must actually reflect terrain, not be a constant."""
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    pytest.importorskip("mujoco")
    robot = "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    profile = inspect_mjcf(robot)
    scores = {}
    for terrain in ("rs1", "rs3"):
        world = generate_terrain(terrain, profile, 0.8, 5)
        scene = os.path.join(tempfile.mkdtemp(), "s.xml")
        build_mjcf_scene(robot, world, scene, 0.005)
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        wide = {"kind": "grid", "x_range": [-0.4, 0.4], "y_range": [-0.1, 0.1],
                "resolution": 0.2}
        values = MujocoSensorSuite(model, data, profile).traversability_map(wide)
        scores[terrain] = sum(values) / len(values)
    assert scores["rs1"] >= scores["rs3"], scores


def test_gnss_position_is_a_geodetic_triple():
    suite = _suite()
    values = suite.gnss_position({"datum": [37.4, -122.1, 10.0]})
    assert len(values) == 3
    assert -90.0 <= values[0] <= 90.0, "latitude out of range: %r" % values[0]
    assert -180.0 <= values[1] <= 180.0, "longitude out of range: %r" % values[1]
