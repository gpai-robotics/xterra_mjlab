"""One shared CanonicalState builder for every simulator backend.

The original defect was three adapters emitting different key names for the
same quantity, so one manifest could not run on two backends. Preventing that
recurring means the assembly lives in ONE place and every backend calls it.
"""

import math

import pytest


def test_shared_builder_derives_gravity_and_body_velocity():
    """Golden values under a 30 degree pitch, derived by hand.

    R^T @ [0,0,-1] = [sin30, 0, -cos30] = [0.5, 0, -0.8660254]
    R^T @ [1,0,0]  = [cos30, 0,  sin30] = [0.8660254, 0, 0.5]
    """
    from robustness_lab.canonical_state import build_canonical_state
    half = math.radians(30.0) / 2.0
    state = build_canonical_state(
        joint_order=["a", "b"],
        joint_position=[0.1, 0.2],
        joint_velocity=[0.3, 0.4],
        base_position_w=[1.0, 2.0, 3.0],
        base_orientation_wxyz=[math.cos(half), 0.0, math.sin(half), 0.0],
        base_linear_velocity_w=[1.0, 0.0, 0.0],
        base_angular_velocity_w=[0.0, 0.0, 0.0],
        velocity_command=[0.5, 0.0, 0.2],
        previous_action=[0.0, 0.0],
        episode_time=0.5,
    )
    assert state.projected_gravity_b == pytest.approx(
        [0.5, 0.0, -0.8660254037844387], abs=1e-9)
    assert state.base_linear_velocity_b == pytest.approx(
        [0.8660254037844387, 0.0, 0.5], abs=1e-9)
    assert state.joint_position == pytest.approx([0.1, 0.2], abs=1e-9)
    assert state.velocity_command == pytest.approx([0.5, 0.0, 0.2], abs=1e-9)


def test_builder_rejects_joint_values_that_do_not_match_joint_order():
    from robustness_lab.canonical_state import build_canonical_state
    half = 0.0
    with pytest.raises(ValueError):
        build_canonical_state(
            joint_order=["a", "b", "c"],
            joint_position=[0.1, 0.2],
            joint_velocity=[0.3, 0.4, 0.5],
            base_position_w=[0.0, 0.0, 0.0],
            base_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            base_linear_velocity_w=[0.0, 0.0, 0.0],
            base_angular_velocity_w=[0.0, 0.0, 0.0],
            velocity_command=[0.0, 0.0, 0.0],
            previous_action=[0.0, 0.0, 0.0],
            episode_time=half,
        )


def test_unsupported_fields_stay_none_not_zero():
    """A zero-filled unsupported field is indistinguishable from a reading."""
    from robustness_lab.canonical_state import build_canonical_state
    state = build_canonical_state(
        joint_order=["a"],
        joint_position=[0.0],
        joint_velocity=[0.0],
        base_position_w=[0.0, 0.0, 0.0],
        base_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        base_linear_velocity_w=[0.0, 0.0, 0.0],
        base_angular_velocity_w=[0.0, 0.0, 0.0],
        velocity_command=[0.0, 0.0, 0.0],
        previous_action=[0.0],
        episode_time=0.0,
    )
    assert state.height_scan is None
    assert state.goal_vector_b is None


def test_every_simulator_backend_uses_the_shared_builder():
    """Structural guard against the three-different-key-names defect returning."""
    import inspect
    from robustness_lab.backends import isaacgym, isaaclab, mujoco
    for module in (mujoco, isaaclab, isaacgym):
        source = inspect.getsource(module)
        assert "build_canonical_state" in source, (
            "%s must assemble state through the shared builder" % module.__name__)


def test_state_exposes_a_mapping_the_compiler_can_consume():
    """The compiler takes a mapping of canonical field -> values."""
    from robustness_lab.canonical_state import build_canonical_state
    from robustness_lab.observation.compiler import compile_terms
    state = build_canonical_state(
        joint_order=["a"],
        joint_position=[0.25],
        joint_velocity=[0.5],
        base_position_w=[0.0, 0.0, 0.4],
        base_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        base_linear_velocity_w=[0.0, 0.0, 0.0],
        base_angular_velocity_w=[0.0, 0.0, 0.0],
        velocity_command=[0.6, 0.0, 0.0],
        previous_action=[0.0],
        episode_time=0.0,
    )
    plan = compile_terms([
        {"source": "projected_gravity_b", "size": 3},
        {"source": "joint_position", "transform": [{"scale": 4.0}], "size": 1},
    ])
    assert plan.build(state) == pytest.approx([0.0, 0.0, -1.0, 1.0], abs=1e-9)
