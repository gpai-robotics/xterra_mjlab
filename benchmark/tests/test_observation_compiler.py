"""Golden-value tests for the observation compiler.

Every constant here was derived by hand. A stub or placeholder cannot
reproduce them. These encode the transforms every published legged-locomotion
policy applies and which schema v1 could not express at all.
"""

import pytest


def test_scale_transform_is_applied():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "base_angular_velocity_b",
                           "transform": [{"scale": 0.25}], "size": 3}])
    assert plan.build({"base_angular_velocity_b": [0.4, -0.8, 1.2]}) == pytest.approx(
        [0.1, -0.2, 0.3], abs=1e-9)


def test_per_element_scale():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command",
                           "transform": [{"scale": [2.0, 2.0, 0.25]}], "size": 3}])
    assert plan.build({"velocity_command": [1.0, 0.5, 4.0]}) == pytest.approx(
        [2.0, 1.0, 1.0], abs=1e-9)


def test_subtract_named_default_pose():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms(
        [{"source": "joint_position",
          "transform": [{"subtract": "default_joint_position"}], "size": 3}],
        references={"default_joint_position": [0.0, 0.8, -1.5]},
    )
    assert plan.build({"joint_position": [0.1, 0.9, -1.9]}) == pytest.approx(
        [0.1, 0.1, -0.4], abs=1e-9)


def test_joint_order_remap_is_a_permutation_not_a_rejection():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms(
        [{"source": "joint_position", "joint_order": "policy", "size": 4}],
        canonical_joint_order=["A", "B", "C", "D"],
        policy_joint_order=["C", "A", "D", "B"],
    )
    assert plan.build({"joint_position": [1.0, 2.0, 3.0, 4.0]}) == pytest.approx(
        [3.0, 1.0, 4.0, 2.0], abs=1e-9)


def test_history_newest_first_stacking():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2}],
                         history={"frames": 3, "order": "newest_first"})
    plan.build({"velocity_command": [1.0, 2.0]})
    plan.build({"velocity_command": [3.0, 4.0]})
    out = plan.build({"velocity_command": [5.0, 6.0]})
    assert len(out) == 6
    assert out == pytest.approx([5.0, 6.0, 3.0, 4.0, 1.0, 2.0], abs=1e-9)


def test_history_oldest_first_stacking():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2}],
                         history={"frames": 3, "order": "oldest_first"})
    plan.build({"velocity_command": [1.0, 2.0]})
    plan.build({"velocity_command": [3.0, 4.0]})
    assert plan.build({"velocity_command": [5.0, 6.0]}) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], abs=1e-9)


def test_history_buffer_is_first_frame_filled_at_reset():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2}],
                         history={"frames": 3, "order": "newest_first"})
    assert plan.build({"velocity_command": [1.0, 2.0]}) == pytest.approx(
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], abs=1e-9)


def test_reset_clears_history_between_trials():
    """A leaked buffer silently inflates scores on the second trial."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2}],
                         history={"frames": 2, "order": "newest_first"})
    plan.build({"velocity_command": [1.0, 2.0]})
    plan.build({"velocity_command": [3.0, 4.0]})
    plan.reset()
    assert plan.build({"velocity_command": [5.0, 6.0]}) == pytest.approx(
        [5.0, 6.0, 5.0, 6.0], abs=1e-9)


def test_unknown_transform_raises_and_lists_valid_names():
    from robustness_lab.observation.compiler import compile_terms
    with pytest.raises(ValueError, match="scale"):
        compile_terms([{"source": "joint_position",
                        "transform": [{"wobble": 1.0}], "size": 3}])


def test_unknown_source_is_rejected():
    from robustness_lab.observation.compiler import compile_terms
    with pytest.raises(ValueError, match="not_a_real_field"):
        compile_terms([{"source": "not_a_real_field", "size": 3}])


def test_declared_size_mismatch_is_rejected():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 3}])
    with pytest.raises(ValueError):
        plan.build({"velocity_command": [1.0, 2.0]})


def test_reference_45_dim_observation_has_expected_size():
    from robustness_lab.observation.compiler import compile_terms
    terms = [
        {"source": "base_angular_velocity_b", "transform": [{"scale": 0.25}], "size": 3},
        {"source": "projected_gravity_b", "size": 3},
        {"source": "velocity_command", "transform": [{"scale": [2.0, 2.0, 0.25]}], "size": 3},
        {"source": "joint_position", "size": 12},
        {"source": "joint_velocity", "transform": [{"scale": 0.05}], "size": 12},
        {"source": "previous_action", "size": 12},
    ]
    assert compile_terms(terms).size == 45


def test_rma_style_50_frame_history_size():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "joint_position", "size": 12},
                          {"source": "joint_velocity", "size": 12},
                          {"source": "previous_action", "size": 12}],
                         history={"frames": 50, "order": "newest_first"})
    assert plan.size == 1800


def test_clip_transform():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "joint_velocity",
                           "transform": [{"clip": [-1.0, 1.0]}], "size": 4}])
    assert plan.build({"joint_velocity": [-3.0, -0.5, 0.5, 3.0]}) == pytest.approx(
        [-1.0, -0.5, 0.5, 1.0], abs=1e-9)


def test_transform_chain_applies_in_declared_order():
    """subtract then scale differs from scale then subtract; order must be honoured."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms(
        [{"source": "joint_position",
          "transform": [{"subtract": [1.0, 1.0]}, {"scale": 2.0}], "size": 2}])
    assert plan.build({"joint_position": [3.0, 5.0]}) == pytest.approx(
        [4.0, 8.0], abs=1e-9)
