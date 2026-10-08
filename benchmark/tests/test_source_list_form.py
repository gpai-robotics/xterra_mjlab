"""The easiest interface: name your sources in order, sizes are computed.

Guessing a layout from a bare width is inherently ambiguous -- 48 could be
3+3+3+12+12+12+3 or 4+4+4+12+12+12. So do not make the researcher encode
semantics as an integer. Let them LIST the observation sources in the order
their network expects, and derive every width from the robot profile and the
declared sensor geometry.

Numbers stay available for anything unusual; this is the path that needs none.
"""

import pytest

JOINTS12 = ["j%d" % i for i in range(12)]
STANDARD = ["base_linear_velocity_b", "base_angular_velocity_b",
            "projected_gravity_b", "velocity_command",
            "joint_position", "joint_velocity", "previous_action"]


def test_sizes_are_derived_from_the_robot():
    """No size given anywhere: 3+3+3+3 + 12+12+12 = 48."""
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(STANDARD, joint_order=JOINTS12)
    assert plan.size == 48


def test_the_same_list_scales_with_joint_count():
    """An 18-joint robot gets 12 + 3*18 = 66 from the identical list."""
    from robustness_lab.observation.compiler import compile_source_list
    joints = ["j%d" % i for i in range(18)]
    assert compile_source_list(STANDARD, joint_order=joints).size == 66


def test_foot_sources_size_from_the_foot_count():
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(["foot_contact"], joint_order=JOINTS12,
                               foot_names=["FL", "FR", "RL", "RR"])
    assert plan.size == 4


def test_exteroceptive_source_sizes_from_its_geometry():
    from robustness_lab.observation.compiler import compile_source_list
    geometry = {"height_scan": {"kind": "grid", "x_range": [-0.8, 0.8],
                                "y_range": [-0.5, 0.5], "resolution": 0.1}}
    plan = compile_source_list(STANDARD + ["height_scan"], joint_order=JOINTS12,
                               geometry=geometry)
    assert plan.size == 48 + 187


def test_declared_order_is_preserved():
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(["projected_gravity_b", "velocity_command"],
                               joint_order=JOINTS12)
    built = plan.build({"projected_gravity_b": [0.0, 0.0, -1.0],
                        "velocity_command": [0.5, 0.0, 0.1]})
    assert built == pytest.approx([0.0, 0.0, -1.0, 0.5, 0.0, 0.1])


def test_a_source_needing_geometry_without_it_is_rejected():
    from robustness_lab.observation.compiler import compile_source_list
    with pytest.raises(ValueError, match="geometry"):
        compile_source_list(["height_scan"], joint_order=JOINTS12)


def test_unknown_source_is_rejected_by_name():
    from robustness_lab.observation.compiler import compile_source_list
    with pytest.raises(ValueError, match="wobble"):
        compile_source_list(["wobble"], joint_order=JOINTS12)


def test_computed_total_can_be_checked_against_the_artifact():
    """The whole point: the researcher never types a width, and we can still
    verify their list against what the network really accepts."""
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(STANDARD, joint_order=JOINTS12)
    from robustness_lab.observation.gates import check_dimension
    result = check_dimension(compiled_size=plan.size, declared_size=plan.size,
                             artifact_size=48)
    assert result["status"] == "ready"


def test_mismatch_against_the_artifact_is_reported_with_both_numbers():
    from robustness_lab.observation.compiler import compile_source_list
    from robustness_lab.observation.gates import check_dimension
    plan = compile_source_list(STANDARD, joint_order=JOINTS12)
    result = check_dimension(compiled_size=plan.size, declared_size=plan.size,
                             artifact_size=45)
    assert result["status"] == "blocked"
    assert "48" in str(result) and "45" in str(result)


def test_per_source_transforms_still_allowed_in_list_form():
    """Naming sources must not cost the transform vocabulary."""
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        [{"source": "velocity_command", "transform": [{"scale": 2.0}]},
         "projected_gravity_b"], joint_order=JOINTS12)
    built = plan.build({"velocity_command": [1.0, 2.0, 3.0],
                        "projected_gravity_b": [0.0, 0.0, -1.0]})
    assert built == pytest.approx([2.0, 4.0, 6.0, 0.0, 0.0, -1.0])
