"""Action decompiler: network output -> simulator joint targets.

Golden values derived by hand. Convention is clip -> scale -> offset, matching
the legged_gym / RSL-RL action pipeline.
"""

import pytest


def test_clip_then_scale_then_offset():
    from robustness_lab.observation.action import compile_action
    decoder = compile_action({"space": "joint_position_offset", "scale": 0.25,
                              "offset": [0.5, 0.5], "clip": [-1.0, 1.0], "size": 2})
    # raw [1.0, -2.0] -> clip [1.0, -1.0] -> scale [0.25, -0.25] -> offset [0.75, 0.25]
    assert decoder.decode([1.0, -2.0]) == pytest.approx([0.75, 0.25], abs=1e-9)


def test_named_offset_resolves_from_references():
    from robustness_lab.observation.action import compile_action
    decoder = compile_action({"space": "joint_position_offset", "scale": 1.0,
                              "offset": "default_joint_position", "size": 3},
                             references={"default_joint_position": [0.0, 0.8, -1.5]})
    assert decoder.decode([0.1, 0.1, 0.1]) == pytest.approx([0.1, 0.9, -1.4], abs=1e-9)


def test_action_remaps_policy_order_back_to_simulator_order():
    """Inverse of the observation permutation: the policy emits in ITS order,
    the simulator must receive canonical order."""
    from robustness_lab.observation.action import compile_action
    decoder = compile_action({"space": "joint_position_offset", "scale": 1.0, "size": 4,
                              "joint_order": ["C", "A", "D", "B"]},
                             canonical_joint_order=["A", "B", "C", "D"])
    assert decoder.decode([3.0, 1.0, 4.0, 2.0]) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0], abs=1e-9)


def test_wrong_action_length_is_rejected():
    from robustness_lab.observation.action import compile_action
    decoder = compile_action({"space": "joint_position_offset", "scale": 1.0, "size": 4})
    with pytest.raises(ValueError):
        decoder.decode([1.0, 2.0])


def test_unknown_action_space_is_rejected():
    from robustness_lab.observation.action import compile_action
    with pytest.raises(ValueError, match="joint_position_offset"):
        compile_action({"space": "telekinesis", "size": 4})


def test_joint_name_absent_from_robot_is_still_rejected():
    """Remapping replaces rejection-on-ORDER, not rejection-on-UNKNOWN-JOINT."""
    from robustness_lab.observation.action import compile_action
    with pytest.raises(ValueError, match="Q"):
        compile_action({"space": "joint_position_offset", "scale": 1.0, "size": 4,
                        "joint_order": ["C", "A", "Q", "B"]},
                       canonical_joint_order=["A", "B", "C", "D"])
