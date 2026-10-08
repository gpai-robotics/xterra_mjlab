import pytest

from robustness_lab.controllers import validate_controller_mapping


def test_learned_controller_requires_complete_mapping():
    with pytest.raises(ValueError, match="joint_order"):
        validate_controller_mapping({"type": "onnx", "action_mode": "position", "observation_keys": ["joint_position"]}, ["hip"])
    with pytest.raises(ValueError, match="action_mode"):
        validate_controller_mapping({"type": "python", "joint_order": ["hip"]}, ["hip"])
    with pytest.raises(ValueError, match="does not match"):
        validate_controller_mapping({"type": "python", "joint_order": ["knee"], "action_mode": "effort"}, ["hip"])


def test_builtin_hold_controller_needs_no_external_mapping():
    validate_controller_mapping({"type": "pd_hold"}, ["hip"])
