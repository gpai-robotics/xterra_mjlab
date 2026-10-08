"""A schema-v2 policy must be reachable from the CLI, not just from Python.

`controllers.py` builds a CompiledPolicyController and `runtime.py` dispatches
on `controller.type == "compiled_policy"`, but skill.CONTROLLER_TYPES never
listed it. So every documented bring-your-own-policy command --

    robustlab preflight <examiner.yaml>
    robustlab launch    <examiner.yaml>

-- died at skill validation with "unsupported skill controller type
compiled_policy". The generic path existed and could not be reached.
"""

import pytest
import yaml


JOINTS = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
          "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
          "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
          "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]


def _skill(tmp_path, controller):
    path = tmp_path / "skill.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "skill_id": "compiled",
        "sensor_track": "blind",
        "control_frequency_hz": 50,
        "navigation": {"mode": "velocity_command", "route_scope": "straight_only",
                       "assistance": "route_follower"},
        "controller": controller,
        "sensors": [{"name": "velocity_command", "kind": "command",
                     "shape": [3], "rate_hz": 50}],
        "observations": [],
        "action": {"mode": "position", "joint_order": JOINTS, "size": 12},
    }), encoding="utf-8")
    return str(path)


def test_compiled_policy_is_an_accepted_controller_type(tmp_path):
    from robustness_lab.skill import load_skill

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("placeholder", encoding="utf-8")
    skill = load_skill(
        _skill(tmp_path, {"type": "compiled_policy", "manifest": str(manifest)}),
        check_files=False,
    )
    assert skill.controller["type"] == "compiled_policy"


def test_a_compiled_policy_skill_without_a_manifest_is_refused(tmp_path):
    from robustness_lab.skill import load_skill

    with pytest.raises(ValueError, match="manifest"):
        load_skill(_skill(tmp_path, {"type": "compiled_policy"}), check_files=False)


def test_a_missing_manifest_file_is_refused_when_checking_files(tmp_path):
    from robustness_lab.skill import load_skill

    path = _skill(tmp_path, {"type": "compiled_policy",
                             "manifest": str(tmp_path / "absent.yaml")})
    with pytest.raises(ValueError, match="does not exist"):
        load_skill(path, check_files=True)
