"""A policy's PD gains are part of its interface and must come from the manifest.

Found by running a real third-party policy. HuggingFace
diasAiMaster/unitree-go2-velocity-flat publishes:

    stiffness: [20, 20, 40, ...]     damping: [1, 1, 2, ...]

The manifest schema had no field for either, so the benchmark ran the policy
with default gains (25 / 0.5) and it fell immediately on flat ground. That is an
ADAPTER failure reported as a POLICY failure -- the misattribution this project
exists to prevent.

Gains belong with `action`: they define how the emitted joint targets are
actually tracked.
"""

import pytest
import yaml

JOINTS = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
          "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
STIFFNESS = [20.0, 20.0, 40.0] * 4
DAMPING = [1.0, 1.0, 2.0] * 4


def _manifest(tmp_path, action_extra):
    action = {"space": "joint_position_offset", "scale": 0.5,
              "offset": "default_joint_position", "size": 12}
    action.update(action_extra)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "2.0", "policy_id": "gains",
        "runtime": {"format": "torchscript", "path": "p.pt",
                    "control_frequency_hz": 50},
        "canonical_joint_order": JOINTS, "joint_order_policy": JOINTS,
        "references": {"default_joint_position": [0.0] * 12},
        "observation": {"terms": [{"source": "joint_position", "size": 12}],
                        "history": {"frames": 1, "order": "newest_first"},
                        "expected_size": 12},
        "action": action,
    }), encoding="utf-8")
    return str(path)


def test_manifest_carries_per_joint_gains(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest
    path = _manifest(tmp_path, {"stiffness": STIFFNESS, "damping": DAMPING})
    manifest = load_policy_manifest(path, check_files=False)
    assert list(manifest.action.stiffness) == pytest.approx(STIFFNESS)
    assert list(manifest.action.damping) == pytest.approx(DAMPING)


def test_scalar_gains_are_broadcast_to_every_joint(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest
    path = _manifest(tmp_path, {"stiffness": 30.0, "damping": 1.5})
    manifest = load_policy_manifest(path, check_files=False)
    assert list(manifest.action.stiffness) == pytest.approx([30.0] * 12)
    assert list(manifest.action.damping) == pytest.approx([1.5] * 12)


def test_gain_length_mismatch_is_rejected(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest
    path = _manifest(tmp_path, {"stiffness": [20.0, 20.0], "damping": DAMPING})
    with pytest.raises(ValueError, match="stiffness"):
        load_policy_manifest(path, check_files=False)


def test_gains_are_remapped_into_simulator_joint_order(tmp_path):
    """The researcher declares gains in THEIR joint order, like everything else."""
    from robustness_lab.policy_manifest import load_policy_manifest
    reversed_order = list(reversed(JOINTS))
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "2.0", "policy_id": "gains-remap",
        "runtime": {"format": "torchscript", "path": "p.pt",
                    "control_frequency_hz": 50},
        "canonical_joint_order": JOINTS, "joint_order_policy": reversed_order,
        "references": {"default_joint_position": [0.0] * 12},
        "observation": {"terms": [{"source": "joint_position", "size": 12}],
                        "history": {"frames": 1, "order": "newest_first"},
                        "expected_size": 12},
        "action": {"space": "joint_position_offset", "scale": 0.5,
                   "offset": "default_joint_position", "size": 12,
                   "joint_order": reversed_order,
                   "stiffness": STIFFNESS, "damping": DAMPING},
    }), encoding="utf-8")
    manifest = load_policy_manifest(str(path), check_files=False)
    # declared against reversed order, so canonical order sees them reversed
    assert list(manifest.action.stiffness) == pytest.approx(list(reversed(STIFFNESS)))


def test_missing_gains_are_reported_as_needing_review(tmp_path):
    """Silently substituting a default gain changes the robot's dynamics."""
    from robustness_lab.policy_manifest import load_policy_manifest
    manifest = load_policy_manifest(_manifest(tmp_path, {}), check_files=False)
    assert manifest.action.gains_declared is False


def test_scaffold_flags_gains_as_researcher_supplied(tmp_path):
    from robustness_lab.scaffold import scaffold_policy_manifest
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    artifact = tmp_path / "policy.pt"
    torch.jit.script(nn.Sequential(nn.Linear(45, 12))).save(str(artifact))
    result = scaffold_policy_manifest(str(artifact), joint_order=JOINTS,
                                      output=str(tmp_path / "m.yaml"))
    assert any("gain" in item.lower() or "stiffness" in item.lower()
               for item in result["needs_review"]), result["needs_review"]
