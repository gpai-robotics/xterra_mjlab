"""ONNX policies must actually run, not merely be advertised.

The manifest schema offers `format: torchscript | onnx`, but the artifact gate
and the compiled_policy runtime only understood TorchScript. A real ONNX policy
was refused with a TorchScript zip-reader error -- the same
declared-but-not-implemented defect this project exists to remove.

Found by running a public third-party policy
(HuggingFace diasAiMaster/unitree-go2-velocity-flat, obs[1,45] -> actions[1,12]).
"""

import pytest


def _onnx_artifact(tmp_path, in_features=45, out_features=12):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    import torch.nn as nn
    path = tmp_path / "policy.onnx"
    model = nn.Sequential(nn.Linear(in_features, 32), nn.ELU(), nn.Linear(32, out_features))
    torch.onnx.export(model, torch.zeros(1, in_features), str(path),
                      input_names=["obs"], output_names=["actions"])
    return str(path)


def test_artifact_gate_accepts_a_valid_onnx_policy(tmp_path):
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_onnx_artifact(tmp_path), 45)
    assert result["status"] == "ready", result.get("reason")
    assert result["output_size"] == 12


def test_artifact_gate_rejects_the_wrong_width_for_onnx(tmp_path):
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_onnx_artifact(tmp_path), 48)
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"


def test_onnx_error_does_not_claim_the_file_is_torchscript(tmp_path):
    """The old failure blamed a zip reader, which sent researchers chasing a
    corrupt-file theory instead of an unsupported-format one."""
    from robustness_lab.observation.gates import check_artifact_accepts
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"not a model")
    result = check_artifact_accepts(str(bad), 45)
    assert result["status"] == "blocked"
    assert "PytorchStreamReader" not in str(result.get("reason", ""))


def test_probe_signature_works_for_onnx(tmp_path):
    from robustness_lab.scaffold import probe_artifact_signature
    signature = probe_artifact_signature(_onnx_artifact(tmp_path))
    assert signature["input_count"] == 1
    assert signature["input_names"] == ["obs"]


def test_compiled_policy_runs_an_onnx_artifact(tmp_path):
    """End to end: an ONNX policy must drive the controller."""
    import yaml
    from robustness_lab.controllers import make_controller
    artifact = _onnx_artifact(tmp_path, 6, 3)
    joints = ["a", "b", "c"]
    manifest = tmp_path / "policy.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": "2.0",
        "policy_id": "onnx-probe",
        "runtime": {"format": "onnx", "path": artifact, "control_frequency_hz": 50},
        "canonical_joint_order": joints,
        "joint_order_policy": joints,
        "observation": {
            "terms": [{"source": "velocity_command", "size": 3},
                      {"source": "projected_gravity_b", "size": 3}],
            "history": {"frames": 1, "order": "newest_first"},
            "expected_size": 6,
        },
        "action": {"space": "joint_position_offset", "scale": 1.0, "size": 3},
    }), encoding="utf-8")
    controller = make_controller(
        {"type": "compiled_policy", "manifest": str(manifest)}, action_size=3)
    action = controller.act({"velocity_command": [0.1, 0.2, 0.3],
                             "projected_gravity_b": [0.0, 0.0, -1.0]})
    assert len(action) == 3
    assert all(abs(v) < 1e6 for v in action)


def test_scaffold_drafts_a_manifest_from_an_onnx_artifact(tmp_path):
    from robustness_lab.scaffold import scaffold_policy_manifest
    joints = ["j%d" % i for i in range(12)]
    result = scaffold_policy_manifest(_onnx_artifact(tmp_path), joint_order=joints,
                                      output=str(tmp_path / "m.yaml"))
    assert result["input_size"] == 45
    assert result["output_size"] == 12
