"""Multi-input policies and encoder sub-artifacts.

Measured against the real FoReSight exports committed under assets/cache:

    student : forward(x, scandots_latent) -> Tensor      two tensors
    teacher : forward(x) -> Tensor                        one tensor
    encoder : forward(depth_image, proprioception) -> latent

The compiler currently builds ONE concatenated vector, so the student cannot
run. Supporting it needs named tensors plus the ability to run a declared
sub-artifact to produce one of them.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "assets" / "cache" / "skills" / "go2_extreme_parkour_student"
STUDENT = SKILLS / "exported_deploy" / "policy.pt"
TEACHER = SKILLS / "exported_teacher" / "policy.pt"
ENCODER = SKILLS / "exported_deploy" / "depth_latest.pt"


def _require(path):
    pytest.importorskip("torch")
    if not path.is_file():
        pytest.skip("artifact not present: %s" % path)


def test_probe_reports_a_single_input_policy():
    _require(TEACHER)
    from robustness_lab.scaffold import probe_artifact_signature
    signature = probe_artifact_signature(str(TEACHER))
    assert signature["input_count"] == 1


def test_probe_reports_a_multi_input_policy_and_names_its_tensors():
    _require(STUDENT)
    from robustness_lab.scaffold import probe_artifact_signature
    signature = probe_artifact_signature(str(STUDENT))
    assert signature["input_count"] == 2
    assert signature["input_names"] == ["x", "scandots_latent"]


def test_probe_reports_the_depth_encoder_signature():
    _require(ENCODER)
    from robustness_lab.scaffold import probe_artifact_signature
    signature = probe_artifact_signature(str(ENCODER))
    assert signature["input_count"] == 2
    assert signature["input_names"] == ["depth_image", "proprioception"]


def test_multi_tensor_plan_builds_named_tensors():
    from robustness_lab.observation.compiler import compile_tensors
    plan = compile_tensors([
        {"name": "x", "terms": [{"source": "joint_position", "size": 3},
                                {"source": "velocity_command", "size": 3}]},
        {"name": "scandots_latent", "terms": [{"source": "previous_action", "size": 2}]},
    ])
    built = plan.build({"joint_position": [1.0, 2.0, 3.0],
                        "velocity_command": [0.1, 0.2, 0.3],
                        "previous_action": [9.0, 8.0]})
    assert list(built.keys()) == ["x", "scandots_latent"]
    assert built["x"] == pytest.approx([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], abs=1e-9)
    assert built["scandots_latent"] == pytest.approx([9.0, 8.0], abs=1e-9)


def test_multi_tensor_plan_preserves_declared_order():
    """Tensors are passed positionally to forward(); order is load-bearing."""
    from robustness_lab.observation.compiler import compile_tensors
    plan = compile_tensors([
        {"name": "second", "terms": [{"source": "previous_action", "size": 2}]},
        {"name": "first", "terms": [{"source": "velocity_command", "size": 3}]},
    ])
    built = plan.build({"previous_action": [1.0, 2.0], "velocity_command": [3.0, 4.0, 5.0]})
    assert list(built.keys()) == ["second", "first"]


def test_single_tensor_manifest_still_compiles_unchanged():
    """The existing single-vector path must not regress."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "joint_position", "size": 3}])
    assert plan.build({"joint_position": [1.0, 2.0, 3.0]}) == pytest.approx(
        [1.0, 2.0, 3.0], abs=1e-9)


def test_multi_input_artifact_is_rejected_by_a_single_tensor_manifest():
    """Refuse loudly rather than feeding a 2-input policy one tensor."""
    _require(STUDENT)
    from robustness_lab.observation.gates import check_artifact_arity
    result = check_artifact_arity(str(STUDENT), declared_tensor_count=1)
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"
