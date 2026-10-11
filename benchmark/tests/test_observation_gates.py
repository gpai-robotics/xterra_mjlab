"""Verification gates.

The point of these gates is that a wrong manifest is caught BEFORE any terrain
is scored, and is never charged to the policy. That is the RS11 principle
generalised: a benchmark that scores a broken adapter as a policy failure is
worse than one that refuses to run.
"""

import pytest


# ---------------------------------------------------------------- G1 dimension

def test_dimension_mismatch_is_adapter_unverified_not_policy_failure():
    from robustness_lab.observation.gates import check_dimension
    result = check_dimension(compiled_size=45, declared_size=45, artifact_size=48)
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"
    assert result["outcome_class"] != "policy_fail"


def test_matching_dimensions_pass():
    from robustness_lab.observation.gates import check_dimension
    assert check_dimension(compiled_size=45, declared_size=45,
                           artifact_size=45)["status"] == "ready"


def test_declared_size_disagreeing_with_compiled_size_is_blocked():
    from robustness_lab.observation.gates import check_dimension
    result = check_dimension(compiled_size=45, declared_size=48, artifact_size=45)
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"


def test_artifact_probe_accepts_the_right_size_and_rejects_the_wrong_one(tmp_path):
    """G1's real half: ask the artifact itself, do not trust the manifest."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from robustness_lab.observation.gates import check_artifact_accepts

    model = torch.jit.script(nn.Sequential(nn.Linear(45, 12)))
    path = tmp_path / "policy.pt"
    model.save(str(path))

    good = check_artifact_accepts(str(path), 45)
    assert good["status"] == "ready"
    assert good["output_size"] == 12

    bad = check_artifact_accepts(str(path), 48)
    assert bad["status"] == "blocked"
    assert bad["outcome_class"] == "adapter_unverified"


# ------------------------------------------------------------- G2 finite/scale

def test_non_finite_observation_is_caught():
    from robustness_lab.observation.gates import check_finite
    result = check_finite([1.0, float("nan"), 3.0])
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"


def test_infinite_observation_is_caught():
    from robustness_lab.observation.gates import check_finite
    assert check_finite([1.0, float("inf")])["status"] == "blocked"


def test_finite_observation_passes():
    from robustness_lab.observation.gates import check_finite
    assert check_finite([1.0, -2.0, 0.0])["status"] == "ready"


# --------------------------------------------------------- G3 known-pose replay

def test_wrong_gravity_sign_is_caught_by_known_pose_replay():
    from robustness_lab.observation.gates import check_known_pose
    result = check_known_pose({"projected_gravity_b": [0.0, 0.0, 1.0]})  # flipped
    assert result["status"] == "blocked"
    assert "projected_gravity_b" in result["failing_terms"]
    assert result["outcome_class"] == "adapter_unverified"


def test_correct_known_pose_passes():
    from robustness_lab.observation.gates import check_known_pose
    result = check_known_pose({"projected_gravity_b": [0.0, 0.0, -1.0]})
    assert result["status"] == "ready"
    assert result["failing_terms"] == []


def test_joint_position_far_from_default_is_caught_by_known_pose_replay():
    """At the default pose the default-subtracted joint term must be ~0."""
    from robustness_lab.observation.gates import check_known_pose
    result = check_known_pose({"projected_gravity_b": [0.0, 0.0, -1.0],
                               "joint_position_error": [0.0, 0.9, 0.0]})
    assert result["status"] == "blocked"
    assert "joint_position_error" in result["failing_terms"]


# ------------------------------------------------------------- outcome taxonomy

def test_adapter_unverified_is_a_distinct_outcome_class():
    from robustness_lab.observation.gates import OUTCOME_CLASSES
    assert "adapter_unverified" in OUTCOME_CLASSES
    assert "policy_fail" in OUTCOME_CLASSES
    assert "environment_unsupported" in OUTCOME_CLASSES
