"""`robustlab policy scaffold`: draft a manifest from the uploaded weights.

The point is to minimise what a submitter must write by hand. The tool probes
the artifact, recovers its real input/output widths, and drafts a schema-v2
manifest. It must NOT silently guess the semantic layout -- it flags what the
researcher has to confirm, and the verification gates catch it if they get it
wrong. Generation gives convenience; verification gives correctness.
"""

import pytest
import yaml

JOINTS = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
          "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]


def _artifact(tmp_path, in_features, out_features):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    path = tmp_path / "policy.pt"
    torch.jit.script(
        nn.Sequential(nn.Linear(in_features, 64), nn.ELU(), nn.Linear(64, out_features))
    ).save(str(path))
    return str(path)


def test_scaffold_recovers_real_dimensions_from_the_artifact(tmp_path):
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 45, 12)
    result = scaffold_policy_manifest(artifact, joint_order=JOINTS,
                                      output=str(tmp_path / "policy.yaml"))
    assert result["input_size"] == 45
    assert result["output_size"] == 12


def test_scaffolded_manifest_is_valid_and_compiles_to_the_probed_width(tmp_path):
    """The draft must be loadable, not a template the user has to repair."""
    from robustness_lab.policy_manifest import load_policy_manifest
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 45, 12)
    out = str(tmp_path / "policy.yaml")
    scaffold_policy_manifest(artifact, joint_order=JOINTS, output=out)
    manifest = load_policy_manifest(out, check_files=False)
    assert manifest.observation_plan.size == 45


def test_scaffold_uses_the_robots_joint_order(tmp_path):
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 45, 12)
    out = str(tmp_path / "policy.yaml")
    scaffold_policy_manifest(artifact, joint_order=JOINTS, output=out)
    payload = yaml.safe_load(open(out, encoding="utf-8"))
    assert payload["action"]["size"] == 12
    assert payload["joint_order_policy"] == JOINTS


def test_scaffold_flags_what_the_researcher_must_confirm(tmp_path):
    """A drafted layout is a hypothesis, never an assertion."""
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 45, 12)
    result = scaffold_policy_manifest(artifact, joint_order=JOINTS,
                                      output=str(tmp_path / "policy.yaml"))
    assert result["needs_review"], "the draft must not present a guess as settled"
    assert isinstance(result["needs_review"], list)
    assert len(result["needs_review"]) > 0


def test_scaffold_handles_a_width_it_cannot_decompose(tmp_path):
    """A width that does not fit a known layout must be reported, not faked."""
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 37, 12)
    result = scaffold_policy_manifest(artifact, joint_order=JOINTS,
                                      output=str(tmp_path / "policy.yaml"))
    assert result["input_size"] == 37
    assert result["needs_review"]


def test_scaffold_reports_an_unloadable_artifact_clearly(tmp_path):
    from robustness_lab.scaffold import scaffold_policy_manifest
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a torchscript module")
    with pytest.raises(ValueError, match="broken.pt"):
        scaffold_policy_manifest(str(bad), joint_order=JOINTS,
                                 output=str(tmp_path / "policy.yaml"))


def test_undecomposable_width_never_reports_a_path_it_did_not_write(tmp_path):
    """Silently naming a nonexistent file is worse than writing a skeleton
    or refusing loudly. The researcher must be left with something usable."""
    from pathlib import Path
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 37, 12)
    out = str(tmp_path / "policy.yaml")
    result = scaffold_policy_manifest(artifact, joint_order=JOINTS, output=out)
    assert result["input_size"] == 37
    assert result["needs_review"]
    written = Path(out).exists()
    reported = result.get("path") or result.get("output")
    if reported:
        assert Path(reported).exists(), (
            "scaffold reported a manifest path it never wrote")
    assert written, "an undecomposable width must still leave a skeleton to fill in"


def test_undecomposable_skeleton_records_the_probed_widths(tmp_path):
    """Whatever it writes must at least carry the facts it DID establish."""
    import yaml
    from robustness_lab.scaffold import scaffold_policy_manifest
    artifact = _artifact(tmp_path, 37, 12)
    out = str(tmp_path / "policy.yaml")
    scaffold_policy_manifest(artifact, joint_order=JOINTS, output=out)
    payload = yaml.safe_load(open(out, encoding="utf-8"))
    assert payload["action"]["size"] == 12
    assert "37" in yaml.safe_dump(payload), "the probed input width must be recorded"
