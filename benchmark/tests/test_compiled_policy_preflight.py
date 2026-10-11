"""A compiled_policy skill must reach the schema-v2 observation gates.

`_observation_verification_report` reads `policy_manifest` from the EXAMINER
config. A submission that declares its manifest on the skill -- the natural
place, since the manifest is the policy's, not the examiner's -- silently fell
back to `legacy_skill_contract`, so G1/G2/G3 never ran and the manifest was
never verified against the artifact.

Requiring it in both files is not the fix: two copies of one path is a mismatch
waiting to happen, and nothing would catch a disagreement.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "submissions/go2_blind/examiner.yaml"


def _report():
    if not CONFIG.is_file():
        pytest.skip("go2_blind submission not present")
    if not (ROOT / "assets/cache/go2_mjcf_profile.json").is_file():
        pytest.skip("go2 assets not cached")
    from robustness_lab.preflight import preflight_examination
    return preflight_examination(str(CONFIG), probe_runtime=False)


def test_compiled_policy_skill_reaches_the_v2_observation_gates():
    report = _report()
    verification = report["observation_verification"]
    assert verification.get("mode") != "legacy_skill_contract", (
        "the schema-v2 manifest was never verified: %s" % verification)


def test_the_manifest_is_resolved_from_the_skill_not_the_examiner():
    report = _report()
    verification = report["observation_verification"]
    assert str(verification.get("manifest", "")).endswith("manifest.yaml")


def test_declared_observation_width_is_checked_against_the_artifact():
    """45 + 4500 must be what the ONNX file actually accepts."""
    report = _report()
    verification = report["observation_verification"]
    assert verification.get("status") in {"ready", "blocked"}
    if verification.get("declared_size") is not None:
        assert verification["declared_size"] == 4545
