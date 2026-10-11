"""Preflight must MEASURE the G2/G3 evidence, not be handed it.

`_observation_verification_report` read `observation_gate_evidence` straight out
of the examiner config. Nothing in the codebase produced it, so:

  * every compiled policy was permanently blocked -- `launch` refuses while
    observation_verification is blocked, so no submission could ever run; and
  * the only way through was to paste the numbers into your own config, which
    is the "declared but never verified" pattern this project exists to remove.
    A submitter pasting `[0, 0, -1]` would pass the gravity gate whatever the
    simulator actually produced.

The fix: reset the robot at its default pose on flat ground, build ONE
observation through the submitter's own compiled plan, and feed that to the
gates. Evidence that came from a simulator is labelled `measured`; evidence
that came from a config is labelled `declared` and never silently mistaken for
the other.

The reset is KINEMATIC -- set the pose, read it back, step no physics -- so the
expected values are closed form and cannot be approximated by a stub:

    projected_gravity_b  == [0.0, 0.0, -1.0]   exactly, on a level base
    joint_position_error == 0.0                at the default pose
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
ASSET = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
SUBMISSION = ROOT / "submissions/go2_blind/examiner.yaml"


def _profile():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _require_mujoco():
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")


# ---------------------------------------------------------------------------
# The backend capability
# ---------------------------------------------------------------------------

def test_mujoco_resets_level_and_at_the_default_pose():
    """A kinematic reset is closed form: level base, default joint angles."""
    _require_mujoco()
    from robustness_lab.backends.mujoco import MujocoBackend

    robot = _profile()
    state = MujocoBackend().reset_canonical_state(robot, str(ASSET), "mjcf")
    assert state is not None

    assert list(state.projected_gravity_b) == pytest.approx(
        [0.0, 0.0, -1.0], abs=1e-6), (
        "a level base must project gravity straight down; got %r"
        % (state.projected_gravity_b,))

    expected = [robot.default_joint_positions[name]
                for name in robot.controller_joint_order]
    assert list(state.joint_position) == pytest.approx(expected, abs=1e-6)
    # Kinematic reset: nothing has moved yet.
    assert list(state.joint_velocity) == pytest.approx([0.0] * len(expected), abs=1e-9)


def test_a_backend_that_cannot_reset_returns_none_rather_than_guessing():
    from robustness_lab.backends.fake import FakeBackend

    assert FakeBackend().reset_canonical_state(_profile(), "", "mjcf") is None


# ---------------------------------------------------------------------------
# Evidence measurement
# ---------------------------------------------------------------------------

def _measured():
    _require_mujoco()
    from robustness_lab.observation.evidence import measure_observation_evidence
    from robustness_lab.policy_manifest import load_policy_manifest

    manifest_path = ROOT / "submissions/go2_blind/manifest.yaml"
    if not manifest_path.is_file():
        pytest.skip("go2_blind submission not present")
    manifest = load_policy_manifest(str(manifest_path), check_files=False)
    return measure_observation_evidence(
        "mujoco", manifest, _profile(), str(ASSET), "mjcf")


def test_measured_evidence_carries_both_gate_inputs():
    evidence = _measured()
    assert evidence is not None
    assert evidence["provenance"] == "measured"
    assert "compiled_observation" in evidence
    assert "known_pose_terms" in evidence


def test_the_measured_observation_is_the_submitters_own_compiled_width():
    """Built through the manifest's plan, so it is 45 + 4500 for this bundle."""
    evidence = _measured()
    observation = evidence["compiled_observation"]
    sizes = sorted(len(values) for values in observation.values()) \
        if isinstance(observation, dict) else [len(observation)]
    assert sizes == [45, 4500]


def test_measured_known_pose_terms_hit_the_closed_form_values():
    from robustness_lab.observation.gates import check_known_pose

    terms = _measured()["known_pose_terms"]
    assert list(terms["projected_gravity_b"]) == pytest.approx(
        [0.0, 0.0, -1.0], abs=1e-6)
    assert check_known_pose(terms)["status"] == "ready"


def test_an_unavailable_backend_measures_nothing_rather_than_inventing_it():
    from robustness_lab.observation.evidence import measure_observation_evidence
    from robustness_lab.policy_manifest import load_policy_manifest

    manifest_path = ROOT / "submissions/go2_blind/manifest.yaml"
    if not manifest_path.is_file():
        pytest.skip("go2_blind submission not present")
    manifest = load_policy_manifest(str(manifest_path), check_files=False)
    assert measure_observation_evidence(
        "fake", manifest, _profile(), str(ASSET), "mjcf") is None


# ---------------------------------------------------------------------------
# Preflight wiring -- the acceptance criteria
# ---------------------------------------------------------------------------

def _report():
    _require_mujoco()
    if not SUBMISSION.is_file():
        pytest.skip("go2_blind submission not present")
    from robustness_lab.preflight import preflight_examination
    return preflight_examination(str(SUBMISSION), probe_runtime=False)


def test_preflight_measures_the_evidence_itself():
    checks = _report()["observation_verification"]["checks"]
    assert checks["finiteness"]["status"] == "ready", checks["finiteness"]
    assert checks["known_pose_replay"]["status"] == "ready", (
        checks["known_pose_replay"])


def test_measured_evidence_is_labelled_as_measured():
    """A report must never let declared numbers read as simulator numbers."""
    checks = _report()["observation_verification"]["checks"]
    assert checks["finiteness"]["provenance"] == "measured"
    assert checks["known_pose_replay"]["provenance"] == "measured"


def test_observation_verification_passes_for_the_reference_submission():
    report = _report()
    assert report["observation_verification"]["status"] == "ready", (
        report["observation_verification"])


def test_declared_evidence_is_labelled_declared_and_never_measured(tmp_path):
    """Config-supplied evidence still works, but is reported for what it is."""
    import yaml
    from robustness_lab.io import load_data
    from robustness_lab.preflight import preflight_examination

    if not SUBMISSION.is_file():
        pytest.skip("go2_blind submission not present")
    raw = dict(load_data(str(SUBMISSION)))
    # A backend that cannot measure, so the declared path is the only one left.
    raw["backend"] = "fake"
    raw["observation_gate_evidence"] = {
        "compiled_observation": [0.0, 1.0, 2.0],
        "known_pose_terms": {"projected_gravity_b": [0.0, 0.0, -1.0]},
    }
    for key in ("robot_profile", "robot_asset", "skill", "runtimes", "output_dir"):
        value = raw.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            raw[key] = str((SUBMISSION.parent / value).resolve())
    config = tmp_path / "examiner.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    checks = preflight_examination(
        str(config), probe_runtime=False)["observation_verification"]["checks"]
    assert checks["finiteness"]["provenance"] == "declared"
    assert checks["known_pose_replay"]["provenance"] == "declared"


def test_no_evidence_from_any_source_still_blocks(tmp_path):
    """The gate must never pass by default when nothing verified anything."""
    import yaml
    from robustness_lab.io import load_data
    from robustness_lab.preflight import preflight_examination

    if not SUBMISSION.is_file():
        pytest.skip("go2_blind submission not present")
    raw = dict(load_data(str(SUBMISSION)))
    raw["backend"] = "fake"
    raw.pop("observation_gate_evidence", None)
    for key in ("robot_profile", "robot_asset", "skill", "runtimes", "output_dir"):
        value = raw.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            raw[key] = str((SUBMISSION.parent / value).resolve())
    config = tmp_path / "examiner.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    verification = preflight_examination(
        str(config), probe_runtime=False)["observation_verification"]
    assert verification["status"] == "blocked"
    assert verification["outcome_class"] == "adapter_unverified"
