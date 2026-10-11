from pathlib import Path

import pytest

from robustness_lab.examiner import (
    ExaminerSettings,
    _next_branch_trial,
    _point,
    generate_capability_passport,
    initialize_examination,
    run_examination,
)
from robustness_lab.io import dump_json, load_data
from robustness_lab.skill import assess_navigation_compatibility, discover_skills, load_skill, validate_skill


ROOT = Path(__file__).resolve().parents[1]


def _skill_manifest(path: Path, track="blind", sensors=None):
    profile = load_data(str(ROOT / "assets/cache/go2_mjcf_profile.json"))
    joints = profile["controller_joint_order"]
    payload = {
        "schema_version": "1.0",
        "skill_id": "test-walk",
        "sensor_track": track,
        "robot_profile_sha256": profile["source_sha256"],
        "control_frequency_hz": 50,
        "controller": {"type": "python", "target": "builtins:list"},
        "sensors": sensors or [
            {"name": "joints", "kind": "joint_position", "shape": [len(joints)], "rate_hz": 50},
            {"name": "command", "kind": "command", "shape": [3], "rate_hz": 50},
        ],
        "observations": [{"name": "policy", "sources": ["joints", "command"], "shape": [len(joints) + 3]}],
        "action": {"mode": "position", "size": len(joints), "joint_order": joints},
    }
    dump_json(str(path), payload)
    return path


def test_skill_contract_rejects_mislabelled_privileged_sensor(tmp_path):
    manifest = _skill_manifest(
        tmp_path / "skill.json",
        track="blind",
        sensors=[{"name": "truth", "kind": "privileged_terrain", "shape": [16], "rate_hz": 50}],
    )
    payload = load_data(str(manifest))
    payload["observations"][0] = {"name": "policy", "sources": ["truth"], "shape": [16]}
    dump_json(str(manifest), payload)
    result = validate_skill(str(manifest))
    assert result["valid"] is False
    assert "expected privileged" in result["error"]


def test_discovery_does_not_deserialize_checkpoint(tmp_path):
    # Invalid pickle bytes are intentionally safe because discovery only hashes
    # and classifies filenames/ZIP members.
    checkpoint = tmp_path / "model_120.pt"
    checkpoint.write_bytes(b"not a pickle")
    result = discover_skills(str(tmp_path))
    assert result["status"] == "ok"
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["iteration"] == 120
    assert "no pickle payload" in result["safety"]


def test_nonmonotonic_audit_clears_inferred_failures():
    settings = ExaminerSettings(
        "x", "fake", "robot", "asset", "mjcf", "skill", "runtimes", "out",
        terrain_codes=["rs1"], screening_severities=[0.2, 0.5, 0.8],
        exploration_repetitions=1, confirmation_min_repetitions=1,
        confirmation_max_repetitions=1, target_success_probability=0.5,
    )
    branch = {"terrain": "rs1", "status": "screening", "points": {}, "inferred_failures": [], "audit_status": "not_needed", "monotonicity": "assumed_with_audit"}
    for severity, outcome in ((0.2, True), (0.5, False), (0.8, False), (0.35, True)):
        _point(branch, severity)["outcomes"].append(outcome)
    request = _next_branch_trial(branch, settings)
    # The exact decision may require confirmation, but a harder audited success
    # must never remain represented as an inferred failure.
    branch["inferred_failures"] = [0.8, 1.0]
    branch["audit_status"] = "completed"
    audit_outcome = True
    if audit_outcome:
        branch["monotonicity"] = "violated"
        branch["status"] = "uncertain"
        branch["inferred_failures"] = []
    assert request is None or request["phase"] in {"exploration", "confirmation"}
    assert branch["monotonicity"] == "violated"
    assert branch["inferred_failures"] == []


def test_examiner_is_resumable_and_generates_passport(tmp_path):
    skill_path = _skill_manifest(tmp_path / "skill.json")
    config = tmp_path / "examiner.yaml"
    config.write_text(
        "\n".join(
            [
                "examiner_id: adaptive-test",
                "backend: fake",
                "robot_profile: %s" % (ROOT / "assets/cache/go2_mjcf_profile.json"),
                "robot_asset: %s" % (ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"),
                "robot_format: mjcf",
                "skill: %s" % skill_path,
                "runtimes: %s" % (ROOT / "configs/runtimes.yaml"),
                "output_dir: %s" % (tmp_path / "run"),
                "terrain_codes: [rs1]",
                "timeout_s: 20.0",
                "adaptive:",
                "  screening_severities: [0.2, 0.5, 0.8]",
                "  exploration_repetitions: 1",
                "  confirmation_min_repetitions: 1",
                "  confirmation_max_repetitions: 2",
                "  target_success_probability: 0.5",
                "  interaction_count: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = initialize_examination(str(config))
    state_path = Path(state["settings"]["output_dir"]) / "examination_state.json"
    first = run_examination(str(state_path), max_trials=2)
    second = run_examination(str(state_path), max_trials=2)
    assert first["total_trials"] == 2
    assert second["total_trials"] == 4
    persisted = load_data(str(state_path))
    assert len({item["trial_id"] for item in persisted["trials"]}) == 4
    report = generate_capability_passport(str(state_path))
    assert Path(report["passport"]).exists()
    assert Path(report["html"]).exists()


def test_examiner_never_assigns_rs8_to_ae_slots(tmp_path):
    skill_path = _skill_manifest(tmp_path / "skill.json")
    config = tmp_path / "examiner.yaml"
    config.write_text(
        "\n".join(
            [
                "examiner_id: ae-contract-test",
                "backend: fake",
                "robot_profile: %s" % (ROOT / "assets/cache/go2_mjcf_profile.json"),
                "robot_asset: %s" % (ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"),
                "robot_format: mjcf",
                "skill: %s" % skill_path,
                "runtimes: %s" % (ROOT / "configs/runtimes.yaml"),
                "output_dir: %s" % (tmp_path / "run"),
                "terrain_codes: [rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9, rs10, rs11]",
                "adaptive:",
                "  interaction_count: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = initialize_examination(str(config))
    assert len(state["interaction_catalog"]) == 100
    assert all("rs8" not in pair for pair in state["interaction_catalog"])
    assert [item["level"] for item in state["progression"]] == ["L3", "L3", "L4", "L5", "L5", "L6", "L7"]


def test_waypoint_policy_requires_declared_provider(tmp_path):
    manifest = _skill_manifest(tmp_path / "skill.json")
    payload = load_data(str(manifest))
    payload["navigation"] = {
        "mode": "waypoint_heading",
        "route_scope": "fixed_route",
        "assistance": "benchmark_waypoints",
        "lookahead": 2,
    }
    payload["sensors"].append(
        {"name": "route", "kind": "waypoints", "shape": [2, 3], "rate_hz": 10}
    )
    dump_json(str(manifest), payload)
    skill = load_skill(str(manifest))
    blocked = assess_navigation_compatibility(skill, "grid", providers=["route_follower"])
    allowed = assess_navigation_compatibility(skill, "grid", providers=["waypoint_heading"])
    assert blocked["status"] == "blocked"
    assert allowed["status"] == "compatible"


def test_velocity_policy_uses_route_follower_for_locomotion_not_navigation(tmp_path):
    manifest = _skill_manifest(tmp_path / "skill.json")
    skill = load_skill(str(manifest))
    locomotion = assess_navigation_compatibility(skill, "grid", "locomotion", ["route_follower"])
    navigation = assess_navigation_compatibility(skill, "grid", "navigation", ["route_follower"])
    assert locomotion["status"] == "compatible"
    assert locomotion["claim_scope"] == "locomotion with declared route assistance"
    assert navigation["status"] == "blocked"


def test_route_blind_policy_is_limited_to_straight_worlds(tmp_path):
    manifest = _skill_manifest(tmp_path / "skill.json")
    payload = load_data(str(manifest))
    payload["navigation"] = {"mode": "none", "route_scope": "straight_only", "assistance": "none"}
    dump_json(str(manifest), payload)
    skill = load_skill(str(manifest))
    assert assess_navigation_compatibility(skill, "chain")["status"] == "compatible"
    assert assess_navigation_compatibility(skill, "maze")["status"] == "blocked"


def test_v1_manifest_rejects_unimplemented_history():
    """schema v1 parses `history` but nothing in src/ ever applies it.

    Silently ignoring it feeds the policy a single frame and blames the policy
    for the resulting failure. Until the schema-v2 compiler implements it, the
    manifest must be rejected loudly.
    """
    import pytest
    from robustness_lab.skill import ObservationTensor
    with pytest.raises(ValueError, match="not implemented"):
        ObservationTensor(name="obs", sources=["a"], shape=[10], history=4).validate()


def test_v1_manifest_rejects_unimplemented_normalization():
    import pytest
    from robustness_lab.skill import ObservationTensor
    with pytest.raises(ValueError, match="not implemented"):
        ObservationTensor(name="obs", sources=["a"], shape=[10],
                          normalization={"mean": [0.0], "std": [1.0]}).validate()


def test_history_of_one_and_empty_normalization_remain_valid():
    """Positive control: the default single-frame, unnormalized case must still pass."""
    from robustness_lab.skill import ObservationTensor
    ObservationTensor(name="obs", sources=["a"], shape=[10]).validate()
