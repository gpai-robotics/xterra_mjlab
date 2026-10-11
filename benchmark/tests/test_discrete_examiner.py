import json
import subprocess
from pathlib import Path

import pytest

from robustness_lab.discrete_examiner import (
    _materialize_pairs,
    _settle,
    initialize_discrete_examination,
    report_discrete,
    run_discrete_examination,
)
from robustness_lab.evidence import build_dashboard_data
from robustness_lab.io import dump_json, load_data
from robustness_lab.worlds import DISCRETE_CHALLENGES, apply_challenge_friction, generate_ae, generate_ae_pairs, generate_terrain
from robustness_lab.models import RobotProfile


ROOT = Path(__file__).resolve().parents[1]


def _fixture_config(tmp_path: Path, terrain_codes=None) -> Path:
    profile_path = ROOT / "assets/cache/go2_mjcf_profile.json"
    profile = load_data(str(profile_path))
    skill = tmp_path / "skill.json"
    dump_json(
        str(skill),
        {
            "schema_version": "1.0",
            "skill_id": "discrete-fake-walk",
            "sensor_track": "blind",
            "control_frequency_hz": 50,
            "controller": {"type": "python", "target": "builtins:list", "command_velocity_mps": 0.8},
            "sensors": [{"name": "command", "kind": "command", "shape": [3], "rate_hz": 50}],
            "observations": [{"name": "policy", "sources": ["command"], "shape": [3]}],
            "action": {"mode": "position", "size": len(profile["controller_joint_order"]), "joint_order": profile["controller_joint_order"]},
        },
    )
    config = tmp_path / "examiner.json"
    dump_json(
        str(config),
        {
            "schema_version": "2.0",
            "examiner_id": "discrete-test",
            "scheduler": "discrete_pruned_v2",
            "backend": "fake",
            "robot_profile": str(profile_path),
            "robot_asset": str(ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"),
            "robot_format": "mjcf",
            "skill": str(skill),
            "runtimes": str(ROOT / "configs/runtimes.yaml"),
            "output_dir": str(tmp_path / "run"),
            "terrain_codes": terrain_codes or list(DISCRETE_CHALLENGES),
            "seed": 101,
            "timeout_s": 8,
            "discrete": {
                "screening_repetitions": 3,
                "confirmation_repetitions": 10,
                "confirmation_successes": 7,
                "infrastructure_retries": 2,
                "fixed_pair_static_friction": 0.5,
                "fixed_pair_dynamic_friction": 0.4,
            },
        },
    )
    return config


def test_discrete_catalogue_has_99_primitives_and_breadth_first_order(tmp_path):
    state = initialize_discrete_examination(str(_fixture_config(tmp_path)))
    primitives = [item for item in state["conditions"].values() if item["kind"] == "primitive"]
    assert len(primitives) == 99
    assert state["catalogue"]["potential_ordered_pairs"] == 110
    assert state["catalogue"]["potential_pair_conditions"] == 330
    first = state["condition_order"][1:12]
    assert first == ["L1-rs%d-g050-f075" % index for index in range(1, 12)]
    assert state["condition_order"][12:23] == ["L1-rs%d-g050-f050" % index for index in range(1, 12)]
    assert Path(state["settings"]["output_dir"], "dashboard/index.html").is_file()


def test_custom_geometry_and_friction_schedule_is_versioned_and_counted(tmp_path):
    config = _fixture_config(tmp_path, ["rs1", "rs2"])
    raw = load_data(str(config))
    raw["discrete"].update(
        {
            "protocol_variant_id": "two-by-two-study",
            "geometry_levels": [
                {"label": "central", "value": 0.4},
                {"label": "boundary", "value": 0.8},
            ],
            "friction_levels": [
                {"label": "traction", "static": 0.7, "dynamic": 0.55},
                {"label": "qualification", "static": 0.45, "dynamic": 0.35},
            ],
            "qualification_friction_label": "qualification",
            "fixed_pair_static_friction": 0.45,
            "fixed_pair_dynamic_friction": 0.35,
        }
    )
    dump_json(str(config), raw)
    state = initialize_discrete_examination(str(config))
    assert state["scheduler"]["protocol_variant_id"] == "two-by-two-study"
    assert state["catalogue"]["primitive_conditions"] == 8
    assert state["catalogue"]["potential_pair_conditions"] == 4
    for item in state["conditions"].values():
        if item["kind"] == "primitive":
            item["status"] = "passed" if item["friction_label"] == "qualification" else "failed"
    _materialize_pairs(state)
    pairs = [item for item in state["conditions"].values() if item["kind"] == "pair"]
    assert len(pairs) == 4
    assert all(item["status"] == "pending" for item in pairs)
    assert all(item["static_friction"] == 0.45 and item["dynamic_friction"] == 0.35 for item in pairs)


def test_reference_identifier_rejects_changed_pinned_values(tmp_path):
    config = _fixture_config(tmp_path, ["rs1", "rs2"])
    raw = load_data(str(config))
    raw["discrete"]["geometry_levels"] = [
        {"label": "medium", "value": 0.40},
        {"label": "easy", "value": 0.20},
        {"label": "hard", "value": 0.80},
    ]
    dump_json(str(config), raw)
    with pytest.raises(ValueError, match="distinct protocol_variant_id"):
        initialize_discrete_examination(str(config))


def test_rs8_is_a_pair_member_and_self_pairs_are_excluded(tmp_path):
    pairs = generate_ae_pairs(DISCRETE_CHALLENGES, include_self=False)
    assert len(pairs) == 110
    assert ("rs8", "rs1") in pairs
    assert all(left != right for left, right in pairs)
    profile = RobotProfile.from_dict(load_data(str(ROOT / "assets/cache/go2_mjcf_profile.json")))
    world = generate_ae(profile, "rs8", "rs1", 0.5, 9, challenge_codes=DISCRETE_CHALLENGES)
    apply_challenge_friction(world, 0.5, 0.4)
    assert world.metadata["challenge_codes"] == list(DISCRETE_CHALLENGES)
    assert world.materials["rough_low"].static_friction == 0.5
    assert world.materials["nominal"].static_friction == 0.9


def test_three_survivors_create_six_directed_pairs_at_each_geometry(tmp_path):
    state = initialize_discrete_examination(str(_fixture_config(tmp_path)))
    state["conditions"]["L0-flat"]["status"] = "passed"
    survivors = {"rs2", "rs3", "rs4"}
    for item in state["conditions"].values():
        if item["kind"] == "primitive":
            item["status"] = "passed" if item["terrain"] in survivors and item["static_friction"] == 0.5 else "failed"
    _materialize_pairs(state)
    pairs = [item for item in state["conditions"].values() if item["kind"] == "pair"]
    pending = [item for item in pairs if item["status"] == "pending"]
    assert len(pairs) == 330
    assert len(pending) == 18
    assert {tuple(item["pair"]) for item in pending} == {("rs2", "rs3"), ("rs3", "rs2"), ("rs2", "rs4"), ("rs4", "rs2"), ("rs3", "rs4"), ("rs4", "rs3")}
    assert all(item["static_friction"] == 0.5 and item["dynamic_friction"] == 0.4 for item in pending)


def test_repetition_rule_and_infrastructure_separation():
    condition = {"id": "x", "kind": "pair", "critical": True, "attempts": ["a", "b", "c"], "status": "running", "target_repetitions": 3}
    state = {"settings": {"screening_repetitions": 3, "confirmation_repetitions": 10, "confirmation_successes": 7, "infrastructure_retries": 2}, "trials": [
        {"trial_id": "a", "valid_behavior": True, "success": True},
        {"trial_id": "b", "valid_behavior": True, "success": True},
        {"trial_id": "c", "valid_behavior": True, "success": True},
    ]}
    _settle(condition, state)
    assert condition["status"] == "confirming"
    assert condition["target_repetitions"] == 10
    condition = {"id": "y", "kind": "primitive", "critical": False, "attempts": ["e"], "status": "running", "target_repetitions": 3}
    state["trials"] = [{"trial_id": "e", "valid_behavior": False, "success": None, "status": "error", "terminal_reason": "simulator_error"}]
    _settle(condition, state)
    assert condition["status"] == "running"


def test_fake_qualification_builds_snapshots_video_telemetry_and_dashboard(tmp_path):
    state = initialize_discrete_examination(str(_fixture_config(tmp_path, ["rs1"])))
    state_path = Path(state["settings"]["output_dir"]) / "examination_state.json"
    result = run_discrete_examination(str(state_path), max_trials=3)
    assert result["executed"] == 3
    persisted = load_data(str(state_path))
    qualification = persisted["conditions"]["L0-flat"]
    assert qualification["status"] == "passed"
    representative = next(item for item in persisted["trials"] if item["trial_id"] == qualification["representative_pass"])
    assert len(representative["media"]["snapshots"]) == 3
    video = Path(persisted["settings"]["output_dir"]) / representative["media"]["video"]["path"]
    assert video.is_file() and video.stat().st_size > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(video)], capture_output=True, text=True, check=True)
    assert any(item["codec_type"] == "video" for item in json.loads(probe.stdout)["streams"])
    assert Path(representative["run_dir"], "telemetry.jsonl").is_file()
    dashboard = Path(persisted["settings"]["output_dir"], "dashboard/index.html")
    assert dashboard.is_file() and "Condition explorer" in dashboard.read_text(encoding="utf-8")


def test_dashboard_does_not_count_errors_as_policy_failures(tmp_path):
    root = tmp_path / "run"
    state = {
        "examiner_id": "x", "status": "running", "stage": "L1", "scheduler": {"selection_policy": "discrete_pruned_v2"},
        "skill": {"skill_id": "p"}, "settings": {"backend": "fake"},
        "conditions": {"c": {"id": "c", "level": "L1", "kind": "primitive", "status": "blocked", "attempts": ["t"]}},
        "trials": [{"trial_id": "t", "condition_id": "c", "valid_behavior": False, "success": None, "status": "error", "terminal_reason": "simulator_error", "diagnosis": {"classification": "infrastructure"}, "metrics": {}, "media": {}}],
    }
    data = build_dashboard_data(state, root)
    assert data["trial_counts"]["behavioral_failures"] == 0
    assert data["trial_counts"]["infrastructure"] == 1
    assert data["conditions"][0]["failures"] == 0


def test_dashboard_recovers_route_distance_from_native_checkpoints(tmp_path):
    profile = RobotProfile.from_dict(load_data(str(ROOT / "assets/cache/go2_mjcf_profile.json")))
    world = generate_terrain("rs1", profile, 0.5, 17)
    world_path = tmp_path / "world.json"
    dump_json(str(world_path), world)
    state = {
        "examiner_id": "progress", "status": "completed", "stage": "L1", "scheduler": {},
        "skill": {"skill_id": "p"}, "settings": {"backend": "foresight_native"},
        "conditions": {"c": {"id": "c", "level": "L1", "kind": "primitive", "terrain": "rs1", "status": "failed", "attempts": ["t"]}},
        "trials": [{
            "trial_id": "t", "condition_id": "c", "valid_behavior": True, "success": False,
            "status": "completed", "terminal_reason": "route_incomplete", "world": str(world_path),
            "metrics": {"progress_fraction": 0.25}, "native_summary": {"max_goal_index": 3, "num_goals": 7, "steps": 1000},
            "diagnosis": {}, "media": {},
        }],
    }
    data = build_dashboard_data(state, tmp_path)
    trial = data["trials"][0]
    assert trial["metrics"]["progress_fraction"] == pytest.approx(0.5)
    assert trial["metrics"]["estimated_route_distance_m"] == pytest.approx(trial["metrics"]["route_length_m"] * 0.5)
    assert trial["performance_evidence"]["distance_is_estimated"] is True


def test_resume_is_idempotent_and_report_exports_linked_evidence(tmp_path):
    state = initialize_discrete_examination(str(_fixture_config(tmp_path, ["rs1"])))
    state_path = Path(state["settings"]["output_dir"]) / "examination_state.json"
    run_discrete_examination(str(state_path), max_trials=1)
    run_discrete_examination(str(state_path), max_trials=2)
    persisted = load_data(str(state_path))
    trial_ids = [item["trial_id"] for item in persisted["trials"]]
    assert len(trial_ids) == len(set(trial_ids)) == 3

    before = state_path.read_bytes()
    export = tmp_path / "portable-report"
    result = report_discrete(str(state_path), str(export))
    assert state_path.read_bytes() == before
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "href='./${esc(" in html
    assert Path(result["html"]).is_file()
    assert (export / "evidence").is_dir()
    assert (export / "examination_state.json").is_file()
    dashboard_data = load_data(str(export / "dashboard_data.json"))
    media_path = dashboard_data["pass_evidence"][0]["media"]["video"]["path"]
    assert (export / media_path).is_file()


def test_behavioral_failures_keep_snapshots_videos_and_dashboard_diagnostics(tmp_path):
    config = _fixture_config(tmp_path, ["rs1"])
    raw = load_data(str(config))
    raw["timeout_s"] = 0.10
    skill_path = Path(raw["skill"])
    skill = load_data(str(skill_path))
    skill["controller"]["command_velocity_mps"] = 0.0
    dump_json(str(skill_path), skill)
    dump_json(str(config), raw)

    state = initialize_discrete_examination(str(config))
    state_path = Path(state["settings"]["output_dir"]) / "examination_state.json"
    result = run_discrete_examination(str(state_path), max_trials=3)
    assert result["status"] == "stopped"
    persisted = load_data(str(state_path))
    assert len(persisted["trials"]) == 3
    for trial in persisted["trials"]:
        assert trial["valid_behavior"] and trial["success"] is False
        assert len(trial["media"]["snapshots"]) == 3
        assert Path(persisted["settings"]["output_dir"], trial["media"]["video"]["path"]).is_file()
        assert trial["diagnosis"]["failure_category"] == "stuck/progress"
        assert "telemetry_window" in trial["diagnosis"]
    dashboard_data = load_data(str(Path(persisted["settings"]["output_dir"], "dashboard/dashboard_data.json")))
    assert dashboard_data["trial_counts"]["behavioral_failures"] == 3
    assert dashboard_data["trial_counts"]["infrastructure"] == 0
    assert dashboard_data["failure_clusters"][0]["count"] == 3
    assert all(item["media"]["video"] for item in dashboard_data["failures"])
