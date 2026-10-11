import json
import sys
from pathlib import Path

import yaml

from robustness_lab.io import dump_json
from robustness_lab.aggregate import aggregate_runs
from robustness_lab.report import generate_aggregate_report, generate_report
from robustness_lab.robot import inspect_urdf
from robustness_lab.runtime import execute_episode
from robustness_lab.worlds import generate_terrain


FIXTURES = Path(__file__).parent / "fixtures"


def test_subprocess_episode_and_report(tmp_path):
    robot_asset = FIXTURES / "mini_quadruped.urdf"
    profile = inspect_urdf(str(robot_asset))
    world = generate_terrain("rs1", profile, 0.2, 11)
    profile_path, world_path = tmp_path / "robot.json", tmp_path / "world.json"
    dump_json(str(profile_path), profile)
    dump_json(str(world_path), world)
    run_dir = tmp_path / "run"
    episode = {
        "episode_id": "e2e",
        "backend": "fake",
        "robot_profile": str(profile_path),
        "robot_asset": str(robot_asset),
        "robot_format": "urdf",
        "world": str(world_path),
        "controller": {"type": "pd_hold", "command_velocity_mps": 1.0},
        "output_dir": str(run_dir),
        "timeout_s": 15.0,
    }
    episode_path = tmp_path / "episode.yaml"
    episode_path.write_text(yaml.safe_dump(episode), encoding="utf-8")
    runtimes_path = tmp_path / "runtimes.yaml"
    runtimes_path.write_text(yaml.safe_dump({"runtimes": {"fake": {"python": sys.executable}}}), encoding="utf-8")
    manifest = execute_episode(str(episode_path), str(runtimes_path))
    assert manifest.status == "completed"
    assert (run_dir / "metrics.json").exists()
    report = tmp_path / "report.html"
    generate_report(str(run_dir), str(report))
    assert "Robustness Lab" in report.read_text(encoding="utf-8")
    stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["hashes"]["robot_asset"] == profile.source_sha256
    assert len(stored["hashes"]["code"]) == 64

    skipped_dir = tmp_path / "skipped"
    skipped_dir.mkdir()
    skipped_manifest = dict(stored)
    skipped_manifest["status"] = "skipped"
    skipped_manifest["episode_spec"] = dict(stored["episode_spec"], backend="isaaclab")
    dump_json(str(skipped_dir / "manifest.json"), skipped_manifest)
    dump_json(str(skipped_dir / "resolved_episode.json"), skipped_manifest["episode_spec"])
    dump_json(
        str(skipped_dir / "episode_log.json"),
        {
            "episode_id": "skipped",
            "backend": "isaaclab",
            "status": "skipped",
            "terminal_reason": "backend_unavailable",
            "diagnostics": {"reason": "GPU unavailable"},
        },
    )
    skipped_report = tmp_path / "skipped.html"
    generate_report(str(skipped_dir), str(skipped_report))
    assert "GPU unavailable" in skipped_report.read_text(encoding="utf-8")
    aggregate = aggregate_runs([str(run_dir), str(skipped_dir)])
    assert aggregate["status_counts"]["isaaclab"]["skipped"] == 1
    aggregate_path = tmp_path / "aggregate.json"
    dump_json(str(aggregate_path), aggregate)
    dashboard = tmp_path / "dashboard.html"
    generate_aggregate_report(str(aggregate_path), str(dashboard))
    assert "Capability Dashboard" in dashboard.read_text(encoding="utf-8")


def test_runtime_without_python_key_falls_back_to_sys_executable(tmp_path):
    import sys, yaml
    from robustness_lab.doctor import run_doctor
    config = tmp_path / "runtimes.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "runtimes": {"coordinator": {"imports": [], "required": True, "requires_gpu": False}},
    }))
    report = run_doctor(str(config))
    assert report["runtimes"]["coordinator"]["status"] == "available"
    assert report["runtimes"]["coordinator"]["executable"] == sys.executable


def test_runtime_expands_environment_variables(tmp_path, monkeypatch):
    import sys, yaml
    from robustness_lab.doctor import run_doctor
    monkeypatch.setenv("RL_TEST_PY", sys.executable)
    config = tmp_path / "runtimes.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "runtimes": {"coordinator": {"python": "${RL_TEST_PY}", "imports": [], "required": True}},
    }))
    assert run_doctor(str(config))["runtimes"]["coordinator"]["status"] == "available"


def test_required_runtime_with_missing_import_is_not_ready(tmp_path):
    """A runtime whose declared imports do not resolve must not be reported ready.

    doctor previously set status from whether the *interpreter* ran, ignoring the
    imports probe entirely, so a required backend whose SDK was absent still
    produced ready=True. preflight consumes this for its simulator_runtime gate.
    """
    import sys, yaml
    from robustness_lab.doctor import run_doctor
    config = tmp_path / "runtimes.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "runtimes": {"coordinator": {
            "python": sys.executable,
            "imports": ["a_module_that_does_not_exist_xyz"],
            "required": True,
            "requires_gpu": False,
        }},
    }))
    report = run_doctor(str(config))
    entry = report["runtimes"]["coordinator"]
    assert entry["imports"]["a_module_that_does_not_exist_xyz"] is False
    assert entry["status"] != "available"
    assert report["ready"] is False


def test_runtime_with_satisfied_imports_is_still_ready(tmp_path):
    """Positive control: the fix must not simply mark everything unavailable."""
    import sys, yaml
    from robustness_lab.doctor import run_doctor
    config = tmp_path / "runtimes.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "runtimes": {"coordinator": {
            "python": sys.executable,
            "imports": ["json"],
            "required": True,
            "requires_gpu": False,
        }},
    }))
    report = run_doctor(str(config))
    assert report["runtimes"]["coordinator"]["status"] == "available"
    assert report["ready"] is True
