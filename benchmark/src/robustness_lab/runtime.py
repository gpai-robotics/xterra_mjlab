"""Subprocess runtime coordinator and auditable run manifests."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .events import DEFAULT_THRESHOLDS, detect_events
from .canonical_state import AVAILABILITY_CLASSES
from .doctor import expand_runtime_path, run_doctor
from .io import dump_json, load_data, runtime_fingerprint, sha256_data, sha256_file, sha256_tree
from .metrics import compute_episode_metrics
from .models import EpisodeLog, EpisodeSpec, RobotProfile, RunManifest, WorldSpec
from .observation.gates import check_deployability


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(expand_runtime_path(value))
    if path.is_absolute():
        return str(path.resolve())
    local = (config_dir / path).resolve()
    if local.exists() or str(value).startswith("."):
        return str(local)
    return str((project_root() / path).resolve())


def load_episode(path: str, output_override: Optional[str] = None) -> Tuple[EpisodeSpec, Dict[str, Any]]:
    raw = load_data(path)
    config_dir = Path(path).resolve().parent
    resolved = dict(raw)
    for key in ("robot_profile", "robot_asset", "world"):
        resolved[key] = _resolve_path(raw[key], config_dir)
    controller = dict(raw["controller"])
    for key in ("path", "manifest"):
        value = controller.get(key)
        if value:
            controller[key] = _resolve_path(str(value), config_dir)
    resolved["controller"] = controller
    if output_override:
        resolved["output_dir"] = str(Path(output_override).resolve())
    else:
        resolved["output_dir"] = _resolve_path(raw.get("output_dir", "runs/%s" % raw["episode_id"]), config_dir)
    episode = EpisodeSpec.from_dict(resolved)
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(episode.event_thresholds)
    episode.event_thresholds = thresholds
    return episode, raw


def _runtime_command(episode: EpisodeSpec, runtimes_path: str) -> Tuple[list, Dict[str, str]]:
    settings = load_data(runtimes_path).get("runtimes", {}).get(episode.backend, {})
    module_args = ["-m", "robustness_lab.backends.worker", "--episode", str(Path(episode.output_dir) / "resolved_episode.json")]
    if settings.get("launcher"):
        # Isaac Lab's launcher accepts `-p <python-file>`, not `python -m`.
        # Launching worker.py directly would put its package directory on
        # sys.path, where backends/isaaclab.py shadows Isaac Lab's top-level
        # package. A run-local shim keeps that directory out of sys.path[0].
        source_root = str(project_root() / "src")
        worker_launcher = Path(episode.output_dir) / "worker_launcher.py"
        worker_launcher.parent.mkdir(parents=True, exist_ok=True)
        worker_launcher.write_text(
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from robustness_lab.backends.worker import main\n"
            "raise SystemExit(main())\n" % source_root,
            encoding="utf-8",
        )
        command = [expand_runtime_path(settings["launcher"])] + settings.get("worker_args", ["-p"]) + [str(worker_launcher), "--episode", module_args[-1]]
    else:
        executable = expand_runtime_path(settings.get("python", sys.executable))
        command = [executable] + module_args
    environment = dict(os.environ)
    source_root = str(project_root() / "src")
    environment["PYTHONPATH"] = source_root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return command, environment


def _controller_observation_report(
    controller: Dict[str, Any], config_dir: Path
) -> Dict[str, Any]:
    """Describe the declared inputs recorded with a compiled-policy episode."""

    if controller.get("type") != "compiled_policy" or not controller.get("manifest"):
        return {}
    manifest_path = Path(str(controller["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = (config_dir / manifest_path).resolve()
    try:
        # Preflight owns the schema-aware collection of direct and encoder inputs.
        from .preflight import _declared_policy_sources

        sources = _declared_policy_sources(load_data(str(manifest_path)))
        report = check_deployability(sources, AVAILABILITY_CLASSES)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "status": "unverified",
            "sources": [],
            "classes_used": [],
            "sensor_track": "unverified",
            "claim_scope": (
                "unverified information availability; not a deployment claim"
            ),
            "reason": str(error),
            "manifest": str(manifest_path),
        }
    report.update({"sources": sources, "manifest": str(manifest_path)})
    return report


def execute_episode(config_path: str, runtimes_path: Optional[str] = None, output_override: Optional[str] = None) -> RunManifest:
    episode, raw = load_episode(config_path, output_override)
    run_dir = Path(episode.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # A configured run directory represents one current attempt. Remove only
    # known generated artifacts so a retry can never inherit stale metrics or
    # telemetry from an earlier successful worker.
    for name in ("episode_log.json", "metrics.json", "worker_status.json", "telemetry.jsonl", "telemetry.parquet", "resolved_scene.xml", "MUJOCO_LOG.TXT"):
        artifact = run_dir / name
        if artifact.is_file():
            artifact.unlink()
    runtimes_path = runtimes_path or str(project_root() / "configs" / "runtimes.yaml")
    robot = RobotProfile.from_dict(load_data(episode.robot_profile))
    world = WorldSpec.from_dict(load_data(episode.world))
    if world.robot_profile_sha256 != robot.source_sha256:
        raise ValueError("world was scaled for a different RobotProfile; regenerate it for this robot model")
    dump_json(str(run_dir / "source_episode.json"), raw)
    dump_json(str(run_dir / "resolved_episode.json"), episode)
    run_id = "%s-%s" % (episode.episode_id, uuid.uuid4().hex[:8])
    hashes = {
        "code": sha256_tree(str(project_root() / "src")),
        "episode": sha256_data(episode.to_dict()),
        "robot_profile": sha256_file(episode.robot_profile),
        "robot_asset": sha256_file(episode.robot_asset) if Path(episode.robot_asset).is_file() else "missing",
        "world": sha256_file(episode.world),
        "controller": sha256_data(episode.controller),
        "runtimes": sha256_file(runtimes_path),
    }
    for key in ("path", "checkpoint", "model"):
        controller_asset = episode.controller.get(key)
        if controller_asset and Path(str(controller_asset)).is_file():
            hashes["controller_%s" % key] = sha256_file(str(controller_asset))
    manifest = RunManifest(
        run_id=run_id,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="running",
        episode_spec=episode.to_dict(),
        resolved_paths={"config": str(Path(config_path).resolve()), "run_dir": str(run_dir), "runtimes": str(Path(runtimes_path).resolve())},
        hashes=hashes,
        runtime=runtime_fingerprint(),
        comparability={"status": "unassessed", "requires": ["robot_conformance.json", "effective_physics.json"]},
        observation_deployability=_controller_observation_report(
            episode.controller, Path(config_path).resolve().parent
        ),
    )
    dump_json(str(run_dir / "manifest.json"), manifest)
    runtime_settings = load_data(runtimes_path).get("runtimes", {}).get(episode.backend, {})
    if runtime_settings.get("requires_gpu"):
        diagnostics = run_doctor(runtimes_path)
        if diagnostics.get("gpu", {}).get("status") != "available":
            reason = diagnostics.get("gpu", {}).get("reason", "GPU unavailable")
            log = EpisodeLog(episode.episode_id, episode.backend, "skipped", "backend_unavailable", diagnostics={"reason": reason, "preflight": "gpu"})
            dump_json(str(run_dir / "episode_log.json"), log)
            dump_json(str(run_dir / "worker_status.json"), {"status": "skipped", "terminal_reason": "backend_unavailable", "exit_code": 0})
            manifest.status = "skipped"
            manifest.runtime["preflight"] = diagnostics
            manifest.artifacts["episode_log"] = str(run_dir / "episode_log.json")
            dump_json(str(run_dir / "manifest.json"), manifest)
            return manifest
    command, environment = _runtime_command(episode, runtimes_path)
    manifest.runtime["worker_command"] = command
    manifest.runtime["worker_cwd"] = str(run_dir)
    try:
        # MuJoCo's default log path is relative to cwd; keep that mutable log
        # inside this episode's run directory when trials execute concurrently.
        completed = subprocess.run(command, env=environment, cwd=str(run_dir), capture_output=True, text=True, timeout=episode.timeout_s + 120.0)
        manifest.runtime["worker_returncode"] = completed.returncode
        manifest.runtime["worker_stdout"] = completed.stdout[-10000:]
        manifest.runtime["worker_stderr"] = completed.stderr[-10000:]
    except subprocess.TimeoutExpired as error:
        manifest.status = "error"
        manifest.runtime["error"] = "worker timeout: %s" % error
        timeout_log = EpisodeLog(
            episode.episode_id,
            episode.backend,
            "error",
            "simulator_error",
            diagnostics={"error": str(error), "phase": "worker_timeout"},
        )
        dump_json(str(run_dir / "episode_log.json"), timeout_log)
        dump_json(
            str(run_dir / "worker_status.json"),
            {"status": "error", "terminal_reason": "simulator_error", "exit_code": 1},
        )
        manifest.artifacts["episode_log"] = str(run_dir / "episode_log.json")
        dump_json(str(run_dir / "manifest.json"), manifest)
        return manifest
    log_path = run_dir / "episode_log.json"
    if not log_path.exists():
        manifest.runtime["error"] = "worker produced no episode_log.json"
        missing_log = EpisodeLog(
            episode.episode_id,
            episode.backend,
            "error",
            "simulator_error",
            diagnostics={
                "error": "worker terminated without producing episode_log.json",
                "phase": "worker_process",
                "worker_returncode": completed.returncode,
                "worker_stdout": completed.stdout[-10000:],
                "worker_stderr": completed.stderr[-10000:],
            },
        )
        dump_json(str(log_path), missing_log)
        dump_json(
            str(run_dir / "worker_status.json"),
            {"status": "error", "terminal_reason": "simulator_error", "exit_code": completed.returncode or 1},
        )
    log = EpisodeLog.from_dict(load_data(str(log_path)))
    manifest.runtime["backend_diagnostics"] = log.diagnostics
    manifest.artifacts["episode_log"] = str(log_path)
    if log.samples:
        detect_events(log, world, robot, episode.event_thresholds)
        metrics = compute_episode_metrics(log, world, robot)
        dump_json(str(run_dir / "episode_log.json"), log)
        dump_json(str(run_dir / "metrics.json"), {"episode_id": episode.episode_id, "metrics": [metric.__dict__ for metric in metrics]})
        manifest.artifacts["metrics"] = str(run_dir / "metrics.json")
    manifest.status = log.status
    manifest.artifacts.update(log.diagnostics.get("telemetry_artifacts", {}))
    dump_json(str(run_dir / "manifest.json"), manifest)
    return manifest


def recompute_metrics(run_dir: str) -> Dict[str, Any]:
    root = Path(run_dir).resolve()
    episode = EpisodeSpec.from_dict(load_data(str(root / "resolved_episode.json")))
    robot = RobotProfile.from_dict(load_data(episode.robot_profile))
    world = WorldSpec.from_dict(load_data(episode.world))
    log = EpisodeLog.from_dict(load_data(str(root / "episode_log.json")))
    detect_events(log, world, robot, episode.event_thresholds)
    metrics = compute_episode_metrics(log, world, robot)
    result = {"episode_id": episode.episode_id, "metrics": [metric.__dict__ for metric in metrics]}
    dump_json(str(root / "episode_log.json"), log)
    dump_json(str(root / "metrics.json"), result)
    return result
