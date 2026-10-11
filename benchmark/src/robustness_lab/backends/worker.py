"""Isolated simulator worker entry point."""

from __future__ import annotations

import argparse
from importlib import metadata
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from robustness_lab.backends.base import BackendUnavailable
from robustness_lab.io import dump_json, load_data, sha256_data, write_telemetry
from robustness_lab.models import EpisodeLog, EpisodeSpec, RobotProfile, WorldSpec


def _backend(name: str):
    if name == "fake":
        from robustness_lab.backends.fake import FakeBackend

        return FakeBackend()
    if name == "mujoco":
        from robustness_lab.backends.mujoco import MujocoBackend

        return MujocoBackend()
    if name == "isaacgym":
        from robustness_lab.backends.isaacgym import IsaacGymBackend

        return IsaacGymBackend()
    if name == "isaaclab":
        from robustness_lab.backends.isaaclab import IsaacLabBackend

        return IsaacLabBackend()
    raise ValueError("unknown backend: %s" % name)


def _write_worker_result(output_dir: Path, log: EpisodeLog, exit_code: int) -> None:
    dump_json(str(output_dir / "episode_log.json"), log)
    dump_json(
        str(output_dir / "worker_status.json"),
        {
            "status": log.status,
            "terminal_reason": log.terminal_reason,
            "exit_code": exit_code,
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    arguments = parser.parse_args(argv)
    episode = EpisodeSpec.from_dict(load_data(arguments.episode))
    output_dir = Path(episode.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    worker_started = time.perf_counter()
    # Native simulator startup and shutdown can terminate the interpreter
    # without returning to Python. Leave a valid terminal record before crossing
    # that boundary, then replace it with the real result on a normal return.
    provisional_log = EpisodeLog(
        episode.episode_id,
        episode.backend,
        "error",
        "simulator_error",
        diagnostics={
            "error": "simulator process terminated before worker finalization",
            "phase": "backend_run",
            "provisional": True,
        },
    )
    _write_worker_result(output_dir, provisional_log, 1)
    try:
        robot = RobotProfile.from_dict(load_data(episode.robot_profile))
        world = WorldSpec.from_dict(load_data(episode.world))
        log = _backend(episode.backend).run(episode, robot, world)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        distributions = sorted(
            (distribution.metadata.get("Name", "unknown"), distribution.version)
            for distribution in metadata.distributions()
        )
        log.diagnostics.update(
            {
                "process_user_cpu_s": float(usage.ru_utime - usage_started.ru_utime),
                "process_system_cpu_s": float(usage.ru_stime - usage_started.ru_stime),
                "process_peak_rss_mib": float(usage.ru_maxrss) / 1024.0,
                "worker_wall_time_s": time.perf_counter() - worker_started,
                "environment": {
                    "python": sys.version,
                    "executable": sys.executable,
                    "platform": platform.platform(),
                    "package_set_sha256": sha256_data(distributions),
                    "package_count": len(distributions),
                },
            }
        )
        exit_code = 0
    except BackendUnavailable as error:
        log = EpisodeLog(episode.episode_id, episode.backend, "skipped", "backend_unavailable", diagnostics={"reason": str(error)})
        exit_code = 0
    except BaseException as error:  # simulator libraries may raise SystemExit
        log = EpisodeLog(
            episode.episode_id,
            episode.backend,
            "error",
            "simulator_error",
            diagnostics={
                "error": str(error) or type(error).__name__,
                "exception_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        exit_code = 1
    artifacts = write_telemetry(str(output_dir / "telemetry"), [sample.__dict__ for sample in log.samples])
    log.diagnostics["telemetry_artifacts"] = artifacts
    _write_worker_result(output_dir, log, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
