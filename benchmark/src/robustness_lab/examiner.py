"""Compute-aware adaptive locomotion capability examination.

The examiner is deliberately stateful and append-only at the evidence level.
It reuses canonical worlds/episodes and chooses the next trial from observed
outcomes instead of materializing a huge Cartesian product up front.
"""

from __future__ import annotations

import datetime as dt
import html
import itertools
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .diagnosis import cluster_diagnoses, diagnose_run
from .benchmark import _graph_world
from .io import atomic_write_text, dump_json, load_data, sha256_data, sha256_file
from .models import RobotProfile, WorldSpec
from .runtime import execute_episode
from .skill import SENSOR_KINDS, SkillPackage, assess_navigation_compatibility, load_skill
from .statistics import capability_curve, wilson_interval
from .worlds import AE_CHALLENGES, TERRAIN_CODES, generate_ae, generate_terrain


EXAMINER_SCHEMA_VERSION = "1.0"


# Coarse skill-manifest sensor kinds are admitted only when every canonical
# field needed to construct that sensor is available. ``None`` marks kinds
# whose canonical representation belongs to a later schema phase.
SENSOR_KIND_CANONICAL_REQUIREMENTS: Dict[str, Optional[FrozenSet[str]]] = {
    "joint_position": frozenset({"joint_position"}),
    "joint_velocity": frozenset({"joint_velocity"}),
    "previous_action": frozenset({"previous_action"}),
    "imu": frozenset({"base_angular_velocity_b", "projected_gravity_b"}),
    "base_state": frozenset(
        {
            "base_position_w",
            "base_orientation_wxyz",
            "base_linear_velocity_b",
            "base_angular_velocity_b",
        }
    ),
    "command": frozenset({"velocity_command"}),
    "waypoints": frozenset({"waypoints_b"}),
    "goal_pose": frozenset({"goal_vector_b", "goal_distance", "goal_heading"}),
    "rgb_camera": None,
    "depth_camera": None,
    "lidar": None,
    "height_scan": None,
    "privileged_terrain": None,
    "privileged_state": None,
}

PROGRESSION_STAGES = (
    {"id": "L3-chain-M", "level": "L3", "topology": "chain", "horizon": "M", "difficulty": 0.5, "pair_mode": "cycle", "requires_all": []},
    {"id": "L3-grid-M", "level": "L3", "topology": "grid", "horizon": "M", "difficulty": 0.5, "pair_mode": "cycle", "requires_all": []},
    {"id": "L4-random-L", "level": "L4", "topology": "random_connected", "horizon": "L", "difficulty": 0.8, "pair_mode": "random", "requires_all": ["L3-grid-M"]},
    {"id": "L5-endurance-100", "level": "L5", "topology": "chain", "horizon": "XL", "difficulty": 0.5, "pair_mode": "cycle", "requires_all": ["L3-chain-M"]},
    {"id": "L5-endurance-200", "level": "L5", "topology": "chain", "horizon": "Endurance", "difficulty": 0.5, "pair_mode": "cycle", "requires_all": ["L5-endurance-100"]},
    {"id": "L6-stressed-L", "level": "L6", "topology": "chain", "horizon": "L", "difficulty": 0.8, "pair_mode": "adaptive", "requires_all": ["L3-chain-M"], "stressors": [{"kind": "push_impulse_ns", "magnitude": 8.0}]},
    {"id": "L7-locked-ood-L", "level": "L7", "topology": "random_connected", "horizon": "L", "difficulty": 0.9, "pair_mode": "random", "requires_any": ["L4-random-L", "L5-endurance-100"], "locked": True},
)


@dataclass
class ExaminerSettings:
    examiner_id: str
    backend: str
    robot_profile: str
    robot_asset: str
    robot_format: str
    skill: str
    runtimes: str
    output_dir: str
    terrain_codes: List[str] = field(default_factory=lambda: list(TERRAIN_CODES))
    screening_severities: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    minimum_severity: float = 0.0
    maximum_severity: float = 1.0
    boundary_tolerance: float = 0.05
    exploration_repetitions: int = 1
    confirmation_min_repetitions: int = 3
    confirmation_max_repetitions: int = 12
    target_success_probability: float = 0.7
    pruning_audit_fraction: float = 0.1
    interaction_count: int = 10
    seed: int = 1701
    timeout_s: float = 20.0
    max_trials: int = 0

    def validate(self) -> None:
        if self.backend not in {"fake", "mujoco", "isaaclab", "isaacgym", "foresight_native"}:
            raise ValueError("unsupported examiner backend %s" % self.backend)
        if self.robot_format not in {"urdf", "mjcf", "usd"}:
            raise ValueError("unsupported robot format %s" % self.robot_format)
        if not self.terrain_codes or any(code not in TERRAIN_CODES for code in self.terrain_codes):
            raise ValueError("terrain_codes must be selected from RS1-RS11")
        if sorted(set(self.screening_severities)) != sorted(self.screening_severities):
            raise ValueError("screening severities must be unique")
        if any(value < self.minimum_severity or value > self.maximum_severity for value in self.screening_severities):
            raise ValueError("screening severities lie outside configured bounds")
        if self.boundary_tolerance <= 0 or self.exploration_repetitions < 1:
            raise ValueError("boundary tolerance and repetitions must be positive")
        if self.confirmation_min_repetitions < 1 or self.confirmation_max_repetitions < self.confirmation_min_repetitions:
            raise ValueError("invalid confirmation repetition bounds")
        if not 0.0 < self.target_success_probability < 1.0:
            raise ValueError("target_success_probability must be between zero and one")
        if not 0.0 <= self.pruning_audit_fraction <= 1.0:
            raise ValueError("pruning_audit_fraction must lie in [0,1]")


def _resolve(value: str, base: Path) -> str:
    path = Path(value)
    return str(path.resolve()) if path.is_absolute() else str((base / path).resolve())


def load_examiner_settings(path: str) -> ExaminerSettings:
    raw = load_data(path)
    base = Path(path).resolve().parent
    paths = {name: _resolve(raw[name], base) for name in ("robot_profile", "robot_asset", "skill", "runtimes")}
    output = _resolve(raw.get("output_dir", "runs/examinations/%s" % raw.get("examiner_id", "examination")), base)
    options = dict(raw.get("adaptive", {}))
    value = ExaminerSettings(
        examiner_id=raw.get("examiner_id", ""),
        backend=raw.get("backend", "isaaclab"),
        robot_profile=paths["robot_profile"],
        robot_asset=paths["robot_asset"],
        robot_format=raw.get("robot_format", "usd"),
        skill=paths["skill"],
        runtimes=paths["runtimes"],
        output_dir=output,
        terrain_codes=list(raw.get("terrain_codes", TERRAIN_CODES)),
        screening_severities=[float(item) for item in options.get("screening_severities", [0.2, 0.5, 0.8])],
        minimum_severity=float(options.get("minimum_severity", 0.0)),
        maximum_severity=float(options.get("maximum_severity", 1.0)),
        boundary_tolerance=float(options.get("boundary_tolerance", 0.05)),
        exploration_repetitions=int(options.get("exploration_repetitions", 1)),
        confirmation_min_repetitions=int(options.get("confirmation_min_repetitions", 3)),
        confirmation_max_repetitions=int(options.get("confirmation_max_repetitions", 12)),
        target_success_probability=float(options.get("target_success_probability", 0.7)),
        pruning_audit_fraction=float(options.get("pruning_audit_fraction", 0.1)),
        interaction_count=int(options.get("interaction_count", 10)),
        seed=int(raw.get("seed", 1701)),
        timeout_s=float(raw.get("timeout_s", 20.0)),
        max_trials=int(raw.get("max_trials", 0)),
    )
    value.validate()
    return value


def _flat_world(robot: RobotProfile, seed: int) -> WorldSpec:
    world = generate_terrain("rs1", robot, 0.0, seed, {"length_m": robot.body_length_m * 3.0})
    world.name = "FLAT-QUALIFICATION-s%d" % seed
    world.terrain_type = "qualification_flat"
    world.materials["low_friction"] = world.materials["nominal"]
    for geometry in world.geometry:
        geometry.material = "nominal"
    for region in world.regions:
        if region.kind == "low_friction":
            region.kind = "flat_qualification"
    world.metadata.update({"difficulty": 0.0, "qualification": True})
    world.validate()
    return world


def _branch_template(code: str) -> Dict[str, Any]:
    return {
        "terrain": code,
        "status": "pending",
        "points": {},
        "measured_boundary": None,
        "boundary_interval": None,
        "inferred_failures": [],
        "audit_status": "not_needed",
        "monotonicity": "assumed_with_audit",
    }


def _balanced_pairs(codes: List[str], count: int, seed: int) -> Tuple[List[List[str]], List[List[str]]]:
    all_pairs = [[left, right] for left, right in itertools.product(codes, repeat=2)]
    if count >= len(all_pairs):
        return all_pairs, all_pairs
    # Deterministic coverage: rotate the right side so both positions are balanced.
    selected = []
    offset = seed % len(codes)
    for index in range(count):
        left = codes[index % len(codes)]
        right = codes[(index // len(codes) + index + offset) % len(codes)]
        selected.append([left, right])
    return all_pairs, selected


def _source_owns_observations(skill: SkillPackage, backend: str) -> bool:
    """Return whether the launched source task constructs the observations."""
    return backend == "foresight_native" and skill.controller.get("type") == "native_rsl_rl"


def _backend_sensor_gate(skill: SkillPackage, backend: str, runtimes_path: str) -> Dict[str, Any]:
    from .backends.fake import FakeBackend
    from .backends.isaacgym import IsaacGymBackend
    from .backends.isaaclab import IsaacLabBackend
    from .backends.mujoco import MujocoBackend

    backend_types = {
        "fake": FakeBackend,
        "mujoco": MujocoBackend,
        "isaaclab": IsaacLabBackend,
        "isaacgym": IsaacGymBackend,
    }
    backend_type = backend_types.get(backend)
    canonical_fields = (
        backend_type.probe().observation_capabilities() if backend_type is not None else frozenset()
    )
    requested = {item.kind for item in skill.sensors if item.required}
    if _source_owns_observations(skill, backend):
        # The native integration launches the policy's own Isaac Lab task,
        # which constructs its observation vector instead of using our bus.
        supported = set(SENSOR_KINDS)
    else:
        supported = {
            kind
            for kind in SENSOR_KINDS
            if SENSOR_KIND_CANONICAL_REQUIREMENTS.get(kind) is not None
            and SENSOR_KIND_CANONICAL_REQUIREMENTS[kind].issubset(canonical_fields)
        }
    missing = sorted(requested - supported)
    runtime = load_data(runtimes_path).get("runtimes", {}).get(backend, {})
    navigation_providers = {
        "fake": {"waypoint_heading", "waypoint_positions", "goal_pose", "route_follower"},
        "foresight_native": {"waypoint_heading"},
        "mujoco": {"route_follower"},
        "isaaclab": {"route_follower"},
        "isaacgym": {"route_follower"},
    }.get(backend, set())
    navigation = assess_navigation_compatibility(skill, "chain", "locomotion", sorted(navigation_providers))
    compatible = not missing and navigation["status"] == "compatible"
    return {
        "status": "compatible" if compatible else "blocked",
        "backend": backend,
        "requested_sensor_kinds": sorted(requested),
        "supported_sensor_kinds": sorted(supported),
        "missing_sensor_kinds": missing,
        "navigation": navigation,
        "runtime_features": runtime.get("features", {}),
        "claim_scope": (
            "fake validates orchestration only"
            if backend == "fake"
            else ("source-native observation and route adapter" if backend == "foresight_native" else "declared sensor availability")
        ),
    }


def initialize_examination(config_path: str, output_override: Optional[str] = None) -> Dict[str, Any]:
    from .discrete_examiner import initialize_discrete_examination, is_discrete_config

    if is_discrete_config(config_path):
        return initialize_discrete_examination(config_path, output_override)
    settings = load_examiner_settings(config_path)
    if output_override:
        settings.output_dir = str(Path(output_override).resolve())
    root = Path(settings.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    robot = RobotProfile.from_dict(load_data(settings.robot_profile))
    skill = load_skill(settings.skill)
    if skill.robot_profile_sha256 and skill.robot_profile_sha256 != robot.source_sha256:
        raise ValueError("skill was declared for a different RobotProfile")
    # RS8 is the repeated-gap transition used around an AE cell, not one of
    # its purple/brown challenge slots. Primitive screening still covers all
    # RS1-RS11; ordered interactions use the canonical ten-member AE set.
    interaction_codes = [code for code in settings.terrain_codes if code in AE_CHALLENGES]
    all_pairs, selected_pairs = _balanced_pairs(interaction_codes, settings.interaction_count, settings.seed)
    sensor_gate = _backend_sensor_gate(skill, settings.backend, settings.runtimes)
    navigation_providers = sensor_gate.get("navigation", {}).get("available_providers", [])
    progression = []
    for item in PROGRESSION_STAGES:
        navigation = assess_navigation_compatibility(skill, item["topology"], "locomotion", navigation_providers)
        progression.append({**item, "status": "pending" if navigation["status"] == "compatible" else "blocked", "trials": [], "navigation": navigation})
    state = {
        "schema_version": EXAMINER_SCHEMA_VERSION,
        "examiner_id": settings.examiner_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ready" if sensor_gate["status"] == "compatible" else "blocked",
        "stage": "qualification" if sensor_gate["status"] == "compatible" else "compatibility",
        "config_path": str(Path(config_path).resolve()),
        "settings": settings.__dict__,
        "hashes": {
            "config": sha256_file(str(Path(config_path).resolve())),
            "robot_profile": sha256_file(settings.robot_profile),
            "robot_asset": sha256_file(settings.robot_asset),
            "skill_manifest": sha256_file(settings.skill),
        },
        "skill": {"skill_id": skill.skill_id, "sensor_track": skill.sensor_track, "controller_type": skill.controller.get("type"), "navigation": skill.navigation_contract()},
        "compatibility": sensor_gate,
        "qualification": {"status": "pending", "trials": []},
        "branches": {code: _branch_template(code) for code in settings.terrain_codes},
        "interaction_catalog": all_pairs,
        "selected_interactions": [{"pair": pair, "status": "pending", "trials": []} for pair in selected_pairs],
        "progression": progression,
        "trials": [],
        "next_trial_index": 0,
        "scheduler": {"selection_policy": "adaptive-boundary-v1", "pruning_audit_fraction": settings.pruning_audit_fraction},
    }
    if sensor_gate["status"] != "compatible":
        reasons = []
        if sensor_gate["missing_sensor_kinds"]:
            reasons.append("backend does not provide required sensor kinds: %s" % ", ".join(sensor_gate["missing_sensor_kinds"]))
        reasons.extend(sensor_gate.get("navigation", {}).get("reasons", []))
        state["block_reason"] = "; ".join(reasons) or "policy/backend contract is incompatible"
    dump_json(str(root / "examination_state.json"), state)
    return state


def _point(branch: Dict[str, Any], severity: float) -> Dict[str, Any]:
    key = "%.6f" % severity
    return branch["points"].setdefault(key, {"severity": severity, "outcomes": [], "trial_ids": [], "phase": "exploration", "decision": "unresolved"})


def _decision(point: Dict[str, Any], settings: ExaminerSettings, confirmation: bool = False) -> str:
    outcomes = [bool(value) for value in point["outcomes"]]
    if not outcomes:
        return "unresolved"
    required = settings.confirmation_min_repetitions if confirmation else settings.exploration_repetitions
    if len(outcomes) < required:
        return "unresolved"
    if not confirmation:
        return "preliminary_pass" if sum(outcomes) / len(outcomes) >= settings.target_success_probability else "preliminary_fail"
    lower, upper = wilson_interval(sum(outcomes), len(outcomes))
    if lower >= settings.target_success_probability:
        return "pass"
    if upper < settings.target_success_probability:
        return "fail"
    if len(outcomes) >= settings.confirmation_max_repetitions:
        return "uncertain"
    return "unresolved"


def _exploration_points(branch: Dict[str, Any]) -> List[Tuple[float, str]]:
    result = []
    for value in branch["points"].values():
        if value["outcomes"]:
            rate = sum(bool(item) for item in value["outcomes"]) / len(value["outcomes"])
            result.append((float(value["severity"]), "pass" if rate >= 0.5 else "fail"))
    return sorted(result)


def _next_branch_trial(branch: Dict[str, Any], settings: ExaminerSettings) -> Optional[Dict[str, Any]]:
    for severity in settings.screening_severities:
        point = _point(branch, severity)
        if len(point["outcomes"]) < settings.exploration_repetitions:
            branch["status"] = "screening"
            return {"severity": severity, "phase": "exploration"}
    points = _exploration_points(branch)
    passes = [severity for severity, decision in points if decision == "pass"]
    failures = [severity for severity, decision in points if decision == "fail"]
    if passes and not failures:
        endpoint = _point(branch, settings.maximum_severity)
        if len(endpoint["outcomes"]) < settings.exploration_repetitions:
            return {"severity": settings.maximum_severity, "phase": "exploration"}
        if sum(endpoint["outcomes"]) / len(endpoint["outcomes"]) >= 0.5:
            boundary = settings.maximum_severity
            interval = [settings.maximum_severity, settings.maximum_severity]
        else:
            failures.append(settings.maximum_severity)
            boundary, interval = max(passes), [max(passes), min(failures)]
    elif failures and not passes:
        endpoint = _point(branch, settings.minimum_severity)
        if len(endpoint["outcomes"]) < settings.exploration_repetitions:
            return {"severity": settings.minimum_severity, "phase": "exploration"}
        if sum(endpoint["outcomes"]) / len(endpoint["outcomes"]) < 0.5:
            boundary, interval = settings.minimum_severity, [settings.minimum_severity, min(failures)]
        else:
            passes.append(settings.minimum_severity)
            boundary, interval = max(passes), [max(passes), min(failures)]
    elif passes and failures:
        lower = max([value for value in passes if value < min(failures)] or [settings.minimum_severity])
        upper = min([value for value in failures if value > lower] or [settings.maximum_severity])
        if upper - lower > settings.boundary_tolerance:
            midpoint = round((lower + upper) / 2.0, 6)
            point = _point(branch, midpoint)
            if len(point["outcomes"]) < settings.exploration_repetitions:
                branch["status"] = "boundary_search"
                return {"severity": midpoint, "phase": "exploration"}
            # The measured midpoint was added to ``points`` only on the next call.
            return _next_branch_trial(branch, settings)
        boundary, interval = lower, [lower, upper]
    else:
        return None

    point = _point(branch, boundary)
    point["phase"] = "confirmation"
    decision = _decision(point, settings, confirmation=True)
    point["decision"] = decision
    if decision == "unresolved":
        branch["status"] = "confirming"
        return {"severity": boundary, "phase": "confirmation"}
    branch["status"] = "completed" if decision in {"pass", "fail"} else "uncertain"
    branch["measured_boundary"] = boundary
    branch["boundary_interval"] = interval
    branch["confirmation_decision"] = decision
    branch["inferred_failures"] = [
        round(value, 3)
        for value in (settings.screening_severities + [settings.maximum_severity])
        if value > interval[1] and "%.6f" % value not in branch["points"]
    ]
    if branch["inferred_failures"]:
        branch["audit_status"] = "pending"
    return None


def _next_trial(state: Dict[str, Any], settings: ExaminerSettings) -> Optional[Dict[str, Any]]:
    if state["qualification"]["status"] == "pending":
        return {"kind": "qualification", "severity": 0.0, "phase": "qualification"}
    if state["qualification"]["status"] != "passed":
        state["status"], state["stage"] = "stopped", "qualification_failed"
        return None
    state["stage"] = "primitive_capabilities"
    for code in settings.terrain_codes:
        branch = state["branches"][code]
        if branch["status"] not in {"completed", "uncertain", "unsupported"}:
            request = _next_branch_trial(branch, settings)
            if request:
                return {"kind": "primitive", "terrain": code, **request}
    # Audits are deterministic and sparse. At least one inferred branch is
    # audited when any pruning occurred and the configured fraction is nonzero.
    pending_audits = [state["branches"][code] for code in settings.terrain_codes if state["branches"][code].get("audit_status") == "pending"]
    audit_budget = int(math.ceil(len(pending_audits) * settings.pruning_audit_fraction)) if pending_audits else 0
    completed_audits = sum(branch.get("audit_status") == "completed" for branch in state["branches"].values())
    if completed_audits < audit_budget:
        branch = pending_audits[completed_audits % len(pending_audits)]
        severity = max(branch["inferred_failures"])
        branch["audit_status"] = "running"
        return {"kind": "audit", "terrain": branch["terrain"], "severity": severity, "phase": "pruning_audit"}
    for branch in pending_audits:
        branch["audit_status"] = "not_selected"
    state["stage"] = "interactions"
    boundaries = {code: state["branches"][code].get("measured_boundary") for code in settings.terrain_codes}
    for interaction in state["selected_interactions"]:
        if interaction["status"] == "pending":
            left, right = interaction["pair"]
            if any(state["branches"][code]["status"] == "unsupported" for code in (left, right)):
                interaction["status"] = "pruned"
                interaction["reason"] = "one or both primitive terrain families are unsupported by this backend fidelity tier"
                continue
            available = [value for value in (boundaries.get(left), boundaries.get(right)) if isinstance(value, (int, float))]
            severity = min(available) if available else 0.5
            return {"kind": "interaction", "terrain": "ae", "pair": [left, right], "severity": severity, "phase": "interaction"}
    passed_pairs = [item["pair"] for item in state["selected_interactions"] if item.get("success") is True]
    state["stage"] = "composed_progression"
    progression = state.get("progression", [])
    by_id = {item["id"]: item for item in progression}
    for item in progression:
        if item["status"] != "pending":
            continue
        if not passed_pairs:
            item["status"] = "pruned"
            item["reason"] = "no AE interaction passed; no valid edge library exists"
            continue
        required_all = [by_id[value]["status"] for value in item.get("requires_all", [])]
        required_any = [by_id[value]["status"] for value in item.get("requires_any", [])]
        if any(value in {"failed", "pruned", "blocked", "skipped"} for value in required_all):
            item["status"] = "pruned"
            item["reason"] = "a required parent stage did not pass"
            continue
        if required_all and any(value != "passed" for value in required_all):
            continue
        if required_any and not any(value == "passed" for value in required_any):
            if all(value in {"failed", "pruned", "blocked", "skipped"} for value in required_any):
                item["status"] = "pruned"
                item["reason"] = "none of the alternative parent stages passed"
            continue
        if item.get("stressors") and settings.backend == "foresight_native":
            item["status"] = "skipped"
            item["reason"] = "source-native bridge does not yet apply canonical stressors"
            continue
        return {
            "kind": "progression",
            "terrain": "aee",
            "severity": item["difficulty"],
            "phase": item["level"],
            "progression_id": item["id"],
            "topology": item["topology"],
            "horizon": item["horizon"],
            "pair_mode": item["pair_mode"],
            "allowed_pairs": passed_pairs,
            "stressors": item.get("stressors", []),
            "locked": item.get("locked", False),
        }
    state["status"], state["stage"] = "completed", "complete"
    return None


def _trial_world(request: Dict[str, Any], robot: RobotProfile, seed: int) -> WorldSpec:
    if request["kind"] == "qualification":
        return _flat_world(robot, seed)
    if request["kind"] == "interaction":
        return generate_ae(robot, request["pair"][0], request["pair"][1], request["severity"], seed)
    if request["kind"] == "progression":
        return _graph_world(
            robot,
            request["topology"],
            request["horizon"],
            request["severity"],
            seed,
            request["pair_mode"],
            request["allowed_pairs"],
        )
    return generate_terrain(request["terrain"], robot, request["severity"], seed)


def _metric_value(run_dir: Path, name: str) -> Optional[float]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    for item in load_data(str(path)).get("metrics", []):
        if item.get("name") == name and item.get("valid"):
            return item.get("value")
    return None


def _execute_foresight_native(
    world_path: Path,
    run_dir: Path,
    seed: int,
    skill: SkillPackage,
) -> Dict[str, Any]:
    """Execute a source-native RSL-RL policy without rebuilding its inputs."""
    controller = skill.controller
    checkpoint = controller.get("checkpoint") or controller.get("path")
    task = controller.get("task")
    if not checkpoint or not task:
        return {
            "status": "error",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "native_controller_contract_incomplete",
            "diagnosis": {"status": "unavailable", "classification": "infrastructure", "reason": "native controller needs checkpoint and task"},
        }
    project = Path(__file__).resolve().parents[2]
    launcher = project / "scripts/run_foresight_world.sh"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "GPU_INDEX": str(controller.get("gpu_index", 1)),
            "VIDEO_LENGTH": str(int(controller.get("evaluation_steps", 1000))),
            "RUN_SEED": str(seed),
            "RECORD_VIDEO": "1" if controller.get("record_video_each_trial", False) else "0",
            # Material binding is a simulator-adapter responsibility. It is
            # enabled independently of policy configuration.
            "PER_REGION_MATERIALS": "1",
        }
    )
    if controller.get("source_root"):
        environment["FORESIGHT_ROOT"] = str(controller["source_root"])
    command = ["bash", str(launcher), str(world_path), str(checkpoint), str(run_dir), str(task)]
    log_path = run_dir / "native_runner.log"
    wall_timeout = float(controller.get("wall_timeout_s", 1800.0))
    bridge_error_path = run_dir / "bridge_error.json"
    timed_out = False
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=str(project), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + wall_timeout
        while process.poll() is None:
            if bridge_error_path.is_file():
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                break
            time.sleep(0.2)
    if timed_out:
        return {
            "status": "error",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "native_runner_timeout",
            "diagnosis": {"status": "unavailable", "classification": "infrastructure", "reason": "source-native runner exceeded wall timeout"},
            "native_log": str(log_path),
        }
    if bridge_error_path.is_file():
        bridge_error = load_data(str(bridge_error_path))
        return {
            "status": "error",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "native_checkpoint_error",
            "diagnosis": {"status": "unavailable", "classification": "infrastructure", "reason": bridge_error.get("message", "native bridge error"), "bridge_error": str(bridge_error_path)},
            "native_log": str(log_path),
        }
    result_path = run_dir / "result.json"
    if process.returncode != 0 or not result_path.is_file():
        return {
            "status": "error",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "native_runner_error",
            "diagnosis": {"status": "unavailable", "classification": "infrastructure", "reason": "source-native runner returned %d or omitted result.json" % process.returncode},
            "native_log": str(log_path),
        }
    native = load_data(str(result_path))
    environment_support = native.get("environment_physics", {}).get("support", {})
    if environment_support.get("status") == "environment_unsupported":
        return {
            "status": "skipped",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "environment_physics_unsupported",
            "diagnosis": {
                "status": "unavailable",
                "classification": "environment_capability",
                "reason": environment_support.get("reason", "simulator adapter cannot reproduce the requested environment physics"),
                "environment_support": environment_support,
                "policy_was_evaluated": False,
            },
            "native_log": str(log_path),
            "native_result": str(result_path),
            "native_summary": native,
        }
    fidelity = native.get("material_fidelity", {})
    if not fidelity.get("capability_claim_eligible", False):
        return {
            "status": "skipped",
            "success": None,
            "valid_behavior": False,
            "terminal_reason": "physics_material_fidelity_unverified",
            "diagnosis": {
                "status": "unavailable",
                "classification": "infrastructure",
                "reason": "simulator adapter did not verify the requested per-region material binding",
                "material_fidelity": fidelity,
                "policy_was_evaluated": False,
            },
            "native_log": str(log_path),
            "native_result": str(result_path),
            "native_summary": native,
        }
    success = bool(native.get("success"))
    terminal = "goal" if success else "route_incomplete"
    diagnosis = {
        "status": "limited",
        "classification": "success" if success else "behavioral/progress",
        "terminal_reason": terminal,
        "event_chain": ["goal reached"] if success else ["final native goal was not reached within the evaluation window"],
        "likely_factors": [],
        "claim_scope": "native summary has route progress and terminations but not canonical per-foot telemetry",
    }
    return {
        "status": "completed",
        "success": success,
        "valid_behavior": True,
        "terminal_reason": terminal,
        "diagnosis": diagnosis,
        "native_log": str(log_path),
        "native_result": str(result_path),
        "native_summary": native,
    }


def _execute_request(state: Dict[str, Any], request: Dict[str, Any], settings: ExaminerSettings, skill: SkillPackage, robot: RobotProfile) -> Dict[str, Any]:
    index = int(state["next_trial_index"])
    seed = settings.seed + index * 7919
    suffix = request.get("terrain", request["kind"])
    trial_id = "%s-%04d-%s-s%.3f" % (settings.examiner_id, index, suffix, request["severity"])
    root = Path(settings.output_dir)
    world_path = root / "worlds" / (trial_id + ".json")
    episode_path = root / "episodes" / (trial_id + ".json")
    run_dir = root / "trials" / trial_id
    world = _trial_world(request, robot, seed)
    dump_json(str(world_path), world)
    episode = {
        "episode_id": trial_id,
        "backend": settings.backend,
        "robot_profile": settings.robot_profile,
        "robot_asset": settings.robot_asset,
        "robot_format": settings.robot_format,
        "world": str(world_path),
        "controller": skill.controller_config(),
        "output_dir": str(run_dir),
        "seed": seed,
        "timeout_s": settings.timeout_s,
        "headless": True,
        "record_video": False,
        "stressors": request.get("stressors", []),
        "metadata": {"examiner_id": settings.examiner_id, "request": request, "sensor_track": skill.sensor_track, "severity": request["severity"]},
    }
    dump_json(str(episode_path), episode)
    if settings.backend == "foresight_native":
        execution = _execute_foresight_native(world_path, run_dir, seed, skill)
        status = execution["status"]
        valid_behavior = execution["valid_behavior"]
        outcome = execution["success"]
        diagnosis = execution["diagnosis"]
    else:
        manifest = execute_episode(str(episode_path), settings.runtimes)
        status = manifest.status
        success_value = _metric_value(run_dir, "success")
        valid_behavior = manifest.status in {"completed", "failed"} and success_value is not None
        outcome = bool(success_value) if valid_behavior else None
        diagnosis = diagnose_run(str(run_dir))
    record = {
        "trial_id": trial_id,
        "index": index,
        "request": request,
        "seed": seed,
        "status": status,
        "terminal_reason": execution.get("terminal_reason") if settings.backend == "foresight_native" else diagnosis.get("terminal_reason"),
        "success": outcome,
        "valid_behavior": valid_behavior,
        "world": str(world_path),
        "episode": str(episode_path),
        "run_dir": str(run_dir),
        "diagnosis": diagnosis,
    }
    if settings.backend == "foresight_native":
        record.update({key: execution[key] for key in ("native_log", "native_result", "native_summary") if key in execution})
    state["trials"].append(record)
    state["next_trial_index"] = index + 1
    if request["kind"] == "qualification":
        state["qualification"]["trials"].append(trial_id)
        state["qualification"]["status"] = "passed" if outcome is True else ("failed" if outcome is False else "blocked")
    elif request["kind"] in {"primitive", "audit"}:
        branch = state["branches"][request["terrain"]]
        if status == "skipped":
            branch["status"] = "unsupported"
            branch["unsupported_kind"] = (
                "environment"
                if diagnosis.get("classification") == "environment_capability"
                else "simulator_integration"
            )
            branch["unsupported_reason"] = diagnosis.get("reason", record["terminal_reason"])
            return record
        point = _point(branch, request["severity"])
        point["trial_ids"].append(trial_id)
        if outcome is not None:
            point["outcomes"].append(outcome)
        if request["kind"] == "audit":
            expected_failure = outcome is False
            branch["audit_status"] = "completed"
            branch["audit_outcome"] = "supports_monotonicity" if expected_failure else "violation"
            if not expected_failure:
                branch["monotonicity"] = "violated"
                branch["status"] = "uncertain"
                branch["inferred_failures"] = []
    elif request["kind"] == "interaction":
        for interaction in state["selected_interactions"]:
            if interaction["pair"] == request["pair"] and interaction["status"] == "pending":
                interaction["trials"].append(trial_id)
                interaction["status"] = "completed" if outcome is not None else "blocked"
                interaction["success"] = outcome
                break
    else:
        stage = next(item for item in state.get("progression", []) if item["id"] == request["progression_id"])
        stage["trials"].append(trial_id)
        stage["status"] = "passed" if outcome is True else ("failed" if outcome is False else "blocked")
        stage["success"] = outcome
    return record


def run_examination(state_path: str, max_trials: Optional[int] = None) -> Dict[str, Any]:
    state_file = Path(state_path).resolve()
    state = load_data(str(state_file))
    from .discrete_examiner import is_discrete_state, run_discrete_examination

    if is_discrete_state(state):
        return run_discrete_examination(str(state_file), max_trials)
    settings = ExaminerSettings(**state["settings"])
    settings.validate()
    robot = RobotProfile.from_dict(load_data(settings.robot_profile))
    skill = load_skill(settings.skill)
    limit = max_trials if max_trials is not None else settings.max_trials
    executed = 0
    while state["status"] not in {"completed", "stopped", "blocked"}:
        if limit and executed >= limit:
            break
        request = _next_trial(state, settings)
        if request is None:
            break
        record = _execute_request(state, request, settings, skill, robot)
        executed += 1
        if not record["valid_behavior"] and record["status"] == "error":
            state["status"] = "blocked"
            state["block_reason"] = record["diagnosis"].get("terminal_reason", record["status"])
        if request["kind"] == "qualification" and not record["valid_behavior"]:
            state["status"] = "blocked"
            state["block_reason"] = record["diagnosis"].get("reason", record["status"])
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump_json(str(state_file), state)
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump_json(str(state_file), state)
    return {"status": state["status"], "stage": state["stage"], "executed": executed, "total_trials": len(state["trials"]), "state": str(state_file)}


def examination_status(state_path: str) -> Dict[str, Any]:
    state = load_data(state_path)
    from .discrete_examiner import discrete_status, is_discrete_state

    if is_discrete_state(state):
        return discrete_status(state_path)
    counts: Dict[str, int] = {}
    for trial in state.get("trials", []):
        counts[trial["status"]] = counts.get(trial["status"], 0) + 1
    return {
        "examiner_id": state["examiner_id"],
        "status": state["status"],
        "stage": state["stage"],
        "trial_count": len(state.get("trials", [])),
        "trial_statuses": counts,
        "qualification": state["qualification"]["status"],
        "completed_branches": sum(item["status"] == "completed" for item in state["branches"].values()),
        "uncertain_branches": sum(item["status"] == "uncertain" for item in state["branches"].values()),
        "total_branches": len(state["branches"]),
    }


def diagnose_examination(state_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    state = load_data(state_path)
    from .discrete_examiner import diagnose_discrete, is_discrete_state

    if is_discrete_state(state):
        return diagnose_discrete(state_path, output_path)
    records = [item for item in state.get("trials", []) if item.get("diagnosis")]
    counterfactuals = []
    by_terrain: Dict[str, List[Dict[str, Any]]] = {}
    for item in records:
        terrain = item.get("request", {}).get("terrain")
        if terrain and item.get("success") is not None:
            by_terrain.setdefault(terrain, []).append(item)
    for terrain, items in by_terrain.items():
        failures = [item for item in items if item["success"] is False]
        successes = [item for item in items if item["success"] is True]
        for failure in failures:
            lower = [item for item in successes if item["request"]["severity"] < failure["request"]["severity"]]
            if lower:
                control = max(lower, key=lambda item: item["request"]["severity"])
                counterfactuals.append({"classification": failure["diagnosis"]["classification"], "terrain": terrain, "failed_severity": failure["request"]["severity"], "control_severity": control["request"]["severity"], "failed_run": failure["run_dir"], "control_run": control["run_dir"], "interpretation": "failure appeared after increasing only the terrain severity within the same family"})
    result = cluster_diagnoses(records, counterfactuals)
    result["state"] = str(Path(state_path).resolve())
    result["counterfactual_count"] = len(counterfactuals)
    destination = output_path or str(Path(state_path).resolve().parent / "failure_diagnosis.json")
    dump_json(destination, result)
    return result


def _branch_summary(branch: Dict[str, Any]) -> Dict[str, Any]:
    points = []
    raw_points = []
    for item in branch.get("points", {}).values():
        outcomes = [bool(value) for value in item.get("outcomes", [])]
        if outcomes:
            raw_points.extend((float(item["severity"]), outcome) for outcome in outcomes)
            lower, upper = wilson_interval(sum(outcomes), len(outcomes))
            points.append({"severity": item["severity"], "success_rate": sum(outcomes) / len(outcomes), "trials": len(outcomes), "lower_ci": lower, "upper_ci": upper, "phase": item.get("phase"), "decision": item.get("decision")})
    curve = capability_curve(raw_points) if raw_points else None
    return {"terrain": branch["terrain"], "status": branch["status"], "boundary": branch.get("measured_boundary"), "boundary_interval": branch.get("boundary_interval"), "confirmation_decision": branch.get("confirmation_decision"), "monotonicity": branch.get("monotonicity"), "audit": branch.get("audit_outcome", branch.get("audit_status")), "measured_points": sorted(points, key=lambda item: item["severity"]), "inferred_failures": branch.get("inferred_failures", []), "descriptive_curve": curve}


def generate_capability_passport(state_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    state = load_data(state_path)
    from .discrete_examiner import is_discrete_state, report_discrete

    if is_discrete_state(state):
        return report_discrete(state_path, output_dir)
    root = Path(output_dir).resolve() if output_dir else Path(state_path).resolve().parent / "passport"
    root.mkdir(parents=True, exist_ok=True)
    diagnosis = diagnose_examination(state_path, str(root / "failure_diagnosis.json"))
    branches = [_branch_summary(state["branches"][code]) for code in state["settings"]["terrain_codes"]]
    measured = sum(len(item["measured_points"]) for item in branches)
    inferred = sum(len(item["inferred_failures"]) for item in branches)
    passport = {
        "schema_version": "1.0",
        "examiner_id": state["examiner_id"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": state["status"],
        "stage": state["stage"],
        "skill": state["skill"],
        "qualification": state["qualification"],
        "evidence_counts": {"measured_points": measured, "inferred_failures": inferred, "trials": len(state["trials"]), "failure_clusters": diagnosis["cluster_count"]},
        "capabilities": branches,
        "interactions": state["selected_interactions"],
        "failure_diagnosis": diagnosis,
        "compatibility": state.get("compatibility", {}),
        "claims": {
            "measured_and_inferred_separated": True,
            "formal_proof": False,
            "locomotion_capability": state["settings"]["backend"] != "fake" and state["qualification"]["status"] == "passed" and measured > 0,
            "statement": (
                "Fake-backend results validate scheduler, metric, diagnosis, and report wiring only; they are not locomotion capability evidence."
                if state["settings"]["backend"] == "fake"
                else (
                    "Capability limits are empirical statistical evidence for the declared policy, robot, sensor track, simulator and configuration."
                    if measured > 0
                    else "No primitive capability point was measured; qualification and structured skips/errors are integration evidence only."
                )
            ),
        },
        "provenance": {"state": str(Path(state_path).resolve()), "state_sha256": sha256_file(str(Path(state_path).resolve())), "hashes": state["hashes"]},
    }
    dump_json(str(root / "capability_passport.json"), passport)
    rows = []
    for item in branches:
        interval = item["boundary_interval"] or [None, None]
        rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s–%s</td><td>%d</td><td>%s</td></tr>" % (html.escape(item["terrain"].upper()), html.escape(item["status"]), "—" if item["boundary"] is None else "%.3f" % item["boundary"], "—" if interval[0] is None else "%.3f" % interval[0], "—" if interval[1] is None else "%.3f" % interval[1], len(item["measured_points"]), html.escape(str(item["audit"]))))
    clusters = "".join("<li><strong>%s</strong>: %d cases; confidence %s</li>" % (html.escape(item["classification"]), item["count"], html.escape(item["confidence"])) for item in diagnosis["clusters"])
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Capability Passport</title><style>body{font-family:Inter,system-ui;background:#eef2f6;color:#102a43;margin:0}.wrap{max-width:1200px;margin:auto;padding:28px}.hero{background:#102a43;color:#fff;padding:26px;border-radius:14px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card,table,.panel{background:#fff;border-radius:10px;padding:14px}strong.big{font-size:27px;display:block}table{width:100%%;border-collapse:collapse;overflow:hidden}th,td{padding:9px;border-bottom:1px solid #d9e2ec;text-align:left}th{background:#dfe7ef}.warn{color:#9b2c2c}</style></head><body><div class='wrap'><div class='hero'><h1>Locomotion Capability Passport</h1><p>%s · %s track · %s</p></div><div class='panel'><h2>Evidence scope</h2><p class='warn'>%s</p></div><div class='cards'><div class='card'><strong class='big'>%d</strong>executed trials</div><div class='card'><strong class='big'>%d</strong>measured points</div><div class='card'><strong class='big'>%d</strong>inferred points</div><div class='card'><strong class='big'>%d</strong>failure clusters</div></div><h2>Primitive capability branches</h2><table><tr><th>Terrain</th><th>Status</th><th>Boundary</th><th>Bracket</th><th>Measured points</th><th>Pruning audit</th></tr>%s</table><div class='panel'><h2>Failure analysis</h2><ul>%s</ul><p class='warn'>Inferred failures are never presented as measured episodes. This passport is empirical evidence, not a formal proof for arbitrary neural policies.</p></div><div class='panel'><h2>Reproducibility</h2><p>Raw source: <code>%s</code></p></div></div></body></html>""" % (html.escape(state["skill"]["skill_id"]), html.escape(state["skill"]["sensor_track"]), html.escape(state["status"]), html.escape(passport["claims"]["statement"]), len(state["trials"]), measured, inferred, diagnosis["cluster_count"], "".join(rows), clusters, html.escape(str(Path(state_path).resolve())))
    atomic_write_text(str(root / "index.html"), page)
    return {"status": "ok", "passport": str(root / "capability_passport.json"), "html": str(root / "index.html"), "trials": len(state["trials"]), "measured_points": measured, "inferred_failures": inferred}
