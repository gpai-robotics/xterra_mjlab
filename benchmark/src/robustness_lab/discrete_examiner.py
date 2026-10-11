"""Breadth-first discrete benchmark with conservative descendant pruning."""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .benchmark import _graph_world
from .diagnosis import cluster_diagnoses, diagnose_run
from .evidence import refresh_evidence
from .events import quaternion_to_rpy
from .io import dump_json, load_data, sha256_data, sha256_file
from .models import RobotProfile, WorldSpec
from .episode_budget import (
    DEFAULT_MAXIMUM_S,
    DEFAULT_MINIMUM_S,
    DEFAULT_REFERENCE_SPEED_MPS,
    DEFAULT_SLACK,
    episode_timeout_s,
)
from .runtime import execute_episode
from .scratch import cleanup_stale_isaac_scratch
from .skill import SkillPackage, load_skill
from .statistics import wilson_interval
from .worlds import DISCRETE_CHALLENGES, apply_challenge_friction, generate_ae, generate_ae_pairs, generate_terrain
from .worlds.modifiers import apply_geometry_modifiers


SCHEMA_VERSION = "2.0"
SCHEDULER = "discrete_pruned_v2"
DEFAULT_PARALLEL_TRIALS = 1
DEFAULT_GEOMETRY_LEVELS = (("medium", 0.50), ("easy", 0.25), ("hard", 0.75))
DEFAULT_FRICTION_LEVELS = (("high", 0.75, 0.60), ("moderate", 0.50, 0.40), ("low", 0.25, 0.20))
TERMINAL = {"passed", "failed", "pruned", "unsupported", "blocked"}
# (level, name, topology, horizon, stressors[, geometry modifier])
#
# L6 is the stressed level and L7 the out-of-distribution one, so the two axes
# the adapter supports but nothing exercised: control/perception latency lives
# at L6. Banking and curvature are single-terrain properties and live at L1 as
# variants of the terrains that passed (see L1_VARIANTS), not here.
PROGRESSION = (
    ("L3", "chain-M", "chain", "M", []),
    ("L3", "grid-M", "grid", "M", []),
    ("L4", "ring-L", "ring", "L", []),
    ("L4", "tree-L", "tree", "L", []),
    ("L4", "random-L", "random_connected", "L", []),
    ("L5", "endurance-100", "chain", "XL", []),
    ("L5", "endurance-200", "chain", "Endurance", []),
    ("L6", "stressed-L", "chain", "L", [{"type": "push_impulse_ns", "value": 8.0, "duration_s": 0.1}]),
    ("L6", "action-latency-L", "chain", "L", [{"type": "action_latency_s", "value": 0.02}]),
    ("L6", "perception-latency-L", "chain", "L", [{"type": "perception_latency_s", "value": 0.04}]),
    ("L7", "locked-ood-L", "random_connected", "L", []),
)


# Single-terrain banking and curvature, matching the shipped gallery
# (configs/galleries/go2_curvature_camber.yaml). Materialised at L1 for every
# plain terrain that PASSED -- a terrain the policy cannot walk flat is not
# re-run banked -- so the cost is proportional to what the policy earned.
L1_VARIANTS = (
    ("left-curve-left-low", {"camber_deg": -12.0, "curve_angle_deg": 65.0, "curve_segments": 28}),
    ("right-curve-right-low", {"camber_deg": 12.0, "curve_angle_deg": -65.0, "curve_segments": 28}),
)


def progression_modifier(template: Sequence[Any]) -> Dict[str, Any]:
    """Geometry modifier for a progression template, empty when it has none."""

    return dict(template[5]) if len(template) > 5 and template[5] else {}


def resolve_worker_count(requested: int) -> int:
    """Clamp a requested worker count to the CPUs available on this machine."""
    value = int(requested)
    if value < 1:
        raise ValueError("parallel_trials must be positive")
    return min(value, max(1, os.cpu_count() or 1))


def _evidence_worker_count(state: Dict[str, Any]) -> int:
    """Return the examination's configured evidence-rendering concurrency."""
    return int(state.get("settings", {}).get("parallel_trials", DEFAULT_PARALLEL_TRIALS))


def is_discrete_config(path: str) -> bool:
    raw = load_data(path)
    return str(raw.get("scheduler", raw.get("adaptive", {}).get("scheduler", ""))).lower() == SCHEDULER


def is_discrete_state(state: Dict[str, Any]) -> bool:
    return state.get("scheduler", {}).get("selection_policy") == SCHEDULER


def _resolve(value: str, base: Path) -> str:
    path = Path(value)
    return str(path.resolve()) if path.is_absolute() else str((base / path).resolve())


def _configured_levels(options: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    geometry_raw = options.get("geometry_levels")
    friction_raw = options.get("friction_levels")
    geometry = (
        [{"label": label, "value": value} for label, value in DEFAULT_GEOMETRY_LEVELS]
        if geometry_raw is None else [dict(item) for item in geometry_raw]
    )
    friction = (
        [{"label": label, "static": static, "dynamic": dynamic} for label, static, dynamic in DEFAULT_FRICTION_LEVELS]
        if friction_raw is None else [dict(item) for item in friction_raw]
    )
    if not geometry or not friction:
        raise ValueError("geometry_levels and friction_levels must be non-empty")
    for item in geometry:
        item["label"] = str(item["label"])
        item["value"] = float(item["value"])
        if not 0.0 <= item["value"] <= 1.0:
            raise ValueError("geometry level values must be in [0, 1]")
    for item in friction:
        item["label"] = str(item["label"])
        item["static"] = float(item["static"])
        item["dynamic"] = float(item["dynamic"])
        if not 0.0 <= item["dynamic"] <= item["static"] <= 2.0:
            raise ValueError("friction levels require 0 <= dynamic <= static <= 2")
    if len({item["label"] for item in geometry}) != len(geometry) or len({round(item["value"], 3) for item in geometry}) != len(geometry):
        raise ValueError("geometry level labels and values must be unique")
    if len({item["label"] for item in friction}) != len(friction) or len({round(item["static"], 3) for item in friction}) != len(friction):
        raise ValueError("friction level labels and static values must be unique")
    return geometry, friction


def _geometry_levels(settings: Dict[str, Any]) -> List[Tuple[str, float]]:
    values = settings.get("geometry_levels") or [{"label": label, "value": value} for label, value in DEFAULT_GEOMETRY_LEVELS]
    return [(str(item["label"]), float(item["value"])) for item in values]


def _friction_levels(settings: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    values = settings.get("friction_levels") or [{"label": label, "static": static, "dynamic": dynamic} for label, static, dynamic in DEFAULT_FRICTION_LEVELS]
    return [(str(item["label"]), float(item["static"]), float(item["dynamic"])) for item in values]


def _settings(config_path: str, output_override: Optional[str] = None) -> Dict[str, Any]:
    raw = load_data(config_path)
    base = Path(config_path).resolve().parent
    paths = {name: _resolve(raw[name], base) for name in ("robot_profile", "robot_asset", "skill", "runtimes")}
    output = str(Path(output_override).resolve()) if output_override else _resolve(raw.get("output_dir", "runs/discrete_examiner/%s" % raw.get("examiner_id", "examination")), base)
    options = dict(raw.get("discrete", {}))
    geometry_levels, friction_levels = _configured_levels(options)
    value = {
        "examiner_id": str(raw.get("examiner_id", "discrete-examination")),
        "backend": str(raw.get("backend", "fake")),
        "robot_profile": paths["robot_profile"], "robot_asset": paths["robot_asset"], "robot_format": str(raw.get("robot_format", "mjcf")),
        "skill": paths["skill"], "runtimes": paths["runtimes"], "output_dir": output,
        "terrain_codes": [str(item).lower() for item in raw.get("terrain_codes", DISCRETE_CHALLENGES)],
        "seed": int(raw.get("seed", 1701)), "timeout_s": float(raw.get("timeout_s", 20.0)), "max_trials": int(raw.get("max_trials", 0)),
        "protocol_variant_id": str(options.get("protocol_variant_id", "reference")),
        "geometry_levels": geometry_levels, "friction_levels": friction_levels,
        "qualification_friction_label": str(options.get("qualification_friction_label", "moderate")),
        "event_thresholds": {str(name): float(threshold) for name, threshold in dict(raw.get("event_thresholds", options.get("event_thresholds", {}))).items()},
        "screening_repetitions": int(options.get("screening_repetitions", 3)), "confirmation_repetitions": int(options.get("confirmation_repetitions", 10)),
        "confirmation_successes": int(options.get("confirmation_successes", 7)), "infrastructure_retries": int(options.get("infrastructure_retries", 2)),
        "parallel_trials": int(options.get("parallel_trials", DEFAULT_PARALLEL_TRIALS)),
        "fixed_pair_static_friction": float(options.get("fixed_pair_static_friction", 0.50)), "fixed_pair_dynamic_friction": float(options.get("fixed_pair_dynamic_friction", 0.40)),
        "native_batch_size": int(options.get("native_batch_size", 32)),
        "gpu_reserve_mb": int(options.get("gpu_reserve_mb", 8192)), "gpu_per_env_mb": int(options.get("gpu_per_env_mb", 512)),
        "ram_reserve_gb": float(options.get("ram_reserve_gb", 16.0)), "ram_per_env_mb": int(options.get("ram_per_env_mb", 512)),
        "disk_reserve_gb": float(options.get("disk_reserve_gb", 20.0)), "disk_per_trial_mb": float(options.get("disk_per_trial_mb", 2.0)),
        "isaac_scratch_reserve_mb": float(options.get("isaac_scratch_reserve_mb", 512.0)),
        "capture_native_telemetry": bool(options.get("capture_native_telemetry", True)),
        "native_telemetry_stride": int(options.get("native_telemetry_stride", 10)),
        "record_video_all_trials": bool(options.get("record_video_all_trials", False)),
    }
    if value["backend"] not in {"fake", "mujoco", "isaaclab", "isaacgym", "foresight_native"}:
        raise ValueError("unsupported discrete examiner backend")
    if set(value["terrain_codes"]) - set(DISCRETE_CHALLENGES):
        raise ValueError("terrain_codes must use RS1-RS11")
    if value["screening_repetitions"] < 1 or value["confirmation_repetitions"] < value["screening_repetitions"]:
        raise ValueError("confirmation_repetitions must be at least screening_repetitions")
    if not 1 <= value["confirmation_successes"] <= value["confirmation_repetitions"]:
        raise ValueError("confirmation_successes must be within the confirmation trial count")
    resolve_worker_count(value["parallel_trials"])
    if value["protocol_variant_id"] == "reference" and (
        value["screening_repetitions"] != 3 or value["confirmation_repetitions"] != 10 or value["confirmation_successes"] != 7
    ):
        raise ValueError("the reference protocol pins the 3/10 repetition and 7/10 scheduling gate; use a distinct protocol_variant_id to change it")
    reference_geometry = [{"label": label, "value": value_} for label, value_ in DEFAULT_GEOMETRY_LEVELS]
    reference_friction = [{"label": label, "static": static, "dynamic": dynamic} for label, static, dynamic in DEFAULT_FRICTION_LEVELS]
    if value["protocol_variant_id"] == "reference" and (
        geometry_levels != reference_geometry
        or friction_levels != reference_friction
        or value["qualification_friction_label"] != "moderate"
        or value["fixed_pair_static_friction"] != 0.50
        or value["fixed_pair_dynamic_friction"] != 0.40
        or value["event_thresholds"]
    ):
        raise ValueError("the reference protocol pins geometry and friction schedules; use a distinct protocol_variant_id for custom values")
    if value["qualification_friction_label"] not in {item["label"] for item in friction_levels}:
        raise ValueError("qualification_friction_label must name a configured friction level")
    if value["native_batch_size"] < 1:
        raise ValueError("native_batch_size must be positive")
    if value["native_telemetry_stride"] < 1:
        raise ValueError("native_telemetry_stride must be positive")
    if min(value["gpu_per_env_mb"], value["ram_per_env_mb"], value["disk_per_trial_mb"], value["isaac_scratch_reserve_mb"]) <= 0:
        raise ValueError("resource-per-environment estimates must be positive")
    return value


def _condition(identifier: str, level: str, kind: str, world_seed: int, **values: Any) -> Dict[str, Any]:
    return {
        "id": identifier, "level": level, "kind": kind, "status": "pending", "attempts": [], "target_repetitions": 3,
        "evidence_grade": "pending", "world_seed": int(world_seed), "pruned_by": [], "prune_reason": "", **values,
    }


def _primitive_conditions(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    conditions: Dict[str, Any] = {}
    order: List[str] = []
    seed = settings["seed"]
    qualification = _condition("L0-flat", "L0", "qualification", seed - 1, geometry_label="flat", geometry_level=0.0, critical=True, target_repetitions=settings["screening_repetitions"])
    conditions[qualification["id"]] = qualification
    order.append(qualification["id"])
    index = 0
    for geometry_label, geometry_level in _geometry_levels(settings):
        for friction_label, static, dynamic in _friction_levels(settings):
            for terrain in settings["terrain_codes"]:
                identifier = "L1-%s-g%03d-f%03d" % (terrain, round(100 * geometry_level), round(100 * static))
                item = _condition(
                    identifier, "L1", "primitive", seed + index, terrain=terrain, geometry_label=geometry_label,
                    geometry_level=geometry_level, friction_label=friction_label, static_friction=static, dynamic_friction=dynamic,
                    critical=friction_label == settings["qualification_friction_label"], target_repetitions=settings["screening_repetitions"],
                )
                conditions[identifier] = item
                order.append(identifier)
                index += 1
    return conditions, order


def initialize_discrete_examination(config_path: str, output_override: Optional[str] = None) -> Dict[str, Any]:
    settings = _settings(config_path, output_override)
    root = Path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    robot = RobotProfile.from_dict(load_data(settings["robot_profile"]))
    skill = load_skill(settings["skill"])
    if skill.robot_profile_sha256 and skill.robot_profile_sha256 != robot.source_sha256:
        raise ValueError("skill was declared for a different RobotProfile")
    from .examiner import _backend_sensor_gate

    compatibility = _backend_sensor_gate(skill, settings["backend"], settings["runtimes"])
    conditions, order = _primitive_conditions(settings)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    state = {
        "schema_version": SCHEMA_VERSION, "examiner_id": settings["examiner_id"], "created_at": now, "updated_at": now,
        "status": "ready" if compatibility["status"] == "compatible" else "blocked", "stage": "L0-qualification" if compatibility["status"] == "compatible" else "compatibility",
        "config_path": str(Path(config_path).resolve()), "settings": settings,
        "hashes": {"config": sha256_file(str(Path(config_path).resolve())), "robot_profile": sha256_file(settings["robot_profile"]), "robot_asset": sha256_file(settings["robot_asset"]), "skill_manifest": sha256_file(settings["skill"])},
        "skill": {"skill_id": skill.skill_id, "sensor_track": skill.sensor_track, "controller_type": skill.controller.get("type"), "navigation": skill.navigation_contract()},
        "compatibility": compatibility, "scheduler": {"selection_policy": SCHEDULER, "protocol_variant_id": settings["protocol_variant_id"], "breadth_first": True, "continuous_search": False},
        "catalogue": {"primitive_conditions": len(settings["terrain_codes"]) * len(settings["geometry_levels"]) * len(settings["friction_levels"]), "potential_ordered_pairs": len(settings["terrain_codes"]) * (len(settings["terrain_codes"]) - 1), "potential_pair_conditions": len(settings["terrain_codes"]) * (len(settings["terrain_codes"]) - 1) * len(settings["geometry_levels"]), "include_self_pairs": False, "include_rs8": "rs8" in settings["terrain_codes"]},
        "conditions": conditions, "condition_order": order, "trials": [], "next_trial_index": 0, "next_batch_index": 0, "active_batch": None,
        "materialized": {"L1": True, "L2": False, "L3": False, "L4": False, "L5": False, "L6": False, "L7": False},
        "level_results": {}, "counterfactuals": [], "resource_history": [],
    }
    if compatibility["status"] != "compatible":
        reasons = list(compatibility.get("navigation", {}).get("reasons", []))
        if compatibility.get("missing_sensor_kinds"):
            reasons.append("missing sensor kinds: %s" % ", ".join(compatibility["missing_sensor_kinds"]))
        state["block_reason"] = "; ".join(reasons) or "policy/backend contract is incompatible"
    state_path = root / "examination_state.json"
    dump_json(str(state_path), state)
    refresh_evidence(str(state_path), workers=_evidence_worker_count(state))
    return load_data(str(state_path))


def _flat_world(robot: RobotProfile, seed: int) -> WorldSpec:
    world = generate_terrain("rs1", robot, 0.0, seed, {"length_m": robot.body_length_m * 3.0})
    world.name = "FLAT-QUALIFICATION-s%d" % seed
    world.terrain_type = "qualification_flat"
    world.materials["low_friction"] = copy.deepcopy(world.materials["nominal"])
    for geometry in world.geometry:
        geometry.material = "nominal"
    world.metadata.update({"difficulty": 0.0, "qualification": True})
    world.validate()
    return world


def _build_world(condition: Dict[str, Any], robot: RobotProfile, settings: Dict[str, Any]) -> WorldSpec:
    kind = condition["kind"]
    seed = int(condition["world_seed"])
    if kind == "qualification":
        return _flat_world(robot, seed)
    if kind in {"primitive", "variant"}:
        world = generate_terrain(condition["terrain"], robot, condition["geometry_level"], seed)
        modifier = condition.get("modifier") or {}
        if modifier:
            world = apply_geometry_modifiers(
                world,
                float(modifier.get("camber_deg", 0.0)),
                float(modifier.get("curve_angle_deg", 0.0)),
                int(modifier.get("curve_segments", 24)),
            )
        return apply_challenge_friction(world, condition["static_friction"], condition["dynamic_friction"])
    if kind in {"pair", "counterfactual"}:
        world = generate_ae(robot, condition["pair"][0], condition["pair"][1], condition["geometry_level"], seed, challenge_codes=settings["terrain_codes"])
        return apply_challenge_friction(world, settings["fixed_pair_static_friction"], settings["fixed_pair_dynamic_friction"])
    world = _graph_world(
        robot, condition["topology"], condition["horizon"], condition["geometry_level"], seed, "cycle",
        condition["allowed_pairs"], challenge_codes=settings["terrain_codes"], include_self_pairs=False,
    )
    modifier = condition.get("modifier") or {}
    if modifier:
        # A real spatial sweep, not a visual tilt: the modifier segments long
        # boxes so collision geometry follows the arc. Recorded on the world as
        # `geometry_modifier` so the run is auditable.
        world = apply_geometry_modifiers(
            world,
            float(modifier.get("camber_deg", 0.0)),
            float(modifier.get("curve_angle_deg", 0.0)),
            int(modifier.get("curve_segments", 24)),
        )
    return apply_challenge_friction(world, settings["fixed_pair_static_friction"], settings["fixed_pair_dynamic_friction"])


def _metrics(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    return {item["name"]: float(item["value"]) for item in load_data(str(path)).get("metrics", []) if item.get("valid") and isinstance(item.get("value"), (int, float))}


def _enhance_diagnosis(diagnosis: Dict[str, Any], run_dir: Path, world: WorldSpec) -> Dict[str, Any]:
    value = copy.deepcopy(diagnosis)
    if value.get("classification") == "success" or value.get("failure_category") == "success":
        return value
    log_path = run_dir / "episode_log.json"
    if not log_path.is_file():
        return value
    log = load_data(str(log_path))
    events = log.get("events", [])
    trigger = next((item for item in events if item.get("event_type") in {"slip_start", "body_contact", "fall", "stuck"}), events[-1] if events else None)
    if not trigger:
        return value
    time_s = float(trigger.get("time_s", 0.0))
    samples = [item for item in log.get("samples", []) if time_s - 2.0 <= float(item.get("time_s", 0.0)) <= time_s + 2.0]
    stride = max(1, len(samples) // 41)
    window = []
    for item in samples[::stride]:
        roll, pitch, _ = quaternion_to_rpy(item.get("base_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))
        window.append({"time_s": item.get("time_s"), "base_position_m": item.get("base_position_m"), "base_speed_mps": sum(float(x) ** 2 for x in item.get("base_linear_velocity_mps", [0.0, 0.0])[:2]) ** 0.5, "roll_rad": roll, "pitch_rad": pitch})
    position = min(log.get("samples", []), key=lambda item: abs(float(item.get("time_s", 0.0)) - time_s)).get("base_position_m", [0.0, 0.0, 0.0]) if log.get("samples") else [0.0, 0.0, 0.0]
    region = next((item for item in world.regions if all(item.minimum[axis] <= position[axis] <= item.maximum[axis] for axis in range(3))), None)
    region_name = trigger.get("region") or (region.name if region else "")
    component = ""
    if region_name.startswith("purple"):
        component = "left/purple RS"
    elif region_name.startswith("brown"):
        component = "right/brown RS"
    elif "transition" in region_name:
        component = "RS8 transition"
    elif region_name:
        component = region_name
    value.update({"trigger": {"event": trigger.get("event_type"), "time_s": time_s, "region": region_name, "pair_component": component, "position_m": position}, "telemetry_window_s": [-2.0, 2.0], "telemetry_window": window})
    return value


def _prepare_trial(
    state: Dict[str, Any],
    condition: Dict[str, Any],
    robot: RobotProfile,
    skill: SkillPackage,
    attempt_offset: int = 0,
) -> Dict[str, Any]:
    """Persist one logical trial and return its resolved paths and identifiers."""
    settings = state["settings"]
    root = Path(settings["output_dir"])
    attempt_index = len(condition["attempts"]) + int(attempt_offset)
    seed = settings["seed"] + state["condition_order"].index(condition["id"]) * 1009 + attempt_index * 7919
    trial_id = "%s-a%02d" % (condition["id"], attempt_index + 1)
    world_path = root / "worlds" / (condition["id"] + ".json")
    episode_path = root / "episodes" / (trial_id + ".json")
    run_dir = root / "trials" / trial_id
    if not world_path.is_file():
        world = _build_world(condition, robot, settings)
        dump_json(str(world_path), world)
    else:
        world = WorldSpec.from_dict(load_data(str(world_path)))
    episode = {
        "episode_id": trial_id,
        "backend": settings["backend"],
        "robot_profile": settings["robot_profile"],
        "robot_asset": settings["robot_asset"],
        "robot_format": settings["robot_format"],
        "world": str(world_path),
        "controller": skill.controller_config(),
        "output_dir": str(run_dir),
        "seed": seed,
        # Scaled to the route this world actually requires. A single global
        # budget made every composed level impossible: 68 m in 20 s needs
        # 3.4 m/s. See episode_budget for the measured figures.
        "timeout_s": episode_timeout_s(
            world,
            reference_speed_mps=float(settings.get(
                "timeout_reference_speed_mps", DEFAULT_REFERENCE_SPEED_MPS)),
            slack=float(settings.get("timeout_slack", DEFAULT_SLACK)),
            minimum_s=float(settings.get("min_timeout_s", DEFAULT_MINIMUM_S)),
            maximum_s=float(settings.get("max_timeout_s", DEFAULT_MAXIMUM_S)),
        ),
        "headless": True,
        # The same setting the native batch path reads. Hardcoding False here
        # made record_video_all_trials a silently ignored option on every
        # simulator-adapter backend.
        "record_video": bool(settings.get("record_video_all_trials", False)),
        "event_thresholds": settings.get("event_thresholds", {}),
        "stressors": condition.get("stressors", []),
        "metadata": {"examiner_id": state["examiner_id"], "condition_id": condition["id"], "level": condition["level"]},
    }
    dump_json(str(episode_path), episode)
    return {
        "trial_id": trial_id,
        "attempt_index": attempt_index,
        "seed": seed,
        "world": world,
        "world_path": world_path,
        "episode_path": episode_path,
        "run_dir": run_dir,
    }


def _execute_condition(
    state: Dict[str, Any],
    condition: Dict[str, Any],
    robot: RobotProfile,
    skill: SkillPackage,
    attempt_index: Optional[int] = None,
    prepared: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    settings = state["settings"]
    root = Path(settings["output_dir"])
    if attempt_index is None:
        attempt_index = len(condition["attempts"])
    if prepared is None:
        prepared = _prepare_trial(
            state,
            condition,
            robot,
            skill,
            attempt_index - len(condition["attempts"]),
        )
    trial_id, seed = prepared["trial_id"], prepared["seed"]
    world, world_path = prepared["world"], prepared["world_path"]
    episode_path, run_dir = prepared["episode_path"], prepared["run_dir"]
    native_summary: Any = None
    if settings["backend"] == "foresight_native":
        from .examiner import _execute_foresight_native
        from .doctor import run_doctor

        preflight = run_doctor(settings["runtimes"])
        if preflight.get("gpu", {}).get("status") != "available":
            execution = {
                "status": "skipped", "success": None, "valid_behavior": False, "terminal_reason": "backend_unavailable",
                "diagnosis": {"status": "unavailable", "classification": "infrastructure", "terminal_reason": "backend_unavailable", "reason": preflight.get("gpu", {}).get("reason", "GPU unavailable"), "preflight": "gpu"},
            }
        else:
            previous = skill.controller.get("record_video_each_trial")
            skill.controller["record_video_each_trial"] = True
            try:
                execution = _execute_foresight_native(world_path, run_dir, seed, skill)
            finally:
                if previous is None:
                    skill.controller.pop("record_video_each_trial", None)
                else:
                    skill.controller["record_video_each_trial"] = previous
        status, valid, success, diagnosis = execution["status"], execution["valid_behavior"], execution["success"], execution["diagnosis"]
        native_summary = execution.get("native_summary")
        metric_values = {}
        if isinstance(native_summary, dict):
            goals = float(native_summary.get("num_goals", 0) or 0)
            metric_values["progress_fraction"] = min(1.0, float(native_summary.get("max_goal_index", 0) or 0) / max(1.0, goals - 1.0)) if goals else 0.0
            if success is True:
                metric_values["progress_fraction"] = 1.0
            for source_name, metric_name in (
                ("evaluation_duration_s", "completion_time"), ("aggregate_path_distance_m", "aggregate_path_distance_m"),
                ("mean_base_speed_mps", "mean_base_speed_mps"), ("max_base_speed_mps", "max_base_speed_mps"),
                ("max_abs_roll_rad", "max_abs_roll_rad"), ("max_abs_pitch_rad", "max_abs_pitch_rad"),
                ("telemetry_sample_count", "telemetry_sample_count"),
            ):
                if isinstance(native_summary.get(source_name), (int, float)):
                    metric_values[metric_name] = float(native_summary[source_name])
    else:
        manifest = execute_episode(str(episode_path), settings["runtimes"])
        status = manifest.status
        metric_values = _metrics(run_dir)
        success_value = metric_values.get("success")
        valid = manifest.status in {"completed", "failed"} and success_value is not None
        success = bool(success_value) if valid else None
        diagnosis = diagnose_run(str(run_dir))
    diagnosis = _enhance_diagnosis(diagnosis, run_dir, world)
    terminal_reason = diagnosis.get("terminal_reason")
    if not terminal_reason:
        terminal_reason = execution.get("terminal_reason", status) if settings["backend"] == "foresight_native" else status
    record = {
        "trial_id": trial_id, "condition_id": condition["id"], "level": condition["level"], "kind": condition["kind"], "attempt_index": attempt_index,
        "seed": seed, "status": status, "started_at": started_at, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "terminal_reason": terminal_reason,
        "success": success, "valid_behavior": valid, "world": str(world_path), "world_hash": sha256_file(str(world_path)), "episode": str(episode_path), "run_dir": str(run_dir),
        "metrics": metric_values, "diagnosis": diagnosis, "hashes": {"robot": state["hashes"]["robot_profile"], "controller": state["hashes"]["skill_manifest"], "world": sha256_file(str(world_path))}, "media": {},
    }
    if settings["backend"] == "foresight_native":
        for key in ("native_log", "native_result", "native_summary"):
            if key in execution:
                record[key] = execution[key]
        result = execution.get("native_summary")
        if isinstance(result, dict) and result.get("video"):
            record["media"]["video"] = {"path": str(Path(result["video"]).resolve().relative_to(root)), "sha256": result.get("video_sha256"), "source": "simulator"}
        if isinstance(result, dict) and result.get("trajectory") and Path(result["trajectory"]).is_file():
            record["telemetry"] = {"trajectory_json": str(Path(result["trajectory"]).resolve().relative_to(root))}
    return record


def _settle(condition: Dict[str, Any], state: Dict[str, Any]) -> None:
    by_id = {item["trial_id"]: item for item in state["trials"]}
    attempts = [by_id[value] for value in condition["attempts"]]
    valid = [item for item in attempts if item.get("valid_behavior") and isinstance(item.get("success"), bool)]
    infrastructure = [item for item in attempts if item not in valid]
    if attempts and attempts[-1] in infrastructure:
        last = attempts[-1]
        if last.get("status") == "skipped":
            condition["status"] = "unsupported"
            condition["unsupported_kind"] = (
                "environment"
                if last.get("diagnosis", {}).get("classification") == "environment_capability"
                else "simulator_integration"
            )
            condition["prune_reason"] = last.get("diagnosis", {}).get("reason", last.get("terminal_reason", "unsupported"))
            return
        if len(infrastructure) >= state["settings"]["infrastructure_retries"]:
            condition["status"] = "blocked"
            condition["prune_reason"] = "infrastructure failed after %d attempts" % len(infrastructure)
            return
    screening = state["settings"]["screening_repetitions"]
    confirmation = state["settings"]["confirmation_repetitions"]
    successes = sum(item["success"] for item in valid)
    if len(valid) < screening:
        condition["status"] = "running"
        return
    if condition["kind"] == "qualification":
        condition["status"] = "passed" if successes >= 2 else "failed"
        condition["evidence_grade"] = "qualification-3"
        return
    mixed = 0 < successes < screening
    requires_confirmation = mixed or (condition.get("critical") and successes == screening)
    if len(valid) == screening and requires_confirmation:
        condition["target_repetitions"] = confirmation
        condition["status"] = "confirming"
        return
    if len(valid) < condition["target_repetitions"]:
        condition["status"] = "confirming"
        return
    if condition["target_repetitions"] == confirmation:
        condition["status"] = "passed" if successes >= state["settings"]["confirmation_successes"] else "failed"
        condition["evidence_grade"] = "confirmed-%d" % confirmation
    else:
        condition["status"] = "passed" if successes == screening else "failed"
        condition["evidence_grade"] = "screening-%d" % screening
    lower, upper = wilson_interval(successes, len(valid))
    condition["summary"] = {"valid": len(valid), "successes": successes, "failures": len(valid) - successes, "success_rate": successes / len(valid), "lower_ci": lower, "upper_ci": upper}


def _materialize_l1_variants(state: Dict[str, Any]) -> None:
    """Banked/curved copies of every plain single terrain that passed.

    Diagnostic only: pairs and later levels keep building from the plain
    primitives, so a variant verdict never changes what L2 is assembled from.
    """

    settings = state["settings"]
    index = len(state["condition_order"])
    for identifier in list(state["condition_order"]):
        parent = state["conditions"][identifier]
        if parent["level"] != "L1" or parent["kind"] != "primitive" or parent["status"] != "passed":
            continue
        for name, modifier in L1_VARIANTS:
            variant_id = "%s-%s" % (parent["id"], name)
            if variant_id in state["conditions"]:
                continue
            item = _condition(
                variant_id, "L1", "variant", settings["seed"] + index,
                terrain=parent["terrain"], geometry_label=parent["geometry_label"],
                geometry_level=parent["geometry_level"], friction_label=parent.get("friction_label"),
                static_friction=parent["static_friction"], dynamic_friction=parent["dynamic_friction"],
                modifier=dict(modifier), parents=[parent["id"]], critical=False,
                target_repetitions=settings["screening_repetitions"],
            )
            state["conditions"][variant_id] = item
            state["condition_order"].append(variant_id)
            index += 1
    state["materialized"]["L1-variants"] = True


def _materialize_pairs(state: Dict[str, Any]) -> None:
    settings = state["settings"]
    qualification_label = settings.get("qualification_friction_label", "moderate")
    moderate: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for item in state["conditions"].values():
        if item["kind"] == "primitive" and item.get("friction_label") == qualification_label:
            moderate[(item["terrain"], float(item["geometry_level"]))] = item
    index = len(state["condition_order"])
    for geometry_label, geometry_level in _geometry_levels(settings):
        for left, right in generate_ae_pairs(settings["terrain_codes"], include_self=False):
            identifier = "L2-%s-to-%s-g%03d" % (left, right, round(100 * geometry_level))
            parents = [moderate[(left, geometry_level)], moderate[(right, geometry_level)]]
            item = _condition(identifier, "L2", "pair", settings["seed"] + index, pair=[left, right], geometry_label=geometry_label, geometry_level=geometry_level, static_friction=settings["fixed_pair_static_friction"], dynamic_friction=settings["fixed_pair_dynamic_friction"], critical=True, parents=[value["id"] for value in parents], target_repetitions=settings["screening_repetitions"])
            if any(parent["status"] != "passed" for parent in parents):
                item["status"] = "pruned"
                item["pruned_by"] = [parent["id"] for parent in parents if parent["status"] != "passed"]
                item["prune_reason"] = "one or both single-terrain geometry levels failed the configured qualification-friction gate"
            state["conditions"][identifier] = item
            state["condition_order"].append(identifier)
            index += 1
    state["materialized"]["L2"] = True


def _stage_outcome(state: Dict[str, Any], level: str) -> str:
    members = [item for item in state["conditions"].values() if item["level"] == level and item["kind"] in {"progression", "counterfactual"}]
    active = [item for item in members if item["status"] not in {"pruned", "unsupported", "blocked"}]
    if any(item["status"] not in TERMINAL for item in members):
        return "running"
    if not active:
        return "pruned"
    return "passed" if sum(item["status"] == "passed" for item in active) / len(active) >= 0.70 else "failed"


def _passed_pair_library(state: Dict[str, Any]) -> Dict[float, List[List[str]]]:
    removed = {tuple(item["pair"]) for item in state["conditions"].values() if item["kind"] == "counterfactual" and item["status"] == "failed"}
    result: Dict[float, List[List[str]]] = {}
    for item in state["conditions"].values():
        if item["kind"] == "pair" and item["status"] == "passed" and tuple(item["pair"]) not in removed:
            result.setdefault(float(item["geometry_level"]), []).append(list(item["pair"]))
    return result


def _materialize_progression_level(state: Dict[str, Any], level: str, robot: RobotProfile) -> None:
    settings = state["settings"]
    library = _passed_pair_library(state)
    templates = [item for item in PROGRESSION if item[0] == level]
    index = len(state["condition_order"])
    for template in templates:
        _, name, topology, horizon, stressors = template[:5]
        modifier = progression_modifier(template)
        for geometry_label, geometry_level in _geometry_levels(settings):
            pairs = library.get(geometry_level, [])
            if not pairs:
                continue
            probe = _graph_world(robot, topology, horizon, geometry_level, settings["seed"] + index, "cycle", pairs, settings["terrain_codes"], False)
            capacity = max(1, len(probe.metadata.get("graph_edges", [])))
            for chunk_index in range(0, len(pairs), capacity):
                chunk = pairs[chunk_index : chunk_index + capacity]
                identifier = "%s-%s-g%03d-b%03d" % (level, name, round(100 * geometry_level), chunk_index // capacity)
                item = _condition(identifier, level, "progression", settings["seed"] + index, topology=topology, horizon=horizon, geometry_label=geometry_label, geometry_level=geometry_level, allowed_pairs=chunk, covered_pairs=chunk, stressors=stressors, modifier=modifier, critical=True, locked=level == "L7", target_repetitions=settings["screening_repetitions"])
                state["conditions"][identifier] = item
                state["condition_order"].append(identifier)
                index += 1
    state["materialized"][level] = True


def _materialize_counterfactual(state: Dict[str, Any], failed: Dict[str, Any]) -> None:
    if failed["kind"] != "progression" or failed.get("counterfactual_created"):
        return
    failed["counterfactual_created"] = True
    pair = list(failed.get("covered_pairs", [])[0]) if failed.get("covered_pairs") else None
    if not pair:
        return
    identifier = "%s-cf-%s-to-%s" % (failed["id"], pair[0], pair[1])
    item = _condition(identifier, failed["level"], "counterfactual", state["settings"]["seed"] + len(state["condition_order"]), pair=pair, geometry_label=failed["geometry_label"], geometry_level=failed["geometry_level"], static_friction=state["settings"]["fixed_pair_static_friction"], dynamic_friction=state["settings"]["fixed_pair_dynamic_friction"], critical=True, parent_world=failed["id"], target_repetitions=state["settings"]["screening_repetitions"])
    state["conditions"][identifier] = item
    insert_at = state["condition_order"].index(failed["id"]) + 1
    state["condition_order"].insert(insert_at, identifier)
    state["counterfactuals"].append(identifier)


def _next_condition(state: Dict[str, Any], robot: RobotProfile) -> Optional[Dict[str, Any]]:
    qualification = state["conditions"]["L0-flat"]
    if qualification["status"] not in TERMINAL:
        state["stage"] = "L0-qualification"
        return qualification
    if qualification["status"] != "passed":
        state["status"] = "stopped" if qualification["status"] == "failed" else "blocked"
        state["stage"] = "qualification_failed" if qualification["status"] == "failed" else "qualification_blocked"
        return None
    l1 = [state["conditions"][value] for value in state["condition_order"] if state["conditions"][value]["level"] == "L1"]
    for item in l1:
        if item["kind"] == "primitive" and item["status"] not in TERMINAL:
            state["stage"] = "L1-single-terrains"
            return item
    if not state["materialized"].get("L1-variants"):
        _materialize_l1_variants(state)
    for identifier in state["condition_order"]:
        item = state["conditions"][identifier]
        if item["level"] == "L1" and item["kind"] == "variant" and item["status"] not in TERMINAL:
            state["stage"] = "L1-camber-curvature"
            return item
    if not state["materialized"]["L2"]:
        _materialize_pairs(state)
    for identifier in state["condition_order"]:
        item = state["conditions"][identifier]
        if item["level"] == "L2" and item["status"] not in TERMINAL:
            state["stage"] = "L2-ordered-pairs"
            return item
    if not _passed_pair_library(state):
        for level in ("L3", "L4", "L5", "L6", "L7"):
            state["materialized"][level] = True
            state["level_results"][level] = "pruned"
        state["status"], state["stage"] = "completed", "complete"
        return None
    previous = "L2"
    for level in ("L3", "L4", "L5", "L6", "L7"):
        # No aggregate stage gate. A level used to be cancelled outright when
        # the previous one scored below 70%, which deleted four levels of the
        # benchmark over one 53.8% result -- including the combinations the
        # policy had already proved it could handle.
        #
        # Pruning stays element-wise instead: _passed_pair_library builds each
        # level only from pairs that PASSED, and drops any pair whose
        # counterfactual later failed. If nothing passed, the library is empty
        # and the level comes out empty on its own -- absent by evidence rather
        # than by threshold.
        if not state["materialized"][level]:
            _materialize_progression_level(state, level, robot)
        for identifier in list(state["condition_order"]):
            item = state["conditions"][identifier]
            if item["level"] == level and item["status"] not in TERMINAL:
                state["stage"] = "%s-composed" % level
                return item
        outcome = _stage_outcome(state, level)
        state["level_results"][level] = outcome
        previous = level
    state["status"], state["stage"] = "completed", "complete"
    return None


def plan_ready_trials(
    state: Dict[str, Any],
    workers: int,
    in_flight: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Plan attempts from the earliest unlocked level without crossing gates.

    ``in_flight`` maps a condition id to the attempt indices already dispatched
    but not yet recorded. Without it a worker refilled mid-flight would re-plan
    an attempt that is currently running, giving two trials the same seed and
    overshooting the repetition target.
    """

    capacity = resolve_worker_count(workers)
    running = {key: set(value) for key, value in (in_flight or {}).items()}
    conditions = state.get("conditions", {})
    order = [identifier for identifier in state.get("condition_order", []) if identifier in conditions]
    order.extend(identifier for identifier in conditions if identifier not in order)

    qualifications = [conditions[identifier] for identifier in order if conditions[identifier].get("level") == "L0"]
    if qualifications:
        unfinished = [condition for condition in qualifications if condition.get("status") not in TERMINAL]
        if unfinished:
            candidates = unfinished
        elif any(condition.get("status") != "passed" for condition in qualifications):
            return []
        else:
            candidates = []
    else:
        candidates = []

    if not candidates:
        active = [conditions[identifier] for identifier in order if conditions[identifier].get("status") not in TERMINAL]
        if not active:
            return []
        level = active[0].get("level")
        candidates = [condition for condition in active if condition.get("level") == level]

    plan: List[Dict[str, Any]] = []
    for condition in candidates:
        parents = [conditions.get(identifier) for identifier in condition.get("parents", [])]
        if any(parent is None or parent.get("status") not in TERMINAL for parent in parents):
            continue
        if any(parent.get("status") != "passed" for parent in parents):
            continue
        attempt_ids = condition.get("attempts", [])
        attempt_count = len(attempt_ids)
        by_id = {item["trial_id"]: item for item in state.get("trials", [])}
        attempts = [by_id[value] for value in attempt_ids if value in by_id]
        valid_count = sum(1 for item in attempts if item.get("valid_behavior") and isinstance(item.get("success"), bool))
        
        active = running.get(condition["id"], set())
        target = int(condition.get("target_repetitions", 0))
        remaining = max(0, target - valid_count - len(active))
        attempt_index = attempt_count
        while remaining > 0:
            if attempt_index in active:
                attempt_index += 1
                continue
            plan.append({"condition_id": condition["id"], "attempt_index": attempt_index})
            attempt_index += 1
            remaining -= 1
            if len(plan) >= capacity:
                return plan
    return plan


def _attach_matched_evidence(state: Dict[str, Any]) -> None:
    conditions = state["conditions"]
    trials = state["trials"]
    passes = [item for item in trials if item.get("success") is True]
    for failure in [item for item in trials if item.get("success") is False]:
        source = conditions[failure["condition_id"]]
        candidates = []
        for candidate in passes:
            other = conditions[candidate["condition_id"]]
            same_family = source.get("terrain") == other.get("terrain") if source.get("terrain") else source.get("pair") == other.get("pair")
            if same_family:
                distance = abs(float(source.get("geometry_level", 0.0)) - float(other.get("geometry_level", 0.0))) + abs(float(source.get("static_friction", 0.5)) - float(other.get("static_friction", 0.5)))
                candidates.append((candidate.get("seed") != failure.get("seed"), distance, candidate))
        if candidates:
            matched = min(candidates, key=lambda item: (item[0], item[1]))[2]
            failure["diagnosis"]["matched_success"] = {"trial_id": matched["trial_id"], "condition_id": matched["condition_id"], "same_seed": matched.get("seed") == failure.get("seed"), "interpretation": "nearest passing condition in the same terrain/pair family"}


def _valid_attempt_count(state: Dict[str, Any], condition: Dict[str, Any]) -> int:
    by_id = {item["trial_id"]: item for item in state["trials"]}
    return sum(
        bool(by_id[trial_id].get("valid_behavior")) and isinstance(by_id[trial_id].get("success"), bool)
        for trial_id in condition["attempts"]
        if trial_id in by_id
    )


def _available_ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _resource_capacity(state: Dict[str, Any], skill: SkillPackage, requested: int) -> Tuple[int, Dict[str, Any]]:
    """Choose a conservative batch size from live GPU, RAM, and disk budgets."""
    settings = state["settings"]
    root = Path(settings["output_dir"])
    disk = shutil.disk_usage(root)
    scratch_root = Path(str(skill.controller.get("isaac_scratch_root", "/tmp"))).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_disk = shutil.disk_usage(scratch_root)
    ram_available = _available_ram_bytes()
    gpu_index = int(skill.controller.get("gpu_index", 1))
    gpu_free_mb = 0
    gpu_error = ""
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10.0, check=True,
        )
        gpu_free_mb = int(probe.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        gpu_error = str(error)
    gib = 1024 ** 3
    gpu_capacity = max(0, (gpu_free_mb - int(settings["gpu_reserve_mb"])) // int(settings["gpu_per_env_mb"])) if gpu_free_mb else 1
    ram_budget_mb = max(0.0, ram_available / 1024 ** 2 - float(settings["ram_reserve_gb"]) * 1024)
    ram_capacity = int(ram_budget_mb // int(settings["ram_per_env_mb"])) if ram_available else 1
    disk_budget_mb = max(0.0, disk.free / 1024 ** 2 - float(settings["disk_reserve_gb"]) * 1024)
    disk_capacity = int(disk_budget_mb // float(settings["disk_per_trial_mb"]))
    scratch_free_after_reserve_mb = scratch_disk.free / 1024 ** 2 - float(settings.get("isaac_scratch_reserve_mb", 512.0))
    if scratch_free_after_reserve_mb < 0:
        disk_capacity = 0
    capacity = min(int(requested), int(settings["native_batch_size"]), gpu_capacity, ram_capacity, disk_capacity)
    if settings.get("record_video_all_trials", False):
        # The source recorder captures one selected vector environment. Serial
        # execution is therefore required to retain an honest video per trial.
        capacity = min(capacity, 1)
    evidence = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu_index": gpu_index, "gpu_free_mb": gpu_free_mb,
        "gpu_probe_error": gpu_error, "ram_available_gb": ram_available / gib, "disk_free_gb": disk.free / gib,
        "isaac_scratch_root": str(scratch_root), "isaac_scratch_free_gb": scratch_disk.free / gib,
        "configured_cap": int(settings["native_batch_size"]), "requested": int(requested),
        "limits": {"gpu": gpu_capacity, "ram": ram_capacity, "disk": disk_capacity}, "selected": max(0, capacity),
        "reserves": {"gpu_mb": settings["gpu_reserve_mb"], "ram_gb": settings["ram_reserve_gb"], "disk_gb": settings["disk_reserve_gb"], "isaac_scratch_mb": settings.get("isaac_scratch_reserve_mb", 512.0)},
    }
    state.setdefault("resource_history", []).append(evidence)
    state["latest_resource_decision"] = evidence
    return max(0, capacity), evidence


def _plan_native_batch(
    state: Dict[str, Any], robot: RobotProfile, skill: SkillPackage, capacity: int
) -> List[Dict[str, Any]]:
    """Plan only independent attempts from the currently unlocked level."""
    first = _next_condition(state, robot)
    if first is None or capacity <= 0:
        return []
    level = first["level"]
    candidates = [
        state["conditions"][identifier]
        for identifier in state["condition_order"]
        if state["conditions"][identifier]["level"] == level
        and state["conditions"][identifier]["status"] not in TERMINAL
    ]
    plan: List[Dict[str, Any]] = []
    batch_steps: Optional[int] = None
    # At most one attempt per condition is allowed in a vector batch. This
    # makes repetitions occur in separate Isaac launches with distinct global
    # RNG seeds instead of pretending env indices are independent seeds.
    for condition in candidates:
        needed = int(condition["target_repetitions"]) - _valid_attempt_count(state, condition)
        if needed <= 0:
            continue
        prepared = _prepare_trial(state, condition, robot, skill, 0)
        waypoints = prepared["world"].routes[0].waypoints if prepared["world"].routes else []
        route_length = sum(
            sum((float(right[axis]) - float(left[axis])) ** 2 for axis in range(3)) ** 0.5
            for left, right in zip(waypoints, waypoints[1:])
        )
        control_hz = max(1.0, float(skill.control_frequency_hz))
        route_steps = int(route_length / 0.5 * control_hz * 1.5 + 0.999)
        prepared["steps"] = max(int(skill.controller.get("evaluation_steps", 1000)), route_steps)
        if batch_steps is None:
            batch_steps = prepared["steps"]
        if prepared["steps"] != batch_steps:
            continue
        plan.append({"condition_id": condition["id"], **prepared})
        if len(plan) >= capacity:
            break
    return plan


def _batch_item_json(item: Dict[str, Any], skill: SkillPackage, state: Dict[str, Any]) -> Dict[str, Any]:
    controller = skill.controller
    run_dir = Path(item["run_dir"])
    profile = RobotProfile.from_dict(load_data(state["settings"]["robot_profile"]))
    condition = state["conditions"][item["condition_id"]]
    return {
        "trial_id": item["trial_id"],
        "world": str(item["world_path"]),
        "bridge_manifest": str(run_dir / "native_bridge_manifest.json"),
        "summary_json": str(run_dir / "summary.json"),
        "video_dir": str(run_dir / "video"),
        "checkpoint": str(controller.get("checkpoint") or controller.get("path")),
        "task": str(controller.get("task")),
        "seed": int(item["seed"]),
        "steps": int(item.get("steps", controller.get("evaluation_steps", 1000))),
        "record_video": bool(state["settings"].get("record_video_all_trials", False)),
        "capture_telemetry": bool(state["settings"].get("capture_native_telemetry", True)),
        "telemetry_stride": int(state["settings"].get("native_telemetry_stride", 10)),
        "control_frequency_hz": float(skill.control_frequency_hz),
        "stressors": condition.get("stressors", []),
        "robot_mass_kg": float(profile.total_mass_kg),
    }


def _native_batch_record(
    state: Dict[str, Any], planned: Dict[str, Any], runner_error: Optional[str] = None
) -> Dict[str, Any]:
    root = Path(state["settings"]["output_dir"])
    condition = state["conditions"][planned["condition_id"]]
    run_dir = Path(planned["run_dir"])
    result_path = run_dir / "result.json"
    native: Dict[str, Any] = {}
    if runner_error:
        status, valid, success, terminal = "error", False, None, "native_batch_error"
        diagnosis = {"status": "unavailable", "classification": "infrastructure", "terminal_reason": terminal, "reason": runner_error}
    elif not result_path.is_file():
        status, valid, success, terminal = "error", False, None, "native_batch_result_missing"
        diagnosis = {"status": "unavailable", "classification": "infrastructure", "terminal_reason": terminal, "reason": "batch completed without result.json"}
    else:
        native = load_data(str(result_path))
        fidelity = native.get("material_fidelity", {})
        environment_support = native.get("environment_physics", {}).get("support", {})
        if int(native.get("world_switch_violations", 0) or 0) > 0:
            status, valid, success, terminal = "error", False, None, "world_assignment_drift"
            diagnosis = {
                "status": "unavailable", "classification": "infrastructure", "terminal_reason": terminal,
                "reason": "vector environment changed benchmark world during episode reset",
                "world_switch_violations": native.get("world_switch_violations"),
            }
        elif environment_support.get("status") == "environment_unsupported":
            status, valid, success, terminal = "skipped", False, None, "environment_physics_unsupported"
            diagnosis = {
                "status": "unavailable", "classification": "environment_capability", "terminal_reason": terminal,
                "reason": environment_support.get("reason", "simulator adapter cannot reproduce the requested environment physics"),
                "environment_support": environment_support, "policy_was_evaluated": False,
            }
        elif not fidelity.get("capability_claim_eligible", False):
            status, valid, success, terminal = "skipped", False, None, "physics_material_fidelity_unverified"
            diagnosis = {
                "status": "unavailable", "classification": "infrastructure", "terminal_reason": terminal,
                "reason": "simulator adapter did not verify the requested per-region material binding",
                "material_fidelity": fidelity, "policy_was_evaluated": False,
            }
        else:
            status, valid, success = "completed", True, bool(native.get("success"))
            terminal = "goal" if success else "route_incomplete"
            diagnosis = {
                "status": "limited", "classification": "success" if success else "behavioral/progress", "terminal_reason": terminal,
                "event_chain": ["goal reached"] if success else ["final native goal was not reached within the evaluation window"],
                "likely_factors": [], "claim_scope": "native summary has route progress and terminations but not canonical per-foot telemetry",
            }
    goals = float(native.get("num_goals", 0) or 0)
    progress = min(1.0, float(native.get("max_goal_index", 0) or 0) / max(1.0, goals - 1.0)) if goals else 0.0
    if success is True:
        progress = 1.0
    metrics = {"progress_fraction": progress}
    for source_name, metric_name in (
        ("evaluation_duration_s", "completion_time"),
        ("aggregate_path_distance_m", "aggregate_path_distance_m"),
        ("mean_base_speed_mps", "mean_base_speed_mps"),
        ("max_base_speed_mps", "max_base_speed_mps"),
        ("max_abs_roll_rad", "max_abs_roll_rad"),
        ("max_abs_pitch_rad", "max_abs_pitch_rad"),
        ("telemetry_sample_count", "telemetry_sample_count"),
    ):
        if isinstance(native.get(source_name), (int, float)):
            metrics[metric_name] = float(native[source_name])
    world_path = Path(planned["world_path"])
    record = {
        "trial_id": planned["trial_id"], "condition_id": condition["id"], "level": condition["level"], "kind": condition["kind"],
        "attempt_index": int(planned.get("attempt_index", len(condition.get("attempts", [])))),
        "started_at": str(planned.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat()),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": int(native.get("seed", planned["seed"])), "requested_seed": int(planned["seed"]),
        "seed_scope": native.get("effective_seed_scope", "single native launch"),
        "status": status, "terminal_reason": terminal, "success": success, "valid_behavior": valid,
        "world": str(world_path), "world_hash": sha256_file(str(world_path)), "episode": str(planned["episode_path"]), "run_dir": str(run_dir),
        "metrics": metrics, "diagnosis": diagnosis,
        "hashes": {"robot": state["hashes"]["robot_profile"], "controller": state["hashes"]["skill_manifest"], "world": sha256_file(str(world_path))},
        "media": {}, "native_result": str(result_path), "native_summary": native,
        "native_log": str(Path(state["settings"]["output_dir"]) / "native_batches" / state.get("last_batch_id", "") / "runner.log"),
    }
    if native.get("video"):
        video = Path(native["video"])
        if video.is_file():
            record["media"]["video"] = {"path": str(video.resolve().relative_to(root)), "sha256": native.get("video_sha256"), "source": "simulator"}
    trajectory = native.get("trajectory")
    if trajectory and Path(trajectory).is_file():
        record["telemetry"] = {"trajectory_json": str(Path(trajectory).resolve().relative_to(root))}
    return record


def _persist_trial_record(state: Dict[str, Any], record: Dict[str, Any]) -> None:
    if any(item["trial_id"] == record["trial_id"] for item in state["trials"]):
        return
    condition = state["conditions"][record["condition_id"]]
    state["trials"].append(record)
    condition["attempts"].append(record["trial_id"])
    state["next_trial_index"] += 1
    _settle(condition, state)
    if condition["status"] == "failed":
        _materialize_counterfactual(state, condition)


def _execute_native_batch(
    state: Dict[str, Any], state_file: Path, robot: RobotProfile, skill: SkillPackage, capacity: int
) -> int:
    """Execute and import one crash-auditable parallel Isaac environment batch."""
    plan = _plan_native_batch(state, robot, skill, capacity)
    if not plan:
        return 0
    controller = skill.controller
    if not (controller.get("checkpoint") or controller.get("path")) or not controller.get("task"):
        raise ValueError("native controller needs checkpoint and task")
    batch_index = int(state.get("next_batch_index", 0))
    batch_id = "batch-%05d" % batch_index
    root = Path(state["settings"]["output_dir"])
    batch_dir = root / "native_batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = {"schema_version": "1.0", "batch_id": batch_id, "items": [_batch_item_json(item, skill, state) for item in plan]}
    dump_json(str(manifest_path), manifest)
    serial_plan = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in item.items()
            if key != "world"
        }
        for item in plan
    ]
    state["active_batch"] = {"batch_id": batch_id, "manifest": str(manifest_path), "plan": serial_plan, "status": "planned"}
    state["status"] = "running"
    state["next_batch_index"] = batch_index + 1
    state["last_batch_id"] = batch_id
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump_json(str(state_file), state)
    launcher = Path(__file__).resolve().parents[2] / "scripts" / "run_foresight_batch.sh"
    environment = os.environ.copy()
    environment["GPU_INDEX"] = str(controller.get("gpu_index", 1))
    # Physics materials belong to the backend and must not depend on a policy
    # manifest flag.
    environment["PER_REGION_MATERIALS"] = "1"
    environment["ROBUSTLAB_ISAAC_SCRATCH_ROOT"] = str(controller.get("isaac_scratch_root", "/tmp"))
    environment["KEEP_ISAAC_TEMP"] = "1" if controller.get("keep_isaac_temp", False) else "0"
    if controller.get("source_root"):
        environment["FORESIGHT_ROOT"] = str(controller["source_root"])
    log_path = batch_dir / "runner.log"
    runner_error: Optional[str] = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(["bash", str(launcher), str(manifest_path)], cwd=str(Path(__file__).resolve().parents[2]), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=float(controller.get("wall_timeout_s", 1800.0)))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
            runner_error = "native batch exceeded wall timeout"
        if process.returncode and not runner_error:
            runner_error = "native batch runner returned %d; see %s" % (process.returncode, log_path)
    state["active_batch"]["status"] = "finished" if not runner_error else "error"
    state["active_batch"]["wall_time_s"] = time.monotonic() - started
    for planned in serial_plan:
        record = _native_batch_record(state, planned, runner_error)
        _persist_trial_record(state, record)
        _attach_matched_evidence(state)
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump_json(str(state_file), state)
        refresh_evidence(str(state_file), workers=_evidence_worker_count(state))
    state["active_batch"] = None
    dump_json(str(state_file), state)
    return len(serial_plan)


def run_discrete_examination(state_path: str, max_trials: Optional[int] = None) -> Dict[str, Any]:
    state_file = Path(state_path).resolve()
    state = load_data(str(state_file))
    if not is_discrete_state(state):
        raise ValueError("state is not a discrete_pruned_v2 examination")
    settings = state["settings"]
    settings.setdefault("native_batch_size", 32)
    settings.setdefault("isaac_scratch_reserve_mb", 512.0)
    settings.setdefault("parallel_trials", DEFAULT_PARALLEL_TRIALS)
    state.setdefault("next_batch_index", 0)
    state.setdefault("active_batch", None)
    state.setdefault("last_batch_id", "")
    if state.get("status") == "blocked" and state.get("stage") == "qualification_blocked":
        last = state.get("trials", [])[-1] if state.get("trials") else {}
        if last.get("terminal_reason") == "backend_unavailable":
            from .doctor import run_doctor

            if run_doctor(settings["runtimes"]).get("gpu", {}).get("status") == "available":
                state["conditions"]["L0-flat"]["status"] = "pending"
                state["status"], state["stage"] = "ready", "L0-qualification"
    robot = RobotProfile.from_dict(load_data(settings["robot_profile"]))
    skill = load_skill(settings["skill"])
    if settings["backend"] == "foresight_native":
        scratch_cleanup = cleanup_stale_isaac_scratch(str(skill.controller.get("isaac_scratch_root", "/tmp")), stale_after_s=0.0)
        if scratch_cleanup.get("removed_count") or scratch_cleanup.get("legacy"):
            state.setdefault("scratch_cleanup_history", []).append({"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), **scratch_cleanup})
            state["scratch_cleanup_history"] = state["scratch_cleanup_history"][-20:]
        if state.get("status") == "blocked" and state.get("stage") == "resource_blocked":
            recovered_capacity, _ = _resource_capacity(state, skill, 1)
            if recovered_capacity >= 1:
                state["status"], state["stage"] = "ready", "resource_recovered"
                state.pop("block_reason", None)
        dump_json(str(state_file), state)
    limit = max_trials if max_trials is not None else settings.get("max_trials", 0)
    executed = 0
    # If the coordinator was interrupted after Isaac finished, import its
    # immutable result files. Otherwise record the interrupted launch as
    # infrastructure evidence and let the normal retry rule decide what follows.
    if settings["backend"] == "foresight_native" and state.get("active_batch"):
        active = state["active_batch"]
        status_path = Path(active["manifest"]).parent / "batch_status.json"
        completed = status_path.is_file() and load_data(str(status_path)).get("status") == "completed"
        error = None if completed else "previous native batch was interrupted before a completed batch status was persisted"
        state["last_batch_id"] = active["batch_id"]
        for planned in active.get("plan", []):
            record = _native_batch_record(state, planned, error)
            _persist_trial_record(state, record)
            executed += 1
        state["active_batch"] = None
        dump_json(str(state_file), state)
        refresh_evidence(str(state_file), workers=_evidence_worker_count(state))
    while state["status"] not in {"completed", "stopped", "blocked"}:
        if limit and executed >= limit:
            break
        if settings["backend"] == "foresight_native":
            requested = int(settings["native_batch_size"])
            if limit:
                requested = min(requested, limit - executed)
            capacity, resource_evidence = _resource_capacity(state, skill, requested)
            if capacity < 1:
                state["status"], state["stage"] = "blocked", "resource_blocked"
                state["block_reason"] = "insufficient GPU, RAM, or disk budget for one native environment"
                dump_json(str(state_file), state)
                refresh_evidence(str(state_file), workers=_evidence_worker_count(state))
                break
            count = _execute_native_batch(state, state_file, robot, skill, capacity)
            if not count:
                break
            state = load_data(str(state_file))
            executed += count
            continue
        condition = _next_condition(state, robot)
        if condition is None:
            break
        requested = resolve_worker_count(settings["parallel_trials"])
        state["status"] = "running"
        # Rolling dispatch: refill a worker the moment it frees, rather than
        # waiting for a whole batch. A batch ran at the pace of its slowest
        # trial and drained the pool to idle at every tail; measured on a
        # 60-trial sweep at 24 workers, the pool was idle 65% of the run.
        #
        # The only barrier kept is the level boundary, which the schedule
        # genuinely needs: promotion to confirmation and the pruning of L2 by
        # L1 both depend on the previous level's verdicts.
        in_flight: Dict[str, Any] = {}
        pending: Dict[Any, Dict[str, Any]] = {}
        completed = 0
        with ThreadPoolExecutor(max_workers=requested) as executor:
            while True:
                room = requested - len(pending)
                if limit:
                    # `executed` only advances once this level drains, so the
                    # trials already completed inside this loop must be counted
                    # here too or the limit never binds.
                    room = min(room, limit - executed - completed - len(pending))
                plan = plan_ready_trials(state, room, in_flight) if room > 0 else []
                for item in plan:
                    planned_condition = state["conditions"][item["condition_id"]]
                    prepared = _prepare_trial(
                        state,
                        planned_condition,
                        robot,
                        skill,
                        item["attempt_index"] - len(planned_condition["attempts"]),
                    )
                    future = executor.submit(
                        _execute_condition,
                        state,
                        planned_condition,
                        robot,
                        skill,
                        item["attempt_index"],
                        prepared,
                    )
                    pending[future] = item
                    in_flight.setdefault(item["condition_id"], set()).add(
                        item["attempt_index"]
                    )
                if not pending:
                    break
                done, _ = futures_wait(
                    list(pending), return_when=FIRST_COMPLETED
                )
                finished = []
                for future in done:
                    item = pending.pop(future)
                    in_flight[item["condition_id"]].discard(item["attempt_index"])
                    if not in_flight[item["condition_id"]]:
                        in_flight.pop(item["condition_id"], None)
                    finished.append(future.result())
                order_index = {
                    identifier: index
                    for index, identifier in enumerate(state["condition_order"])
                }
                finished.sort(
                    key=lambda record: (
                        order_index.get(record["condition_id"], 0),
                        int(record["attempt_index"]),
                    )
                )
                for record in finished:
                    _persist_trial_record(state, record)
                    _attach_matched_evidence(state)
                state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                dump_json(str(state_file), state)
                completed += len(finished)
        if not completed:
            break
        # Evidence rendering is serial work with nothing in flight, so it runs
        # once per level rather than once per batch.
        refresh_evidence(str(state_file), workers=_evidence_worker_count(state))
        state = load_data(str(state_file))
        executed += completed
    # Materialize terminal/pruned descendants even when the last trial closed a stage.
    if state["status"] not in {"stopped", "blocked"}:
        _next_condition(state, robot)
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump_json(str(state_file), state)
    dashboard = refresh_evidence(str(state_file), workers=_evidence_worker_count(state))
    return {"status": state["status"], "stage": state["stage"], "executed": executed, "total_trials": len(state["trials"]), "conditions": len(state["conditions"]), "state": str(state_file), "dashboard": dashboard["dashboard"]}


def discrete_status(state_path: str) -> Dict[str, Any]:
    state = load_data(state_path)
    conditions = list(state.get("conditions", {}).values())
    return {
        "examiner_id": state["examiner_id"], "status": state["status"], "stage": state["stage"], "trial_count": len(state.get("trials", [])),
        "condition_count": len(conditions), "passed": sum(item["status"] == "passed" for item in conditions), "failed": sum(item["status"] == "failed" for item in conditions),
        "pruned": sum(item["status"] == "pruned" for item in conditions), "unsupported": sum(item["status"] == "unsupported" for item in conditions),
        "dashboard": str(Path(state_path).resolve().parent / "dashboard" / "index.html"),
    }


def diagnose_discrete(state_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    state = load_data(state_path)
    records = [item for item in state.get("trials", []) if item.get("diagnosis")]
    counterfactuals = []
    for item in state.get("conditions", {}).values():
        if item["kind"] == "counterfactual" and item.get("attempts"):
            counterfactuals.append({"classification": "behavioral/composition", "terrain": item.get("pair"), "control_run": item["attempts"], "failed_run": item.get("parent_world"), "interpretation": "isolated pair replay after a composed-world failure"})
    result = cluster_diagnoses(records, counterfactuals)
    destination = output_path or str(Path(state_path).resolve().parent / "failure_diagnosis.json")
    dump_json(destination, result)
    result["output"] = str(Path(destination).resolve())
    return result


def report_discrete(state_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    state = load_data(state_path)
    result = refresh_evidence(state_path, workers=_evidence_worker_count(state))
    diagnosis = diagnose_discrete(state_path, str(Path(state_path).resolve().parent / "dashboard" / "failure_diagnosis.json"))
    result["diagnosis"] = diagnosis["output"]
    if output_dir:
        destination = Path(output_dir).resolve()
        examination_root = Path(state_path).resolve().parent
        if examination_root in destination.parents:
            raise ValueError("report output must not be nested inside the examination run")
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(result["dashboard"])
        target = destination / "index.html"
        # The in-run dashboard lives one directory below its evidence.  A
        # portable export places the HTML and evidence at the same root.
        target.write_text(source.read_text(encoding="utf-8").replace("../", "./"), encoding="utf-8")
        for name in ("evidence", "worlds", "episodes", "trials", "native_batches"):
            source_dir = examination_root / name
            if source_dir.is_dir():
                shutil.copytree(str(source_dir), str(destination / name), dirs_exist_ok=True)
        for name in ("dashboard_data.json", "conditions.csv", "failure_diagnosis.json"):
            source_file = source.parent / name
            if source_file.is_file():
                shutil.copy2(str(source_file), str(destination / name))
        preview_dir = source.parent / "world_previews"
        if preview_dir.is_dir():
            shutil.copytree(str(preview_dir), str(destination / "dashboard" / "world_previews"), dirs_exist_ok=True)
        shutil.copy2(str(Path(state_path).resolve()), str(destination / "examination_state.json"))
        result["html"] = str(target)
    else:
        result["html"] = result["dashboard"]
    return result
