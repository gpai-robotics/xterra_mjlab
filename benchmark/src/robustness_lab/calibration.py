"""Backend calibration fixtures and effective-physics analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

from .io import dump_json, load_data
from .models import GeometrySpec, MaterialSpec, Pose, RegionSpec, RobotProfile, RouteSpec, WorldSpec


CALIBRATION_KINDS = ("static_drop", "restitution_drop", "friction_incline", "slide", "contact_compliance", "actuator_step", "timing")


def _percentile(values: List[float], probability: float) -> Any:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _effective_measurements(kind: str, log: Dict[str, Any], world: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    samples = log.get("samples", [])
    if not samples:
        return {"valid": False, "reason": "calibration log has no samples"}
    times = [float(item["time_s"]) for item in samples]
    heights = [float(item["base_position_m"][2]) for item in samples]
    horizontal_speeds = [math.hypot(*[float(value) for value in item["base_linear_velocity_mps"][:2]]) for item in samples]
    vertical_speeds = [float(item["base_linear_velocity_mps"][2]) for item in samples]
    tail = max(1, len(samples) // 5)
    measured: Dict[str, Any] = {
        "valid": True,
        "sample_count": len(samples),
        "duration_s": max(times[-1] - times[0], 0.0),
    }
    if kind == "static_drop":
        measured.update(
            {
                "resting_base_height_m": sum(heights[-tail:]) / tail,
                "resting_vertical_speed_rms_mps": math.sqrt(sum(value * value for value in vertical_speeds[-tail:]) / tail),
                "maximum_drop_m": max(heights) - min(heights),
            }
        )
    elif kind == "restitution_drop":
        impact_index = min(range(len(heights)), key=lambda index: heights[index])
        rebound_height = max(heights[impact_index:]) - heights[impact_index]
        drop_height = max(heights[: impact_index + 1]) - heights[impact_index]
        measured.update(
            {
                "impact_time_s": times[impact_index],
                "drop_height_m": drop_height,
                "rebound_height_m": rebound_height,
                "effective_restitution": math.sqrt(max(rebound_height, 0.0) / drop_height) if drop_height > 1e-9 else None,
            }
        )
    elif kind == "friction_incline":
        angle = abs(float(world.get("resolved_parameters", {}).get("surface_angle_rad", 0.0)))
        moving = [index for index, value in enumerate(horizontal_speeds) if value > 0.02]
        measured.update(
            {
                "motion_onset_s": times[moving[0]] if moving else None,
                "slid": bool(moving),
                "tan_incline": math.tan(angle),
                "effective_static_friction_bound": "below" if moving else "at_least",
            }
        )
    elif kind == "slide":
        pairs = [(times[index], horizontal_speeds[index]) for index in range(len(samples)) if horizontal_speeds[index] > 0.01]
        if len(pairs) >= 2:
            mean_t = sum(item[0] for item in pairs) / len(pairs)
            mean_v = sum(item[1] for item in pairs) / len(pairs)
            denominator = sum((item[0] - mean_t) ** 2 for item in pairs)
            slope = sum((item[0] - mean_t) * (item[1] - mean_v) for item in pairs) / denominator if denominator else 0.0
            measured.update({"deceleration_mps2": max(0.0, -slope), "effective_dynamic_friction": max(0.0, -slope) / 9.80665})
        else:
            measured.update({"valid": False, "reason": "no measurable sliding interval"})
    elif kind == "contact_compliance":
        tail_height = sum(heights[-tail:]) / tail
        measured.update({"minimum_base_height_m": min(heights), "settled_base_height_m": tail_height, "settled_height_variation_m": max(heights[-tail:]) - min(heights[-tail:])})
    elif kind == "actuator_step":
        joint_names = sorted({name for item in samples for name in item.get("joint_position_rad", {})})
        excursions = {}
        for name in joint_names:
            values = [float(item.get("joint_position_rad", {}).get(name, 0.0)) for item in samples]
            excursions[name] = max(values) - min(values)
        measured.update({"maximum_joint_excursion_rad": max(excursions.values()) if excursions else None, "joint_excursion_rad": excursions})
    elif kind == "timing":
        inference = [float(item.get("inference_latency_s", 0.0)) for item in samples]
        simulation = [float(item.get("simulator_step_s", 0.0)) for item in samples if float(item.get("simulator_step_s", 0.0)) > 0.0]
        measured.update(
            {
                "inference_latency_p95_s": _percentile(inference, 0.95),
                "simulation_step_p95_s": _percentile(simulation, 0.95),
                "configured_physics_dt_s": float(episode.get("dt_s", 0.0)),
                "configured_control_period_s": float(episode.get("dt_s", 0.0)) * int(episode.get("control_decimation", 1)),
            }
        )
    return measured


def calibration_world(profile: RobotProfile, kind: str, seed: int = 0) -> WorldSpec:
    if kind not in CALIBRATION_KINDS:
        raise ValueError("unknown calibration kind: %s" % kind)
    length, width, thickness = max(3.0, profile.body_length_m * 8.0), max(2.0, profile.body_width_m * 8.0), 0.1
    angle = math.radians(20.0) if kind == "friction_incline" else 0.0
    rise = length * math.sin(angle)
    material = MaterialSpec("calibration", 0.6, 0.5, 0.2 if kind == "restitution_drop" else 0.0, compliance=0.02 if kind == "contact_compliance" else 0.0, damping=5.0)
    floor = GeometrySpec("calibration-surface", "box", [length, width, thickness], Pose([length / 2.0, 0.0, rise / 2.0 - thickness / 2.0], [0.0, -angle, 0.0]), "calibration", "calibration_surface")
    height = profile.standing_height_m * (2.0 if kind in {"static_drop", "restitution_drop"} else 1.0)
    world = WorldSpec(
        name="calibration-%s" % kind,
        terrain_type="calibration",
        seed=seed,
        robot_profile_sha256=profile.source_sha256,
        dimensions_m=[length, width, max(3.0, height * 2.0)],
        spawn_pose=Pose([profile.body_length_m, 0.0, height]),
        goal_pose=Pose([length - profile.body_length_m, 0.0, profile.standing_height_m + rise]),
        materials={"calibration": material},
        geometry=[floor],
        regions=[RegionSpec("calibration", kind, [0.0, -width / 2.0, -1.0], [length, width / 2.0, 4.0], True)],
        routes=[RouteSpec("calibration", [[profile.body_length_m, 0.0, 0.0], [length - profile.body_length_m, 0.0, rise]], ["calibration"], "calibration")],
        normalized_parameters={"kind": kind},
        resolved_parameters={"surface_angle_rad": angle, "configured_static_friction": material.static_friction, "configured_dynamic_friction": material.dynamic_friction, "configured_restitution": material.restitution, "configured_compliance": material.compliance},
        metadata={"calibration_kind": kind},
    )
    world.validate()
    return world


def generate_calibration_suite(profile: RobotProfile, output_dir: str) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    worlds = []
    for index, kind in enumerate(CALIBRATION_KINDS):
        world = calibration_world(profile, kind, index)
        destination = root / (kind + ".json")
        dump_json(str(destination), world)
        worlds.append({"kind": kind, "world": str(destination)})
    suite = {"schema_version": "1.0", "suite_id": "backend-calibration", "worlds": worlds, "required_before_cross_sim": True}
    dump_json(str(root / "suite.json"), suite)
    return suite


def analyze_calibration(run_dirs: List[str], output_path: str) -> Dict[str, Any]:
    backends: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        root = Path(run_dir).resolve()
        episode = load_data(str(root / "resolved_episode.json"))
        world = load_data(episode["world"])
        log = load_data(str(root / "episode_log.json"))
        metrics = load_data(str(root / "metrics.json")) if (root / "metrics.json").exists() else {"metrics": []}
        kind = world.get("metadata", {}).get("calibration_kind", "unknown")
        backends.setdefault(episode["backend"], {})[kind] = {
            "status": log.get("status"),
            "terminal_reason": log.get("terminal_reason"),
            "configured": world.get("resolved_parameters", {}),
            "metrics": {item["name"]: item.get("value") for item in metrics.get("metrics", []) if item.get("valid")},
            "effective": _effective_measurements(kind, log, world, episode),
            "run_dir": str(root),
        }
    required = set(CALIBRATION_KINDS)
    completeness = {backend: sorted(required - set(results)) for backend, results in backends.items()}
    invalid = {
        backend: sorted(kind for kind, record in results.items() if record["status"] != "completed" or not record["effective"].get("valid"))
        for backend, results in backends.items()
    }
    result = {"schema_version": "1.0", "backends": backends, "complete": bool(backends) and all(not missing for missing in completeness.values()) and all(not failures for failures in invalid.values()), "missing_fixtures": completeness, "invalid_fixtures": invalid, "warning": "Configuration values are not treated as effective physics; inspect measured fixture metrics."}
    dump_json(output_path, result)
    return result
