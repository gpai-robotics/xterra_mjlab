"""Locomotion skill contracts and safe artifact discovery.

The benchmark never guesses a policy's observation or action convention.  A
``SkillPackage`` describes that boundary explicitly while discovery only
classifies artifacts; it never deserializes an untrusted PyTorch checkpoint.
"""

from __future__ import annotations

import math
import re
import statistics
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io import load_data, sha256_file


SKILL_SCHEMA_VERSION = "1.0"
SENSOR_KINDS = {
    "joint_position",
    "joint_velocity",
    "previous_action",
    "imu",
    "base_state",
    "command",
    "waypoints",
    "goal_pose",
    "rgb_camera",
    "depth_camera",
    "lidar",
    "height_scan",
    "privileged_terrain",
    "privileged_state",
}
SENSOR_TRACKS = {"blind", "exteroceptive", "privileged"}
CONTROLLER_TYPES = {"python", "torchscript", "onnx", "native_rsl_rl", "compiled_policy"}
NAVIGATION_MODES = {"waypoint_heading", "waypoint_positions", "goal_pose", "velocity_command", "autonomous", "none"}
ROUTE_SCOPES = {"straight_only", "fixed_route", "route_choice"}


@dataclass
class SensorRequirement:
    name: str
    kind: str
    shape: List[int]
    rate_hz: float
    frame: str = "robot"
    required: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("sensor name is required")
        if self.kind not in SENSOR_KINDS:
            raise ValueError("unsupported sensor kind %s" % self.kind)
        if self.rate_hz <= 0:
            raise ValueError("sensor %s rate_hz must be positive" % self.name)
        if not self.shape or any(not isinstance(value, int) or value <= 0 for value in self.shape):
            raise ValueError("sensor %s shape must contain positive integers" % self.name)


@dataclass
class ObservationTensor:
    name: str
    sources: List[str]
    shape: List[int]
    dtype: str = "float32"
    history: int = 1
    normalization: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.sources:
            raise ValueError("observation tensor name and sources are required")
        if not self.shape or any(not isinstance(value, int) or value <= 0 for value in self.shape):
            raise ValueError("observation tensor %s has an invalid shape" % self.name)
        if self.dtype not in {"float16", "float32", "float64", "uint8", "int64"}:
            raise ValueError("unsupported observation dtype %s" % self.dtype)
        if self.history < 1:
            raise ValueError("observation history must be at least one")
        if self.history > 1 or self.normalization:
            raise ValueError("history/normalization are declared in schema v1 but not implemented; use schema v2")


@dataclass
class SkillPackage:
    skill_id: str
    controller: Dict[str, Any]
    sensors: List[SensorRequirement]
    observations: List[ObservationTensor]
    action: Dict[str, Any]
    control_frequency_hz: float
    sensor_track: str
    navigation: Dict[str, Any] = field(default_factory=dict)
    robot_profile_sha256: str = ""
    recurrent_state: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SKILL_SCHEMA_VERSION
    manifest_path: str = ""

    def validate(self, check_files: bool = True) -> None:
        if not self.skill_id:
            raise ValueError("skill_id is required")
        if self.sensor_track not in SENSOR_TRACKS:
            raise ValueError("sensor_track must be blind, exteroceptive, or privileged")
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        navigation = self.navigation_contract()
        if navigation["mode"] not in NAVIGATION_MODES:
            raise ValueError("unsupported navigation mode %s" % navigation["mode"])
        if navigation["route_scope"] not in ROUTE_SCOPES:
            raise ValueError("unsupported navigation route_scope %s" % navigation["route_scope"])
        if int(navigation.get("lookahead", 0)) < 0:
            raise ValueError("navigation lookahead must be non-negative")
        controller_type = self.controller.get("type")
        if controller_type not in CONTROLLER_TYPES:
            raise ValueError("unsupported skill controller type %s" % controller_type)
        sensor_names = set()
        for sensor in self.sensors:
            sensor.validate()
            if sensor.name in sensor_names:
                raise ValueError("duplicate sensor name %s" % sensor.name)
            sensor_names.add(sensor.name)
        privileged = any(sensor.kind.startswith("privileged_") for sensor in self.sensors)
        exteroceptive = any(sensor.kind in {"rgb_camera", "depth_camera", "lidar", "height_scan"} for sensor in self.sensors)
        expected_track = "privileged" if privileged else ("exteroceptive" if exteroceptive else "blind")
        if self.sensor_track != expected_track:
            raise ValueError("sensor_track %s conflicts with declared sensors; expected %s" % (self.sensor_track, expected_track))
        for observation in self.observations:
            observation.validate()
            missing = set(observation.sources) - sensor_names
            if missing:
                raise ValueError("observation %s references missing sensors %s" % (observation.name, sorted(missing)))
        joint_order = self.action.get("joint_order", [])
        if not joint_order or len(set(joint_order)) != len(joint_order):
            raise ValueError("action.joint_order must be explicit and unique")
        if self.action.get("mode") not in {"position", "effort"}:
            raise ValueError("action.mode must be position or effort")
        if int(self.action.get("size", len(joint_order))) != len(joint_order):
            raise ValueError("action size does not match joint_order")
        if controller_type in {"torchscript", "onnx", "native_rsl_rl"}:
            path = self.controller.get("path") or self.controller.get("checkpoint")
            if not path:
                raise ValueError("%s skill requires a path/checkpoint" % controller_type)
            if check_files and not Path(path).is_file():
                raise ValueError("skill artifact does not exist: %s" % path)
        if controller_type == "python" and not self.controller.get("target"):
            raise ValueError("python skill requires controller.target")
        if controller_type == "compiled_policy":
            # The schema-v2 manifest carries the observation and action
            # contract, so it is this controller's artifact.
            manifest = self.controller.get("manifest")
            if not manifest:
                raise ValueError("compiled_policy skill requires controller.manifest")
            if check_files and not Path(manifest).is_file():
                raise ValueError("skill artifact does not exist: %s" % manifest)

        sensor_kinds = {sensor.kind for sensor in self.sensors if sensor.required}
        if navigation["mode"] in {"waypoint_heading", "waypoint_positions"} and "waypoints" not in sensor_kinds:
            raise ValueError("waypoint navigation requires a declared waypoints sensor")
        if navigation["mode"] == "goal_pose" and "goal_pose" not in sensor_kinds:
            raise ValueError("goal_pose navigation requires a declared goal_pose sensor")
        if navigation["mode"] == "velocity_command" and "command" not in sensor_kinds:
            raise ValueError("velocity_command navigation requires a declared command sensor")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def controller_config(self) -> Dict[str, Any]:
        """Translate the public skill contract into the existing controller contract."""
        result = dict(self.controller)
        result["joint_order"] = list(self.action["joint_order"])
        result["action_mode"] = self.action["mode"]
        result["control_frequency_hz"] = self.control_frequency_hz
        result["sensor_track"] = self.sensor_track
        result["navigation"] = self.navigation_contract()
        result["sensor_requirements"] = [asdict(item) for item in self.sensors]
        result["observation_tensors"] = [asdict(item) for item in self.observations]
        return result

    def navigation_contract(self) -> Dict[str, Any]:
        """Return the explicit route interface, with a safe legacy default."""
        if self.navigation:
            result = dict(self.navigation)
        else:
            kinds = {sensor.kind for sensor in self.sensors}
            result = {
                "mode": "velocity_command" if "command" in kinds else "none",
                "route_scope": "straight_only",
                "assistance": "route_follower" if "command" in kinds else "none",
                "lookahead": 0,
                "legacy_inferred": True,
            }
        result.setdefault("route_scope", "fixed_route" if result.get("mode") in {"waypoint_heading", "waypoint_positions", "goal_pose"} else "straight_only")
        result.setdefault("assistance", "benchmark_waypoints" if result.get("mode") in {"waypoint_heading", "waypoint_positions", "goal_pose"} else ("route_follower" if result.get("mode") == "velocity_command" else "none"))
        result.setdefault("lookahead", 0)
        return result


def load_skill(path: str, check_files: bool = True) -> SkillPackage:
    raw = load_data(path)
    manifest = Path(path).resolve()
    controller = dict(raw.get("controller", {}))
    for key in ("path", "checkpoint", "manifest"):
        value = controller.get(key)
        if value and not Path(str(value)).is_absolute():
            controller[key] = str((manifest.parent / str(value)).resolve())
    value = SkillPackage(
        skill_id=raw.get("skill_id", ""),
        controller=controller,
        sensors=[SensorRequirement(**item) for item in raw.get("sensors", [])],
        observations=[ObservationTensor(**item) for item in raw.get("observations", [])],
        action=dict(raw.get("action", {})),
        control_frequency_hz=float(raw.get("control_frequency_hz", 0.0)),
        sensor_track=raw.get("sensor_track", ""),
        navigation=dict(raw.get("navigation", {})),
        robot_profile_sha256=raw.get("robot_profile_sha256", ""),
        recurrent_state=dict(raw.get("recurrent_state", {})),
        provenance=dict(raw.get("provenance", {})),
        schema_version=str(raw.get("schema_version", SKILL_SCHEMA_VERSION)),
        manifest_path=str(manifest),
    )
    value.validate(check_files=check_files)
    return value


def assess_navigation_compatibility(
    skill: SkillPackage,
    topology: str,
    evaluation: str = "locomotion",
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Gate route delivery without pretending every policy accepts waypoints.

    ``locomotion`` evaluation may use a declared route follower. ``navigation``
    evaluation measures route choice and therefore cannot hide a planner behind
    a policy that only accepts velocity commands.
    """
    navigation = skill.navigation_contract()
    mode = navigation["mode"]
    available = set(providers or ())
    branching = topology in {"tree", "grid", "maze", "random_connected"}
    required_provider = {
        "waypoint_heading": "waypoint_heading",
        "waypoint_positions": "waypoint_positions",
        "goal_pose": "goal_pose",
        "velocity_command": "route_follower",
    }.get(mode)
    reasons: List[str] = []
    if required_provider and required_provider not in available:
        reasons.append("missing navigation provider: %s" % required_provider)
    if mode == "none" and topology != "chain":
        reasons.append("policy declares no route input and is limited to straight courses")
    if evaluation == "navigation" and branching:
        if navigation["route_scope"] != "route_choice" or mode in {"velocity_command", "none"}:
            reasons.append("policy cannot make an unaided route choice on branching topology")
    return {
        "status": "compatible" if not reasons else "blocked",
        "mode": mode,
        "route_scope": navigation["route_scope"],
        "assistance": navigation["assistance"],
        "topology": topology,
        "evaluation": evaluation,
        "required_provider": required_provider,
        "available_providers": sorted(available),
        "reasons": reasons,
        "claim_scope": "locomotion with declared route assistance" if evaluation == "locomotion" and not reasons else ("unaided route choice" if evaluation == "navigation" and not reasons else "not comparable"),
    }


def validate_skill(path: str) -> Dict[str, Any]:
    try:
        skill = load_skill(path)
        artifact = skill.controller.get("path") or skill.controller.get("checkpoint")
        return {
            "valid": True,
            "skill_id": skill.skill_id,
            "sensor_track": skill.sensor_track,
            "controller_type": skill.controller.get("type"),
            "artifact_sha256": sha256_file(artifact) if artifact and Path(artifact).is_file() else None,
            "directly_runnable": skill.controller.get("type") in {"python", "torchscript", "onnx"},
            "source_native_runnable": skill.controller.get("type") == "native_rsl_rl" and bool(skill.controller.get("task")),
            "navigation": skill.navigation_contract(),
            "sensor_count": len(skill.sensors),
            "observation_count": len(skill.observations),
            "action_size": len(skill.action.get("joint_order", [])),
        }
    except (OSError, ValueError, TypeError) as error:
        return {"valid": False, "error": str(error)}


def probe_skill(path: str, warmup: int = 3, repetitions: int = 20) -> Dict[str, Any]:
    """Load a portable artifact and verify declared tensor I/O with zero data.

    This validates executability and shapes only.  It deliberately makes no
    claim that zero observations are physically meaningful or that the policy
    can locomote in the target simulator.
    """
    skill = load_skill(path)
    kind = skill.controller.get("type")
    if kind not in {"torchscript", "onnx"}:
        return {"status": "unsupported", "skill_id": skill.skill_id, "reason": "artifact probing currently supports torchscript and onnx skills"}
    input_names = list(skill.controller.get("input_tensors", [item.name for item in skill.observations]))
    observation_index = {item.name: item for item in skill.observations}
    missing = set(input_names) - set(observation_index)
    if missing:
        raise ValueError("controller input_tensors reference undeclared observations %s" % sorted(missing))
    from .artifacts import load_artifact

    import numpy as np

    started = time.perf_counter()
    artifact = load_artifact(
        skill.controller["path"],
        runtime_format=kind,
        device=skill.controller.get("device", "cpu"),
        providers=skill.controller.get("providers", ["CPUExecutionProvider"]),
    )
    if len(artifact.input_names) != len(input_names):
        raise ValueError(
            "%s input count does not match skill observation tensors"
            % artifact.label
        )
    tensors = [
        np.zeros([1] + observation_index[name].shape, dtype=np.float32)
        for name in input_names
    ]

    def infer():
        return artifact.run(tensors)

    backend_version = artifact.runtime_version
    for _ in range(max(0, warmup)):
        infer()
    latencies = []
    output = None
    for _ in range(max(1, repetitions)):
        tick = time.perf_counter()
        output = infer()
        latencies.append((time.perf_counter() - tick) * 1000.0)
    shape = list(output.shape)
    finite = bool(np.isfinite(output).all())
    expected_actions = len(skill.action["joint_order"])
    valid_shape = bool(shape and shape[-1] == expected_actions)
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]
    return {
        "status": "passed" if finite and valid_shape else "failed",
        "skill_id": skill.skill_id,
        "controller_type": kind,
        "artifact": skill.controller["path"],
        "artifact_sha256": sha256_file(skill.controller["path"]),
        "runtime_version": backend_version,
        "input_tensors": [{"name": name, "shape": [1] + observation_index[name].shape} for name in input_names],
        "output_shape": shape,
        "expected_action_size": expected_actions,
        "finite_output": finite,
        "latency_ms": {"median": statistics.median(latencies), "p95": p95, "samples": len(latencies), "including_load_ms": (time.perf_counter() - started) * 1000.0},
        "claim_scope": "artifact executability and tensor-shape validation only; no locomotion capability claim",
    }


def _archive_members(path: Path) -> List[str]:
    try:
        if zipfile.is_zipfile(str(path)):
            with zipfile.ZipFile(str(path)) as archive:
                return archive.namelist()
    except (OSError, zipfile.BadZipFile):
        pass
    return []


def _checkpoint_iteration(path: Path) -> Optional[int]:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def _candidate(path: Path, root: Path) -> Dict[str, Any]:
    members = _archive_members(path)
    portable = [name for name in members if name.endswith(("policy.pt", "policy.onnx", "depth_latest.onnx", "depth_latest.pt"))]
    has_training_state = any(name.endswith("data.pkl") for name in members) or path.name.startswith("model_")
    role = "student" if "student" in path.name.lower() else ("teacher" if "teacher" in str(path).lower() else "unknown")
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(str(path)),
        "iteration": _checkpoint_iteration(path),
        "role_hint": role,
        "format": "bundle_with_exports" if portable else ("rsl_rl_checkpoint" if has_training_state else "unknown_pytorch"),
        "portable_members": portable,
        "directly_runnable": False,
        "required_action": "extract and describe portable export" if portable else "provide native RSL-RL task adapter or export policy",
    }


def discover_skills(root_path: str, limit: int = 50) -> Dict[str, Any]:
    """Find policy artifacts without importing or unpickling them.

    For RSL-RL run directories only the highest-iteration checkpoint is listed;
    this prevents thousands of intermediate files from obscuring usable skills.
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError("skill search root does not exist: %s" % root)
    selected: List[Path] = []
    run_groups: Dict[Path, List[Path]] = {}
    for path in root.rglob("*.pt"):
        if not path.is_file():
            continue
        if _checkpoint_iteration(path) is not None:
            run_groups.setdefault(path.parent, []).append(path)
        else:
            selected.append(path)
    for paths in run_groups.values():
        selected.append(max(paths, key=lambda item: _checkpoint_iteration(item) or -1))
    candidates = [_candidate(path, root) for path in sorted(selected)]
    candidates.sort(
        key=lambda item: (
            bool(item["portable_members"]),
            item["role_hint"] == "student",
            item["iteration"] or -1,
            item["size_bytes"],
        ),
        reverse=True,
    )
    return {
        "status": "ok",
        "schema_version": "1.0",
        "root": str(root),
        "candidate_count": len(candidates),
        "returned_count": min(len(candidates), max(0, limit)),
        "candidates": candidates[: max(0, limit)],
        "safety": "artifacts were classified from paths and ZIP members only; no pickle payload was executed",
    }


def extract_portable_members(bundle_path: str, output_dir: str, members: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Extract only policy/depth exports from a trusted local bundle."""
    source = Path(bundle_path).resolve()
    destination = Path(output_dir).resolve()
    if not zipfile.is_zipfile(str(source)):
        raise ValueError("bundle is not a ZIP-compatible policy bundle")
    allowed_suffixes = ("policy.pt", "policy.onnx", "depth_latest.pt", "depth_latest.onnx", "env.yaml", "agent.yaml")
    extracted = []
    with zipfile.ZipFile(str(source)) as archive:
        available = [name for name in archive.namelist() if name.endswith(allowed_suffixes)]
        requested = list(members) if members is not None else available
        invalid = sorted(set(requested) - set(available))
        if invalid:
            raise ValueError("requested unavailable or unsafe bundle members: %s" % invalid)
        destination.mkdir(parents=True, exist_ok=True)
        for name in requested:
            source_parts = Path(name).parts
            # Keep the logical bundle group (params/exported_deploy/
            # exported_teacher) so identically named policy files can never
            # overwrite one another.
            group = source_parts[-2] if len(source_parts) >= 2 else "artifacts"
            if group not in {"params", "exported_deploy", "exported_teacher"}:
                group = "artifacts"
            target = destination / group / source_parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source_handle, target.open("wb") as target_handle:
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    target_handle.write(chunk)
            extracted.append({"member": name, "path": str(target), "sha256": sha256_file(str(target))})
    return {"status": "ok", "bundle": str(source), "output_dir": str(destination), "extracted": extracted}
