"""Standalone readiness checks for a complete RobustLab examination.

Preflight deliberately separates three responsibilities:

* the policy contract describes observations, navigation input, and actions;
* the simulator adapter owns geometry, materials, and environment effects;
* the scheduler owns repetitions, pruning, evidence, and resumption.

No policy inference or physics step is executed here.
"""

from __future__ import annotations

import datetime as dt
import shlex
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .doctor import expand_runtime_path, run_doctor
from .environment_physics import (
    DEPTH_RESISTIVE_EFFECT,
    assess_environment_support,
    world_physics_plan,
)
from .io import dump_json, load_data, sha256_file
from .models import RobotProfile
from .observation.gates import (
    check_artifact_accepts,
    check_artifact_accepts_tensors,
    check_deployability,
    check_dimension,
    check_finite,
    check_known_pose,
    resolve_allowed_classes,
)
from .observation.evidence import measure_observation_evidence
from .policy_manifest import load_policy_manifest
from .robot import validate_default_pose
from .skill import load_skill
from .worlds import generate_terrain


BACKEND_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "fake": {
        "runtime": "fake",
        "robot_formats": ["urdf", "mjcf", "xml", "usd"],
        "environment_effects": [],
        "claim_scope": "scheduler and reporting only",
    },
    "mujoco": {
        "runtime": "mujoco",
        "robot_formats": ["mjcf", "xml"],
        # Must match MujocoBackend.probe().environment_effects exactly; two
        # independent lists of the same fact drift apart, and the drift shows up
        # as a terrain reported Unsupported by preflight but supported by the run.
        "environment_effects": ["linear_velocity_drag", DEPTH_RESISTIVE_EFFECT],
        "claim_scope": "MuJoCo locomotion",
    },
    "isaaclab": {
        "runtime": "isaaclab",
        "robot_formats": ["usd", "urdf"],
        "environment_effects": ["linear_velocity_drag"],
        "claim_scope": "Isaac Lab locomotion",
    },
    "isaacgym": {
        "runtime": "isaacgym",
        "robot_formats": ["urdf"],
        "environment_effects": ["linear_velocity_drag"],
        "claim_scope": "Isaac Gym locomotion",
    },
    "foresight_native": {
        "runtime": "isaaclab",
        # The source-native task loads its own robot. This asset is used for
        # morphology scaling and provenance, so any canonical format is valid.
        "robot_formats": ["urdf", "mjcf", "xml", "usd"],
        "environment_effects": ["linear_velocity_drag"],
        "claim_scope": "source-native Isaac Lab locomotion",
    },
}


def _resolve(value: str, base: Path) -> str:
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def _file_record(path: str, role: str) -> Dict[str, Any]:
    candidate = Path(path)
    present = candidate.is_file()
    return {
        "role": role,
        "path": str(candidate),
        "status": "ready" if present else "missing",
        "sha256": sha256_file(str(candidate)) if present else None,
    }


def _runtime_static_status(config: Dict[str, Any], runtime_name: str) -> Dict[str, Any]:
    settings = config.get("runtimes", {}).get(runtime_name)
    if not settings:
        return {"status": "missing", "runtime": runtime_name, "reason": "runtime is not declared in runtimes.yaml"}
    declared = settings.get("python") or settings.get("launcher")
    # A runtime that declares no interpreter runs in this one, exactly as
    # runtime._runtime_command and doctor._probe_python already assume. Without
    # this, the shipped mujoco runtime -- which deliberately declares none --
    # reads as missing on every machine.
    executable = expand_runtime_path(declared) if declared else sys.executable
    available = bool(
        executable
        and (Path(str(executable)).exists() or shutil.which(str(executable)))
    )
    return {
        "status": "ready" if available else "missing",
        "runtime": runtime_name,
        "executable": executable,
        "requires_gpu": bool(settings.get("requires_gpu", False)),
        "features": settings.get("features", {}),
        "reason": "" if available else "configured executable was not found",
    }


def _representative_environment_report(
    robot: RobotProfile,
    terrain_codes: Iterable[str],
    supported_effects: Iterable[str],
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    for index, terrain in enumerate(terrain_codes):
        world = generate_terrain(str(terrain).lower(), robot, 0.5, 7100 + index)
        plan = world_physics_plan(world)
        reports[str(terrain).lower()] = {
            "required_effects": plan["required_effects"],
            "requested_tier": plan["requested_tier"],
            **assess_environment_support(plan, supported_effects),
        }
    missing = sorted(
        {
            effect
            for report in reports.values()
            for effect in report.get("missing_effects", [])
        }
    )
    return {
        "status": "ready" if not missing else "blocked",
        "implementation_owner": "simulator_adapter",
        "policy_interface_modified": False,
        "supported_effects": sorted(set(supported_effects)),
        "missing_effects": missing,
        "terrains": reports,
    }


def _adapter_block(reason: str, **details: Any) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "outcome_class": "adapter_unverified",
        "reason": reason,
        **details,
    }


def _declared_policy_sources(manifest: Mapping[str, Any]) -> List[str]:
    """Collect canonical inputs from direct terms and referenced encoders."""

    if not isinstance(manifest, Mapping):
        raise ValueError("policy manifest must be a mapping")
    observation = manifest.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("policy manifest observation must be a mapping")

    term_groups = []
    source_groups = []
    referenced_encoders = set()
    if "tensors" in observation:
        tensors = observation.get("tensors")
        if isinstance(tensors, (str, bytes)) or not isinstance(tensors, Sequence):
            raise ValueError("policy manifest observation tensors must be a sequence")
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, Mapping):
                raise ValueError(
                    "policy manifest observation tensor %d must be a mapping"
                    % index
                )
            if "terms" in tensor:
                term_groups.append(tensor.get("terms"))
            if "from_encoder" in tensor:
                referenced_encoders.add(tensor.get("from_encoder"))
    else:
        if "sources" in observation:
            source_groups.append(observation.get("sources"))
        else:
            term_groups.append(observation.get("terms"))

    sources = set()
    for terms in term_groups:
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
            raise ValueError("policy manifest observation terms must be a sequence")
        for index, term in enumerate(terms):
            if not isinstance(term, Mapping):
                raise ValueError(
                    "policy manifest observation term %d must be a mapping" % index
                )
            source = term.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    "policy manifest observation term %d requires a source" % index
                )
            sources.add(source)

    for declared_sources in source_groups:
        if isinstance(declared_sources, (str, bytes)) or not isinstance(
            declared_sources, Sequence
        ):
            raise ValueError("policy manifest observation sources must be a sequence")
        for index, entry in enumerate(declared_sources):
            source = entry if isinstance(entry, str) else (
                entry.get("source") if isinstance(entry, Mapping) else None
            )
            if not isinstance(source, str) or not source:
                raise ValueError(
                    "policy manifest observation source %d requires a source" % index
                )
            sources.add(source)

    encoders = manifest.get("encoders", [])
    if isinstance(encoders, (str, bytes)) or not isinstance(encoders, Sequence):
        raise ValueError("policy manifest encoders must be a sequence")
    for index, encoder in enumerate(encoders):
        if not isinstance(encoder, Mapping):
            raise ValueError("policy manifest encoder %d must be a mapping" % index)
        if encoder.get("name") not in referenced_encoders:
            continue
        inputs = encoder.get("inputs")
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ValueError("policy manifest encoder inputs must be a sequence")
        for input_index, input_spec in enumerate(inputs):
            if not isinstance(input_spec, Mapping):
                raise ValueError(
                    "policy manifest encoder input %d must be a mapping" % input_index
                )
            source = input_spec.get("name")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    "policy manifest encoder input %d requires a name" % input_index
                )
            sources.add(source)
    return sorted(sources)


def _observation_deployability_report(
    raw: Mapping[str, Any],
    base: Path,
    mode: str,
) -> Dict[str, Any]:
    """Build the run-level gate for observation availability classes."""

    allowed_classes = resolve_allowed_classes(mode)
    details: Dict[str, Any] = {
        "mode": mode,
        "allowed_classes": sorted(allowed_classes),
        "sources": [],
    }
    manifest_value = raw.get("policy_manifest")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = (base / manifest_path).resolve()
        details["manifest"] = str(manifest_path)
        try:
            manifest_raw = load_data(str(manifest_path))
            sources = _declared_policy_sources(manifest_raw)
        except (KeyError, OSError, TypeError, ValueError) as error:
            return _adapter_block(
                "policy observation sources could not be checked: %s" % error,
                rejected=[],
                classes_used=[],
                sensor_track="unverified",
                claim_scope=(
                    "unverified information availability; not a deployment claim"
                ),
                **details,
            )
        details["sources"] = sources

    try:
        result = check_deployability(details["sources"], allowed_classes)
    except ValueError as error:
        return _adapter_block(
            "policy observation sources could not be checked: %s" % error,
            rejected=[],
            classes_used=[],
            sensor_track="unverified",
            claim_scope="unverified information availability; not a deployment claim",
            **details,
        )
    result.update(details)
    return result


def _observation_verification_report(
    raw: Mapping[str, Any],
    base: Path,
    compatibility: Mapping[str, Any],
    backend_name: str,
    robot: Optional[RobotProfile],
    robot_asset: Optional[str],
    robot_format: str,
) -> Dict[str, Any]:
    """Build the preflight observation gate without changing legacy contracts.

    Schema-v1 skills predate the canonical observation compiler and remain under
    the existing ``observation_action_bridge`` gate.  A configuration that names
    a schema-v2 ``policy_manifest`` opts into all three verification gates.
    Simulator-measured evidence is preferred; declared evidence is retained as
    an explicitly labelled fallback for backends that cannot reset.
    """

    manifest_value = raw.get("policy_manifest")
    if not manifest_value:
        if compatibility.get("status") != "compatible":
            return _adapter_block(
                "the existing observation/action bridge is incompatible",
                mode="legacy_skill_contract",
            )
        return {
            "status": "ready",
            "mode": "legacy_skill_contract",
            "reason": "schema-v1 skill remains covered by observation_action_bridge",
        }

    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = (base / manifest_path).resolve()
    try:
        manifest_raw = load_data(str(manifest_path))
        manifest = load_policy_manifest(str(manifest_path))
        multi_tensor_sizes = getattr(manifest.observation_plan, "sizes", None)
        declared_size = (
            sum(multi_tensor_sizes.values())
            if multi_tensor_sizes is not None
            else int(manifest_raw["observation"]["expected_size"])
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return _adapter_block(
            "policy observation manifest could not be verified: %s" % error,
            manifest=str(manifest_path),
        )

    runtime_format = manifest.runtime.get("format")
    if multi_tensor_sizes is not None:
        artifact_probe = check_artifact_accepts_tensors(
            manifest.runtime["path"],
            list(multi_tensor_sizes.values()),
            runtime_format=runtime_format,
        )
        dimension = {
            "status": artifact_probe["status"],
            "compiled_size": manifest.observation_plan.size,
            "declared_size": declared_size,
            "tensor_sizes": dict(multi_tensor_sizes),
        }
        if artifact_probe["status"] != "ready":
            dimension.update(
                _adapter_block(
                    "multi-tensor artifact dimensions could not be verified"
                )
            )
    else:
        artifact_probe = check_artifact_accepts(
            manifest.runtime["path"],
            declared_size,
            runtime_format=runtime_format,
        )
        dimension = check_dimension(
            manifest.observation_plan.size,
            declared_size,
            artifact_probe.get("input_size")
            if artifact_probe["status"] == "ready"
            else None,
            width_is_discriminating=artifact_probe.get(
                "width_is_discriminating"
            ),
        )

    measured_evidence = None
    if robot is not None and robot_asset:
        measured_evidence = measure_observation_evidence(
            backend_name,
            manifest,
            robot,
            robot_asset,
            robot_format,
        )

    declared_evidence = raw.get("observation_gate_evidence")
    if isinstance(measured_evidence, Mapping):
        evidence = measured_evidence
        provenance = "measured"
    elif declared_evidence is None or declared_evidence == {}:
        evidence = {}
        provenance = "unavailable"
    elif not isinstance(declared_evidence, Mapping):
        finite = _adapter_block(
            "compiled_observation evidence is required for the finiteness gate",
            provenance="unavailable",
        )
        known_pose = _adapter_block(
            "known_pose_terms evidence is required for the known-pose replay gate",
            failing_terms=[],
            provenance="unavailable",
        )
        return _adapter_block(
            "observation_gate_evidence must be a mapping",
            manifest=str(manifest_path),
            artifact_probe=artifact_probe,
            dimension=dimension,
            checks={
                "artifact_probe": artifact_probe,
                "dimension": dimension,
                "finiteness": finite,
                "known_pose_replay": known_pose,
            },
        )
    else:
        evidence = declared_evidence
        provenance = "declared"
    finite = (
        check_finite(evidence["compiled_observation"])
        if "compiled_observation" in evidence
        else _adapter_block(
            "compiled_observation evidence is required for the finiteness gate"
        )
    )
    known_pose = (
        check_known_pose(evidence["known_pose_terms"])
        if "known_pose_terms" in evidence
        else _adapter_block(
            "known_pose_terms evidence is required for the known-pose replay gate",
            failing_terms=[],
        )
    )
    finite["provenance"] = provenance
    known_pose["provenance"] = provenance
    checks = {
        "artifact_probe": artifact_probe,
        "dimension": dimension,
        "finiteness": finite,
        "known_pose_replay": known_pose,
    }
    blocked = [name for name, result in checks.items() if result["status"] != "ready"]
    if blocked:
        return _adapter_block(
            "observation verification blocked at: %s" % ", ".join(blocked),
            manifest=str(manifest_path),
            checks=checks,
        )
    return {
        "status": "ready",
        "manifest": str(manifest_path),
        "checks": checks,
    }


def preflight_examination(
    config_path: str,
    output: Optional[str] = None,
    probe_runtime: bool = True,
) -> Dict[str, Any]:
    """Validate a full examination contract without launching an episode."""
    source = Path(config_path).resolve()
    raw = load_data(str(source))
    base = source.parent
    backend = str(raw.get("backend", "fake"))
    robot_format = str(raw.get("robot_format", "")).lower()
    observation_mode = str(raw.get("observation_mode", "deployable_only"))
    contract = BACKEND_CONTRACTS.get(backend)
    if not contract:
        raise ValueError("unknown backend %s" % backend)
    resolve_allowed_classes(observation_mode)

    required_paths = {
        key: _resolve(str(raw[key]), base)
        for key in ("robot_profile", "robot_asset", "skill", "runtimes")
        if raw.get(key)
    }
    if raw.get("policy_manifest"):
        required_paths["policy_manifest"] = _resolve(
            str(raw["policy_manifest"]), base
        )
    files = {name: _file_record(path, name) for name, path in required_paths.items()}
    missing_configuration = sorted(set(("robot_profile", "robot_asset", "skill", "runtimes")) - set(required_paths))
    files_ready = not missing_configuration and all(item["status"] == "ready" for item in files.values())

    policy: Dict[str, Any] = {"status": "blocked", "policy_was_executed": False}
    robot: Dict[str, Any] = {"status": "blocked"}
    environment: Dict[str, Any] = {"status": "blocked", "implementation_owner": "simulator_adapter"}
    compatibility: Dict[str, Any] = {"status": "blocked"}
    profile = None
    skill = None

    validation_error: Optional[str] = None
    if files_ready:
        try:
            profile = RobotProfile.from_dict(load_data(required_paths["robot_profile"]))
            skill = load_skill(required_paths["skill"])
        except (KeyError, OSError, TypeError, ValueError) as error:
            validation_error = str(error)
            profile = None
            skill = None

    if files_ready and profile is not None and skill is not None:
        if skill.robot_profile_sha256 and skill.robot_profile_sha256 != profile.source_sha256:
            policy = {
                "status": "blocked",
                "policy_was_executed": False,
                "reason": "skill manifest names a different RobotProfile hash",
            }
        else:
            artifact = skill.controller.get("path") or skill.controller.get("checkpoint")
            policy = {
                "status": "ready",
                "policy_was_executed": False,
                "skill_id": skill.skill_id,
                "controller_type": skill.controller.get("type"),
                "sensor_track": skill.sensor_track,
                "required_sensors": sorted({item.kind for item in skill.sensors if item.required}),
                "navigation": skill.navigation_contract(),
                "action_mode": skill.action.get("mode"),
                "action_size": len(skill.action.get("joint_order", [])),
                "artifact": _file_record(str(artifact), "policy_artifact") if artifact else None,
            }
            if policy["artifact"] and policy["artifact"]["status"] != "ready":
                policy["status"] = "blocked"
                policy["reason"] = "declared policy artifact is missing"

        format_ready = robot_format in contract["robot_formats"]
        default_pose = validate_default_pose(profile)
        robot_ready = format_ready and default_pose["status"] == "ready"
        if not format_ready:
            robot_reason = "robot format is not accepted by this backend contract"
        elif default_pose["status"] != "ready":
            robot_reason = "default joint pose is invalid"
        else:
            robot_reason = ""
        robot = {
            "status": "ready" if robot_ready else "blocked",
            "robot_id": profile.name,
            "format": robot_format,
            "accepted_formats": contract["robot_formats"],
            "profile_source_sha256": profile.source_sha256,
            "reason": robot_reason,
            "default_pose": default_pose,
        }

        from .examiner import _backend_sensor_gate

        compatibility = _backend_sensor_gate(skill, backend, required_paths["runtimes"])
        terrain_codes = raw.get("terrain_codes", ["rs%d" % index for index in range(1, 12)])
        environment = _representative_environment_report(profile, terrain_codes, contract["environment_effects"])

        if backend == "foresight_native":
            source_root = Path(str(skill.controller.get("source_root", "")))
            native_play = source_root / "scripts" / "rsl_rl" / "play.py"
            environment["native_source"] = {
                "status": "ready" if native_play.is_file() else "missing",
                "source_root": str(source_root),
                "play_script": str(native_play),
            }
    elif validation_error:
        policy = {
            "status": "blocked",
            "policy_was_executed": False,
            "reason": "robot or policy contract is invalid: %s" % validation_error,
        }
        robot = {"status": "blocked", "reason": "robot or policy contract could not be loaded"}

    runtime_config = load_data(required_paths["runtimes"]) if files.get("runtimes", {}).get("status") == "ready" else {"runtimes": {}}
    runtime_name = str(contract["runtime"])
    runtime = _runtime_static_status(runtime_config, runtime_name)
    runtime_probe = None
    if probe_runtime and runtime["status"] == "ready":
        doctor = run_doctor(required_paths["runtimes"])
        runtime_probe = doctor.get("runtimes", {}).get(runtime_name, {})
        runtime["probe"] = runtime_probe
        if runtime_probe.get("status") != "available":
            runtime["status"] = "blocked"
            runtime["reason"] = runtime_probe.get("reason") or runtime_probe.get("stderr") or "runtime probe failed"
        if runtime.get("requires_gpu") and doctor.get("gpu", {}).get("status") != "available":
            runtime["status"] = "blocked"
            runtime["reason"] = doctor.get("gpu", {}).get("reason", "GPU is unavailable")

    native_source_ready = environment.get("native_source", {}).get("status", "ready") == "ready"
    if not raw.get("policy_manifest") and skill is not None:
        # The manifest belongs to the policy, so a submission declares it on the
        # skill. Requiring it on the examiner too would be a second copy of one
        # path, and a disagreement between them would be silent.
        skill_manifest = skill.controller.get("manifest")
        if skill_manifest:
            raw = dict(raw)
            raw["policy_manifest"] = skill_manifest
    observation_verification = _observation_verification_report(
        raw,
        base,
        compatibility,
        backend,
        profile,
        required_paths.get("robot_asset"),
        robot_format,
    )
    observation_deployability = _observation_deployability_report(
        raw, base, observation_mode
    )
    controller_type = policy.get("controller_type")
    controller_gate_setting = None
    if files_ready and skill is not None:
        controller_gate_setting = skill.controller.get("flat_ground_gate")
    compiled_policy_run = (
        controller_type == "compiled_policy" or bool(raw.get("policy_manifest"))
    )
    flat_ground_gate_enabled = (
        bool(controller_gate_setting)
        if controller_gate_setting is not None
        else compiled_policy_run
    )
    flat_ground_gate = {
        "status": "ready",
        "enabled": flat_ground_gate_enabled,
        "phase": "before_terrain_scoring",
        "reason": (
            "compiled-policy locomotion is qualified on a short flat-ground run"
            if flat_ground_gate_enabled
            else "flat-ground qualification is disabled for this legacy controller"
        ),
    }
    gates = {
        "configuration_files": files_ready,
        "robot_contract": robot.get("status") == "ready",
        "policy_contract": policy.get("status") == "ready",
        "observation_action_bridge": compatibility.get("status") == "compatible",
        "simulator_runtime": runtime.get("status") == "ready",
        "environment_physics": environment.get("status") == "ready" and native_source_ready,
        "observation_verification": observation_verification,
        "observation_deployability": observation_deployability,
        "flat_ground_gate": flat_ground_gate,
    }
    ready = all(
        value.get("status") == "ready" if isinstance(value, Mapping) else value
        for value in gates.values()
    )
    command = "robustlab examine start %s" % shlex.quote(str(source))
    observation_claim_scope = observation_deployability["claim_scope"]
    sensor_track = observation_deployability["sensor_track"]
    backend_claim_scope = contract["claim_scope"]
    report = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "policy_was_executed": False,
        "config": str(source),
        "backend": backend,
        "claim_scope": "%s; %s" % (backend_claim_scope, observation_claim_scope),
        "backend_claim_scope": backend_claim_scope,
        "observation_mode": observation_mode,
        "sensor_track": sensor_track,
        "observation_claim_scope": observation_claim_scope,
        "gates": gates,
        "files": files,
        "missing_configuration_fields": missing_configuration,
        "validation_error": validation_error,
        "robot": robot,
        "policy": policy,
        "compatibility": compatibility,
        "runtime": runtime,
        "environment_physics": environment,
        "observation_verification": observation_verification,
        "observation_deployability": observation_deployability,
        "flat_ground_gate": flat_ground_gate,
        "outcome_class": observation_verification.get("outcome_class")
        or observation_deployability.get("outcome_class"),
        "next_command": command if ready else None,
        "meaning": "ready means the declared interfaces and backend capabilities are present; locomotion performance remains unmeasured until execution",
    }
    destination = output or str(Path(raw.get("output_dir", base / "runs")) / "preflight.json")
    destination_path = Path(destination)
    if not destination_path.is_absolute():
        destination_path = (base / destination_path).resolve()
    report["output"] = str(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(str(destination_path), report)
    return report
