"""Schema-v2 policy manifests and their compiled observation/action boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io import load_data
from .models import RobotProfile
from .observation.action import ActionDecoder, compile_action
from .observation.compiler import (
    MultiTensorPlan,
    ObservationPlan,
    compile_encoder_tensor,
    compile_source_list,
    compile_terms,
)
from .observation.encoders import CompiledEncoder, compile_encoder


POLICY_SCHEMA_VERSION = "2.0"
SUPPORTED_RUNTIME_FORMATS = ("torchscript", "onnx", "adapter", "native")


@dataclass(frozen=True)
class PolicyManifest:
    """A validated v2 manifest with compiled observation and action pipelines."""

    policy_id: str
    observation_plan: Any
    action: ActionDecoder
    runtime: Dict[str, Any]
    encoders: Dict[str, CompiledEncoder] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = POLICY_SCHEMA_VERSION
    manifest_path: str = ""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return int(value)


def _runtime_config(
    value: Any, manifest_path: Path, check_files: bool
) -> Dict[str, Any]:
    runtime = dict(_mapping(value, "runtime"))
    runtime_format = runtime.get("format")
    if runtime_format not in SUPPORTED_RUNTIME_FORMATS:
        raise ValueError(
            "unsupported runtime format %r; supported values are: %s"
            % (runtime_format, ", ".join(SUPPORTED_RUNTIME_FORMATS))
        )
    artifact_value = runtime.get("path")
    if not isinstance(artifact_value, str) or not artifact_value:
        raise ValueError("runtime.path is required")
    artifact = Path(artifact_value)
    if not artifact.is_absolute():
        artifact = manifest_path.parent / artifact
    artifact = artifact.resolve()
    if check_files and not artifact.is_file():
        raise ValueError("policy artifact does not exist: %s" % artifact)
    runtime["path"] = str(artifact)

    frequency = runtime.get("control_frequency_hz")
    if isinstance(frequency, bool):
        raise ValueError("runtime.control_frequency_hz must be positive")
    try:
        numeric_frequency = float(frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime.control_frequency_hz must be positive") from exc
    if numeric_frequency <= 0.0:
        raise ValueError("runtime.control_frequency_hz must be positive")
    runtime["control_frequency_hz"] = numeric_frequency
    return runtime


def _profile_default_pose(profile: Any) -> Optional[List[float]]:
    """The bound robot's stance, ordered by its controller joint order."""

    if profile is None:
        return None
    order = list(getattr(profile, "controller_joint_order", None) or [])
    stance = dict(getattr(profile, "default_joint_positions", None) or {})
    if not order or not stance:
        return None
    if any(name not in stance for name in order):
        return None
    return [float(stance[name]) for name in order]


def _reference_values(
    raw: Mapping[str, Any], action_size: int, profile: Any = None
) -> Tuple[Dict[str, Any], List[str]]:
    supplied = raw.get("references", {})
    references = dict(_mapping(supplied, "references"))

    # The default pose belongs to the selected robot rather than the policy
    # artifact, so it comes from the bound robot profile.  It is never
    # invented: a manifest that names default_joint_position without either a
    # bound profile or an explicit list is refused at compile time.  Filling it
    # with zeros instead would make `subtract: default_joint_position` a no-op
    # and feed absolute joint angles to a policy trained on offsets -- a large
    # constant error at exactly the right width, which no later gate can see.
    profiled = _profile_default_pose(profile)
    declared = references.get("default_joint_position")
    notes: List[str] = []
    if declared is not None and profiled is not None:
        try:
            declared_values = [float(value) for value in declared]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "references.default_joint_position must be a list of numbers"
            ) from exc
        differs = len(declared_values) != len(profiled) or any(
            abs(left - right) > 1e-6
            for left, right in zip(declared_values, profiled)
        )
        if differs:
            # A reference pose that is a PERMUTATION of the robot's is not a
            # design choice: the same angles in a different order means the
            # manifest was written against a different joint ordering, which
            # would scramble every leg. A genuinely different pose is normal --
            # the reference pose a policy was trained to offset from belongs to
            # the policy, not to the robot -- so that is recorded, not refused.
            same_multiset = len(declared_values) == len(profiled) and sorted(
                round(value, 6) for value in declared_values
            ) == sorted(round(value, 6) for value in profiled)
            if same_multiset:
                raise ValueError(
                    "references.default_joint_position holds the robot "
                    "profile's stance in a different order, which means the "
                    "manifest is written against a different joint ordering: "
                    "manifest %r, robot %r"
                    % (declared_values, profiled)
                )
            notes.append(
                "references.default_joint_position differs from the robot "
                "profile stance (manifest %r, robot %r); using the manifest "
                "value, which is the pose the policy was trained to offset "
                "from" % (declared_values, profiled)
            )
    elif declared is None and profiled is not None:
        references["default_joint_position"] = profiled
    return references, notes


def _encoders_config(
    value: Any, manifest_path: Path
) -> Dict[str, CompiledEncoder]:
    if value is None:
        return {}
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("encoders must be a sequence of mappings")
    compiled = {}
    for index, raw_spec in enumerate(value):
        if not isinstance(raw_spec, Mapping):
            raise ValueError("encoder %d must be a mapping" % index)
        spec = dict(raw_spec)
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("encoder %d name must be a non-empty string" % index)
        if name in compiled:
            raise ValueError("encoder names must be unique; duplicate %r" % name)
        artifact_value = spec.get("path")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise ValueError("encoder %r path is required" % name)
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            artifact = manifest_path.parent / artifact
        spec["path"] = str(artifact.resolve())
        compiled[name] = compile_encoder(spec)
    return compiled


def _robot_profile(value: Any, manifest_path: Path) -> Optional[RobotProfile]:
    if value is None:
        return None
    if isinstance(value, RobotProfile):
        return value
    if isinstance(value, Mapping):
        return RobotProfile.from_dict(dict(value))
    if isinstance(value, (str, Path)):
        profile_path = Path(value)
        if not profile_path.is_absolute():
            profile_path = manifest_path.parent / profile_path
        return RobotProfile.from_dict(load_data(str(profile_path.resolve())))
    raise ValueError("robot_profile must be a RobotProfile, mapping, or path")


def _matching_profile_names(
    declared: Any,
    profiled: Optional[Sequence[str]],
    label: str,
) -> Any:
    if declared is None:
        return None if profiled is None else list(profiled)
    if profiled is not None and list(declared) != list(profiled):
        raise ValueError("%s does not match the robot profile" % label)
    return declared


def _observation_tensors(
    tensors: Any,
    encoders: Mapping[str, CompiledEncoder],
    references: Mapping[str, Any],
    canonical_joint_order: Any,
    policy_joint_order: Any,
    allow_custom: bool,
) -> MultiTensorPlan:
    if isinstance(tensors, (str, bytes)) or not isinstance(tensors, Sequence):
        raise ValueError("observation tensors must be a sequence of mappings")
    plans = []
    names = set()
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, Mapping):
            raise ValueError("observation tensor %d must be a mapping" % index)
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "observation tensor %d name must be a non-empty string" % index
            )
        if name in names:
            raise ValueError(
                "observation tensor names must be unique; duplicate %r" % name
            )
        names.add(name)

        encoder_name = tensor.get("from_encoder")
        if encoder_name is None:
            if "terms" not in tensor:
                raise ValueError(
                    "observation tensor %r requires terms or from_encoder" % name
                )
            plan = compile_terms(
                tensor.get("terms"),
                references=references,
                canonical_joint_order=canonical_joint_order,
                policy_joint_order=policy_joint_order,
                history=tensor.get("history"),
                allow_custom=allow_custom,
            )
            if "size" in tensor:
                declared_size = _positive_int(
                    tensor.get("size"), "observation tensor %r size" % name
                )
                if declared_size != plan.size:
                    raise ValueError(
                        "observation tensor %r declares size %d, but its terms produce %d"
                        % (name, declared_size, plan.size)
                    )
        else:
            if "terms" in tensor:
                raise ValueError(
                    "observation tensor %r cannot declare both terms and from_encoder"
                    % name
                )
            if not isinstance(encoder_name, str) or encoder_name not in encoders:
                raise ValueError(
                    "observation tensor %r references unknown encoder %r"
                    % (name, encoder_name)
                )
            plan = compile_encoder_tensor(
                encoders[encoder_name],
                tensor.get("transform", []),
                tensor.get("size"),
                references=references,
                allow_custom=allow_custom,
            )
        plans.append((name, plan))
    if not plans:
        raise ValueError("observation tensors must not be empty")
    return MultiTensorPlan(plans)


def load_policy_manifest(
    path: str,
    check_files: bool = True,
    robot_profile: Any = None,
) -> PolicyManifest:
    """Load and compile a schema-v2 policy manifest."""

    raw = load_data(path)
    version = str(raw.get("schema_version", ""))
    if version != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported policy schema_version %r" % version)

    policy_id = raw.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id is required")

    manifest_path = Path(path).resolve()
    runtime = _runtime_config(raw.get("runtime"), manifest_path, check_files)
    encoders = _encoders_config(raw.get("encoders"), manifest_path)
    observation = _mapping(raw.get("observation"), "observation")
    allow_custom = observation.get("allow_custom", False)
    if not isinstance(allow_custom, bool):
        raise ValueError("observation.allow_custom must be a boolean")
    action_config = dict(_mapping(raw.get("action"), "action"))
    action_size = _positive_int(action_config.get("size"), "action size")
    profile = _robot_profile(
        robot_profile if robot_profile is not None else raw.get("robot_profile"),
        manifest_path,
    )
    references, reference_notes = _reference_values(raw, action_size, profile)

    policy_joint_order = raw.get("joint_order_policy")
    canonical_joint_order = _matching_profile_names(
        raw.get("canonical_joint_order"),
        None if profile is None else profile.controller_joint_order,
        "canonical_joint_order",
    )
    foot_names = _matching_profile_names(
        observation.get(
            "foot_names",
            raw.get("foot_names", raw.get("canonical_foot_order")),
        ),
        None if profile is None else profile.foot_links,
        "foot_names",
    )
    if policy_joint_order is not None:
        if isinstance(policy_joint_order, (str, bytes)) or not isinstance(
            policy_joint_order, Sequence
        ):
            raise ValueError("joint_order_policy must be a sequence of joint names")
        if any(not isinstance(name, str) or not name for name in policy_joint_order):
            raise ValueError("joint_order_policy must contain non-empty joint names")
        action_config.setdefault("joint_order", list(policy_joint_order))

    # Older scaffold consumers read its generated, sized ``terms`` directly.
    # A marked scaffold carries that mirror for compatibility, but compilation
    # still uses ``sources``. Unmarked researcher manifests may never declare
    # both forms.
    scaffold_terms_mirror = (
        observation.get("_scaffold_source_list") is True
        and "sources" in observation
        and "terms" in observation
    )
    declared_forms = [
        key
        for key in ("sources", "terms", "tensors")
        if key in observation and not (key == "terms" and scaffold_terms_mirror)
    ]
    if len(declared_forms) > 1:
        raise ValueError(
            "observation cannot declare both %s"
            % " and ".join(declared_forms)
        )

    if "tensors" in observation:
        observation_plan = _observation_tensors(
            observation.get("tensors"),
            encoders,
            references,
            canonical_joint_order,
            policy_joint_order,
            allow_custom,
        )
        if "expected_size" in observation:
            expected_size = _positive_int(
                observation.get("expected_size"), "observation expected_size"
            )
            if expected_size != observation_plan.size:
                raise ValueError(
                    "observation expected_size is %d, but the compiled size is %d"
                    % (expected_size, observation_plan.size)
                )
    elif "sources" in observation:
        # Source mappings may declare a per-entry ``size`` override. The
        # compiler derives widths only for neighboring entries that omit it.
        observation_plan = compile_source_list(
            observation.get("sources"),
            joint_order=canonical_joint_order,
            foot_names=foot_names,
            geometry=observation.get("geometry"),
            references=references,
            policy_joint_order=policy_joint_order,
            history=observation.get("history"),
            allow_custom=allow_custom,
        )
        if "expected_size" in observation:
            expected_size = _positive_int(
                observation.get("expected_size"), "observation expected_size"
            )
            if expected_size != observation_plan.size:
                raise ValueError(
                    "observation expected_size is %d, but the compiled size is %d"
                    % (expected_size, observation_plan.size)
                )
    else:
        observation_plan = compile_terms(
            observation.get("terms"),
            references=references,
            canonical_joint_order=canonical_joint_order,
            policy_joint_order=policy_joint_order,
            history=observation.get("history"),
            allow_custom=allow_custom,
        )
        expected_size = _positive_int(
            observation.get("expected_size"), "observation expected_size"
        )
        if expected_size != observation_plan.size:
            raise ValueError(
                "observation expected_size is %d, but the compiled size is %d"
                % (expected_size, observation_plan.size)
            )

    action = compile_action(
        action_config,
        references=references,
        canonical_joint_order=canonical_joint_order,
    )
    return PolicyManifest(
        policy_id=policy_id,
        observation_plan=observation_plan,
        action=action,
        runtime=runtime,
        encoders=encoders,
        warnings=reference_notes,
        manifest_path=str(manifest_path),
    )


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "SUPPORTED_RUNTIME_FORMATS",
    "PolicyManifest",
    "load_policy_manifest",
]
