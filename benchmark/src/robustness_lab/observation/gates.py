"""Verification gates for the policy observation and action boundary.

Failures in this module describe an adapter whose correctness has not been
established.  They must never be interpreted as evidence about policy quality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from robustness_lab.artifacts import load_artifact
from robustness_lab.canonical_state import AVAILABILITY_CLASSES, CANONICAL_FIELDS


OUTCOME_CLASSES = frozenset(
    {
        "policy_pass",
        "policy_fail",
        "adapter_unverified",
        "environment_unsupported",
        "infrastructure_error",
        "pruned",
    }
)

DEPLOYABLE_AVAILABILITY_CLASSES = frozenset(
    {"onboard", "onboard_derived", "command"}
)

WIDTH_DISCRIMINATION_WARNING = (
    "artifact accepts multiple observation widths, so width acceptance cannot "
    "confirm this manifest; the known-pose and flat-ground gates must verify "
    "the observation construction"
)

ROUTE_ASSISTANCE_SOURCES = frozenset(
    {"waypoints_b", "goal_vector_b", "goal_heading", "goal_distance"}
)


def _validated_sources(sources: Iterable[str]) -> set[str]:
    source_names = set(sources)
    unknown = sorted(source_names - set(CANONICAL_FIELDS))
    if unknown:
        raise ValueError(
            "unknown canonical observation source(s): %s" % ", ".join(unknown)
        )
    return source_names


def derive_sensor_track(sources: Iterable[str]) -> str:
    """Derive the strongest perception track from declared canonical sources."""

    source_names = _validated_sources(sources)
    if any(
        CANONICAL_FIELDS[name]["availability"] == "simulator_only"
        for name in source_names
    ):
        return "privileged"
    if any(CANONICAL_FIELDS[name].get("perception") is True for name in source_names):
        return "exteroceptive"
    return "blind"


def resolve_allowed_classes(mode: str) -> set[str]:
    """Resolve a run's observation mode to its allowed availability classes."""

    if mode == "deployable_only":
        return set(DEPLOYABLE_AVAILABILITY_CLASSES)
    if mode == "unrestricted":
        return set(AVAILABILITY_CLASSES)
    raise ValueError("unknown observation mode %r" % mode)


def claim_scope_for_classes(classes: Iterable[str]) -> str:
    """Describe the claim supported by the information a policy consumed."""

    classes_used = set(classes)
    unknown = sorted(classes_used - set(AVAILABILITY_CLASSES))
    if unknown:
        raise ValueError(
            "unknown availability class(es): %s" % ", ".join(unknown)
        )
    if "simulator_only" in classes_used:
        return "simulation/privileged evaluation only; not a deployment claim"
    if "infrastructure" in classes_used:
        return (
            "infrastructure-assisted evaluation; not a field-autonomous "
            "deployment claim"
        )
    return "deployable evaluation using onboard information and commands"


def claim_scope_for_sources(sources: Iterable[str]) -> str:
    """Describe perception, route assistance, and deployment claim scope."""

    source_names = _validated_sources(sources)
    track = derive_sensor_track(source_names)
    route_scope = (
        "route-assisted by declared goal/waypoint commands"
        if source_names & ROUTE_ASSISTANCE_SOURCES
        else "no route assistance"
    )
    classes = {
        CANONICAL_FIELDS[name]["availability"] for name in source_names
    }
    return "%s sensor track; %s; %s" % (
        track,
        route_scope,
        claim_scope_for_classes(classes),
    )


def _blocked(reason: str, **details: Any) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "outcome_class": "adapter_unverified",
        "reason": reason,
        **details,
    }


def check_flat_ground(
    distance_m: float,
    duration_s: float,
    required_distance_m: float,
    fell: bool = False,
) -> Dict[str, Any]:
    """Qualify an observation/action adapter before terrain is scored.

    Failure here is deliberately an adapter outcome: flat ground cannot tell us
    whether the policy is incapable or its simulator mapping is wrong.
    """

    try:
        distance = float(distance_m)
        duration = float(duration_s)
        required_distance = float(required_distance_m)
    except (TypeError, ValueError):
        return _blocked(
            "flat-ground setup evidence is invalid; the adapter is unverified "
            "and terrain results are not scored",
            distance_m=distance_m,
            duration_s=duration_s,
            required_distance_m=required_distance_m,
            fell=bool(fell),
        )

    details = {
        "distance_m": distance,
        "duration_s": duration,
        "required_distance_m": required_distance,
        "fell": bool(fell),
    }
    valid_evidence = (
        math.isfinite(distance)
        and math.isfinite(duration)
        and math.isfinite(required_distance)
        and duration > 0.0
        and required_distance >= 0.0
    )
    if not valid_evidence or fell or distance < 0.0 or distance < required_distance:
        return _blocked(
            "policy did not walk on flat ground; the observation or actuation "
            "setup is unverified, so terrain results are not scored",
            **details,
        )
    return {"status": "ready", **details}


def check_dimension(
    compiled_size: int,
    declared_size: int,
    artifact_size: int,
    width_is_discriminating: Optional[bool] = None,
) -> Dict[str, Any]:
    """Require compiler, manifest, and probed artifact dimensions to agree."""

    sizes = {
        "compiled_size": compiled_size,
        "declared_size": declared_size,
        "artifact_size": artifact_size,
    }
    if len(set(sizes.values())) != 1:
        return _blocked(
            "observation dimensions disagree: compiled_size=%r, "
            "declared_size=%r, artifact_size=%r"
            % (compiled_size, declared_size, artifact_size),
            **sizes,
        )
    result = {"status": "ready", **sizes}
    if width_is_discriminating is False:
        result["warning"] = WIDTH_DISCRIMINATION_WARNING
    return result


def check_exteroceptive_support(
    required: Iterable[str], supported: Iterable[str]
) -> Dict[str, Any]:
    """Block when an adapter cannot produce every required sensor field."""

    missing = sorted(set(required) - set(supported))
    if missing:
        return _blocked(
            "adapter does not support required exteroceptive fields: %s"
            % ", ".join(missing),
            missing=missing,
        )
    return {"status": "ready", "missing": []}


def check_deployability(
    sources: Iterable[str], allowed: Iterable[str]
) -> Dict[str, Any]:
    """Check that every observation source has an allowed availability class."""

    source_names = _validated_sources(sources)

    allowed_classes = set(allowed)
    classes_by_source = {
        name: CANONICAL_FIELDS[name]["availability"] for name in source_names
    }
    classes_used = sorted(set(classes_by_source.values()))
    rejected = sorted(
        name
        for name, availability in classes_by_source.items()
        if availability not in allowed_classes
    )
    details = {
        "rejected": rejected,
        "classes_used": classes_used,
        "sensor_track": derive_sensor_track(source_names),
        "claim_scope": claim_scope_for_sources(source_names),
    }
    if rejected:
        rejected_details = ", ".join(
            "%s (%s)" % (name, classes_by_source[name]) for name in rejected
        )
        return _blocked(
            "observation sources use disallowed availability classes: %s"
            % rejected_details,
            **details,
        )
    return {"status": "ready", **details}


def check_artifact_accepts(
    path: str, input_size: int, runtime_format: Optional[str] = None
) -> Dict[str, Any]:
    """Execute a policy and determine whether input width is discriminating."""

    try:
        artifact = load_artifact(path, runtime_format=runtime_format)
        observation = np.zeros((1, input_size), dtype=np.float32)
        output = artifact.run([observation])
        shape = tuple(int(value) for value in output.shape)
        if not shape:
            raise ValueError("artifact returned a scalar rather than an action tensor")
    except Exception as error:
        return _blocked(
            "artifact %r did not accept input_size=%r: %s"
            % (path, input_size, error),
            path=path,
            input_size=input_size,
        )

    far_offset = max(8, input_size // 16)
    offsets = (
        -2 * far_offset,
        -far_offset,
        -3,
        -1,
        0,
        1,
        3,
        far_offset,
        2 * far_offset,
    )
    probe_widths = sorted(
        {input_size + offset for offset in offsets if input_size + offset > 0}
    )
    accepted_widths = [input_size]
    for probe_width in probe_widths:
        if probe_width == input_size:
            continue
        try:
            probe = np.zeros((1, probe_width), dtype=np.float32)
            probe_output = artifact.run([probe])
            probe_shape = tuple(int(value) for value in probe_output.shape)
            if probe_shape:
                accepted_widths.append(probe_width)
        except Exception:
            continue
    accepted_widths.sort()
    width_is_discriminating = accepted_widths == [input_size]
    return {
        "status": "ready",
        "path": path,
        "input_size": input_size,
        "output_size": shape[-1],
        "width_is_discriminating": width_is_discriminating,
        "accepted_widths": accepted_widths,
        **(
            {}
            if width_is_discriminating
            else {"warning": WIDTH_DISCRIMINATION_WARNING}
        ),
    }


def check_artifact_accepts_tensors(
    path: str,
    input_sizes: Sequence[int],
    runtime_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a multi-input portable policy once with zero tensors."""

    sizes = [int(value) for value in input_sizes]
    try:
        artifact = load_artifact(path, runtime_format=runtime_format)
        inputs = [
            np.zeros((1, input_size), dtype=np.float32)
            for input_size in sizes
        ]
        output = artifact.run(inputs)
        shape = tuple(int(value) for value in output.shape)
        if not shape:
            raise ValueError("artifact returned a scalar rather than an action tensor")
    except Exception as error:
        return _blocked(
            "artifact %r did not accept input_sizes=%r: %s"
            % (path, sizes, error),
            path=path,
            input_sizes=sizes,
        )
    return {
        "status": "ready",
        "path": path,
        "input_sizes": sizes,
        "output_size": shape[-1],
    }


def check_artifact_arity(
    artifact_path: str,
    declared_tensor_count: int,
    runtime_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Require a manifest to declare every positional artifact input."""

    try:
        artifact = load_artifact(artifact_path, runtime_format=runtime_format)
    except ValueError as error:
        return _blocked(str(error), path=artifact_path)

    artifact_input_count = len(artifact.input_names)
    input_names = list(artifact.input_names)
    details = {
        "path": artifact_path,
        "declared_tensor_count": declared_tensor_count,
        "artifact_input_count": artifact_input_count,
        "input_names": input_names,
    }
    if artifact_input_count != declared_tensor_count:
        return _blocked(
            "artifact input count=%r does not match declared tensor count=%r; "
            "artifact input names=%r"
            % (artifact_input_count, declared_tensor_count, input_names),
            **details,
        )
    return {"status": "ready", **details}


def _indexed_values(
    values: Any, index: Tuple[Any, ...] = ()
) -> Iterable[Tuple[Tuple[Any, ...], float]]:
    if isinstance(values, Real):
        yield index, float(values)
        return
    if isinstance(values, Mapping):
        for name, value in values.items():
            yield from _indexed_values(value, index + (str(name),))
        return
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("values must be a number or a sequence of numbers")
    for position, value in enumerate(values):
        yield from _indexed_values(value, index + (position,))


def _public_index(index: Tuple[Any, ...]) -> Any:
    if len(index) == 1:
        return index[0]
    return list(index)


def check_finite(values: Any) -> Dict[str, Any]:
    """Block observations containing NaN or either infinity."""

    try:
        offending = [
            _public_index(index)
            for index, value in _indexed_values(values)
            if not math.isfinite(value)
        ]
    except (TypeError, ValueError) as error:
        return _blocked(str(error), offending_indices=[])
    if offending:
        return _blocked(
            "observation contains non-finite values at indices %r" % offending,
            offending_indices=offending,
        )
    return {"status": "ready", "offending_indices": []}


# A scalar expectation is broadcast across the observed term.  A sequence has
# an exact expected shape.  Adding another closed-form replay check is therefore
# a data-table edit rather than a new term-specific control-flow branch.
KNOWN_POSE_EXPECTATIONS = {
    "projected_gravity_b": (0.0, 0.0, -1.0),
    "joint_position_error": 0.0,
}


def _matches_expectation(values: Any, expectation: Any, tolerance: float) -> bool:
    try:
        observed = [value for _, value in _indexed_values(values)]
    except (TypeError, ValueError):
        return False
    if isinstance(expectation, Real):
        expected = [float(expectation)] * len(observed)
    else:
        expected = [value for _, value in _indexed_values(expectation)]
    return len(observed) == len(expected) and all(
        math.isfinite(value) and abs(value - target) <= tolerance
        for value, target in zip(observed, expected)
    )


def check_known_pose(
    terms: Mapping[str, Any], tolerance: float = 1e-3
) -> Dict[str, Any]:
    """Compare canonical default-pose observations with closed-form values."""

    if not isinstance(terms, Mapping):
        return _blocked(
            "known-pose terms must be a mapping",
            failing_terms=list(KNOWN_POSE_EXPECTATIONS),
        )
    failing = [
        name
        for name, expectation in KNOWN_POSE_EXPECTATIONS.items()
        if name in terms
        and not _matches_expectation(terms[name], expectation, tolerance)
    ]
    if failing:
        return _blocked(
            "known-pose replay failed for terms: %s" % ", ".join(failing),
            failing_terms=failing,
        )
    return {"status": "ready", "failing_terms": []}


__all__ = [
    "OUTCOME_CLASSES",
    "KNOWN_POSE_EXPECTATIONS",
    "DEPLOYABLE_AVAILABILITY_CLASSES",
    "claim_scope_for_classes",
    "claim_scope_for_sources",
    "derive_sensor_track",
    "check_artifact_accepts",
    "check_artifact_arity",
    "check_deployability",
    "check_dimension",
    "check_exteroceptive_support",
    "check_finite",
    "check_known_pose",
    "resolve_allowed_classes",
]
