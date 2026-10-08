"""Draft schema-v2 policy manifests from probed portable artifacts.

The observation declarations produced here are hypotheses.  Artifact probing
can establish tensor widths, but it cannot recover the semantic meaning or
preprocessing of each input feature.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .artifacts import load_artifact
from .io import dump_json, dump_yaml, load_data
from .models import RobotProfile
from .observation.compiler import compile_source_list
from .observation.gates import check_artifact_accepts
from .policy_manifest import load_policy_manifest


def probe_artifact_signature(
    artifact_path: str, runtime_format: Optional[str] = None
) -> Dict[str, Any]:
    """Return the ordered input signature of a portable artifact."""

    artifact = load_artifact(
        str(artifact_path), runtime_format=runtime_format
    )
    input_names = list(artifact.input_names)

    return {
        "input_count": len(input_names),
        "input_names": input_names,
    }


def _load_profile(value: Any) -> Optional[RobotProfile]:
    if value is None:
        return None
    if isinstance(value, RobotProfile):
        return value
    if isinstance(value, (str, Path)):
        return RobotProfile.from_dict(load_data(str(value)))
    raise ValueError("robot_profile must be a RobotProfile or a profile path")


def _joint_names(
    joint_order: Optional[Sequence[str]], profile: Optional[RobotProfile]
) -> List[str]:
    supplied = None if joint_order is None else list(joint_order)
    profiled = None if profile is None else list(profile.controller_joint_order)
    if supplied is not None and profiled is not None and supplied != profiled:
        raise ValueError(
            "joint_order does not match RobotProfile.controller_joint_order"
        )
    names = supplied if supplied is not None else profiled
    if not names:
        raise ValueError("joint_order or robot_profile is required")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_order must contain non-empty joint names")
    if len(set(names)) != len(names):
        raise ValueError("joint_order must not contain duplicate names")
    return names


def _probe_artifact(path: str) -> Tuple[int, int, str]:
    artifact_name = Path(path).name
    artifact = load_artifact(path)
    candidates = artifact.candidate_input_widths()

    if not candidates:
        raise ValueError(
            "could not determine an input width to probe for %s artifact %s"
            % (artifact.label, artifact_name)
        )

    failures = []
    for input_size in candidates:
        result = check_artifact_accepts(
            path, input_size, runtime_format=artifact.format
        )
        if result.get("status") == "ready":
            return input_size, int(result["output_size"]), artifact.format
        failures.append(result.get("reason", "probe failed"))
    raise ValueError(
        "could not verify an input width for %s artifact %s: %s"
        % (artifact.label, artifact_name, "; ".join(failures))
    )


def _proprioceptive_terms(
    joint_count: int,
    include_base_linear_velocity: bool,
    default_joint_positions: Sequence[float],
) -> List[Dict[str, Any]]:
    terms: List[Dict[str, Any]] = []
    if include_base_linear_velocity:
        terms.append({"source": "base_linear_velocity_b", "size": 3})
    terms.extend(
        [
            {
                "source": "base_angular_velocity_b",
                "size": 3,
                "transform": [{"scale": 0.25}],
            },
            {"source": "projected_gravity_b", "size": 3},
            {
                "source": "velocity_command",
                "size": 3,
                "transform": [{"scale": [2.0, 2.0, 0.25]}],
            },
            {
                "source": "joint_position",
                "size": joint_count,
                "transform": [{"subtract": list(default_joint_positions)}],
            },
            {
                "source": "joint_velocity",
                "size": joint_count,
                "transform": [{"scale": 0.05}],
            },
            {
                "source": "previous_action",
                "size": joint_count,
            },
        ]
    )
    return terms


def _height_scan_shape(size: int) -> Tuple[int, int]:
    """Choose an x-by-y factor pair near the 17-by-11 reference aspect."""

    target_aspect = 17.0 / 11.0
    candidates = []
    for y_count in range(1, math.isqrt(size) + 1):
        if size % y_count == 0:
            x_count = size // y_count
            candidates.append(
                (abs(x_count / float(y_count) - target_aspect), x_count, y_count)
            )
    _, x_count, y_count = min(candidates)
    return x_count, y_count


def _height_scan_range(point_count: int, resolution: float) -> List[float]:
    half_span = round((point_count - 1) * resolution / 2.0, 12)
    lower = 0.0 if half_span == 0.0 else -half_span
    return [lower, half_span]


def _height_scan_term(size: int) -> Dict[str, Any]:
    x_count, y_count = _height_scan_shape(size)
    resolution = 0.1
    return {
        "source": "height_scan",
        "size": size,
        "geometry": {
            "kind": "grid",
            "x_range": _height_scan_range(x_count, resolution),
            "y_range": _height_scan_range(y_count, resolution),
            "resolution": resolution,
        },
    }


def _observation_terms(
    input_size: int,
    joint_count: int,
    default_joint_positions: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    layout_a = _proprioceptive_terms(
        joint_count, False, default_joint_positions
    )
    layout_b = _proprioceptive_terms(
        joint_count, True, default_joint_positions
    )
    layouts = [
        ("A (proprioception without base linear velocity)", layout_a),
        ("B (proprioception with base linear velocity)", layout_b),
    ]
    for name, terms in layouts:
        if sum(int(term["size"]) for term in terms) == input_size:
            return terms, name

    # Retain both established 187-point forms before resolving the otherwise
    # ambiguous choice between A + (R + 3) and B + R.
    for name, terms in layouts:
        if input_size - sum(int(term["size"]) for term in terms) == 187:
            return (
                terms + [_height_scan_term(187)],
                "%s plus the 187-point legged_gym height scan" % name[0],
            )

    # Layout B is the common full-proprioception form. Prefer it when both
    # bases leave a positive remainder, while still allowing A-only widths.
    for name, terms in reversed(layouts):
        remainder = input_size - sum(int(term["size"]) for term in terms)
        if remainder > 0:
            return (
                terms + [_height_scan_term(remainder)],
                "%s plus an inferred %d-point height scan" % (name[0], remainder),
            )

    # A tensor width cannot establish observation semantics or exteroceptive
    # geometry. Returning no terms lets the caller report that human input is
    # required without manufacturing a plausible but false declaration.
    return [], None


def _default_positions(profile: Optional[RobotProfile], joints: Sequence[str]) -> List[float]:
    if profile is None:
        return [0.0] * len(joints)
    return [float(profile.default_joint_positions.get(name, 0.0)) for name in joints]


def _computed_source_list(
    terms: Sequence[Dict[str, Any]],
    input_size: int,
    joints: Sequence[str],
    foot_names: Sequence[str],
    default_positions: Sequence[float],
) -> Optional[List[Dict[str, Any]]]:
    sources = [
        {key: value for key, value in term.items() if key != "size"}
        for term in terms
    ]
    if not sources:
        return None
    try:
        plan = compile_source_list(
            sources,
            joint_order=joints,
            foot_names=foot_names,
            references={"default_joint_position": list(default_positions)},
            policy_joint_order=joints,
        )
    except ValueError:
        return None
    return sources if plan.size == input_size else None


def scaffold_policy_manifest(
    artifact_path: str,
    joint_order: Optional[Sequence[str]],
    output: str,
    robot_profile: Any = None,
) -> Dict[str, Any]:
    """Probe an artifact and write a draft or an incomplete review skeleton."""

    profile = _load_profile(robot_profile)
    joints = _joint_names(joint_order, profile)
    default_positions = _default_positions(profile, joints)
    input_size, output_size, runtime_format = _probe_artifact(str(artifact_path))
    if output_size != len(joints):
        raise ValueError(
            "artifact output_size=%d does not match joint_order length %d"
            % (output_size, len(joints))
        )

    terms, assumed_layout = _observation_terms(
        input_size, len(joints), default_positions
    )
    decomposed = assumed_layout is not None
    sources = _computed_source_list(
        terms,
        input_size,
        joints,
        [] if profile is None else profile.foot_links,
        default_positions,
    )
    needs_review = [
        "Confirm the hypothesized observation term semantics and ordering.",
        "Confirm every observation transform scale and preprocessing operation.",
        "Confirm the action scale, offset, clipping, stiffness/damping gains "
        "from the researcher's deployment config, and control frequency.",
    ]
    if decomposed:
        needs_review.insert(
            0,
            "Confirm assumed observation layout %s." % assumed_layout,
        )
        if any(term.get("source") == "height_scan" for term in terms):
            needs_review.insert(
                1,
                "Confirm the height-scan grid inferred from the artifact input "
                "width; multiple grid footprints have the same point count.",
            )
    else:
        needs_review.insert(
            0,
            "The observation layout for input width %d could not be inferred "
            "and must be written by hand."
            % input_size,
        )

    artifact = Path(artifact_path).resolve()
    destination = Path(output).resolve()
    manifest = {
        "schema_version": "2.0",
        "policy_id": artifact.stem,
        "runtime": {
            "format": runtime_format,
            "path": str(artifact),
            "control_frequency_hz": (
                float(profile.control_frequency_hz) if profile is not None else 50.0
            ),
        },
        "canonical_joint_order": joints,
        "joint_order_policy": joints,
        "references": {
            "default_joint_position": default_positions,
        },
        "observation": dict(
            (
                {
                    "sources": sources,
                    # Compatibility mirror for callers of scaffold releases
                    # that consumed these inferred terms programmatically.
                    "terms": terms,
                    "_scaffold_source_list": True,
                }
                if sources is not None
                else {"terms": terms}
            ),
            history={"frames": 1, "order": "newest_first"},
            expected_size=input_size,
        ),
        "action": {
            "space": "joint_position_offset",
            "scale": 0.25,
            "offset": "default_joint_position",
            "clip": [-100.0, 100.0],
            "stiffness": None,
            "damping": None,
            "size": output_size,
        },
        "needs_review": needs_review,
    }

    if destination.suffix.lower() in {".yaml", ".yml"}:
        dump_yaml(str(destination), manifest)
    else:
        dump_json(str(destination), manifest)

    if decomposed:
        compiled = load_policy_manifest(str(destination), check_files=False)
        if compiled.observation_plan.size != input_size:
            raise ValueError(
                "drafted manifest compiled to %d inputs, but artifact accepted %d"
                % (compiled.observation_plan.size, input_size)
            )

    result = {
        "input_size": input_size,
        "output_size": output_size,
        "needs_review": needs_review,
        "path": str(destination),
        "output": str(destination),
        "terms": terms,
    }
    if sources is not None:
        result["sources"] = sources
    if not decomposed:
        result.update(
            {
                "status": "blocked",
                "outcome_class": "adapter_unverified",
                "reason": needs_review[0],
            }
        )
    return result


__all__ = ["probe_artifact_signature", "scaffold_policy_manifest"]
