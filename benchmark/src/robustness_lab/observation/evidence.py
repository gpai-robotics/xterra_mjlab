"""Simulator-measured evidence for observation verification gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Dict, Optional

from ..backends.base import BackendUnavailable
from ..backends.worker import _backend


def _plain(value: Any) -> Any:
    """Convert compiled observations to JSON-serialisable containers."""

    if isinstance(value, Mapping):
        return {str(name): _plain(values) for name, values in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        return float(value)
    raise TypeError(
        "observation contains a non-JSON value of type %s"
        % type(value).__name__
    )


def measure_observation_evidence(
    backend_name,
    manifest,
    robot,
    robot_asset,
    robot_format,
) -> Optional[Dict[str, Any]]:
    """Measure G2/G3 evidence by resetting the simulator, or return None."""

    try:
        backend = _backend(str(backend_name))
        capabilities = backend.probe()
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return None
    if not capabilities.available:
        return None

    try:
        state = backend.reset_canonical_state(
            robot, robot_asset, robot_format
        )
    except (
        AttributeError,
        BackendUnavailable,
        ImportError,
        ModuleNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None
    if state is None:
        return None

    try:
        manifest.observation_plan.reset()
        compiled_observation = manifest.observation_plan.build(state)
        compiled_observation = _plain(compiled_observation)
        defaults = [
            float(robot.default_joint_positions[name])
            for name in robot.controller_joint_order
        ]
        joint_position_error = [
            float(position) - default
            for position, default in zip(state.joint_position, defaults)
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return {
        "provenance": "measured",
        "compiled_observation": compiled_observation,
        "known_pose_terms": {
            "projected_gravity_b": _plain(state.projected_gravity_b),
            "joint_position_error": joint_position_error,
        },
    }
