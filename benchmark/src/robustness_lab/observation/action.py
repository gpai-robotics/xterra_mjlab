"""Compile policy action declarations into simulator-target decoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

from . import transforms


SUPPORTED_ACTION_SPACES = ("joint_position_offset",)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return int(value)


def _copy_sequence(value: Any, label: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("%s must be a sequence of numbers" % label)
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain only numbers" % label) from exc


def _resolve_offset(
    value: Any, references: Mapping[str, Any], size: int
) -> Tuple[List[float], bool]:
    """Return the offset and whether it came from a named reference.

    The distinction matters because the two are written in different joint
    orders: a named reference such as ``default_joint_position`` is the robot's,
    in canonical order, while an explicit list is the submitter's own and
    follows the policy's order, as ``stiffness`` and ``damping`` do.
    """

    named = isinstance(value, str)
    if named:
        if value not in references:
            raise ValueError("unknown offset reference %r" % value)
        value = references[value]
    if value is None:
        return [0.0] * size, False
    result = _copy_sequence(value, "offset")
    if len(result) != size:
        raise ValueError("offset has length %d, expected %d" % (len(result), size))
    return result, named


def _resolve_gain(
    value: Any,
    label: str,
    size: int,
    inverse_permutation: Optional[Sequence[int]],
) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        result = [float(value)] * size
    else:
        result = _copy_sequence(value, label)
        if len(result) != size:
            raise ValueError(
                "%s has length %d, expected %d" % (label, len(result), size)
            )
    if inverse_permutation is not None:
        result = [result[index] for index in inverse_permutation]
    return tuple(result)


def _inverse_permutation(
    policy_joint_order: Any,
    canonical_joint_order: Optional[Sequence[str]],
    size: int,
) -> Optional[Tuple[int, ...]]:
    if canonical_joint_order is not None and (
        isinstance(canonical_joint_order, (str, bytes))
        or not isinstance(canonical_joint_order, Sequence)
    ):
        raise ValueError("canonical_joint_order must be a sequence of joint names")
    if policy_joint_order is None:
        if canonical_joint_order is not None and len(canonical_joint_order) != size:
            raise ValueError(
                "canonical_joint_order has length %d, expected %d"
                % (len(canonical_joint_order), size)
            )
        return None
    if isinstance(policy_joint_order, (str, bytes)) or not isinstance(
        policy_joint_order, Sequence
    ):
        raise ValueError("joint_order must be a sequence of joint names")
    if canonical_joint_order is None:
        raise ValueError("joint_order requires canonical_joint_order")

    policy = list(policy_joint_order)
    canonical = list(canonical_joint_order)
    if len(policy) != size:
        raise ValueError(
            "joint_order has length %d, expected action size %d" % (len(policy), size)
        )
    if len(canonical) != size:
        raise ValueError(
            "canonical_joint_order has length %d, expected action size %d"
            % (len(canonical), size)
        )
    if any(not isinstance(name, str) or not name for name in policy):
        raise ValueError("joint_order must contain non-empty joint names")
    if any(not isinstance(name, str) or not name for name in canonical):
        raise ValueError(
            "canonical_joint_order must contain non-empty joint names"
        )
    if len(set(policy)) != len(policy):
        raise ValueError("joint_order must not contain duplicate names")
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical_joint_order must not contain duplicate names")

    canonical_names = set(canonical)
    for name in policy:
        if name not in canonical_names:
            raise ValueError(
                "policy joint %r is absent from canonical_joint_order" % name
            )
    policy_names = set(policy)
    for name in canonical:
        if name not in policy_names:
            raise ValueError("canonical joint %r is absent from joint_order" % name)

    policy_indices = {name: index for index, name in enumerate(policy)}
    return tuple(policy_indices[name] for name in canonical)


class ActionDecoder:
    """Decode raw policy output into joint targets in canonical joint order."""

    def __init__(
        self,
        size: int,
        scale: Any,
        offset: Sequence[float],
        clip: Optional[Sequence[float]],
        inverse_permutation: Optional[Sequence[int]],
        stiffness: Optional[Sequence[float]] = None,
        damping: Optional[Sequence[float]] = None,
        motor: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.size = size
        self._scale = scale
        self._offset = tuple(float(value) for value in offset)
        self._clip = None if clip is None else tuple(float(value) for value in clip)
        self._inverse_permutation = (
            None
            if inverse_permutation is None
            else tuple(int(index) for index in inverse_permutation)
        )
        self.stiffness = (
            None if stiffness is None else tuple(float(value) for value in stiffness)
        )
        self.damping = (
            None if damping is None else tuple(float(value) for value in damping)
        )
        self.gains_declared = self.stiffness is not None and self.damping is not None
        # Isaac Lab DCMotor envelope the policy was trained against, or None
        # for a plain clip to the actuator's ctrlrange.
        self.motor = None if motor is None else dict(motor)

    def decode(self, raw_action: Sequence[Real]) -> List[float]:
        """Apply clip, scale, offset, then remap policy order to canonical order."""

        values = _copy_sequence(raw_action, "raw action")
        if len(values) != self.size:
            raise ValueError(
                "raw action has length %d, expected %d" % (len(values), self.size)
            )
        if self._clip is not None:
            values = transforms.clip(values, self._clip)
        values = transforms.scale(values, self._scale)
        values = [
            value + offset for value, offset in zip(values, self._offset)
        ]
        if self._inverse_permutation is not None:
            values = [values[index] for index in self._inverse_permutation]
        return values


def compile_action(
    config: Mapping[str, Any],
    references: Optional[Mapping[str, Any]] = None,
    canonical_joint_order: Optional[Sequence[str]] = None,
) -> ActionDecoder:
    """Compile an action configuration into an :class:`ActionDecoder`."""

    if not isinstance(config, Mapping):
        raise ValueError("action config must be a mapping")
    space = config.get("space")
    if space not in SUPPORTED_ACTION_SPACES:
        raise ValueError(
            "unsupported action space %r; supported values are: %s"
            % (space, ", ".join(SUPPORTED_ACTION_SPACES))
        )
    size = _positive_int(config.get("size"), "action size")

    reference_values = {} if references is None else references
    if not isinstance(reference_values, Mapping):
        raise ValueError("references must be a mapping")

    scale_value = config.get("scale", 1.0)
    if isinstance(scale_value, Real):
        scale: Any = float(scale_value)
    else:
        scale = tuple(_copy_sequence(scale_value, "scale"))
    transforms.scale([0.0] * size, scale)
    clip = config.get("clip")
    if clip is not None:
        transforms.clip([], clip)
    offset, offset_is_named = _resolve_offset(
        config.get("offset"), reference_values, size
    )
    inverse = _inverse_permutation(
        config.get("joint_order"), canonical_joint_order, size
    )
    if offset_is_named and inverse is not None:
        # The offset is applied before the inverse permutation, so a canonical
        # reference has to be scattered into the policy's joint order first.
        # Without this a zero action commands a scrambled pose -- the right
        # width, the wrong legs.
        reordered = [0.0] * size
        for canonical_index, policy_index in enumerate(inverse):
            reordered[policy_index] = offset[canonical_index]
        offset = reordered
    stiffness = _resolve_gain(config.get("stiffness"), "stiffness", size, inverse)
    damping = _resolve_gain(config.get("damping"), "damping", size, inverse)
    if (stiffness is None) != (damping is None):
        raise ValueError("stiffness and damping must be declared together")
    motor = _resolve_motor(config.get("motor"))
    if motor is not None and stiffness is None:
        raise ValueError("action.motor needs stiffness and damping to drive it")
    return ActionDecoder(
        size,
        scale,
        offset,
        clip,
        inverse,
        stiffness=stiffness,
        damping=damping,
        motor=motor,
    )


def _resolve_motor(value: Any) -> Optional[Dict[str, float]]:
    """Validate an optional ``{model: dc_motor, ...}`` torque-speed envelope."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("action.motor must be a mapping")
    if value.get("model") != "dc_motor":
        raise ValueError("action.motor.model must be 'dc_motor'")
    motor = {}
    for key in ("saturation_effort", "effort_limit", "velocity_limit"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, Real) or number <= 0:
            raise ValueError("action.motor.%s must be a positive number" % key)
        motor[key] = float(number)
    return motor


def dc_motor_torque(torque: float, velocity: float, motor: Mapping[str, float]) -> float:
    """Clip a PD torque to Isaac Lab's DCMotor torque-speed envelope.

    Available torque falls linearly from ``saturation_effort`` at rest to zero
    at ``velocity_limit``, and never exceeds ``effort_limit``.
    """

    saturation = motor["saturation_effort"]
    limit = motor["effort_limit"]
    speed_ratio = velocity / motor["velocity_limit"]
    upper = max(min(saturation * (1.0 - speed_ratio), limit), 0.0)
    lower = min(max(saturation * (-1.0 - speed_ratio), -limit), 0.0)
    return min(max(torque, lower), upper)


__all__ = ["SUPPORTED_ACTION_SPACES", "ActionDecoder", "compile_action", "dc_motor_torque"]
