"""Pure transforms used by the observation compiler.

All transforms accept and return plain Python lists.  Keeping this layer free
of simulator and array-library types makes the individual operations easy to
test and gives every backend the same numerical semantics.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, List, Optional

from robustness_lab.canonical_state import (
    projected_gravity_from_quaternion,
    quaternion_to_euler_rpy,
    world_to_body,
)


def _values(values: Sequence[Real]) -> List[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("transform input must be a sequence of numbers")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("transform input must be a sequence of numbers") from exc


def _per_element(argument: Any, length: int, name: str) -> List[float]:
    if isinstance(argument, Real):
        return [float(argument)] * length
    if isinstance(argument, (str, bytes)) or not isinstance(argument, Sequence):
        raise ValueError("%s must be a scalar or a sequence" % name)
    try:
        result = [float(value) for value in argument]
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain only numbers" % name) from exc
    if len(result) != length:
        raise ValueError(
            "%s has length %d, expected %d" % (name, len(result), length)
        )
    return result


def scale(values: Sequence[Real], factor: Any) -> List[float]:
    """Multiply by a scalar or an equally sized sequence."""

    source = _values(values)
    factors = _per_element(factor, len(source), "scale")
    return [value * multiplier for value, multiplier in zip(source, factors)]


def add(values: Sequence[Real], offset: Any) -> List[float]:
    """Add a scalar or an equally sized sequence element by element."""

    source = _values(values)
    offsets = _per_element(offset, len(source), "add")
    return [value + addition for value, addition in zip(source, offsets)]


def negate(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Negate every value."""

    return [-value for value in _values(values)]


def abs(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Return the absolute value of every element."""

    return [math.fabs(value) for value in _values(values)]


def power(values: Sequence[Real], exponent: Any) -> List[float]:
    """Raise every value to the supplied numeric exponent."""

    if isinstance(exponent, bool) or not isinstance(exponent, Real):
        raise ValueError("power exponent must be a number")
    numeric_exponent = float(exponent)
    try:
        return [math.pow(value, numeric_exponent) for value in _values(values)]
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "power is undefined for the supplied values and exponent"
        ) from exc


def tanh(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Apply the hyperbolic tangent element by element."""

    return [math.tanh(value) for value in _values(values)]


def log1p(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Apply ``log(1 + value)``, whose real domain requires values above -1."""

    source = _values(values)
    if any(value <= -1.0 for value in source):
        raise ValueError("log1p requires every input value to be greater than -1")
    return [math.log1p(value) for value in source]


def reciprocal(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Return element-wise reciprocals, rejecting division by zero."""

    source = _values(values)
    if any(value == 0.0 for value in source):
        raise ValueError("reciprocal is undefined for a zero input element")
    return [1.0 / value for value in source]


def pad(values: Sequence[Real], options: Mapping[str, Any]) -> List[float]:
    """Right-pad a vector to ``options['size']`` using ``options['value']``."""

    if not isinstance(options, Mapping) or set(options) != {"size", "value"}:
        raise ValueError("pad requires a mapping containing exactly size and value")
    size = options["size"]
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
        raise ValueError("pad size must be a non-negative integer")
    value = options["value"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("pad value must be a number")
    source = _values(values)
    if len(source) > size:
        raise ValueError(
            "pad input length %d exceeds requested size %d" % (len(source), size)
        )
    return source + [float(value)] * (int(size) - len(source))


def repeat(values: Sequence[Real], count: Any) -> List[float]:
    """Tile the complete input vector ``count`` times."""

    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("repeat count must be a non-negative integer")
    return _values(values) * int(count)


def clamp_norm(values: Sequence[Real], limit: Any) -> List[float]:
    """Scale a vector only when its L2 norm exceeds ``limit``."""

    if isinstance(limit, bool) or not isinstance(limit, Real):
        raise ValueError("clamp_norm limit must be a number")
    numeric_limit = float(limit)
    if not math.isfinite(numeric_limit) or numeric_limit < 0.0:
        raise ValueError("clamp_norm limit must be finite and non-negative")
    source = _values(values)
    norm = math.sqrt(sum(value * value for value in source))
    if not math.isfinite(norm):
        raise ValueError("clamp_norm requires inputs with a finite L2 norm")
    if norm <= numeric_limit or norm == 0.0:
        return source
    multiplier = numeric_limit / norm
    return [value * multiplier for value in source]


def quat_to_rotmat(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Convert a body-to-world wxyz quaternion to a row-major rotation matrix."""

    quaternion = _values(values)
    return [
        component
        for world_axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        for component in world_to_body(quaternion, world_axis)
    ]


def rotate_by_yaw(values: Sequence[Real], angle: Any) -> List[float]:
    """Rotate a three-vector about the positive z axis by ``angle`` radians."""

    if isinstance(angle, bool) or not isinstance(angle, Real):
        raise ValueError("rotate_by_yaw angle must be a number in radians")
    numeric_angle = float(angle)
    if not math.isfinite(numeric_angle):
        raise ValueError("rotate_by_yaw angle must be finite")
    half_angle = -0.5 * numeric_angle
    inverse_yaw = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    return world_to_body(inverse_yaw, _values(values))


def subtract(
    values: Sequence[Real],
    constant: Any,
    references: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    """Subtract a constant sequence, optionally resolved by reference name."""

    if isinstance(constant, str):
        if references is None or constant not in references:
            raise ValueError("unknown subtract reference %r" % constant)
        constant = references[constant]
    source = _values(values)
    offsets = _per_element(constant, len(source), "subtract")
    return [value - offset for value, offset in zip(source, offsets)]


def clip(values: Sequence[Real], bounds: Sequence[Real]) -> List[float]:
    """Clamp every value to inclusive ``[low, high]`` bounds."""

    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
        raise ValueError("clip must be a [low, high] sequence")
    if len(bounds) != 2:
        raise ValueError("clip must contain exactly [low, high]")
    try:
        low, high = (float(value) for value in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("clip bounds must be numbers") from exc
    if low > high:
        raise ValueError("clip low bound must not exceed its high bound")
    return [max(low, min(high, value)) for value in _values(values)]


def select(values: Sequence[Real], indices: Any) -> List[float]:
    """Select ordered indices or a contiguous ``start``/``count`` range."""

    source = _values(values)
    if isinstance(indices, Mapping):
        if set(indices) != {"start", "count"}:
            raise ValueError("select range must contain exactly start and count")
        start = indices["start"]
        count = indices["count"]
        if (
            isinstance(start, bool)
            or not isinstance(start, Integral)
            or isinstance(count, bool)
            or not isinstance(count, Integral)
        ):
            raise ValueError("select range start and count must be integers")
        resolved_start = int(start)
        resolved_count = int(count)
        if resolved_count < 0:
            raise ValueError("select range count must not be negative")
        if resolved_start < 0:
            resolved_start += len(source)
        end = resolved_start + resolved_count
        if resolved_start < 0 or resolved_start > len(source) or end > len(source):
            raise ValueError(
                "select range start=%d count=%d is out of range"
                % (start, count)
            )
        return source[resolved_start:end]
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError("select must be a sequence of indices or a range mapping")
    result = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("select indices must be integers")
        resolved = int(index)
        if resolved < 0:
            resolved += len(source)
        if resolved < 0 or resolved >= len(source):
            raise ValueError("select index %d is out of range" % index)
        result.append(source[resolved])
    return result


def normalize(values: Sequence[Real], stats: Mapping[str, Sequence[Real]]) -> List[float]:
    """Apply element-wise ``(value - mean) / std`` normalization."""

    if not isinstance(stats, Mapping) or "mean" not in stats or "std" not in stats:
        raise ValueError("normalize requires a mapping with mean and std")
    source = _values(values)
    means = _per_element(stats["mean"], len(source), "normalize mean")
    standard_deviations = _per_element(
        stats["std"], len(source), "normalize std"
    )
    if any(value == 0.0 for value in standard_deviations):
        raise ValueError("normalize std values must be non-zero")
    return [
        (value - mean) / std
        for value, mean, std in zip(source, means, standard_deviations)
    ]


def quat_to_projected_gravity(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Convert a body-to-world wxyz quaternion to body-frame gravity."""

    return projected_gravity_from_quaternion(_values(values))


def quat_to_euler(values: Sequence[Real], argument: Any = None) -> List[float]:
    """Convert a body-to-world wxyz quaternion to roll, pitch, yaw."""

    return quaternion_to_euler_rpy(_values(values))


def to_body_frame(values: Sequence[Real], quaternion: Sequence[Real]) -> List[float]:
    """Rotate a three-vector from world coordinates into body coordinates."""

    return world_to_body(_values(quaternion), _values(values))


def one_hot(values: Sequence[Real], options: Mapping[str, int]) -> List[float]:
    """Encode every integer class in ``values`` using ``options['classes']`` bins."""

    if not isinstance(options, Mapping) or "classes" not in options:
        raise ValueError("one_hot requires a mapping with classes")
    classes = options["classes"]
    if isinstance(classes, bool) or not isinstance(classes, Integral) or classes <= 0:
        raise ValueError("one_hot classes must be a positive integer")
    result = []
    for raw_value in _values(values):
        category = int(raw_value)
        if raw_value != category or category < 0 or category >= classes:
            raise ValueError(
                "one_hot value %r is not an integer in [0, %d)"
                % (raw_value, classes)
            )
        encoded = [0.0] * classes
        encoded[category] = 1.0
        result.extend(encoded)
    return result


TRANSFORMS = {
    "scale": scale,
    "add": add,
    "negate": negate,
    "abs": abs,
    "power": power,
    "tanh": tanh,
    "log1p": log1p,
    "reciprocal": reciprocal,
    "pad": pad,
    "repeat": repeat,
    "clamp_norm": clamp_norm,
    "quat_to_rotmat": quat_to_rotmat,
    "rotate_by_yaw": rotate_by_yaw,
    "subtract": subtract,
    "clip": clip,
    "select": select,
    "normalize": normalize,
    "quat_to_projected_gravity": quat_to_projected_gravity,
    "quat_to_euler": quat_to_euler,
    "to_body_frame": to_body_frame,
    "one_hot": one_hot,
}
VALID_TRANSFORMS = tuple(TRANSFORMS) + ("custom",)


def _custom_transform(values: Sequence[Real], argument: Any) -> List[float]:
    if not isinstance(argument, Mapping) or set(argument) != {"target"}:
        raise ValueError("custom transform requires {'target': 'module:function'}")
    target = argument["target"]
    if not isinstance(target, str) or target.count(":") != 1:
        raise ValueError("custom transform target %r must be module:function" % target)
    module_name, attribute_name = target.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("custom transform target %r must be module:function" % target)
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, attribute_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(
            "could not resolve custom transform target %r" % target
        ) from exc
    if not callable(function):
        raise ValueError(
            "custom transform target %r must be callable and return a finite iterable"
            % target
        )
    try:
        raw_result = function(_values(values))
        result = [float(value) for value in raw_result]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "custom transform target %r must return an iterable of numbers" % target
        ) from exc
    if any(not math.isfinite(value) for value in result):
        raise ValueError(
            "custom transform target %r must return only finite values" % target
        )
    return result


def apply_transform(
    name: str,
    values: Sequence[Real],
    argument: Any,
    allow_custom: bool = False,
) -> List[float]:
    """Apply a named transform, with explicit opt-in for arbitrary callables."""

    if name == "custom":
        if allow_custom is not True:
            raise ValueError(
                "custom transforms are disabled unless allow_custom is true"
            )
        return _custom_transform(values, argument)
    if name not in TRANSFORMS:
        raise ValueError(
            "unknown transform %r; valid transforms are: %s"
            % (name, ", ".join(VALID_TRANSFORMS))
        )
    return TRANSFORMS[name](values, argument)


__all__ = [
    "TRANSFORMS",
    "VALID_TRANSFORMS",
    "apply_transform",
    "scale",
    "add",
    "negate",
    "abs",
    "power",
    "tanh",
    "log1p",
    "reciprocal",
    "pad",
    "repeat",
    "clamp_norm",
    "quat_to_rotmat",
    "rotate_by_yaw",
    "subtract",
    "clip",
    "select",
    "normalize",
    "quat_to_projected_gravity",
    "quat_to_euler",
    "to_body_frame",
    "one_hot",
]
