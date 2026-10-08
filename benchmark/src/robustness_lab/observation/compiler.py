"""Compile declarative observation terms into efficient runtime plans."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from robustness_lab.canonical_state import CANONICAL_FIELDS

from . import transforms
from .sensor_geometry import geometry_element_count


Operation = Callable[[List[float]], List[float]]


EXTEROCEPTIVE_FIELDS = frozenset(
    {
        "height_scan",
        "height_scan_per_foot",
        "elevation_map",
        "traversability_map",
        "depth_image",
        "rgb_image",
        "lidar_rays",
        "point_cloud",
    }
)

OPTIONAL_SENSOR_GEOMETRY_FIELDS = frozenset(
    {
        "semantic_segmentation",
        "occupancy_grid",
        "signed_distance_field",
        "stereo_disparity",
        "optical_flow",
        "thermal_image",
    }
)


@dataclass(frozen=True)
class _CompiledTerm:
    source: str
    instance: Optional[str]
    source_size: int
    validate_source_size: bool
    output_size: int
    operations: Tuple[Operation, ...]
    permutation: Optional[Tuple[int, ...]]
    geometry: Optional[Mapping[str, Any]]

    def build(self, state: Mapping[str, Any]) -> List[float]:
        if self.source not in state:
            raise ValueError("state is missing canonical field %r" % self.source)
        state_value = state[self.source]
        value_label = self.source
        if self.instance is not None:
            if not isinstance(state_value, Mapping) or self.instance not in state_value:
                raise ValueError(
                    "canonical field %r is missing instance %r"
                    % (self.source, self.instance)
                )
            state_value = state_value[self.instance]
            value_label = "%s instance %r" % (self.source, self.instance)
        values = _state_values(state_value, value_label)
        if self.validate_source_size and len(values) != self.source_size:
            if self.geometry is not None and self.operations:
                raise ValueError(
                    "canonical field %r has length %d, but its geometry requires %d"
                    % (self.source, len(values), self.source_size)
                )
            raise ValueError(
                "canonical field %r has length %d, declared size is %d"
                % (self.source, len(values), self.source_size)
            )
        for operation in self.operations:
            values = operation(values)
        if self.permutation is not None:
            if len(values) != len(self.permutation):
                raise ValueError(
                    "canonical field %r has transformed length %d, but its joint "
                    "permutation has length %d"
                    % (self.source, len(values), len(self.permutation))
                )
            values = [values[index] for index in self.permutation]
        if len(values) != self.output_size:
            raise ValueError(
                "canonical field %r produced length %d, declared size is %d"
                % (self.source, len(values), self.output_size)
            )
        return values


class ObservationPlan:
    """A compiled observation pipeline with resettable frame history."""

    def __init__(
        self,
        terms: Sequence[_CompiledTerm],
        frame_size: int,
        history_frames: int,
        history_order: str,
        history_layout: str = "frame_major",
    ) -> None:
        self._terms = tuple(terms)
        self._frame_size = frame_size
        self._history_frames = history_frames
        self._history_order = history_order
        self._history_layout = history_layout
        # Where each term sits inside one frame, so term-major stacking can
        # gather a term's whole history contiguously.
        slices = []
        cursor = 0
        for term in self._terms:
            slices.append((cursor, cursor + term.output_size))
            cursor += term.output_size
        self._term_slices = tuple(slices)
        self.size = frame_size * history_frames
        self._history: Deque[List[float]] = deque(maxlen=history_frames)
        self.reset()

    def build(self, state: Mapping[str, Any]) -> List[float]:
        """Build one observation and update the plan's history buffer."""

        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping of canonical fields to values")
        frame = []
        for term in self._terms:
            frame.extend(term.build(state))
        if not self._history:
            # First frame after a reset fills every slot, as Isaac Lab's and
            # mjlab's CircularBuffer do. A zero-filled history is a state the
            # policy never saw in training.
            for _ in range(self._history_frames - 1):
                self._history.append(list(frame))
        self._history.append(frame)
        frames = list(self._history)
        if self._history_order == "newest_first":
            frames.reverse()
        if self._history_layout == "term_major":
            # Isaac Lab / mjlab concatenate each term's whole history before
            # moving to the next term, rather than stacking whole frames.
            return [
                value
                for start, stop in self._term_slices
                for history_frame in frames
                for value in history_frame[start:stop]
            ]
        return [value for history_frame in frames for value in history_frame]

    @property
    def sensor_requests(
        self,
    ) -> Tuple[Tuple[str, Optional[str], Mapping[str, Any]], ...]:
        """Return geometry-bearing sensor terms needed to build this plan."""

        return tuple(
            (term.source, term.instance, term.geometry)
            for term in self._terms
            if term.geometry is not None
        )

    def reset(self) -> None:
        """Clear observed frames; the next build backfills the whole history."""

        self._history.clear()


class MultiTensorPlan:
    """Ordered collection of independently compiled observation tensors."""

    def __init__(self, plans: Sequence[Tuple[str, Any]]) -> None:
        self._plans = tuple(plans)
        self.sizes = OrderedDict((name, plan.size) for name, plan in self._plans)
        self.size = sum(self.sizes.values())

    def build(self, state: Mapping[str, Any]) -> Mapping[str, List[float]]:
        """Build every tensor in its declared positional order."""

        return OrderedDict(
            (name, plan.build(state)) for name, plan in self._plans
        )

    def reset(self) -> None:
        """Reset the frame history of every tensor plan."""

        for _, plan in self._plans:
            plan.reset()

    @property
    def sensor_requests(
        self,
    ) -> Tuple[Tuple[str, Optional[str], Mapping[str, Any]], ...]:
        """Return sensor terms from all tensors in declared tensor order."""

        return tuple(
            request
            for _, plan in self._plans
            for request in plan.sensor_requests
        )


class EncoderTensorPlan:
    """An observation tensor produced by a researcher-supplied encoder."""

    def __init__(
        self,
        encoder: Any,
        operations: Sequence[Operation],
        output_size: int,
    ) -> None:
        self._encoder = encoder
        self._operations = tuple(operations)
        self.size = output_size

    def build(self, state: Mapping[str, Any]) -> List[float]:
        values = self._encoder.run(state)
        for operation in self._operations:
            values = operation(values)
        if len(values) != self.size:
            raise ValueError(
                "encoder-backed observation tensor produced %d values; expected %d"
                % (len(values), self.size)
            )
        return values

    def reset(self) -> None:
        """Encoder-backed tensors have no compiler-owned history."""

        return None

    @property
    def sensor_requests(
        self,
    ) -> Tuple[Tuple[str, Optional[str], Mapping[str, Any]], ...]:
        return ()


def _state_values(value: Any, source: str) -> List[float]:
    if isinstance(value, Real):
        return [float(value)]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("canonical field %r must be a number or sequence" % source)
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "canonical field %r must contain only numbers" % source
        ) from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return int(value)


def _copy_argument(argument: Any) -> Any:
    if isinstance(argument, Mapping):
        return {key: _copy_argument(value) for key, value in argument.items()}
    if isinstance(argument, Sequence) and not isinstance(argument, (str, bytes)):
        return [_copy_argument(value) for value in argument]
    return argument


def _resolve_reference(argument: Any, references: Mapping[str, Any], name: str) -> Any:
    if isinstance(argument, str):
        if argument not in references:
            raise ValueError("unknown %s reference %r" % (name, argument))
        return _copy_argument(references[argument])
    return _copy_argument(argument)


def _compiled_operation(
    name: str,
    argument: Any,
    references: Mapping[str, Any],
    input_size: Optional[int],
    allow_custom: bool,
) -> Tuple[Operation, Optional[int]]:
    if name in ("subtract", "normalize", "to_body_frame"):
        resolved = _resolve_reference(argument, references, name)
        if (
            name == "to_body_frame"
            and isinstance(resolved, Mapping)
            and "quaternion" in resolved
        ):
            resolved = resolved["quaternion"]
    else:
        resolved = _copy_argument(argument)

    size_is_known = input_size is not None
    if size_is_known:
        probe_values = [0.0] * input_size
    elif name == "select" and isinstance(resolved, Mapping):
        start = resolved.get("start")
        count = resolved.get("count")
        if (
            isinstance(start, Integral)
            and not isinstance(start, bool)
            and isinstance(count, Integral)
            and not isinstance(count, bool)
            and count >= 0
        ):
            probe_size = (
                int(start) + int(count)
                if start >= 0
                else max(-int(start), int(count))
            )
            probe_values = [0.0] * probe_size
        else:
            probe_values = [0.0]
    elif name == "select" and isinstance(resolved, Sequence) and not isinstance(
        resolved, (str, bytes)
    ):
        valid_indices = [
            int(index)
            for index in resolved
            if isinstance(index, Integral) and not isinstance(index, bool)
        ]
        probe_size = max(
            (index + 1 if index >= 0 else -index for index in valid_indices),
            default=0,
        )
        probe_values = [0.0] * probe_size
    elif name == "pad":
        probe_values = []
    elif name in ("scale", "add", "subtract") and isinstance(
        resolved, Sequence
    ) and not isinstance(resolved, (str, bytes)):
        probe_values = [0.0] * len(resolved)
    elif (
        name == "normalize"
        and isinstance(resolved, Mapping)
        and isinstance(resolved.get("mean"), Sequence)
        and not isinstance(resolved.get("mean"), (str, bytes))
    ):
        probe_values = [0.0] * len(resolved["mean"])
    elif name in (
        "quat_to_projected_gravity",
        "quat_to_euler",
        "quat_to_rotmat",
    ):
        probe_values = [1.0, 0.0, 0.0, 0.0]
    elif name in ("rotate_by_yaw", "to_body_frame"):
        probe_values = [0.0, 0.0, 0.0]
    else:
        probe_values = [0.0]
    if name in ("power", "reciprocal"):
        probe_values = [1.0] * len(probe_values)
    elif name in (
        "quat_to_projected_gravity",
        "quat_to_euler",
        "quat_to_rotmat",
    ) and input_size == 4:
        probe_values = [1.0, 0.0, 0.0, 0.0]
    transformed = transforms.apply_transform(
        name, probe_values, resolved, allow_custom=allow_custom
    )
    output_size = len(transformed)
    if not size_is_known and name not in (
        "select",
        "pad",
        "quat_to_projected_gravity",
        "quat_to_euler",
        "quat_to_rotmat",
        "rotate_by_yaw",
        "to_body_frame",
    ):
        constrained_by_argument = name in ("scale", "add", "subtract") and (
            isinstance(resolved, Sequence)
            and not isinstance(resolved, (str, bytes))
        )
        constrained_by_argument = constrained_by_argument or (
            name == "normalize"
            and isinstance(resolved, Mapping)
            and isinstance(resolved.get("mean"), Sequence)
            and not isinstance(resolved.get("mean"), (str, bytes))
        )
        if not constrained_by_argument:
            output_size = None
    return (
        lambda values, transform_name=name, arg=resolved, custom=allow_custom: (
            transforms.apply_transform(
                transform_name, values, arg, allow_custom=custom
            )
        ),
        output_size,
    )


def _joint_permutation(
    canonical_joint_order: Any,
    policy_joint_order: Any,
    term_size: int,
) -> Tuple[int, ...]:
    if canonical_joint_order is None or policy_joint_order is None:
        raise ValueError(
            "joint_order 'policy' requires canonical_joint_order and policy_joint_order"
        )
    canonical = list(canonical_joint_order)
    policy = list(policy_joint_order)
    if len(canonical) != term_size or len(policy) != term_size:
        raise ValueError(
            "joint orders must both have the term's declared size %d" % term_size
        )
    if len(set(canonical)) != len(canonical) or len(set(policy)) != len(policy):
        raise ValueError("joint orders must not contain duplicate names")
    if set(canonical) != set(policy):
        missing = sorted(set(canonical) - set(policy))
        extra = sorted(set(policy) - set(canonical))
        raise ValueError(
            "policy joint order is not a permutation of canonical joint order "
            "(missing=%r, extra=%r)" % (missing, extra)
        )
    indices: Dict[Any, int] = {name: index for index, name in enumerate(canonical)}
    return tuple(indices[name] for name in policy)


def _named_context_count(value: Any, label: str) -> int:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("%s is required and must be a sequence" % label)
    names = list(value)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("%s must contain non-empty names" % label)
    if len(set(names)) != len(names):
        raise ValueError("%s must not contain duplicate names" % label)
    return len(names)


def _natural_source_size(
    source: str,
    joint_order: Any,
    foot_names: Any,
    source_geometry: Any,
) -> int:
    """Resolve one canonical field's metadata-declared natural width."""

    rule = CANONICAL_FIELDS[source].get("natural_size")
    if isinstance(rule, Integral) and not isinstance(rule, bool):
        return _positive_int(rule, "canonical source %r natural size" % source)
    if rule == "joint_count":
        return _named_context_count(joint_order, "joint_order")
    if rule in ("foot_count", "foot_count_x3"):
        foot_count = _named_context_count(foot_names, "foot_names")
        return foot_count * (3 if rule == "foot_count_x3" else 1)
    if rule == "geometry":
        if source_geometry is None:
            raise ValueError(
                "canonical source %r requires geometry to compute its size" % source
            )
        return geometry_element_count(source, source_geometry)
    if rule == "declared":
        raise ValueError(
            "canonical source %r has no natural fixed width; use the "
            "size-declaring terms form" % source
        )
    raise ValueError(
        "canonical source %r has no recognized natural sizing rule" % source
    )


def _transformed_size(
    transform_specs: Any,
    input_size: int,
    references: Mapping[str, Any],
    allow_custom: bool,
) -> int:
    """Use the term compiler's transform inference to obtain output width."""

    if isinstance(transform_specs, (str, bytes)) or not isinstance(
        transform_specs, Sequence
    ):
        raise ValueError("source transform must be a sequence")
    output_size: Optional[int] = input_size
    for spec in transform_specs:
        if not isinstance(spec, Mapping) or len(spec) != 1:
            raise ValueError("each transform must be a single-entry mapping")
        name, argument = next(iter(spec.items()))
        _, output_size = _compiled_operation(
            name, argument, references, output_size, allow_custom
        )
    if output_size is None:
        raise ValueError("source transforms do not determine an output size")
    return output_size


def compile_source_list(
    sources: Sequence[Any],
    joint_order: Optional[Sequence[str]] = None,
    foot_names: Optional[Sequence[str]] = None,
    geometry: Optional[Mapping[str, Any]] = None,
    references: Optional[Mapping[str, Any]] = None,
    policy_joint_order: Optional[Sequence[str]] = None,
    history: Optional[Mapping[str, Any]] = None,
    allow_custom: bool = False,
) -> ObservationPlan:
    """Compile ordered source names with natural or per-entry widths.

    This is an additive convenience layer over :func:`compile_terms`; it only
    resolves omitted source widths and delegates all term behavior, including
    validation of declared widths, to that compiler.
    """

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise ValueError("sources must be a sequence of source names or mappings")
    geometry_values = {} if geometry is None else geometry
    if not isinstance(geometry_values, Mapping):
        raise ValueError("geometry must be a mapping keyed by source name")
    reference_values = {} if references is None else references
    if not isinstance(reference_values, Mapping):
        raise ValueError("references must be a mapping")

    terms = []
    for source_index, entry in enumerate(sources):
        if isinstance(entry, str):
            term: Dict[str, Any] = {"source": entry}
        elif isinstance(entry, Mapping):
            term = {key: _copy_argument(value) for key, value in entry.items()}
        else:
            raise ValueError(
                "source %d must be a source name or mapping" % source_index
            )
        has_size_override = "size" in term
        source = term.get("source")
        if source not in CANONICAL_FIELDS:
            raise ValueError("unknown canonical source %r" % source)

        if "geometry" not in term and source in geometry_values:
            term["geometry"] = _copy_argument(geometry_values[source])
        source_geometry = term.get("geometry")
        if (
            source == "height_scan_per_foot"
            and isinstance(source_geometry, Mapping)
            and "foot_count" not in source_geometry
            and "origins_b" not in source_geometry
            and foot_names is not None
        ):
            source_geometry = dict(source_geometry)
            source_geometry["foot_count"] = _named_context_count(
                foot_names, "foot_names"
            )
            term["geometry"] = source_geometry

        if not has_size_override:
            natural_size = _natural_source_size(
                source, joint_order, foot_names, source_geometry
            )
            term["size"] = _transformed_size(
                term.get("transform", []),
                natural_size,
                reference_values,
                allow_custom,
            )
        terms.append(term)

    return compile_terms(
        terms,
        references=reference_values,
        canonical_joint_order=joint_order,
        policy_joint_order=policy_joint_order,
        history=history,
        allow_custom=allow_custom,
    )


def compile_terms(
    terms: Sequence[Mapping[str, Any]],
    references: Optional[Mapping[str, Any]] = None,
    canonical_joint_order: Optional[Sequence[str]] = None,
    policy_joint_order: Optional[Sequence[str]] = None,
    history: Optional[Mapping[str, Any]] = None,
    allow_custom: bool = False,
) -> ObservationPlan:
    """Compile observation term declarations into an :class:`ObservationPlan`."""

    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise ValueError("terms must be a sequence of mappings")
    reference_values = {} if references is None else references
    if not isinstance(reference_values, Mapping):
        raise ValueError("references must be a mapping")

    compiled_terms = []
    unnamed_multi_instance_sources = set()
    frame_size = 0
    for term_index, term in enumerate(terms):
        if not isinstance(term, Mapping):
            raise ValueError("term %d must be a mapping" % term_index)
        source = term.get("source")
        if source not in CANONICAL_FIELDS:
            raise ValueError("unknown canonical source %r" % source)
        instance = term.get("instance")
        if instance is not None:
            if not isinstance(instance, str) or not instance:
                raise ValueError("term %d instance must be a non-empty string" % term_index)
            if not CANONICAL_FIELDS[source].get("multi_instance", False):
                raise ValueError(
                    "term %d source %r does not support named instances"
                    % (term_index, source)
                )
        elif CANONICAL_FIELDS[source].get("multi_instance", False):
            if source in unnamed_multi_instance_sources:
                raise ValueError(
                    "multi-instance source %r is declared more than once without "
                    "an instance name" % source
                )
            unnamed_multi_instance_sources.add(source)
        declared_size = _positive_int(
            term.get("size"), "term %d size" % term_index
        )
        transform_specs = term.get("transform", [])
        if isinstance(transform_specs, (str, bytes)) or not isinstance(
            transform_specs, Sequence
        ):
            raise ValueError("term %d transform must be a sequence" % term_index)

        geometry_size = None
        if source in EXTEROCEPTIVE_FIELDS:
            if "geometry" not in term or term["geometry"] is None:
                raise ValueError(
                    "term %d source %r requires a geometry declaration"
                    % (term_index, source)
                )
            geometry_size = geometry_element_count(source, term["geometry"])
        elif (
            source in OPTIONAL_SENSOR_GEOMETRY_FIELDS
            and term.get("geometry") is not None
        ):
            geometry_size = geometry_element_count(source, term["geometry"])

        source_size = geometry_size if geometry_size is not None else declared_size
        validate_source_size = geometry_size is not None or not transform_specs
        operations = []
        inferred_size = geometry_size if geometry_size is not None else (
            declared_size if not transform_specs else None
        )
        for spec in transform_specs:
            if not isinstance(spec, Mapping) or len(spec) != 1:
                raise ValueError("each transform must be a single-entry mapping")
            name, argument = next(iter(spec.items()))
            operation, inferred_size = _compiled_operation(
                name, argument, reference_values, inferred_size, allow_custom
            )
            operations.append(operation)

        if (
            geometry_size is not None
            and inferred_size is not None
            and inferred_size != declared_size
        ):
            if transform_specs:
                raise ValueError(
                    "term %d source %r declares size %d, but its geometry and "
                    "transforms produce size %d"
                    % (term_index, source, declared_size, inferred_size)
                )
            raise ValueError(
                "term %d source %r declares size %d, but its geometry "
                "implies size %d"
                % (term_index, source, declared_size, geometry_size)
            )

        joint_order = term.get("joint_order")
        permutation = None
        if joint_order is not None:
            if joint_order != "policy":
                raise ValueError("joint_order must be 'policy' when specified")
            permutation = _joint_permutation(
                canonical_joint_order, policy_joint_order, declared_size
            )

        compiled_terms.append(
            _CompiledTerm(
                source=source,
                instance=instance,
                source_size=source_size,
                validate_source_size=validate_source_size,
                output_size=declared_size,
                operations=tuple(operations),
                permutation=permutation,
                geometry=(
                    None
                    if term.get("geometry") is None
                    else _copy_argument(term["geometry"])
                ),
            )
        )
        frame_size += declared_size

    history_values = {"frames": 1, "layout": "frame_major"}
    if history is not None:
        if not isinstance(history, Mapping):
            raise ValueError("history must be a mapping")
        history_values.update(history)
    history_frames = _positive_int(history_values.get("frames"), "history frames")
    history_order = history_values.get("order")
    if history_frames > 1 and history_order is None:
        # Stacking the wrong way round is the same width as stacking the right
        # way round, so it cannot be caught downstream. Only the submitter
        # knows which end of their buffer is the newest frame.
        raise ValueError(
            "history order must be declared as 'newest_first' or "
            "'oldest_first' when stacking more than one frame; the direction "
            "cannot be inferred from the artifact"
        )
    if history_order is None:
        history_order = "newest_first"
    if history_order not in ("newest_first", "oldest_first"):
        raise ValueError("history order must be 'newest_first' or 'oldest_first'")
    history_layout = history_values.get("layout")
    if history_layout not in ("frame_major", "term_major"):
        raise ValueError(
            "history layout must be 'frame_major' or 'term_major', got %r"
            % (history_layout,)
        )

    return ObservationPlan(
        compiled_terms, frame_size, history_frames, history_order, history_layout
    )


def compile_tensors(
    tensors: Sequence[Mapping[str, Any]],
    references: Optional[Mapping[str, Any]] = None,
    canonical_joint_order: Optional[Sequence[str]] = None,
    policy_joint_order: Optional[Sequence[str]] = None,
    allow_custom: bool = False,
) -> MultiTensorPlan:
    """Compile ordered named tensor declarations into a multi-tensor plan."""

    if isinstance(tensors, (str, bytes)) or not isinstance(tensors, Sequence):
        raise ValueError("tensors must be a sequence of mappings")

    plans = []
    names = set()
    for tensor_index, tensor in enumerate(tensors):
        if not isinstance(tensor, Mapping):
            raise ValueError("tensor %d must be a mapping" % tensor_index)
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tensor %d name must be a non-empty string" % tensor_index)
        if name in names:
            raise ValueError("tensor names must be unique; duplicate name %r" % name)
        names.add(name)
        plan = compile_terms(
            tensor.get("terms"),
            references=references,
            canonical_joint_order=canonical_joint_order,
            policy_joint_order=policy_joint_order,
            history=tensor.get("history"),
            allow_custom=allow_custom,
        )
        plans.append((name, plan))

    return MultiTensorPlan(plans)


def compile_encoder_tensor(
    encoder: Any,
    transform_specs: Any,
    expected_size: Any,
    references: Optional[Mapping[str, Any]] = None,
    allow_custom: bool = False,
) -> EncoderTensorPlan:
    """Compile transforms applied to one encoder's flattened output."""

    if isinstance(transform_specs, (str, bytes)) or not isinstance(
        transform_specs, Sequence
    ):
        raise ValueError("encoder tensor transform must be a sequence")
    reference_values = {} if references is None else references
    operations = []
    output_size = int(encoder.output_size)
    for spec in transform_specs:
        if not isinstance(spec, Mapping) or len(spec) != 1:
            raise ValueError("each transform must be a single-entry mapping")
        name, argument = next(iter(spec.items()))
        operation, output_size = _compiled_operation(
            name, argument, reference_values, output_size, allow_custom
        )
        operations.append(operation)
    declared_size = _positive_int(expected_size, "encoder tensor size")
    if output_size != declared_size:
        raise ValueError(
            "encoder tensor declares size %d, but its transforms produce %d"
            % (declared_size, output_size)
        )
    return EncoderTensorPlan(encoder, operations, output_size)


__all__ = [
    "EXTEROCEPTIVE_FIELDS",
    "OPTIONAL_SENSOR_GEOMETRY_FIELDS",
    "EncoderTensorPlan",
    "MultiTensorPlan",
    "ObservationPlan",
    "compile_encoder_tensor",
    "compile_source_list",
    "compile_tensors",
    "compile_terms",
]
