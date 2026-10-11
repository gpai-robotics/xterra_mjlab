"""Researcher-supplied portable observation encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from robustness_lab.artifacts import LoadedArtifact, load_artifact


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return int(value)


def _shape(value: Any, label: str) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("%s must be a sequence of positive integers" % label)
    result = tuple(
        _positive_int(dimension, "%s dimension" % label) for dimension in value
    )
    if not result:
        raise ValueError("%s must not be empty" % label)
    return result


class CompiledEncoder:
    """Loaded preprocessing weights with a declared positional tensor contract."""

    provided_by = "researcher"

    def __init__(
        self,
        artifact: LoadedArtifact,
        name: str,
        artifact_path: str,
        inputs: Sequence[Tuple[str, Tuple[int, ...]]],
        output_name: str,
        output_size: int,
    ) -> None:
        self._artifact = artifact
        self._artifact_path = artifact_path
        self.name = name
        self.path = artifact_path
        self._inputs = tuple(inputs)
        self.output_name = output_name
        self.output_size = output_size

    def run(self, inputs: Mapping[str, Any]) -> List[float]:
        """Run declared inputs positionally with an implicit batch size of one."""

        if not isinstance(inputs, Mapping):
            raise ValueError("encoder inputs must be a mapping")
        tensors = []
        for name, declared_shape in self._inputs:
            if name not in inputs:
                raise ValueError("encoder input %r is unavailable" % name)
            try:
                tensor = np.asarray(inputs[name], dtype=np.float32)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    "encoder input %r could not be converted to a tensor: %s"
                    % (name, error)
                ) from error
            actual_shape = tuple(int(value) for value in tensor.shape)
            if actual_shape != declared_shape:
                raise ValueError(
                    "encoder input %r has shape %r; declared shape is %r"
                    % (name, actual_shape, declared_shape)
                )
            tensors.append(np.expand_dims(tensor, axis=0))

        output = self._artifact.run(tensors)
        flattened = output.reshape(-1).tolist()
        actual_size = len(flattened)
        if actual_size != self.output_size:
            raise ValueError(
                "encoder artifact %s returned actual width %d; declared output size is %d"
                % (Path(self._artifact_path).name, actual_size, self.output_size)
            )
        return [float(value) for value in flattened]


def compile_encoder(spec: Mapping[str, Any]) -> CompiledEncoder:
    """Load and validate one researcher-supplied portable encoder."""

    if not isinstance(spec, Mapping):
        raise ValueError("encoder spec must be a mapping")
    encoder_name = spec.get("name")
    if not isinstance(encoder_name, str) or not encoder_name:
        raise ValueError("encoder name must be a non-empty string")
    artifact_value = spec.get("path")
    if not isinstance(artifact_value, str) or not artifact_value:
        raise ValueError("encoder path is required")
    artifact = Path(artifact_value)
    if not artifact.is_file():
        raise ValueError("encoder artifact does not exist: %s" % artifact)

    input_specs = spec.get("inputs")
    if isinstance(input_specs, (str, bytes)) or not isinstance(input_specs, Sequence):
        raise ValueError("encoder inputs must be a sequence of mappings")
    if not input_specs:
        raise ValueError("encoder inputs must not be empty")
    declared_inputs = []
    names = set()
    for index, input_spec in enumerate(input_specs):
        if not isinstance(input_spec, Mapping):
            raise ValueError("encoder input %d must be a mapping" % index)
        name = input_spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("encoder input %d name must be a non-empty string" % index)
        if name in names:
            raise ValueError("encoder input names must be unique; duplicate %r" % name)
        names.add(name)
        declared_inputs.append(
            (name, _shape(input_spec.get("shape"), "encoder input %r shape" % name))
        )

    output = spec.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("encoder output must be a mapping")
    output_name = output.get("name")
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("encoder output name must be a non-empty string")
    output_size = _positive_int(output.get("size"), "encoder output size")

    loaded_artifact = load_artifact(
        str(artifact), runtime_format=spec.get("format")
    )
    actual_arity = len(loaded_artifact.input_names)
    if actual_arity != len(declared_inputs):
        raise ValueError(
            "encoder artifact %s accepts %d inputs; %d were declared"
            % (artifact.name, actual_arity, len(declared_inputs))
        )

    return CompiledEncoder(
        loaded_artifact,
        encoder_name,
        str(artifact.resolve()),
        declared_inputs,
        output_name,
        output_size,
    )


__all__ = ["CompiledEncoder", "compile_encoder"]
