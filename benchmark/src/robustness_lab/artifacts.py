"""Shared loading and execution for portable policy artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


def _error_summary(error: Exception) -> str:
    lines = [line for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else error.__class__.__name__


class LoadedArtifact:
    """A TorchScript or ONNX artifact behind one positional-input interface."""

    def __init__(
        self,
        path: str,
        runtime_format: str,
        backend: Any,
        input_names: Sequence[str],
        input_shapes: Sequence[Sequence[Any]],
        device: str = "cpu",
        torch_module: Any = None,
        runtime_version: str = "unknown",
    ) -> None:
        self.path = path
        self.format = runtime_format
        self._backend = backend
        self.input_names = tuple(str(name) for name in input_names)
        self.input_shapes = tuple(tuple(shape) for shape in input_shapes)
        self.device = device
        self._torch = torch_module
        self.runtime_version = runtime_version

    @property
    def label(self) -> str:
        return "ONNX" if self.format == "onnx" else "TorchScript"

    def candidate_input_widths(self) -> List[int]:
        """Return statically discoverable input widths in preference order."""

        candidates: List[int] = []
        if self.format == "onnx":
            for shape in self.input_shapes:
                if shape:
                    value = shape[-1]
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        and value not in candidates
                    ):
                        candidates.append(value)
            return candidates

        for module in self._backend.modules():
            value = getattr(module, "in_features", None)
            if isinstance(value, int) and value > 0 and value not in candidates:
                candidates.append(value)
        for parameter in self._backend.parameters():
            shape = tuple(int(value) for value in parameter.shape)
            if len(shape) == 2 and shape[1] > 0 and shape[1] not in candidates:
                candidates.append(shape[1])
        return candidates

    def run(self, inputs: Sequence[Any]) -> np.ndarray:
        """Execute batched inputs positionally and return the first output."""

        values = list(inputs)
        try:
            if len(values) != len(self.input_names):
                raise ValueError(
                    "artifact accepts %d inputs, but %d were supplied"
                    % (len(self.input_names), len(values))
                )
            if self.format == "onnx":
                payload = {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in zip(self.input_names, values)
                }
                outputs = self._backend.run(None, payload)
                if not outputs:
                    raise ValueError("artifact returned no outputs")
                return np.asarray(outputs[0])

            tensors = [
                self._torch.as_tensor(
                    value, dtype=self._torch.float32, device=self.device
                )
                for value in values
            ]
            with self._torch.inference_mode():
                output = self._backend(*tensors)
            if isinstance(output, (tuple, list)) and output:
                output = output[0]
            elif isinstance(output, Mapping) and len(output) == 1:
                output = next(iter(output.values()))
            if not hasattr(output, "detach"):
                raise ValueError("artifact did not return a tensor")
            return output.detach().cpu().numpy()
        except Exception as error:
            raise ValueError(
                "%s artifact %r did not accept inputs: %s"
                % (self.label, self.path, _error_summary(error))
            ) from error


def load_artifact(
    path: str,
    runtime_format: Optional[str] = None,
    device: str = "cpu",
    providers: Optional[Sequence[str]] = None,
) -> LoadedArtifact:
    """Load a portable artifact, preferring its declaration over its suffix."""

    artifact_path = str(path)
    selected_format = runtime_format
    if selected_format is None:
        selected_format = (
            "onnx" if Path(artifact_path).suffix.lower() == ".onnx" else "torchscript"
        )
    if not isinstance(selected_format, str):
        raise ValueError("artifact format must be 'torchscript' or 'onnx'")
    selected_format = selected_format.lower()
    if selected_format not in {"torchscript", "onnx"}:
        raise ValueError(
            "unsupported artifact format %r; expected 'torchscript' or 'onnx'"
            % selected_format
        )

    if selected_format == "onnx":
        try:
            import onnxruntime as ort
        except (ImportError, ModuleNotFoundError) as error:
            raise ValueError(
                "ONNX artifact support requires the 'onnxruntime' package; "
                "install the robustness-lab 'onnx' extra"
            ) from error
        try:
            options = {} if providers is None else {"providers": list(providers)}
            # A locomotion policy is evaluated one observation at a time, so the
            # default of one compute thread per core buys nothing and costs
            # thread overhead. It also matters under parallel trials: onnxruntime
            # would otherwise start a full thread pool in EVERY worker process,
            # oversubscribing the machine many times over.
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                artifact_path, sess_options=session_options, **options
            )
            inputs = list(session.get_inputs())
        except Exception as error:
            raise ValueError(
                "artifact %r could not be loaded as ONNX: %s"
                % (artifact_path, _error_summary(error))
            ) from error
        return LoadedArtifact(
            artifact_path,
            selected_format,
            session,
            [item.name for item in inputs],
            [item.shape for item in inputs],
            device=device,
            runtime_version=str(getattr(ort, "__version__", "unknown")),
        )

    try:
        import torch

        model = torch.jit.load(artifact_path, map_location=device)
        model.eval()
        arguments = list(model.forward.schema.arguments)
        input_names = [str(argument.name) for argument in arguments[1:]]
    except Exception as error:
        raise ValueError(
            "could not load TorchScript artifact %s: %s"
            % (Path(artifact_path).name, _error_summary(error))
        ) from error
    return LoadedArtifact(
        artifact_path,
        selected_format,
        model,
        input_names,
        [()] * len(input_names),
        device=device,
        torch_module=torch,
        runtime_version=str(getattr(torch, "__version__", "unknown")),
    )


__all__ = ["LoadedArtifact", "load_artifact"]
