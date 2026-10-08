"""Controller plug-in contract shared by simulator workers."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

import numpy as np

from .artifacts import load_artifact
from .canonical_state import CANONICAL_FIELDS, CanonicalState
from .policy_manifest import load_policy_manifest


class Controller:
    def reset(self, context: Dict[str, Any]) -> None:
        del context

    def act(self, observation: Dict[str, Any]) -> List[float]:
        raise NotImplementedError


class ZeroController(Controller):
    def __init__(self, size: int) -> None:
        self.size = size

    def act(self, observation: Dict[str, Any]) -> List[float]:
        del observation
        return [0.0] * self.size


class ConstantController(Controller):
    def __init__(self, action: List[float]) -> None:
        self.action = [float(value) for value in action]

    def act(self, observation: Dict[str, Any]) -> List[float]:
        del observation
        return list(self.action)


class PythonController(Controller):
    def __init__(self, target: str, kwargs: Dict[str, Any]) -> None:
        module_name, attribute = target.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        self.implementation = factory(**kwargs)

    def reset(self, context: Dict[str, Any]) -> None:
        if hasattr(self.implementation, "reset"):
            self.implementation.reset(context)

    def act(self, observation: Dict[str, Any]) -> List[float]:
        result = self.implementation.act(observation) if hasattr(self.implementation, "act") else self.implementation(observation)
        return [float(value) for value in result]


class TorchScriptController(Controller):
    def __init__(self, path: str, observation_keys: List[str], device: str = "cpu", observation_tensors: List[Dict[str, Any]] = None, input_tensors: List[str] = None) -> None:
        self.artifact = load_artifact(
            path, runtime_format="torchscript", device=device
        )
        self.keys = observation_keys
        self.device = device
        self.tensor_specs = {item["name"]: item for item in (observation_tensors or [])}
        self.input_tensors = input_tensors or []

    @staticmethod
    def _flatten(value: Any) -> List[float]:
        if isinstance(value, dict):
            result = []
            for key in sorted(value):
                result.extend(TorchScriptController._flatten(value[key]))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(TorchScriptController._flatten(item))
            return result
        return [float(value)]

    def _declared_inputs(self, observation: Dict[str, Any]):
        tensors = []
        for name in self.input_tensors:
            spec = self.tensor_specs[name]
            values = []
            for source in spec.get("sources", []):
                if source not in observation:
                    raise ValueError("controller observation source %s is unavailable" % source)
                values.extend(self._flatten(observation[source]))
            expected = math.prod(spec["shape"])
            if len(values) != expected:
                raise ValueError("observation tensor %s resolved to %d values; expected %d" % (name, len(values), expected))
            tensors.append(np.asarray(values, dtype=np.float32).reshape([1] + list(spec["shape"])))
        return tensors

    def act(self, observation: Dict[str, Any]) -> List[float]:
        if self.input_tensors:
            output = self.artifact.run(self._declared_inputs(observation))
            return output.squeeze(0).tolist()
        values = []
        for key in self.keys:
            value = observation[key]
            values.extend(value if isinstance(value, list) else [value])
        tensor = np.asarray([values], dtype=np.float32)
        return self.artifact.run([tensor]).squeeze(0).tolist()


class OnnxController(Controller):
    def __init__(self, path: str, observation_keys: List[str], observation_tensors: List[Dict[str, Any]] = None, input_tensors: List[str] = None) -> None:
        self.artifact = load_artifact(path, runtime_format="onnx")
        self.keys = observation_keys
        self.tensor_specs = {item["name"]: item for item in (observation_tensors or [])}
        self.input_tensors = input_tensors or []

    def act(self, observation: Dict[str, Any]) -> List[float]:
        if self.input_tensors:
            if len(self.artifact.input_names) != len(self.input_tensors):
                raise ValueError("ONNX runtime input count does not match declared input_tensors")
            tensors = []
            for contract_name in self.input_tensors:
                spec = self.tensor_specs[contract_name]
                values = []
                for source in spec.get("sources", []):
                    if source not in observation:
                        raise ValueError("controller observation source %s is unavailable" % source)
                    values.extend(TorchScriptController._flatten(observation[source]))
                expected = math.prod(spec["shape"])
                if len(values) != expected:
                    raise ValueError("observation tensor %s resolved to %d values; expected %d" % (contract_name, len(values), expected))
                tensors.append(np.asarray(values, dtype=np.float32).reshape([1] + list(spec["shape"])))
            return self.artifact.run(tensors)[0].tolist()
        values = []
        for key in self.keys:
            value = observation[key]
            values.extend(value if isinstance(value, list) else [value])
        tensor = np.asarray([values], dtype=np.float32)
        return self.artifact.run([tensor])[0].tolist()


class CompiledPolicyController(Controller):
    """Portable policy driven by a schema-v2 observation manifest."""

    def __init__(self, manifest_path: str, action_size: int, device: str = "cpu") -> None:
        manifest = load_policy_manifest(manifest_path)
        if manifest.action.size != action_size:
            raise ValueError(
                "compiled_policy action has %d values; expected %d"
                % (manifest.action.size, action_size)
            )
        self.artifact = load_artifact(
            manifest.runtime["path"],
            runtime_format=manifest.runtime.get("format"),
            device=device,
            providers=manifest.runtime.get("providers"),
        )
        self.plan = manifest.observation_plan
        self.decoder = manifest.action
        self.device = device
        self.observation_size = self.plan.size
        self.artifact_path = manifest.runtime["path"]
        self.previous_action = [0.0] * self.decoder.size

    def reset(self, context: Dict[str, Any]) -> None:
        del context
        self.plan.reset()
        self.previous_action = [0.0] * self.decoder.size

    def act(self, observation: Any) -> List[float]:
        if isinstance(observation, CanonicalState):
            state = {
                name: getattr(observation, name) for name in CANONICAL_FIELDS
            }
        elif isinstance(observation, Mapping):
            state = dict(observation)
        else:
            raise ValueError(
                "compiled_policy observation must be a CanonicalState or mapping"
            )

        # Feedback is owned by the policy boundary so it cannot accidentally be
        # initialized from the robot's default pose or a decoded joint target.
        state["previous_action"] = list(self.previous_action)
        values = self.plan.build(state)
        if isinstance(values, Mapping):
            tensors = [
                np.asarray([tensor_values], dtype=np.float32)
                for tensor_values in values.values()
            ]
        else:
            tensors = [np.asarray([values], dtype=np.float32)]
        output = self.artifact.run(tensors)
        raw = output.squeeze(0).reshape(-1).tolist()
        self.previous_action = [float(value) for value in raw]
        return self.decoder.decode(self.previous_action)


def validate_controller_mapping(config: Dict[str, Any], expected_joint_order: List[str]) -> None:
    """Reject ambiguous learned-controller contracts before simulation starts."""
    kind = config.get("type")
    if kind == "compiled_policy":
        return
    if kind not in {"python", "torchscript", "onnx", "constant"}:
        return
    declared_order = config.get("joint_order")
    if not declared_order:
        raise ValueError("%s controller requires an explicit joint_order" % kind)
    if list(declared_order) != list(expected_joint_order):
        raise ValueError(
            "controller joint_order does not match backend order; declared=%s backend=%s"
            % (list(declared_order), list(expected_joint_order))
        )
    if config.get("action_mode") not in {"position", "effort"}:
        raise ValueError("%s controller requires action_mode: position or effort" % kind)
    if kind in {"torchscript", "onnx"} and not config.get("observation_keys"):
        raise ValueError("%s controller requires a non-empty observation_keys mapping" % kind)


def make_controller(config: Dict[str, Any], action_size: int) -> Controller:
    kind = config.get("type")
    if kind in {"zero", "pd_hold"}:
        return ZeroController(action_size)
    if kind == "constant":
        action = config.get("action", [])
        if len(action) != action_size:
            raise ValueError("constant action has %d values; expected %d" % (len(action), action_size))
        return ConstantController(action)
    if kind == "python":
        if not config.get("target"):
            raise ValueError("python controller requires module:attribute target")
        return PythonController(config["target"], config.get("kwargs", {}))
    if kind == "torchscript":
        return TorchScriptController(config["path"], config.get("observation_keys", []), config.get("device", "cpu"), config.get("observation_tensors", []), config.get("input_tensors", []))
    if kind == "onnx":
        return OnnxController(config["path"], config.get("observation_keys", []), config.get("observation_tensors", []), config.get("input_tensors", []))
    if kind == "compiled_policy":
        if not config.get("manifest"):
            raise ValueError("compiled_policy controller requires a manifest")
        return CompiledPolicyController(
            config["manifest"], action_size, config.get("device", "cpu")
        )
    raise ValueError("unsupported controller type: %s" % kind)


def timed_action(controller: Controller, observation: Dict[str, Any]) -> Tuple[List[float], float]:
    started = time.perf_counter()
    action = controller.act(observation)
    return action, time.perf_counter() - started
