"""Materialize reproducible experiment matrices without running them."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .io import dump_json, dump_yaml, load_data
from .models import RobotProfile
from .sampling import sobol
from .worlds import generate_ae, generate_aee, generate_terrain


DEFAULT_STRESSOR_BOUNDS = {
    "friction_scale": (0.5, 1.2),
    "payload_fraction": (0.0, 0.3),
    "com_offset_m": (-0.04, 0.04),
    "action_latency_s": (0.0, 0.06),
    "perception_latency_s": (0.0, 0.15),
    "depth_dropout": (0.0, 0.35),
    "push_impulse_ns": (0.0, 20.0),
}


def _contains_seed_template(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_seed_template(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_seed_template(item) for item in value)
    return isinstance(value, str) and "{controller_seed}" in value


def _resolve_controller(value: Any, config_dir: Path, controller_seed: Any = None, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _resolve_controller(item, config_dir, controller_seed, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_controller(item, config_dir, controller_seed, key) for item in value]
    if isinstance(value, str):
        resolved = value.replace("{controller_seed}", str(controller_seed)) if controller_seed is not None else value
        if key in {"path", "checkpoint", "model"} and not Path(resolved).is_absolute():
            return str((config_dir / resolved).resolve())
        return resolved
    return value


def materialize_suite(suite_path: str, robot_profile_path: str, backend: str, robot_asset: str, robot_format: str, controller_path: str, output_dir: str) -> Dict[str, Any]:
    suite = load_data(suite_path)
    robot = RobotProfile.from_dict(load_data(robot_profile_path))
    controller_template = load_data(controller_path)
    training_seed_config = suite.get("controller_training_seeds")
    if training_seed_config is None:
        training_seeds = [None]
    elif isinstance(training_seed_config, int):
        training_seeds = list(range(training_seed_config))
    else:
        training_seeds = list(training_seed_config)
    if not training_seeds:
        raise ValueError("controller_training_seeds cannot be empty when declared")
    if len(training_seeds) > 1 and not _contains_seed_template(controller_template):
        raise ValueError("multiple controller_training_seeds require {controller_seed} in a controller model/checkpoint path")
    root = Path(output_dir).resolve()
    worlds_dir, episodes_dir = root / "worlds", root / "episodes"
    worlds_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    repetitions = int(suite.get("repetitions", 1))
    difficulties = [float(value) for value in suite.get("difficulties", [0.5])]
    codes = suite.get("terrain_codes", ["rs1"])
    base_seed = int(suite.get("seed_base", suite.get("seeds", [0])[0]))
    conditions = []
    for difficulty in difficulties:
        for code in codes:
            if code == "ae":
                pairs = suite.get("ae_pairs", [["rs1", "rs5"]])
                if pairs == "all":
                    from .worlds import generate_ae_pairs

                    pairs = generate_ae_pairs()
                variants = [("ae-%s-%s" % tuple(pair), generate_ae(robot, pair[0], pair[1], difficulty, base_seed)) for pair in pairs]
            elif code == "aee":
                layout = suite.get("aee_layout", "long")
                variants = [("aee-%s" % layout, generate_aee(robot, layout, difficulty, base_seed, int(suite.get("aee_count", 12))))]
            else:
                variants = [(code, generate_terrain(code, robot, difficulty, base_seed))]
            for variant, world in variants:
                condition_id = "%s-d%.3f" % (variant, difficulty)
                path = worlds_dir / (condition_id + ".json")
                dump_json(str(path), world)
                conditions.append({"condition_id": condition_id, "difficulty": difficulty, "world": str(path)})
    stressor_cfg = suite.get("stressors", {})
    stressor_samples = [{}]
    if stressor_cfg.get("sampling") == "sobol":
        dimensions = stressor_cfg.get("dimensions", list(DEFAULT_STRESSOR_BOUNDS))
        bounds = {name: DEFAULT_STRESSOR_BOUNDS[name] for name in dimensions}
        stressor_samples = sobol(bounds, int(stressor_cfg.get("samples", max(8, repetitions))), base_seed)
    episodes = []
    for controller_index, controller_seed in enumerate(training_seeds):
        controller = _resolve_controller(copy.deepcopy(controller_template), Path(controller_path).resolve().parent, controller_seed)
        for condition_index, condition in enumerate(conditions):
            for repetition in range(repetitions):
                seed = base_seed + controller_index * 1000000 + condition_index * 10000 + repetition
                stressor_values = stressor_samples[repetition % len(stressor_samples)]
                stressors = [{"type": name, "value": value} for name, value in stressor_values.items()]
                controller_suffix = "-c%s" % controller_seed if controller_seed is not None else ""
                episode_id = "%s-%s%s-r%03d" % (suite["suite_id"], condition["condition_id"], controller_suffix, repetition)
                output = root / "runs" / episode_id
                episode = {
                    "episode_id": episode_id,
                    "backend": backend,
                    "robot_profile": str(Path(robot_profile_path).resolve()),
                    "robot_asset": str(Path(robot_asset).resolve()),
                    "robot_format": robot_format,
                    "world": condition["world"],
                    "controller": controller,
                    "output_dir": str(output),
                    "seed": seed,
                    "dt_s": float(suite.get("dt_s", 0.005)),
                    "control_decimation": int(suite.get("control_decimation", 4)),
                    "timeout_s": float(suite.get("timeout_s", 30.0)),
                    "headless": True,
                    "stressors": stressors,
                    "metadata": {"suite_id": suite["suite_id"], "condition_id": condition["condition_id"], "severity": condition["difficulty"], "repetition": repetition, "controller_training_seed": controller_seed, "locked": bool(suite.get("locked", False))},
                }
                episode_path = episodes_dir / (episode_id + ".yaml")
                dump_yaml(str(episode_path), episode)
                episodes.append(str(episode_path))
    manifest = {"schema_version": "1.0", "suite_id": suite["suite_id"], "source_suite": str(Path(suite_path).resolve()), "robot_profile": str(Path(robot_profile_path).resolve()), "backend": backend, "condition_count": len(conditions), "episode_count": len(episodes), "controller_training_seeds": training_seeds, "conditions": conditions, "episodes": episodes, "locked": bool(suite.get("locked", False))}
    dump_json(str(root / "suite_manifest.json"), manifest)
    return manifest
