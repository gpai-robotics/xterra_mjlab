#!/usr/bin/env python3
"""Run a native FoReSight/RSL-RL policy on a Robustness Lab WorldSpec.

All unknown arguments are forwarded to the source repository's ``play.py``.
That preserves the exact trained observation, estimator, action and checkpoint
loading path while replacing only its terrain generator with our WorldSpec.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _extract_bridge_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--world")
    parser.add_argument("--batch-manifest")
    parser.add_argument("--foresight-root", required=True)
    parser.add_argument("--bridge-manifest")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    bridge, forwarded = parser.parse_known_args(argv)
    if not bridge.world and not bridge.batch_manifest:
        parser.error("one of --world or --batch-manifest is required")
    if bridge.world and not bridge.bridge_manifest:
        parser.error("--bridge-manifest is required with --world")
    return bridge, forwarded


BRIDGE_ARGS, FORWARDED_ARGS = _extract_bridge_arguments(sys.argv[1:])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORESIGHT_ROOT = Path(BRIDGE_ARGS.foresight_root).resolve()
if BRIDGE_ARGS.batch_manifest:
    _batch_document = json.loads(Path(BRIDGE_ARGS.batch_manifest).resolve().read_text(encoding="utf-8"))
    BATCH_ITEMS = list(_batch_document.get("items", []))
    if not BATCH_ITEMS:
        raise SystemExit("batch manifest contains no items")
    WORLD_PATH = Path(BATCH_ITEMS[0]["world"]).resolve()
else:
    BATCH_ITEMS = []
    WORLD_PATH = Path(BRIDGE_ARGS.world).resolve()
if not WORLD_PATH.is_file():
    raise SystemExit("world file does not exist: %s" % WORLD_PATH)
if not (FORESIGHT_ROOT / "scripts/rsl_rl/play.py").is_file():
    raise SystemExit("FoReSight source play script not found under %s" % FORESIGHT_ROOT)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(FORESIGHT_ROOT))
sys.path.insert(0, str(FORESIGHT_ROOT / "parkour_tasks"))
sys.path.insert(0, str(FORESIGHT_ROOT / "scripts/rsl_rl"))
sys.argv = [str(FORESIGHT_ROOT / "scripts/rsl_rl/play.py")] + FORWARDED_ARGS

# Import launches Isaac Sim and defines the source-native policy runner. It does
# not invoke main because it is imported rather than executed as __main__.
# Isaac Lab 2.3 moved `pretrained_checkpoint` from `isaaclab.utils` to
# `isaaclab_rl.utils`. The source play script still imports the old location.
# Register a LAZY proxy: resolving eagerly would pull in isaaclab.utils.mesh,
# which needs `pxr`, and USD bindings only exist once Isaac Sim's app is up.
# Shimming here keeps the policy's own repository untouched.
import types as _types  # noqa: E402

_pretrained = _types.ModuleType("isaaclab.utils.pretrained_checkpoint")


def _get_published_pretrained_checkpoint(*args, **kwargs):
    from isaaclab_rl.utils.pretrained_checkpoint import (  # noqa: WPS433
        get_published_pretrained_checkpoint as _resolved,
    )

    return _resolved(*args, **kwargs)


_pretrained.get_published_pretrained_checkpoint = _get_published_pretrained_checkpoint

# rsl_rl 3.1 moved `EmpiricalNormalization` from `rsl_rl.modules` to
# `rsl_rl.networks`. The source repository still imports the old location, but
# it needs 3.1's `PPO.init_storage` signature (unified `obs`), so downgrading
# rsl_rl is wrong. Re-export the moved symbol instead.
try:  # pragma: no cover - depends on the installed rsl_rl version
    import rsl_rl.modules as _rsl_modules

    if not hasattr(_rsl_modules, "EmpiricalNormalization"):
        from rsl_rl.networks import EmpiricalNormalization as _EmpiricalNormalization

        _rsl_modules.EmpiricalNormalization = _EmpiricalNormalization
except ImportError:
    pass

sys.modules.setdefault("isaaclab.utils.pretrained_checkpoint", _pretrained)

from scripts.rsl_rl import play as source_play  # noqa: E402

from isaaclab.utils import configclass  # noqa: E402
from parkour_isaaclab.terrains import ParkourSubTerrainBaseCfg, ParkourTerrainGeneratorCfg  # noqa: E402
from parkour_isaaclab.envs.parkour_manager_based_rl_env import ParkourManagerBasedRLEnv  # noqa: E402
import torch  # noqa: E402
from robustness_lab.environment_physics import apply_linear_drag, world_physics_plan  # noqa: E402
from robustness_lab.integrations.foresight_world import bridge_manifest, combine_native_result, world_to_foresight_terrain  # noqa: E402
from robustness_lab.integrations.foresight_world import foresight_terrain_requirements, material_face_tag  # noqa: E402
from robustness_lab.io import dump_json, load_data  # noqa: E402
from robustness_lab.models import WorldSpec  # noqa: E402


WORLD_SPEC = WorldSpec.from_dict(load_data(str(WORLD_PATH)))
MATERIAL_BINDING_REPORT = {"status": "pending", "materials": {}, "method": "separate USD collision mesh per physics material"}
VECTOR_WORLD_PATHS = [Path(item["world"]).resolve() for item in BATCH_ITEMS] if BATCH_ITEMS else []
TERRAIN_CALL_INDEX = 0
VECTOR_TRACKER = {
    "max_goal_indices": [], "episode_terminations": [], "world_switch_violations": [], "final_world_indices": [],
    "step": 0, "applied_stressors": [], "last_positions": [], "path_distances_m": [],
    "max_speeds_mps": [], "speed_sums_mps": [], "speed_samples": [], "max_roll_rad": [], "max_pitch_rad": [],
    "final_positions_m": [], "trajectories": [],
}
ENVIRONMENT_PHYSICS_PLANS = []
ENVIRONMENT_PHYSICS_TRACKER = {"application_counts": [], "regions": [], "dt_s": None}


def _refresh_environment_physics_plans():
    """Build backend-local effects without changing policy observations/actions."""
    global ENVIRONMENT_PHYSICS_PLANS
    specs = [WorldSpec.from_dict(load_data(str(path))) for path in VECTOR_WORLD_PATHS] if VECTOR_WORLD_PATHS else [WORLD_SPEC]
    ENVIRONMENT_PHYSICS_PLANS = [
        world_physics_plan(
            spec,
            (-7.0 - float(spec.spawn_pose.position[0]), -float(spec.spawn_pose.position[1]), 0.0),
        )
        for spec in specs
    ]
    ENVIRONMENT_PHYSICS_TRACKER["application_counts"] = [0 for _ in specs]
    ENVIRONMENT_PHYSICS_TRACKER["regions"] = [[] for _ in specs]
    ENVIRONMENT_PHYSICS_TRACKER["dt_s"] = None


def _apply_environment_physics(env):
    """Apply the selected WorldSpec fidelity tier inside the simulator step."""
    if not ENVIRONMENT_PHYSICS_PLANS:
        _refresh_environment_physics_plans()
    robot = env.scene["robot"]
    parkour = env.parkour_manager.get_term("base_parkour")
    origins = getattr(parkour, "env_origins", env.scene.env_origins)
    positions = (robot.data.root_pos_w[:, :3] - origins[:, :3]).detach().cpu().numpy()
    root_velocity = robot.data.root_vel_w.clone()
    dt_s = float(getattr(env, "step_dt", 0.0) or 0.0)
    if dt_s <= 0.0:
        sim_dt = float(getattr(getattr(env.cfg, "sim", None), "dt", 0.005))
        dt_s = sim_dt * max(1, int(getattr(env.cfg, "decimation", 1)))
    ENVIRONMENT_PHYSICS_TRACKER["dt_s"] = dt_s
    changed = []
    for env_index, position in enumerate(positions):
        plan = ENVIRONMENT_PHYSICS_PLANS[min(env_index, len(ENVIRONMENT_PHYSICS_PLANS) - 1)]
        updated, regions = apply_linear_drag(
            position,
            root_velocity[env_index, :3].detach().cpu().tolist(),
            plan.get("effects", []),
            dt_s,
        )
        if not regions:
            continue
        root_velocity[env_index, :3] = torch.tensor(updated, device=root_velocity.device, dtype=root_velocity.dtype)
        ENVIRONMENT_PHYSICS_TRACKER["application_counts"][env_index] += 1
        known = ENVIRONMENT_PHYSICS_TRACKER["regions"][env_index]
        for region in regions:
            if region not in known:
                known.append(region)
        changed.append(env_index)
    if changed:
        env_ids = torch.tensor(changed, device=env.device, dtype=torch.long)
        robot.write_root_velocity_to_sim(root_velocity[env_ids], env_ids=env_ids)


def _install_per_region_material_importer():
    """Create one collision mesh per material after Isaac combines terrain."""
    import trimesh
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils
    import isaaclab.terrains.terrain_importer as terrain_importer_module

    original = terrain_importer_module.create_prim_from_mesh

    def create_with_material_meshes(prim_path, mesh, **kwargs):
        colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        world_specs = [WorldSpec.from_dict(load_data(str(path))) for path in VECTOR_WORLD_PATHS] if VECTOR_WORLD_PATHS else [WORLD_SPEC]
        required = {}
        for spec in world_specs:
            for item in spec.geometry:
                material = spec.materials[item.material]
                identity = "%s:%.9g:%.9g:%.9g" % (
                    item.material, float(material.static_friction), float(material.dynamic_friction), float(material.restitution)
                )
                required[identity] = (item.material, material)
        prim_utils.create_prim(prim_path, "Xform")
        claimed = np.zeros(len(mesh.faces), dtype=bool)
        face_groups = {}
        for identity, (name, material) in sorted(required.items()):
            tag = np.asarray(material_face_tag(name, material), dtype=np.uint8)
            indices = np.flatnonzero(np.all(colors[:, :4] == tag[None, :], axis=1)).astype(np.int64)
            face_groups[identity] = indices
            claimed[indices] = True
        nominal_identity = next((identity for identity, value in required.items() if value[0] == "nominal"), next(iter(required)))
        untagged = np.flatnonzero(~claimed).astype(np.int64)
        if len(untagged):
            face_groups[nominal_identity] = np.concatenate((face_groups[nominal_identity], untagged))
        bound = {}
        for identity, (name, material) in sorted(required.items()):
            tag = np.asarray(material_face_tag(name, material), dtype=np.uint8)
            indices = face_groups[identity]
            suffix = "%02x%02x%02x" % tuple(int(value) for value in tag[:3])
            cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(material.static_friction),
                dynamic_friction=float(material.dynamic_friction),
                restitution=float(material.restitution),
                friction_combine_mode="average",
                restitution_combine_mode="max",
            )
            child_path = "%s/robustlab_%s_%s" % (prim_path, name.replace("-", "_"), suffix)
            if len(indices):
                child_mesh = mesh.submesh([indices], append=True, repair=False)
                if not isinstance(child_mesh, trimesh.Trimesh):
                    child_mesh = trimesh.util.concatenate(child_mesh)
                child_kwargs = dict(kwargs)
                child_kwargs["physics_material"] = cfg
                original(child_path, child_mesh, **child_kwargs)
            bound[identity] = {
                "canonical_name": name,
                "face_count": int(len(indices)),
                "collision_prim": child_path + "/mesh",
                "static_friction": float(material.static_friction),
                "dynamic_friction": float(material.dynamic_friction),
                "restitution": float(material.restitution),
            }
        missing = [name for name, value in bound.items() if value["face_count"] == 0]
        MATERIAL_BINDING_REPORT.update(
            {
                "status": "verified" if not missing else "failed",
                "materials": bound,
                "untagged_faces_assigned_to_nominal": int(len(untagged)),
                "missing_material_faces": missing,
                "rs11_fidelity": "rigid-viscous: lowered rigid patches plus backend velocity drag; deformable mud not applied",
            }
        )

    terrain_importer_module.create_prim_from_mesh = create_with_material_meshes


# Per-material collision meshes are opt-in until flat and friction calibration
# fixtures have passed on the pinned Isaac/PhysX runtime.
PER_REGION_MATERIALS = os.environ.get("PER_REGION_MATERIALS", "1") == "1"
if PER_REGION_MATERIALS:
    _install_per_region_material_importer()


def robustlab_world_terrain(difficulty, cfg, num_goals):
    global TERRAIN_CALL_INDEX
    del difficulty
    world_path = cfg.world_path
    if VECTOR_WORLD_PATHS:
        world_path = str(VECTOR_WORLD_PATHS[TERRAIN_CALL_INDEX])
        TERRAIN_CALL_INDEX += 1
    return world_to_foresight_terrain(
        world_path,
        num_goals,
        size=tuple(cfg.size),
        horizontal_scale=float(cfg.horizontal_scale),
        vertical_scale=float(cfg.vertical_scale),
    )


@configclass
class RobustLabWorldTerrainCfg(ParkourSubTerrainBaseCfg):
    function = robustlab_world_terrain
    world_path: str = ""


_original_parse_env_cfg = source_play.parse_env_cfg


def _parse_env_cfg_with_world(*args, **kwargs):
    global TERRAIN_CALL_INDEX
    cfg = _original_parse_env_cfg(*args, **kwargs)
    TERRAIN_CALL_INDEX = 0
    paths = VECTOR_WORLD_PATHS or [WORLD_PATH]
    all_requirements = [foresight_terrain_requirements(str(path)) for path in paths]
    requirements = {
        "size": [max(item["size"][0] for item in all_requirements), max(item["size"][1] for item in all_requirements)],
        "num_goals": max(item["num_goals"] for item in all_requirements),
    }
    vectorized = len(paths) > 1
    cfg.scene.terrain.terrain_generator = ParkourTerrainGeneratorCfg(
        size=tuple(requirements["size"]),
        border_width=20.0,
        num_rows=len(paths),
        num_cols=1,
        num_goals=int(requirements["num_goals"]),
        horizontal_scale=0.08,
        vertical_scale=0.005,
        slope_threshold=1.5,
        difficulty_range=(0.0, 0.0),
        use_cache=False,
        curriculum=vectorized,
        random_difficulty=False,
        sub_terrains={
            "robustlab_world": RobustLabWorldTerrainCfg(
                proportion=1.0,
                world_path=str(WORLD_PATH),
                horizontal_scale=0.08,
                vertical_scale=0.005,
            )
        },
    )
    cfg.scene.terrain.max_init_terrain_level = len(paths) - 1
    cfg.scene.num_envs = len(paths)
    cfg.parkours.base_parkour.debug_vis = True
    cfg.commands.base_velocity.debug_vis = True
    # Remove evaluation-time randomization so the first integration video is a
    # deterministic policy/world test. Stressors are added by the examiner.
    for event_name in ("randomize_rigid_body_com", "randomize_rigid_body_mass", "push_by_setting_velocity", "physics_material"):
        if hasattr(cfg.events, event_name):
            setattr(cfg.events, event_name, None)
    return cfg


source_play.parse_env_cfg = _parse_env_cfg_with_world


_original_env_step = ParkourManagerBasedRLEnv.step

if BATCH_ITEMS:
    from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter  # noqa: E402
    from parkour_isaaclab.envs.mdp.parkours.parkour_event import ParkourEvent  # noqa: E402

    _original_compute_origins = ParkourTerrainImporter._compute_env_origins_curriculum

    def _ordered_env_origins(self, num_envs, origins):
        if num_envs == origins.shape[0] and origins.shape[1] == 1:
            self.max_terrain_level = origins.shape[0]
            self.terrain_levels = torch.arange(num_envs, device=self.device, dtype=torch.long)
            self.terrain_types = torch.zeros(num_envs, device=self.device, dtype=torch.long)
            return origins[self.terrain_levels, self.terrain_types].clone()
        return _original_compute_origins(self, num_envs, origins)

    ParkourTerrainImporter._compute_env_origins_curriculum = _ordered_env_origins

    _original_resample_parkour = ParkourEvent._resample_command

    def _resample_without_world_switch(self, env_ids):
        """Reset an episode without moving an env to another benchmark world."""
        _original_resample_parkour(self, env_ids)
        fixed_levels = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        terrain_types = self.terrain.terrain_types
        self.terrain.terrain_levels[env_ids] = fixed_levels[env_ids]
        self.env_origins[env_ids] = self.terrain.terrain_origins[fixed_levels[env_ids], terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[fixed_levels[env_ids], terrain_types[env_ids]]
        selected_goals = self.terrain_goals[fixed_levels, terrain_types]
        extended_goals = torch.cat(
            (selected_goals, selected_goals[:, -1:].repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1
        )
        self.env_goals[env_ids] = extended_goals[env_ids]
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        robot_root_pos = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos
        target_norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        next_norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        self.target_yaw = torch.atan2(self.target_pos_rel[:, 1] / (target_norm[:, 0] + 1e-5), self.target_pos_rel[:, 0] / (target_norm[:, 0] + 1e-5))
        self.next_target_yaw = torch.atan2(self.next_target_pos_rel[:, 1] / (next_norm[:, 0] + 1e-5), self.next_target_pos_rel[:, 0] / (next_norm[:, 0] + 1e-5))
        level_cpu = fixed_levels.detach().cpu().numpy()
        type_cpu = terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[level_cpu, type_cpu]
        if self.debug_vis:
            self.env_per_heights = self.total_heights[fixed_levels, terrain_types]

    ParkourEvent._resample_command = _resample_without_world_switch
    def _tracked_env_step(self, action):
        step_index = int(VECTOR_TRACKER["step"])
        _apply_environment_physics(self)
        trigger_step = max(50, int(BATCH_ITEMS[0].get("steps", 1000)) // 3)
        if step_index == trigger_step:
            robot = self.scene["robot"]
            for env_index, item in enumerate(BATCH_ITEMS):
                for stressor in item.get("stressors", []):
                    if stressor.get("type") != "push_impulse_ns":
                        continue
                    impulse = float(stressor.get("value", 0.0))
                    mass = max(1e-6, float(item.get("robot_mass_kg", 1.0)))
                    direction = stressor.get("direction", [-1.0, 0.0, 0.0])
                    norm = sum(float(value) ** 2 for value in direction) ** 0.5 or 1.0
                    velocity = robot.data.root_vel_w[env_index : env_index + 1].clone()
                    for axis in range(3):
                        velocity[0, axis] += impulse / mass * float(direction[axis]) / norm
                    env_ids = torch.tensor([env_index], device=self.device, dtype=torch.long)
                    robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)
                    VECTOR_TRACKER["applied_stressors"].append(
                        {"environment_index": env_index, "step": step_index, "type": "push_impulse_ns", "value": impulse, "direction": direction}
                    )
        result = _original_env_step(self, action)
        VECTOR_TRACKER["step"] = step_index + 1
        parkour = self.parkour_manager.get_term("base_parkour")
        goals = parkour.cur_goal_idx.detach().cpu().tolist()
        if not VECTOR_TRACKER["max_goal_indices"]:
            VECTOR_TRACKER["max_goal_indices"] = list(goals)
            VECTOR_TRACKER["episode_terminations"] = [0 for _ in goals]
            VECTOR_TRACKER["world_switch_violations"] = [0 for _ in goals]
            VECTOR_TRACKER["final_world_indices"] = list(range(len(goals)))
        VECTOR_TRACKER["max_goal_indices"] = [max(left, right) for left, right in zip(VECTOR_TRACKER["max_goal_indices"], goals)]
        terminated = result[2]
        truncated = result[3]
        dones = (terminated | truncated).detach().cpu().reshape(-1).tolist()
        VECTOR_TRACKER["episode_terminations"] = [left + int(right) for left, right in zip(VECTOR_TRACKER["episode_terminations"], dones)]
        levels = parkour.terrain.terrain_levels.detach().cpu().tolist()
        VECTOR_TRACKER["world_switch_violations"] = [
            left + int(level != index) for index, (left, level) in enumerate(zip(VECTOR_TRACKER["world_switch_violations"], levels))
        ]
        VECTOR_TRACKER["final_world_indices"] = list(levels)
        robot = self.scene["robot"]
        origins = getattr(parkour, "env_origins", self.scene.env_origins)
        positions = (robot.data.root_pos_w[:, :3] - origins[:, :3]).detach().cpu().numpy()
        velocities = robot.data.root_lin_vel_w[:, :3].detach().cpu().numpy()
        quaternions = robot.data.root_quat_w[:, :4].detach().cpu().numpy()
        if not VECTOR_TRACKER["last_positions"]:
            count = len(positions)
            VECTOR_TRACKER["last_positions"] = [value.tolist() for value in positions]
            VECTOR_TRACKER["path_distances_m"] = [0.0] * count
            VECTOR_TRACKER["max_speeds_mps"] = [0.0] * count
            VECTOR_TRACKER["speed_sums_mps"] = [0.0] * count
            VECTOR_TRACKER["speed_samples"] = [0] * count
            VECTOR_TRACKER["max_roll_rad"] = [0.0] * count
            VECTOR_TRACKER["max_pitch_rad"] = [0.0] * count
            VECTOR_TRACKER["final_positions_m"] = [value.tolist() for value in positions]
            VECTOR_TRACKER["trajectories"] = [[] for _ in range(count)]
        for env_index, (position, velocity, quaternion) in enumerate(zip(positions, velocities, quaternions)):
            previous = np.asarray(VECTOR_TRACKER["last_positions"][env_index], dtype=np.float64)
            delta = float(np.linalg.norm(position - previous))
            # Ignore discontinuities caused by the source environment resetting
            # an episode to its spawn pose inside the fixed evaluation window.
            if delta < 1.0:
                VECTOR_TRACKER["path_distances_m"][env_index] += delta
            speed = float(np.linalg.norm(velocity))
            VECTOR_TRACKER["max_speeds_mps"][env_index] = max(VECTOR_TRACKER["max_speeds_mps"][env_index], speed)
            VECTOR_TRACKER["speed_sums_mps"][env_index] += speed
            VECTOR_TRACKER["speed_samples"][env_index] += 1
            w, x, y, z = [float(value) for value in quaternion]
            roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
            pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
            VECTOR_TRACKER["max_roll_rad"][env_index] = max(VECTOR_TRACKER["max_roll_rad"][env_index], abs(roll))
            VECTOR_TRACKER["max_pitch_rad"][env_index] = max(VECTOR_TRACKER["max_pitch_rad"][env_index], abs(pitch))
            VECTOR_TRACKER["last_positions"][env_index] = position.tolist()
            VECTOR_TRACKER["final_positions_m"][env_index] = position.tolist()
            item = BATCH_ITEMS[env_index] if BATCH_ITEMS else {"capture_telemetry": True, "telemetry_stride": 10}
            stride = max(1, int(item.get("telemetry_stride", 10)))
            if item.get("capture_telemetry", True) and (step_index % stride == 0 or dones[env_index]):
                VECTOR_TRACKER["trajectories"][env_index].append({
                    "step": step_index + 1, "position_m": position.tolist(), "linear_velocity_mps": velocity.tolist(),
                    "quaternion_wxyz": quaternion.tolist(), "goal_index": int(goals[env_index]), "episode_end": bool(dones[env_index]),
                })
        return result

    ParkourManagerBasedRLEnv.step = _tracked_env_step

else:
    def _environment_physics_step(self, action):
        _apply_environment_physics(self)
        return _original_env_step(self, action)

    ParkourManagerBasedRLEnv.step = _environment_physics_step


def _set_source_argument(name, value):
    setattr(source_play.args_cli, name, value)


def _prepare_manifest(world_path, bridge_path):
    requirements = foresight_terrain_requirements(str(world_path))
    manifest = bridge_manifest(str(world_path), num_goals=requirements["num_goals"], size=tuple(requirements["size"]))
    manifest.update(
        {
            "foresight_root": str(FORESIGHT_ROOT),
            "forwarded_arguments": FORWARDED_ARGS,
            "evidence_scope": "external native policy on canonical Robustness Lab geometry",
            "terrain_requirements": requirements,
        }
    )
    dump_json(str(bridge_path), manifest)


def _finalize_result(summary_path, bridge_path, video_dir, world_spec=None):
    video_path = ""
    candidate = video_dir / "rl-video-step-0.mp4" if video_dir else None
    if candidate and candidate.is_file():
        video_path = str(candidate)
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if BRIDGE_ARGS.benchmark_steps > 0 and not source_play.args_cli.video:
        summary["steps"] = BRIDGE_ARGS.benchmark_steps
        summary["headless_no_video"] = True
    updated_manifest = json.loads(bridge_path.read_text(encoding="utf-8"))
    updated_manifest["material_binding_report"] = MATERIAL_BINDING_REPORT
    active_world = world_spec or WORLD_SPEC
    compliant_used = any(item.material == "compliant" for item in active_world.geometry)
    physics = updated_manifest.get("environment_physics", {})
    physics_plan = physics.get("plan", {})
    required_effects = set(physics_plan.get("required_effects", []))
    supported_effects = {"linear_velocity_drag"}
    missing_effects = sorted(required_effects - supported_effects)
    environment_index = int(summary.get("environment_index", 0) or 0)
    counts = ENVIRONMENT_PHYSICS_TRACKER.get("application_counts", [])
    regions = ENVIRONMENT_PHYSICS_TRACKER.get("regions", [])
    runtime_physics = {
        "status": "verified" if not missing_effects else "environment_unsupported",
        "implementation_owner": "simulator_adapter",
        "policy_interface_modified": False,
        "selected_tier": physics_plan.get("requested_tier", "rigid_contact"),
        "supported_effects": sorted(supported_effects),
        "missing_effects": missing_effects,
        "application_steps": int(counts[environment_index]) if environment_index < len(counts) else 0,
        "regions_entered": list(regions[environment_index]) if environment_index < len(regions) else [],
        "simulation_dt_s": ENVIRONMENT_PHYSICS_TRACKER.get("dt_s"),
        "calibration_scope": "deterministic rigid-viscous approximation; not deformable or particle mud",
    }
    physics["runtime"] = runtime_physics
    physics["support"] = {
        "status": "environment_supported" if not missing_effects else "environment_unsupported",
        "capability_claim_eligible": not missing_effects,
        "requested_tier": physics_plan.get("requested_tier", "rigid_contact"),
        "supported_effects": sorted(supported_effects),
        "missing_effects": missing_effects,
        "implementation_owner": "simulator_adapter",
        "policy_interface_modified": False,
    }
    updated_manifest["environment_physics"] = physics
    summary["environment_physics_runtime"] = runtime_physics
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    physics_eligible = not missing_effects
    if PER_REGION_MATERIALS:
        updated_manifest["material_fidelity"].update(
            {
                "mode": "per_region_collision_meshes",
                "per_region_equivalent": MATERIAL_BINDING_REPORT.get("status") == "verified",
                "capability_claim_eligible": MATERIAL_BINDING_REPORT.get("status") == "verified" and physics_eligible,
                "rs11_claim_eligible": physics_eligible if compliant_used else None,
                "rs11_fidelity_tier": physics_plan.get("requested_tier") if compliant_used else None,
            }
        )
    if MATERIAL_BINDING_REPORT.get("status") == "verified":
        updated_manifest["limitations"] = [
            item for item in updated_manifest.get("limitations", [])
            if item != "single global Isaac terrain material in the current FoReSight importer"
        ]
        updated_manifest["limitations"].append("per-material collision meshes are runtime-verified; effective friction still requires calibration")
    dump_json(str(bridge_path), updated_manifest)
    result = combine_native_result(str(bridge_path), str(summary_path), video_path)
    dump_json(str(bridge_path.parent / "result.json"), result)


def _run_one(item):
    global WORLD_PATH, WORLD_SPEC
    WORLD_PATH = Path(item["world"]).resolve()
    WORLD_SPEC = WorldSpec.from_dict(load_data(str(WORLD_PATH)))
    bridge_path = Path(item["bridge_manifest"]).resolve()
    summary_path = Path(item["summary_json"]).resolve()
    video_dir = Path(item["video_dir"]).resolve() if item.get("video_dir") else None
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    MATERIAL_BINDING_REPORT.clear()
    MATERIAL_BINDING_REPORT.update({"status": "pending", "materials": {}, "method": "separate USD collision mesh per physics material"})
    _refresh_environment_physics_plans()
    _set_source_argument("task", item["task"])
    _set_source_argument("checkpoint", item["checkpoint"])
    _set_source_argument("seed", int(item["seed"]))
    _set_source_argument("summary_json", str(summary_path))
    _set_source_argument("video", bool(item.get("record_video", True)))
    _set_source_argument("video_length", int(item.get("steps", BRIDGE_ARGS.benchmark_steps or 1000)))
    _set_source_argument("video_dir", str(video_dir) if video_dir else None)
    _prepare_manifest(WORLD_PATH, bridge_path)
    source_play.main()
    _finalize_result(summary_path, bridge_path, video_dir)


def _run_batch():
    status_path = Path(BRIDGE_ARGS.batch_manifest).resolve().parent / "batch_status.json"
    checkpoints = {item["checkpoint"] for item in BATCH_ITEMS}
    tasks = {item["task"] for item in BATCH_ITEMS}
    steps = {int(item.get("steps", 1000)) for item in BATCH_ITEMS}
    if len(checkpoints) != 1 or len(tasks) != 1 or len(steps) != 1:
        raise ValueError("a vector batch must use one checkpoint, task, and step count")
    source_play.args_cli.task = next(iter(tasks))
    source_play.args_cli.checkpoint = next(iter(checkpoints))
    source_play.args_cli.seed = int(BATCH_ITEMS[0]["seed"])
    source_play.args_cli.num_envs = len(BATCH_ITEMS)
    source_play.args_cli.video = bool(BATCH_ITEMS[0].get("record_video", False))
    source_play.args_cli.video_length = next(iter(steps))
    source_play.args_cli.record_env_id = 0
    batch_summary = status_path.parent / "batch_summary.json"
    source_play.args_cli.summary_json = str(batch_summary)
    source_play.args_cli.video_dir = BATCH_ITEMS[0].get("video_dir") if source_play.args_cli.video else None
    for item in BATCH_ITEMS:
        bridge_path = Path(item["bridge_manifest"]).resolve()
        bridge_path.parent.mkdir(parents=True, exist_ok=True)
        requirements = foresight_terrain_requirements(item["world"])
        _prepare_manifest(Path(item["world"]).resolve(), bridge_path)
    _refresh_environment_physics_plans()
    VECTOR_TRACKER["max_goal_indices"] = []
    VECTOR_TRACKER["episode_terminations"] = []
    VECTOR_TRACKER["world_switch_violations"] = []
    VECTOR_TRACKER["final_world_indices"] = []
    VECTOR_TRACKER["step"] = 0
    VECTOR_TRACKER["applied_stressors"] = []
    for name in ("last_positions", "path_distances_m", "max_speeds_mps", "speed_sums_mps", "speed_samples", "max_roll_rad", "max_pitch_rad", "final_positions_m", "trajectories"):
        VECTOR_TRACKER[name] = []
    source_play.main()
    completed = []
    for index, item in enumerate(BATCH_ITEMS):
        output = Path(item["bridge_manifest"]).resolve().parent
        world_spec = WorldSpec.from_dict(load_data(item["world"]))
        requirements = foresight_terrain_requirements(item["world"])
        num_goals = int(max(value["num_goals"] for value in [foresight_terrain_requirements(x["world"]) for x in BATCH_ITEMS]))
        max_goal = int(VECTOR_TRACKER["max_goal_indices"][index])
        telemetry_path = output / "trajectory.json"
        trajectory = VECTOR_TRACKER["trajectories"][index] if VECTOR_TRACKER["trajectories"] else []
        if item.get("capture_telemetry", True) and trajectory:
            dump_json(str(telemetry_path), {
                "schema_version": "1.0", "trial_id": item.get("trial_id"),
                "sample_stride_steps": max(1, int(item.get("telemetry_stride", 10))),
                "control_frequency_hz": float(item.get("control_frequency_hz", 0.0) or 0.0),
                "samples": trajectory,
            })
        speed_samples = max(1, int(VECTOR_TRACKER["speed_samples"][index]))
        summary = {
            "checkpoint": item["checkpoint"], "task": item["task"], "seed": int(BATCH_ITEMS[0]["seed"]),
            "requested_seed": int(item["seed"]), "batch_seed": int(BATCH_ITEMS[0]["seed"]),
            "effective_seed_scope": "shared Isaac launch; repetitions are scheduled in separate launches",
            "environment_index": index,
            "assigned_world_index": index, "final_world_index": int(VECTOR_TRACKER["final_world_indices"][index]),
            "world_switch_violations": int(VECTOR_TRACKER["world_switch_violations"][index]),
            "applied_stressors": [value for value in VECTOR_TRACKER["applied_stressors"] if value["environment_index"] == index],
            "steps": next(iter(steps)), "max_goal_index": max_goal, "num_goals": num_goals,
            "success": bool(max_goal >= num_goals - 1),
            "episode_terminations": int(VECTOR_TRACKER["episode_terminations"][index]),
            "control_frequency_hz": float(item.get("control_frequency_hz", 0.0) or 0.0),
            "evaluation_duration_s": next(iter(steps)) / max(1.0, float(item.get("control_frequency_hz", 0.0) or 0.0)),
            "aggregate_path_distance_m": float(VECTOR_TRACKER["path_distances_m"][index]),
            "mean_base_speed_mps": float(VECTOR_TRACKER["speed_sums_mps"][index]) / speed_samples,
            "max_base_speed_mps": float(VECTOR_TRACKER["max_speeds_mps"][index]),
            "max_abs_roll_rad": float(VECTOR_TRACKER["max_roll_rad"][index]),
            "max_abs_pitch_rad": float(VECTOR_TRACKER["max_pitch_rad"][index]),
            "final_base_position_m": VECTOR_TRACKER["final_positions_m"][index],
            "trajectory": str(telemetry_path) if telemetry_path.is_file() else None,
            "telemetry_sample_count": len(trajectory),
        }
        summary_path = Path(item["summary_json"]).resolve()
        dump_json(str(summary_path), summary)
        video_dir = Path(item["video_dir"]).resolve() if index == 0 and source_play.args_cli.video and item.get("video_dir") else None
        _finalize_result(summary_path, Path(item["bridge_manifest"]).resolve(), video_dir, world_spec)
        completed.append({"index": index, "trial_id": item.get("trial_id"), "status": "completed", "result": str(output / "result.json")})
        dump_json(str(status_path), {"status": "running", "completed": completed, "total": len(BATCH_ITEMS)})
    dump_json(str(status_path), {"status": "completed", "completed": completed, "total": len(BATCH_ITEMS), "execution_mode": "parallel_environments"})


try:
    original_is_running = None
    if BRIDGE_ARGS.benchmark_steps > 0 and not source_play.args_cli.video:
        original_is_running = source_play.simulation_app.is_running
        calls = {"count": 0}

        def _bounded_is_running():
            caller = inspect.currentframe().f_back
            in_policy_loop = (
                caller is not None
                and Path(caller.f_code.co_filename).resolve() == Path(source_play.__file__).resolve()
                and caller.f_code.co_name == "main"
                and caller.f_lineno >= 287
            )
            if in_policy_loop:
                calls["count"] += 1
                if calls["count"] > BRIDGE_ARGS.benchmark_steps:
                    return False
            return original_is_running()

        source_play.simulation_app.is_running = _bounded_is_running
    if BATCH_ITEMS:
        _run_batch()
    else:
        _run_one(
            {
                "world": str(WORLD_PATH),
                "bridge_manifest": str(Path(BRIDGE_ARGS.bridge_manifest).resolve()),
                "summary_json": source_play.args_cli.summary_json,
                "video_dir": source_play.args_cli.video_dir,
                "checkpoint": source_play.args_cli.checkpoint,
                "task": source_play.args_cli.task,
                "seed": source_play.args_cli.seed,
                "steps": source_play.args_cli.video_length if source_play.args_cli.video else BRIDGE_ARGS.benchmark_steps,
                "record_video": source_play.args_cli.video,
            }
        )
except BaseException as error:
    error_root = Path(BRIDGE_ARGS.batch_manifest).resolve().parent if BRIDGE_ARGS.batch_manifest else Path(BRIDGE_ARGS.bridge_manifest).resolve().parent
    dump_json(str(error_root / "bridge_error.json"), {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    raise
finally:
    source_play.simulation_app.close()
