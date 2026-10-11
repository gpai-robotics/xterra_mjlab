"""Legacy Isaac Gym Preview 4 adapter.

This adapter deliberately uses the stable gymapi surface instead of importing
Isaac Lab. It supports URDF assets, static box terrain, PD/effort actions, and
canonical root/joint telemetry.
"""

from __future__ import annotations

import math
import time
from importlib.util import find_spec
from pathlib import Path
from typing import List

from .base import Backend, BackendCapabilities, BackendUnavailable
from ..canonical_state import CanonicalState, build_canonical_state, world_to_body
from ..controllers import make_controller, timed_action, validate_controller_mapping
from ..environment_physics import apply_linear_drag, world_physics_plan
from ..models import EpisodeLog, EpisodeSpec, RobotProfile, TelemetrySample, WorldSpec
from ..observation.gates import (
    check_artifact_accepts,
    check_artifact_accepts_tensors,
)
from ..policy_manifest import load_policy_manifest


def _quat_from_rpy(gymapi, rpy):
    roll, pitch, yaw = rpy
    return gymapi.Quat.from_euler_zyx(yaw, pitch, roll)


def _record_xyz(record):
    try:
        return [float(record[axis]) for axis in ("x", "y", "z")]
    except (ValueError, IndexError, TypeError):
        return [float(record[index]) for index in range(3)]


def _record_quat(record):
    try:
        return [float(record[axis]) for axis in ("w", "x", "y", "z")]
    except (ValueError, IndexError, TypeError):
        return [float(record[index]) for index in (3, 0, 1, 2)]


def _assign_xyz(record, values):
    try:
        for axis, value in zip(("x", "y", "z"), values):
            record[axis] = float(value)
    except (ValueError, IndexError, TypeError):
        for index, value in enumerate(values):
            record[index] = float(value)


class IsaacGymBackend(Backend):
    name = "isaacgym"
    _OBSERVATIONS = frozenset(
        {
            "joint_position",
            "joint_velocity",
            "base_position_w",
            "base_orientation_wxyz",
            "base_euler_rpy",
            "projected_gravity_b",
            "base_linear_velocity_b",
            "base_angular_velocity_b",
            "previous_action",
            "foot_contact",
            "foot_contact_force",
            "foot_position_b",
            "foot_velocity_b",
            "episode_time",
            "velocity_command",
        }
    )

    @classmethod
    def probe(cls) -> BackendCapabilities:
        # Importing gymapi loads Isaac Gym's native graphics plugins into the
        # caller. Those plugins conflict with MuJoCo rendering when capability
        # probes and render checks share a process, so discovery must remain
        # side-effect free; run() performs the real import in its worker.
        if find_spec("isaacgym") is not None:
            return BackendCapabilities(
                cls.name,
                True,
                "preview4",
                ["box", "triangle_mesh"],
                deformable=False,
                camera=False,
                contact_forces=True,
                environment_effects=["linear_velocity_drag"],
                observations=cls._OBSERVATIONS,
            )
        return BackendCapabilities(
            cls.name,
            False,
            observations=cls._OBSERVATIONS,
            reason="isaacgym is not importable",
        )

    def run(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        environment_support = self.require_world_support(world)
        physics_plan = world_physics_plan(world)
        if episode.record_video:
            raise BackendUnavailable("Isaac Gym v1 adapter does not yet expose video capture")
        if episode.stressors:
            raise BackendUnavailable("Isaac Gym v1 adapter does not yet apply stressors; refusing to run an altered condition")
        try:
            import numpy as np
            import torch
            from isaacgym import gymapi, gymtorch
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailable(str(error))
        if episode.robot_format != "urdf":
            raise BackendUnavailable("Isaac Gym adapter requires a URDF robot asset")
        compiled_policy = episode.controller.get("type") == "compiled_policy"
        compiled_observation_size = None
        if compiled_policy:
            gate_started = time.perf_counter()
            manifest = load_policy_manifest(episode.controller["manifest"])
            compiled_observation_size = manifest.observation_plan.size
            if hasattr(manifest.observation_plan, "sizes"):
                artifact_gate = check_artifact_accepts_tensors(
                    manifest.runtime["path"],
                    list(manifest.observation_plan.sizes.values()),
                    runtime_format=manifest.runtime.get("format"),
                )
            else:
                artifact_gate = check_artifact_accepts(
                    manifest.runtime["path"],
                    compiled_observation_size,
                    runtime_format=manifest.runtime.get("format"),
                )
            if artifact_gate["status"] != "ready":
                return EpisodeLog(
                    episode_id=episode.episode_id,
                    backend=self.name,
                    status="skipped",
                    terminal_reason="adapter_unverified",
                    diagnostics={
                        "wall_time_s": time.perf_counter() - gate_started,
                        "terrain_fidelity_tier": physics_plan.get(
                            "requested_tier", "rigid_contact"
                        ),
                        "environment_support": environment_support,
                        "policy_interface_modified": False,
                        "observation_size": compiled_observation_size,
                        "previous_action_initialised_to_zero": True,
                        "reason": artifact_gate["reason"],
                        "artifact_gate": artifact_gate,
                    },
                )
        gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = episode.dt_s
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0.0, 0.0, -9.80665)
        params.substeps = int(episode.metadata.get("substeps", 2))
        params.physx.use_gpu = bool(episode.metadata.get("use_gpu", True))
        params.use_gpu_pipeline = bool(episode.metadata.get("use_gpu_pipeline", True))
        compute_device = int(episode.metadata.get("compute_device_id", 0))
        graphics_device = -1 if episode.headless else int(episode.metadata.get("graphics_device_id", 0))
        sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise BackendUnavailable("Isaac Gym could not create a PhysX simulation; check NVIDIA driver/CUDA")
        samples: List[TelemetrySample] = []
        started = time.perf_counter()
        try:
            spacing = max(world.dimensions_m[0], world.dimensions_m[1]) + 2.0
            environment = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, -3.0), gymapi.Vec3(spacing, spacing, 5.0), 1)
            terrain_handles = []
            for geometry in world.geometry:
                if geometry.shape != "box":
                    continue
                options = gymapi.AssetOptions()
                options.fix_base_link = True
                options.disable_gravity = True
                asset = gym.create_box(sim, *geometry.size, options)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(*geometry.pose.position)
                transform.r = _quat_from_rpy(gymapi, geometry.pose.rpy)
                handle = gym.create_actor(environment, asset, transform, "terrain_%s" % geometry.name, 0, 0)
                properties = gym.get_actor_rigid_shape_properties(environment, handle)
                material = world.materials[str(geometry.parameters.get("source_material", geometry.material))]
                for prop in properties:
                    prop.friction = material.dynamic_friction
                    prop.restitution = material.restitution
                gym.set_actor_rigid_shape_properties(environment, handle, properties)
                terrain_handles.append(handle)
            source = Path(episode.robot_asset).resolve()
            asset_root = source.parent
            asset_file = source.name
            # package://go2_description/... is resolved when the root is the
            # directory containing the package folder.
            if "package://" in source.read_text(encoding="utf-8", errors="ignore"):
                package_folder = source.parent.parent
                asset_root = package_folder.parent
                asset_file = str(source.relative_to(asset_root))
            robot_options = gymapi.AssetOptions()
            robot_options.fix_base_link = False
            robot_options.collapse_fixed_joints = False
            position_control = compiled_policy or episode.controller.get("type") == "pd_hold" or episode.controller.get("action_mode") == "position"
            robot_options.default_dof_drive_mode = gymapi.DOF_MODE_POS if position_control else gymapi.DOF_MODE_EFFORT
            robot_options.use_mesh_materials = True
            robot_asset = gym.load_asset(sim, str(asset_root), asset_file, robot_options)
            if robot_asset is None:
                raise BackendUnavailable("Isaac Gym failed to load URDF: %s" % source)
            spawn = gymapi.Transform()
            spawn.p = gymapi.Vec3(*world.spawn_pose.position)
            spawn.r = _quat_from_rpy(gymapi, world.spawn_pose.rpy)
            actor = gym.create_actor(environment, robot_asset, spawn, robot.name, 0, 1)
            dof_names = list(gym.get_asset_dof_names(robot_asset))
            validate_controller_mapping(episode.controller, dof_names)
            body_names = list(gym.get_asset_rigid_body_names(robot_asset))
            body_indices = {name: index for index, name in enumerate(body_names)}
            global_body_indices = {name: gym.find_actor_rigid_body_index(environment, actor, name, gymapi.DOMAIN_SIM) for name in body_names}
            base_index = body_indices.get(robot.base_link, 0)
            canonical_joint_order = None
            canonical_joint_indices = None
            canonical_action_indices = None
            canonical_foot_names = None
            if compiled_policy:
                canonical_joint_order = list(robot.controller_joint_order)
                if len(canonical_joint_order) != len(dof_names) or set(
                    canonical_joint_order
                ) != set(dof_names):
                    raise ValueError(
                        "RobotProfile.controller_joint_order does not match Isaac Gym DOFs; canonical=%s dofs=%s"
                        % (canonical_joint_order, dof_names)
                    )
                dof_indices = {
                    name: index for index, name in enumerate(dof_names)
                }
                canonical_joint_indices = [
                    dof_indices[name] for name in canonical_joint_order
                ]
                canonical_indices = {
                    name: index for index, name in enumerate(canonical_joint_order)
                }
                canonical_action_indices = [
                    canonical_indices[name] for name in dof_names
                ]
                canonical_foot_names = list(robot.foot_links)
                missing_feet = [
                    name
                    for name in canonical_foot_names
                    if name not in body_indices
                    or global_body_indices.get(name, -1) < 0
                ]
                if missing_feet:
                    raise ValueError(
                        "RobotProfile foot links are absent from Isaac Gym body/contact data: %s"
                        % ", ".join(missing_feet)
                    )
            dof_properties = gym.get_asset_dof_properties(robot_asset)
            if position_control:
                dof_properties["driveMode"].fill(gymapi.DOF_MODE_POS)
                dof_properties["stiffness"].fill(float(episode.controller.get("stiffness", 25.0)))
                dof_properties["damping"].fill(float(episode.controller.get("damping", 0.5)))
            else:
                dof_properties["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            gym.set_actor_dof_properties(environment, actor, dof_properties)
            dof_state = gym.get_actor_dof_states(environment, actor, gymapi.STATE_ALL)
            for index, name in enumerate(dof_names):
                dof_state["pos"][index] = robot.default_joint_positions.get(name, 0.0)
                dof_state["vel"][index] = 0.0
            gym.set_actor_dof_states(environment, actor, dof_state, gymapi.STATE_ALL)
            gym.prepare_sim(sim)
            contact_force_tensor = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
            controller = make_controller(episode.controller, len(dof_names))
            controller.reset({"robot": robot.to_dict(), "world": world.to_dict(), "joint_names": dof_names})
            action = [robot.default_joint_positions.get(name, 0.0) for name in dof_names] if episode.controller.get("type") == "pd_hold" else [0.0] * len(dof_names)

            def current_canonical_state(step: int) -> CanonicalState:
                state = gym.get_actor_dof_states(
                    environment, actor, gymapi.STATE_ALL
                )
                bodies = gym.get_actor_rigid_body_states(
                    environment, actor, gymapi.STATE_ALL
                )
                gym.refresh_net_contact_force_tensor(sim)
                root_state = bodies[base_index]
                position_w = _record_xyz(root_state["pose"]["p"])
                quaternion_wxyz = _record_quat(root_state["pose"]["r"])
                linear_velocity_w = _record_xyz(root_state["vel"]["linear"])
                angular_velocity_w = _record_xyz(root_state["vel"]["angular"])
                return build_canonical_state(
                    joint_order=canonical_joint_order,
                    joint_position=[
                        float(state["pos"][index])
                        for index in canonical_joint_indices
                    ],
                    joint_velocity=[
                        float(state["vel"][index])
                        for index in canonical_joint_indices
                    ],
                    base_position_w=position_w,
                    base_orientation_wxyz=quaternion_wxyz,
                    base_linear_velocity_w=linear_velocity_w,
                    base_angular_velocity_w=angular_velocity_w,
                    velocity_command=list(
                        episode.controller.get("command", [0.6, 0.0, 0.0])
                    ),
                    previous_action=list(controller.previous_action),
                    episode_time=float(step) * episode.dt_s,
                    foot_contact=[
                        float(
                            torch.linalg.vector_norm(
                                contact_force_tensor[
                                    global_body_indices[name]
                                ]
                            )
                        )
                        > 1.0
                        for name in canonical_foot_names
                    ],
                    foot_contact_force=[
                        world_to_body(
                            quaternion_wxyz,
                            contact_force_tensor[global_body_indices[name]]
                            .detach()
                            .cpu()
                            .tolist(),
                        )
                        for name in canonical_foot_names
                    ],
                    foot_position_b=[
                        world_to_body(
                            quaternion_wxyz,
                            [
                                value - position_w[axis]
                                for axis, value in enumerate(
                                    _record_xyz(
                                        bodies[body_indices[name]]["pose"]["p"]
                                    )
                                )
                            ],
                        )
                        for name in canonical_foot_names
                    ],
                    foot_velocity_b=[
                        world_to_body(
                            quaternion_wxyz,
                            [
                                value - linear_velocity_w[axis]
                                for axis, value in enumerate(
                                    _record_xyz(
                                        bodies[body_indices[name]]["vel"]["linear"]
                                    )
                                )
                            ],
                        )
                        for name in canonical_foot_names
                    ],
                )

            inference_latency = 0.0
            environment_effect_steps = 0
            status, reason = "failed", "timeout"
            max_steps = int(math.ceil(episode.timeout_s / episode.dt_s))
            travel_direction = 1.0 if world.goal_pose.position[0] >= world.spawn_pose.position[0] else -1.0
            for step in range(max_steps + 1):
                bodies_for_effects = gym.get_actor_rigid_body_states(environment, actor, gymapi.STATE_ALL)
                base_for_effects = bodies_for_effects[base_index]
                updated_velocity, applied_regions = apply_linear_drag(
                    _record_xyz(base_for_effects["pose"]["p"]),
                    _record_xyz(base_for_effects["vel"]["linear"]),
                    physics_plan.get("effects", []),
                    episode.dt_s,
                )
                if applied_regions:
                    _assign_xyz(bodies_for_effects[base_index]["vel"]["linear"], updated_velocity)
                    gym.set_actor_rigid_body_states(environment, actor, bodies_for_effects, gymapi.STATE_VEL)
                    environment_effect_steps += 1
                if step % episode.control_decimation == 0:
                    if compiled_policy:
                        observation = current_canonical_state(step)
                    else:
                        state = gym.get_actor_dof_states(environment, actor, gymapi.STATE_ALL)
                        root_state = gym.get_actor_rigid_body_states(environment, actor, gymapi.STATE_ALL)[0]
                        observation = {"joint_position": state["pos"].tolist(), "joint_velocity": state["vel"].tolist(), "base_position_m": _record_xyz(root_state["pose"]["p"]), "command_velocity_mps": episode.controller.get("command", [0.6, 0.0, 0.0])}
                    if episode.controller.get("type") != "pd_hold":
                        controller_action, inference_latency = timed_action(controller, observation)
                        action = (
                            [
                                controller_action[index]
                                for index in canonical_action_indices
                            ]
                            if compiled_policy
                            else controller_action
                        )
                    if len(action) != len(dof_names):
                        raise ValueError("controller action size does not match Isaac Gym DOFs")
                    if position_control:
                        gym.set_actor_dof_position_targets(environment, actor, np.asarray(action, dtype=np.float32))
                    else:
                        gym.set_actor_dof_efforts(environment, actor, np.asarray(action, dtype=np.float32))
                step_started = time.perf_counter()
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                step_time = time.perf_counter() - step_started
                if step % episode.control_decimation:
                    continue
                state = gym.get_actor_dof_states(environment, actor, gymapi.STATE_ALL)
                bodies = gym.get_actor_rigid_body_states(environment, actor, gymapi.STATE_ALL)
                gym.refresh_net_contact_force_tensor(sim)
                root_state = bodies[body_indices.get(robot.base_link, 0)]
                position = _record_xyz(root_state["pose"]["p"])
                quaternion = _record_quat(root_state["pose"]["r"])
                linear_velocity = _record_xyz(root_state["vel"]["linear"])
                angular_velocity = _record_xyz(root_state["vel"]["angular"])
                foot_positions = {name: _record_xyz(bodies[index]["pose"]["p"]) for name, index in body_indices.items() if name in robot.foot_links}
                foot_velocities = {name: _record_xyz(bodies[index]["vel"]["linear"]) for name, index in body_indices.items() if name in robot.foot_links}
                foot_forces = {name: contact_force_tensor[global_body_indices[name]].detach().cpu().tolist() for name in robot.foot_links if name in global_body_indices}
                foot_contacts = {name: float(torch.linalg.vector_norm(contact_force_tensor[global_body_indices[name]])) > 1.0 for name in foot_forces}
                body_contacts = [name for name in body_names if name not in robot.foot_links and float(torch.linalg.vector_norm(contact_force_tensor[global_body_indices[name]])) > 1.0]
                samples.append(TelemetrySample(step * episode.dt_s, position, quaternion, linear_velocity, angular_velocity, list(episode.controller.get("command", [0.6, 0.0, 0.0])), {name: float(state["pos"][index]) for index, name in enumerate(dof_names)}, {name: float(state["vel"][index]) for index, name in enumerate(dof_names)}, action=[float(value) for value in action], foot_contacts=foot_contacts, foot_positions_m=foot_positions, foot_velocities_mps=foot_velocities, foot_forces_n=foot_forces, body_contacts=body_contacts, inference_latency_s=inference_latency, simulator_step_s=step_time))
                if travel_direction * (position[0] - world.goal_pose.position[0]) >= -0.25 * robot.body_length_m:
                    status, reason = "completed", "goal"
                    break
                if position[2] < 0.2 * robot.standing_height_m:
                    status, reason = "failed", "fall"
                    break
            diagnostics = {"wall_time_s": time.perf_counter() - started, "terrain_actor_count": len(terrain_handles), "contact_force_status": "mapped", "terrain_fidelity_tier": physics_plan.get("requested_tier", "rigid_contact"), "environment_support": environment_support, "environment_effect_steps": environment_effect_steps, "policy_interface_modified": False}
            if compiled_policy:
                diagnostics.update(
                    {
                        "observation_size": compiled_observation_size,
                        "previous_action_initialised_to_zero": True,
                    }
                )
            return EpisodeLog(episode.episode_id, self.name, status, reason, samples, diagnostics=diagnostics)
        finally:
            gym.destroy_sim(sim)
