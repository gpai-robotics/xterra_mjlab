"""Isaac Lab 2.x USD articulation adapter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _quat(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _color(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


_EXTEROCEPTIVE_OBSERVATIONS = frozenset(
    {
        "height_scan",
        "height_scan_per_foot",
        "elevation_map",
        "terrain_normal_b",
        "lidar_rays",
        "point_cloud",
        "range_sensor",
        "occupancy_grid",
        "signed_distance_field",
        "depth_image",
        "rgb_image",
        "semantic_segmentation",
        "stereo_disparity",
        "optical_flow",
    }
)

_TILED_CAMERA_OBSERVATIONS = frozenset(
    {"rgb_image", "semantic_segmentation", "optical_flow"}
)

_RAYCAST_OBSERVATIONS = _EXTEROCEPTIVE_OBSERVATIONS - _TILED_CAMERA_OBSERVATIONS


def _unverified_observation_reason(sources: Iterable[str]) -> str:
    """Explain why implemented Isaac Lab observations cannot be claimed yet."""

    names = sorted(set(sources))
    if len(names) == 1:
        return (
            "Isaac Lab implements observation source %r, but has never produced it "
            "in a run" % names[0]
        )
    return (
        "Isaac Lab implements observation sources %s, but has never produced them "
        "in a run" % ", ".join(repr(name) for name in names)
    )


@dataclass
class IsaacLabCapabilities(BackendCapabilities):
    """Isaac Lab capabilities split by execution evidence."""

    unverified_observations: FrozenSet[str] = field(default_factory=frozenset)

    def unverified_capabilities(self) -> FrozenSet[str]:
        """Implemented canonical fields not yet observed in a completed run."""

        return frozenset(self.unverified_observations)

    def assess_observation_sources(self, requested: Iterable[str]) -> Mapping[str, Any]:
        """Gate requested observations without mislabelling implementations absent."""

        requested_set = frozenset(requested)
        unverified = requested_set & self.unverified_capabilities()
        unsupported = requested_set - self.observation_capabilities() - unverified
        if unverified:
            return {
                "status": "blocked",
                "outcome_class": "adapter_unverified",
                "requested": sorted(requested_set),
                "unverified": sorted(unverified),
                "unsupported": sorted(unsupported),
                "reason": _unverified_observation_reason(unverified),
            }
        if unsupported:
            return {
                "status": "blocked",
                "outcome_class": "adapter_unverified",
                "requested": sorted(requested_set),
                "unverified": [],
                "unsupported": sorted(unsupported),
                "reason": "Isaac Lab does not implement observation sources %s"
                % ", ".join(repr(name) for name in sorted(unsupported)),
            }
        return {
            "status": "ready",
            "requested": sorted(requested_set),
            "unverified": [],
            "unsupported": [],
            "reason": "",
        }


def _compiled_observation_sources(manifest: Any) -> FrozenSet[str]:
    """Collect direct and encoder inputs from a compiled policy manifest."""

    sources = set()

    def visit(plan: Any) -> None:
        for term in getattr(plan, "_terms", ()):
            source = getattr(term, "source", None)
            if isinstance(source, str) and source:
                sources.add(source)
        for _, child_plan in getattr(plan, "_plans", ()):
            visit(child_plan)
        encoder = getattr(plan, "_encoder", None)
        for name, _ in getattr(encoder, "_inputs", ()):
            if isinstance(name, str) and name:
                sources.add(name)

    visit(manifest.observation_plan)
    return frozenset(sources)


def _compiled_sensor_requests(
    manifest: Any,
) -> Tuple[Tuple[str, Optional[str], Mapping[str, Any]], ...]:
    """Return unique native-sensor requests in manifest declaration order."""

    unique_requests = {}
    instance_modes = {}
    for source, instance, geometry in manifest.observation_plan.sensor_requests:
        if source not in _EXTEROCEPTIVE_OBSERVATIONS:
            continue
        mode = instance is not None
        if source in instance_modes and instance_modes[source] != mode:
            raise ValueError(
                "sensor %r mixes named and unnamed instance declarations" % source
            )
        instance_modes[source] = mode
        key = (source, instance)
        if key in unique_requests and unique_requests[key] != geometry:
            raise ValueError(
                "sensor %r instance %r has conflicting geometry declarations" % key
            )
        unique_requests[key] = geometry
    return tuple(
        (source, instance, geometry)
        for (source, instance), geometry in unique_requests.items()
    )


def _rpy_rotation(rpy: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _spawn_raycast_terrain_mesh(world: WorldSpec) -> None:
    """Create one static USD mesh containing the box terrain for ray casters."""

    import omni.usd
    from pxr import UsdGeom

    points = []
    face_counts = []
    face_indices = []
    local_corners = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    triangles = (
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    )
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        rotation = _rpy_rotation(geometry.pose.rpy)
        half_size = [float(value) / 2.0 for value in geometry.size]
        position = [float(value) for value in geometry.pose.position]
        first_vertex = len(points)
        for corner in local_corners:
            local = [corner[axis] * half_size[axis] for axis in range(3)]
            points.append(
                tuple(
                    position[row]
                    + sum(rotation[row][column] * local[column] for column in range(3))
                    for row in range(3)
                )
            )
        for triangle in triangles:
            face_counts.append(3)
            face_indices.extend(first_vertex + index for index in triangle)

    if not points:
        raise ValueError("Isaac Lab ray-cast observations require box terrain geometry")
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain/RobustnessLabRaycastMesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()


class IsaacLabBackend(Backend):
    name = "isaaclab"
    _IMPLEMENTED_OBSERVATIONS = frozenset(
        {
            "joint_position",
            "joint_velocity",
            "joint_torque",
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
    ) | _EXTEROCEPTIVE_OBSERVATIONS
    # No adapter episode has progressed far enough to produce an observation.
    # Keep implementation inventory separate from evidence-backed claims.
    _VERIFIED_OBSERVATIONS = frozenset()
    _UNVERIFIED_OBSERVATIONS = _IMPLEMENTED_OBSERVATIONS
    # Compatibility alias for code that used the former private inventory.
    _OBSERVATIONS = _IMPLEMENTED_OBSERVATIONS

    @classmethod
    def probe(cls) -> IsaacLabCapabilities:
        # Capability discovery runs in coordinator processes that may also use
        # MuJoCo. Avoid loading Isaac Lab's native Omniverse/graphics stack;
        # run() imports it in the isolated simulator worker.
        if find_spec("isaaclab") is not None:
            return IsaacLabCapabilities(
                cls.name,
                True,
                "2.x",
                ["box", "mesh"],
                deformable=False,
                camera=False,
                contact_forces=True,
                environment_effects=["linear_velocity_drag"],
                observations=cls._VERIFIED_OBSERVATIONS,
                unverified_observations=cls._UNVERIFIED_OBSERVATIONS,
            )
        return IsaacLabCapabilities(
            cls.name,
            False,
            observations=cls._VERIFIED_OBSERVATIONS,
            unverified_observations=cls._UNVERIFIED_OBSERVATIONS,
            reason="isaaclab is not importable",
        )

    def run(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        environment_support = self.require_world_support(world)
        physics_plan = world_physics_plan(world)
        if episode.record_video:
            raise BackendUnavailable("Isaac Lab v1 adapter does not yet expose video capture")
        if episode.stressors:
            raise BackendUnavailable("Isaac Lab v1 adapter does not yet apply stressors; refusing to run an altered condition")
        if episode.robot_format != "usd":
            raise BackendUnavailable("Isaac Lab adapter requires a USD robot asset; import URDF to USD first")
        compiled_policy = episode.controller.get("type") == "compiled_policy"
        compiled_observation_size = None
        sensor_requests = ()
        if compiled_policy:
            gate_started = time.perf_counter()
            manifest = load_policy_manifest(episode.controller["manifest"])
            compiled_observation_size = manifest.observation_plan.size
            capability_gate = self.probe().assess_observation_sources(
                _compiled_observation_sources(manifest)
            )
            if capability_gate["status"] != "ready":
                return EpisodeLog(
                    episode_id=episode.episode_id,
                    backend=self.name,
                    status="skipped",
                    terminal_reason=capability_gate["outcome_class"],
                    diagnostics={
                        "wall_time_s": time.perf_counter() - gate_started,
                        "device": str(episode.metadata.get("device", "cuda:0")),
                        "terrain_fidelity_tier": physics_plan.get(
                            "requested_tier", "rigid_contact"
                        ),
                        "environment_support": environment_support,
                        "policy_interface_modified": False,
                        "observation_size": compiled_observation_size,
                        "previous_action_initialised_to_zero": True,
                        "reason": capability_gate["reason"],
                        "capability_gate": capability_gate,
                    },
                )
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
                        "device": str(episode.metadata.get("device", "cuda:0")),
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
            sensor_requests = _compiled_sensor_requests(manifest)
        needs_tiled_camera = any(
            source in _TILED_CAMERA_OBSERVATIONS
            for source, _, _ in sensor_requests
        )
        try:
            from isaaclab.app import AppLauncher

            launcher_options = {"headless": episode.headless}
            if needs_tiled_camera:
                launcher_options["enable_cameras"] = True
            app = AppLauncher(**launcher_options).app
        except Exception as error:
            raise BackendUnavailable("Isaac Lab AppLauncher failed: %s" % error)
        samples: List[TelemetrySample] = []
        started = time.perf_counter()
        sensor_suite = None
        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            device = str(episode.metadata.get("device", "cuda:0"))
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=episode.dt_s, device=device))
            sim_utils.create_prim("/World/Terrain", "Xform")
            for index, geometry in enumerate(world.geometry):
                if geometry.shape != "box":
                    continue
                material = world.materials[str(geometry.parameters.get("source_material", geometry.material))]
                cfg = sim_utils.CuboidCfg(
                    size=tuple(geometry.size),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=material.static_friction, dynamic_friction=material.dynamic_friction, restitution=material.restitution),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_color(material.color)),
                )
                cfg.func("/World/Terrain/G_%05d" % index, cfg, translation=tuple(geometry.pose.position), orientation=_quat(geometry.pose.rpy))
            if any(source in _RAYCAST_OBSERVATIONS for source, _, _ in sensor_requests):
                _spawn_raycast_terrain_mesh(world)
            actuator_effort = max((item.effort_limit or 0.0 for item in robot.actuators), default=0.0) or 40.0
            actuator_velocity = max((item.velocity_limit or 0.0 for item in robot.actuators), default=0.0) or 30.0
            position_control = compiled_policy or episode.controller.get("type") == "pd_hold" or episode.controller.get("action_mode") == "position"
            robot_cfg = ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=episode.robot_asset, activate_contact_sensors=True),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(world.spawn_pose.position), rot=_quat(world.spawn_pose.rpy), joint_pos=robot.default_joint_positions),
                actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit_sim=actuator_effort, velocity_limit_sim=actuator_velocity, stiffness=float(episode.controller.get("stiffness", 25.0)) if position_control else 0.0, damping=float(episode.controller.get("damping", 0.5)) if position_control else 0.0)},
            )
            articulation = Articulation(robot_cfg)
            contact_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/Robot/.*", update_period=0.0, history_length=2, track_air_time=True, debug_vis=False))
            if sensor_requests:
                from .isaaclab_sensors import IsaacLabSensorSuite

                sensor_suite = IsaacLabSensorSuite(
                    articulation,
                    robot,
                    sensor_requests,
                    episode.dt_s,
                )
            sim.reset()
            contact_sensor.reset()
            if sensor_suite is not None:
                sensor_suite.reset()
            joint_names = list(articulation.data.joint_names)
            validate_controller_mapping(episode.controller, joint_names)
            body_names = list(articulation.data.body_names)
            foot_indices = {name: body_names.index(name) for name in robot.foot_links if name in body_names}
            contact_body_names = list(contact_sensor.body_names)
            contact_indices = {name: contact_body_names.index(name) for name in robot.foot_links if name in contact_body_names}
            canonical_joint_order = None
            canonical_joint_indices = None
            canonical_action_indices = None
            canonical_foot_names = None
            if compiled_policy:
                canonical_joint_order = list(robot.controller_joint_order)
                if len(canonical_joint_order) != len(joint_names) or set(
                    canonical_joint_order
                ) != set(joint_names):
                    raise ValueError(
                        "RobotProfile.controller_joint_order does not match Isaac Lab joints; canonical=%s joints=%s"
                        % (canonical_joint_order, joint_names)
                    )
                joint_indices = {
                    name: index for index, name in enumerate(joint_names)
                }
                canonical_joint_indices = [
                    joint_indices[name] for name in canonical_joint_order
                ]
                canonical_indices = {
                    name: index for index, name in enumerate(canonical_joint_order)
                }
                canonical_action_indices = [
                    canonical_indices[name] for name in joint_names
                ]
                canonical_foot_names = list(robot.foot_links)
                missing_feet = [
                    name
                    for name in canonical_foot_names
                    if name not in foot_indices or name not in contact_indices
                ]
                if missing_feet:
                    raise ValueError(
                        "RobotProfile foot links are absent from Isaac Lab body/contact data: %s"
                        % ", ".join(missing_feet)
                    )
            controller = make_controller(episode.controller, len(joint_names))
            controller.reset({"robot": robot.to_dict(), "world": world.to_dict(), "joint_names": joint_names})
            default_action = [robot.default_joint_positions.get(name, 0.0) for name in joint_names]
            action = list(default_action) if episode.controller.get("type") == "pd_hold" else [0.0] * len(joint_names)

            def current_canonical_state(step: int) -> CanonicalState:
                position_w = articulation.data.root_pos_w[0].detach().cpu().tolist()
                quaternion_wxyz = articulation.data.root_quat_w[0].detach().cpu().tolist()
                linear_velocity_w = articulation.data.root_lin_vel_w[0].detach().cpu().tolist()
                angular_velocity_w = articulation.data.root_ang_vel_w[0].detach().cpu().tolist()
                joint_position = articulation.data.joint_pos[0].detach().cpu().tolist()
                joint_velocity = articulation.data.joint_vel[0].detach().cpu().tolist()
                joint_torque = articulation.data.applied_torque[0].detach().cpu().tolist()
                forces_w = contact_sensor.data.net_forces_w[0].detach().cpu()
                state = build_canonical_state(
                    joint_order=canonical_joint_order,
                    joint_position=[
                        joint_position[index] for index in canonical_joint_indices
                    ],
                    joint_velocity=[
                        joint_velocity[index] for index in canonical_joint_indices
                    ],
                    joint_torque=[
                        joint_torque[index] for index in canonical_joint_indices
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
                        float(forces_w[contact_indices[name]].norm()) > 1.0
                        for name in canonical_foot_names
                    ],
                    foot_contact_force=[
                        world_to_body(
                            quaternion_wxyz,
                            forces_w[contact_indices[name]].tolist(),
                        )
                        for name in canonical_foot_names
                    ],
                    foot_position_b=[
                        world_to_body(
                            quaternion_wxyz,
                            [
                                value - position_w[axis]
                                for axis, value in enumerate(
                                    articulation.data.body_pos_w[
                                        0, foot_indices[name]
                                    ]
                                    .detach()
                                    .cpu()
                                    .tolist()
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
                                    articulation.data.body_lin_vel_w[
                                        0, foot_indices[name]
                                    ]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )
                            ],
                        )
                        for name in canonical_foot_names
                    ],
                )
                if sensor_suite is not None:
                    for source, instance, geometry in sensor_requests:
                        values = sensor_suite.read(source, instance, geometry)
                        if instance is None:
                            setattr(state, source, values)
                        else:
                            instances = getattr(state, source)
                            if not isinstance(instances, dict):
                                instances = {}
                                setattr(state, source, instances)
                            instances[instance] = values
                return state

            inference_latency = 0.0
            environment_effect_steps = 0
            status, reason = "failed", "timeout"
            max_steps = int(math.ceil(episode.timeout_s / episode.dt_s))
            travel_direction = 1.0 if world.goal_pose.position[0] >= world.spawn_pose.position[0] else -1.0
            for step in range(max_steps + 1):
                if not app.is_running():
                    status, reason = "error", "simulator_closed"
                    break
                if step % episode.control_decimation == 0:
                    if compiled_policy:
                        observation = current_canonical_state(step)
                    else:
                        observation = {"joint_position": articulation.data.joint_pos[0].detach().cpu().tolist(), "joint_velocity": articulation.data.joint_vel[0].detach().cpu().tolist(), "base_position_m": articulation.data.root_pos_w[0].detach().cpu().tolist(), "command_velocity_mps": episode.controller.get("command", [0.6, 0.0, 0.0])}
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
                    if len(action) != len(joint_names):
                        raise ValueError("controller action size does not match Isaac Lab articulation")
                    action_tensor = torch.tensor([action], device=device, dtype=torch.float32)
                    if position_control:
                        articulation.set_joint_position_target(action_tensor)
                    else:
                        articulation.set_joint_effort_target(action_tensor)
                updated_velocity, applied_regions = apply_linear_drag(
                    articulation.data.root_pos_w[0].detach().cpu().tolist(),
                    articulation.data.root_lin_vel_w[0].detach().cpu().tolist(),
                    physics_plan.get("effects", []),
                    episode.dt_s,
                )
                if applied_regions:
                    root_velocity = articulation.data.root_vel_w.clone()
                    root_velocity[0, :3] = torch.tensor(updated_velocity, device=device, dtype=root_velocity.dtype)
                    articulation.write_root_velocity_to_sim(root_velocity)
                    environment_effect_steps += 1
                articulation.write_data_to_sim()
                step_started = time.perf_counter()
                sim.step(render=needs_tiled_camera or not episode.headless)
                step_time = time.perf_counter() - step_started
                articulation.update(episode.dt_s)
                contact_sensor.update(episode.dt_s)
                if sensor_suite is not None:
                    sensor_suite.update(episode.dt_s)
                if step % episode.control_decimation:
                    continue
                position = articulation.data.root_pos_w[0].detach().cpu().tolist()
                quaternion = articulation.data.root_quat_w[0].detach().cpu().tolist()
                linear_velocity = articulation.data.root_lin_vel_w[0].detach().cpu().tolist()
                angular_velocity = articulation.data.root_ang_vel_w[0].detach().cpu().tolist()
                joint_position = articulation.data.joint_pos[0].detach().cpu().tolist()
                joint_velocity = articulation.data.joint_vel[0].detach().cpu().tolist()
                applied_torque = articulation.data.applied_torque[0].detach().cpu().tolist()
                foot_positions = {name: articulation.data.body_pos_w[0, index].detach().cpu().tolist() for name, index in foot_indices.items()}
                foot_velocities = {name: articulation.data.body_lin_vel_w[0, index].detach().cpu().tolist() for name, index in foot_indices.items()}
                forces = contact_sensor.data.net_forces_w[0].detach().cpu()
                foot_forces = {name: forces[index].tolist() for name, index in contact_indices.items()}
                foot_contacts = {name: float(forces[index].norm()) > 1.0 for name, index in contact_indices.items()}
                body_contacts = [name for index, name in enumerate(contact_body_names) if name not in contact_indices and float(forces[index].norm()) > 1.0]
                samples.append(TelemetrySample(step * episode.dt_s, position, quaternion, linear_velocity, angular_velocity, list(episode.controller.get("command", [0.6, 0.0, 0.0])), dict(zip(joint_names, joint_position)), dict(zip(joint_names, joint_velocity)), dict(zip(joint_names, applied_torque)), [float(value) for value in action], foot_contacts=foot_contacts, foot_positions_m=foot_positions, foot_velocities_mps=foot_velocities, foot_forces_n=foot_forces, body_contacts=body_contacts, inference_latency_s=inference_latency, simulator_step_s=step_time))
                if travel_direction * (position[0] - world.goal_pose.position[0]) >= -0.25 * robot.body_length_m:
                    status, reason = "completed", "goal"
                    break
                if position[2] < 0.2 * robot.standing_height_m:
                    status, reason = "failed", "fall"
                    break
            diagnostics = {"wall_time_s": time.perf_counter() - started, "device": device, "contact_force_status": "mapped", "contact_bodies": contact_body_names, "terrain_fidelity_tier": physics_plan.get("requested_tier", "rigid_contact"), "environment_support": environment_support, "environment_effect_steps": environment_effect_steps, "policy_interface_modified": False}
            if compiled_policy:
                diagnostics.update(
                    {
                        "observation_size": compiled_observation_size,
                        "previous_action_initialised_to_zero": True,
                    }
                )
            return EpisodeLog(episode.episode_id, self.name, status, reason, samples, diagnostics=diagnostics)
        finally:
            if sensor_suite is not None:
                sensor_suite.close()
            app.close()
