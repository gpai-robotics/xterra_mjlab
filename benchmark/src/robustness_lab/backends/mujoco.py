"""MuJoCo worker with generic MJCF terrain injection and controller contract."""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Backend, BackendCapabilities, BackendUnavailable
from ..canonical_state import (
    CanonicalState,
    build_canonical_state,
    quaternion_to_euler_rpy,
    world_to_body,
)
from ..controllers import make_controller, timed_action, validate_controller_mapping
from ..environment_physics import (
    DEPTH_RESISTIVE_EFFECT,
    DEPTH_RESISTIVE_TIER,
    RIGID_VISCOUS_TIER,
    MudFootContact,
    MudParameters,
)
from ..io import sha256_file
from ..models import EpisodeLog, EpisodeSpec, RobotProfile, TelemetrySample, WorldSpec
from ..navigation import RouteFollower
from ..observation.action import dc_motor_torque
from ..observation.gates import (
    check_artifact_accepts,
    check_artifact_accepts_tensors,
    check_flat_ground,
)
from ..policy_manifest import load_policy_manifest
from ..rendering import make_mjcf_camera
from ..worlds import generate_terrain


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def _mjcf_rgba(color: str) -> str:
    """Convert a WorldSpec #RRGGBB material color to MuJoCo RGBA."""
    value = str(color or "").strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            channels = [int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
            return "%.6g %.6g %.6g 1" % tuple(channels)
        except ValueError:
            pass
    return "0.35 0.45 0.55 1"


def _absolutize_assets(root: ET.Element, source_dir: Path) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir"):
            value = compiler.get(attribute)
            if value and not Path(value).is_absolute():
                compiler.set(attribute, str((source_dir / value).resolve()))
    asset = root.find("asset")
    if asset is not None:
        for element in list(asset):
            filename = element.get("file")
            if filename and not Path(filename).is_absolute() and (compiler is None or not compiler.get("meshdir")):
                element.set("file", str((source_dir / filename).resolve()))



def _quat_wxyz_from_rpy(rpy: Sequence[float]) -> Tuple[float, float, float, float]:
    """Intrinsic roll-pitch-yaw (about x, then y, then z) to a MuJoCo wxyz quaternion."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def reached_goal(
    position: Sequence[float],
    goal: Sequence[float],
    spawn: Sequence[float],
    body_length_m: float,
    approach_direction: Optional[Sequence[float]] = None,
    lateral_tolerance_m: Optional[float] = None,
) -> bool:
    """True when the base has crossed the route-aligned goal gate.

    The gate begins a quarter body-length before the goal and is bounded
    laterally.  This preserves the historical straight-course pass point while
    making the test independent of the world's x axis.  ``lateral_tolerance_m``
    should normally be the robot stance width: a base still over the intended
    lane counts, while a robot on a different branch of a grid does not.

    The previous check was a vertical plane at x = goal.x. On a grid world whose
    route runs east then north, the robot walked one edge to (7.71, 0.92) --
    6.9 m from the goal at (7.80, 7.80) -- and was scored as having reached it.
    Every grid, ring and tree world was passable by walking one edge.
    """

    along_tolerance = 0.25 * float(body_length_m)
    lateral_tolerance = (
        along_tolerance
        if lateral_tolerance_m is None
        else float(lateral_tolerance_m)
    )
    dx = float(position[0]) - float(goal[0])
    dy = float(position[1]) - float(goal[1])
    if approach_direction is None:
        ax, ay = float(goal[0]) - float(spawn[0]), float(goal[1]) - float(spawn[1])
    else:
        ax, ay = float(approach_direction[0]), float(approach_direction[1])
    norm = math.hypot(ax, ay)
    if norm <= 1e-9:
        return False
    ax, ay = ax / norm, ay / norm
    along = dx * ax + dy * ay
    lateral = abs(dx * ay - dy * ax)
    return along >= -along_tolerance and lateral <= lateral_tolerance


def route_final_direction(world: WorldSpec) -> Optional[Tuple[float, float]]:
    """Planar direction of the central route's last leg, or None."""

    route = next(
        (
            item
            for item in world.routes
            if item.route_kind in {"central", "graph-route"}
        ),
        None,
    )
    if route is None or len(route.waypoints) < 2:
        return None
    a, b = route.waypoints[-2], route.waypoints[-1]
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    if math.hypot(dx, dy) <= 1e-9:
        return None
    return (dx, dy)


def video_overlay_primitives(
    route: Optional[Any],
    goal: Sequence[float],
    base_position: Sequence[float],
    base_yaw: float,
    command_velocity: Sequence[float],
    next_waypoint_index: Optional[int] = None,
    actual_velocity_world: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Build world-space debug primitives without touching simulation state.

    ``route`` is deliberately optional. The caller supplies it only when a
    route follower is actually producing the policy's command, so a
    constant-velocity episode does not misleadingly look waypoint-guided.
    Both velocity arrows are renderer diagnostics: orange is the commanded
    planar velocity and green is the measured planar base velocity.
    """

    # Keep the route above the surface rather than coplanar with it. The old
    # 6 cm lift and 3.5 cm waypoint radius were effectively sub-pixel in the
    # overview camera on composed worlds.
    lift = 0.14
    primitives: List[Dict[str, Any]] = []
    waypoints = list(route.waypoints) if route is not None else []
    raised = [
        [float(point[0]), float(point[1]), float(point[2]) + lift]
        for point in waypoints
    ]
    for start, end in zip(raised, raised[1:]):
        primitives.append(
            {
                "kind": "line",
                "start": start,
                "end": end,
                "color": [0.0, 0.85, 0.95, 0.9],
            }
        )
    for index, point in enumerate(raised):
        primitives.append(
            {
                "kind": "waypoint",
                "position": point,
                "index": index,
                "color": [0.0, 0.85, 0.95, 0.95],
            }
        )

    if raised and next_waypoint_index is not None:
        index = max(0, min(int(next_waypoint_index), len(raised) - 1))
        primitives.append(
            {
                "kind": "target",
                "position": raised[index],
                "index": index,
                "color": [1.0, 0.85, 0.05, 1.0],
            }
        )

    goal_surface_z = float(goal[2])
    if raised:
        # Goal poses use base height; route points use the terrain surface.
        goal_surface_z = raised[-1][2]
    primitives.append(
        {
            "kind": "goal",
            "position": [float(goal[0]), float(goal[1]), goal_surface_z],
            "color": [1.0, 0.1, 0.2, 1.0],
        }
    )

    forward = float(command_velocity[0]) if command_velocity else 0.0
    lateral = float(command_velocity[1]) if len(command_velocity) > 1 else 0.0
    cosine, sine = math.cos(float(base_yaw)), math.sin(float(base_yaw))
    world_vx = cosine * forward - sine * lateral
    world_vy = sine * forward + cosine * lateral
    arrow_start = [
        float(base_position[0]),
        float(base_position[1]),
        float(base_position[2]) + 0.28,
    ]
    arrow_end = [
        arrow_start[0] + world_vx,
        arrow_start[1] + world_vy,
        arrow_start[2],
    ]
    if math.hypot(world_vx, world_vy) > 1e-3:
        primitives.append(
            {
                "kind": "velocity",
                "start": arrow_start,
                "end": arrow_end,
                "color": [1.0, 0.45, 0.05, 1.0],
            }
        )
    if actual_velocity_world is not None and math.hypot(
        float(actual_velocity_world[0]), float(actual_velocity_world[1])
    ) > 1e-3:
        actual_start = [
            float(base_position[0]),
            float(base_position[1]),
            float(base_position[2]) + 0.42,
        ]
        primitives.append(
            {
                "kind": "actual_velocity",
                "start": actual_start,
                "end": [
                    actual_start[0] + float(actual_velocity_world[0]),
                    actual_start[1] + float(actual_velocity_world[1]),
                    actual_start[2],
                ],
                "color": [0.2, 1.0, 0.25, 1.0],
            }
        )
    return primitives


def add_video_overlays(
    mujoco: Any, scene: Any, primitives: Sequence[Dict[str, Any]]
) -> None:
    """Append renderer-only geoms to an ``MjvScene`` (never to the model)."""

    import numpy as np

    for primitive in primitives:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        kind = primitive["kind"]
        color = np.asarray(primitive["color"], dtype=float)
        if kind in {"line", "velocity", "actual_velocity"}:
            geom_type = (
                mujoco.mjtGeom.mjGEOM_ARROW
                if kind in {"velocity", "actual_velocity"}
                else mujoco.mjtGeom.mjGEOM_LINE
            )
            width = 0.035 if kind in {"velocity", "actual_velocity"} else 0.02
            mujoco.mjv_connector(
                geom,
                geom_type,
                width,
                np.asarray(primitive["start"], dtype=float),
                np.asarray(primitive["end"], dtype=float),
            )
            geom.rgba[:] = color
        else:
            radius = {"waypoint": 0.06, "target": 0.11, "goal": 0.13}[kind]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([radius, radius, radius], dtype=float),
                np.asarray(primitive["position"], dtype=float),
                np.eye(3).reshape(-1),
                color,
            )
        scene.ngeom += 1


def build_mjcf_scene(robot_asset: str, world: WorldSpec, output_path: str, dt_s: float) -> None:
    source = Path(robot_asset).resolve()
    root = ET.parse(str(source)).getroot()
    if root.tag != "mujoco":
        raise ValueError("robot MJCF root must be <mujoco>")
    _absolutize_assets(root, source.parent)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "%.9g" % dt_s)
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nuser_geom", str(max(4, int(size.get("nuser_geom", "0")))))
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    material_classes = list(world.materials)
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        # ``source_material`` keeps older AE manifests physically correct;
        # newer manifests retain the physical material in ``material`` and
        # carry their purple/brown slot color separately.
        physical_material_name = str(geometry.parameters.get("source_material", geometry.material))
        display_material_name = str(geometry.parameters.get("display_material", geometry.material))
        material = world.materials[physical_material_name]
        display_material = world.materials.get(display_material_name, material)
        depth_resistive_mud = (
            geometry.semantic == "marsh"
            and geometry.parameters.get("capability_tier")
            == DEPTH_RESISTIVE_TIER
        )
        attributes = {
            "name": "terrain_%s" % _safe_name(geometry.name),
            "type": "box",
            "pos": " ".join("%.9g" % value for value in geometry.pose.position),
            "euler": " ".join("%.9g" % value for value in geometry.pose.rpy),
            "size": " ".join("%.9g" % (value / 2.0) for value in geometry.size),
            "friction": "%.9g %.9g %.9g" % (material.dynamic_friction, max(material.dynamic_friction * 0.01, 0.0001), max(material.dynamic_friction * 0.001, 0.00001)),
            # MuJoCo has no separate static-friction slot. Preserve exact world
            # ground truth in model-resident geom data: static friction,
            # material class index/count, then restitution. The established
            # sliding coefficient above continues to control contact dynamics.
            "user": "%.9g %d %d %.9g"
            % (
                material.static_friction,
                material_classes.index(physical_material_name),
                len(material_classes),
                material.restitution,
            ),
            "rgba": _mjcf_rgba(display_material.color),
            "contype": "0" if depth_resistive_mud else "1",
            "conaffinity": "0" if depth_resistive_mud else "1",
        }
        restitution = float(material.restitution)
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("material restitution must be within [0, 1]")
        if material.compliance > 0 or restitution > 0:
            if restitution > 0:
                logarithm = math.log(max(restitution, 1e-12))
                damping_ratio = -logarithm / math.sqrt(
                    math.pi * math.pi + logarithm * logarithm
                )
                damping_ratio = max(damping_ratio, 1e-6)
            else:
                damping_ratio = max(material.damping, 1.0)
            attributes["solref"] = "%.9g %.9g" % (
                max(material.compliance, 0.002)
                if material.compliance > 0
                else 0.02,
                damping_ratio,
            )
        ET.SubElement(worldbody, "geom", attributes)
        if depth_resistive_mud:
            # The colored volume is visual-only so feet can penetrate it.  A
            # rigid 1 cm box whose top is at the volume bottom bounds sinkage.
            # Scales with the mud layer: a fixed 1 cm floor under a 2 cm layer is
            # a different terrain than under a 20 cm one. The small floor
            # keeps it thick enough not to tunnel at large timesteps.
            backstop_thickness = max(0.005, 0.25 * float(geometry.size[2]))
            bottom_z = float(geometry.pose.position[2]) - 0.5 * float(
                geometry.size[2]
            )
            backstop = dict(attributes)
            backstop.update(
                {
                    "name": "%s_backstop" % attributes["name"],
                    "pos": "%.9g %.9g %.9g"
                    % (
                        float(geometry.pose.position[0]),
                        float(geometry.pose.position[1]),
                        bottom_z - 0.5 * backstop_thickness,
                    ),
                    "size": "%.9g %.9g %.9g"
                    % (
                        0.5 * float(geometry.size[0]),
                        0.5 * float(geometry.size[1]),
                        0.5 * backstop_thickness,
                    ),
                    "contype": "1",
                    "conaffinity": "1",
                }
            )
            backstop.pop("solref", None)
            ET.SubElement(worldbody, "geom", backstop)
    # Move the first free-root body to the requested spawn. Most robot-only
    # Menagerie models use this convention.
    top_level_bodies = worldbody.findall("body")
    for body in top_level_bodies:
        if body.find("freejoint") is not None or any(joint.get("type") == "free" for joint in body.findall("joint")):
            body.set("pos", " ".join("%.9g" % value for value in world.spawn_pose.position))
            # The orientation too. A cambered world hands us a rolled spawn; a
            # curved one a yawed spawn. Dropping it spawned the robot level on a
            # banked floor -- one side's feet in the air, the other's inside the
            # ground -- before the policy had produced a single action.
            rpy = list(world.spawn_pose.rpy or (0.0, 0.0, 0.0))
            if any(abs(float(value)) > 1e-12 for value in rpy):
                body.set("quat", " ".join("%.9g" % value for value in _quat_wxyz_from_rpy(rpy)))
            break
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


class MujocoBackend(Backend):
    name = "mujoco"
    _OBSERVATIONS = frozenset(
        {
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "joint_acceleration",
            "joint_position_error",
            "base_position_w",
            "base_orientation_wxyz",
            "base_euler_rpy",
            "projected_gravity_b",
            "base_linear_velocity_b",
            "base_angular_velocity_b",
            "base_linear_acceleration_b",
            "base_height",
            "previous_action",
            "foot_contact",
            "foot_contact_force",
            "foot_position_b",
            "foot_velocity_b",
            "foot_slip",
            "self_collision",
            "actuator_saturation",
            "imu_bias",
            "episode_time",
            "time_remaining",
            "gait_phase",
            "gait_clock",
            "odometry_position",
            "odometry_heading",
            "magnetometer",
            "velocity_command",
            "height_scan",
            "height_scan_per_foot",
            "elevation_map",
            "terrain_normal_b",
            "lidar_rays",
            "point_cloud",
            "depth_image",
            "rgb_image",
            "semantic_segmentation",
            "occupancy_grid",
            "signed_distance_field",
            "stereo_disparity",
            "optical_flow",
            "range_sensor",
            "goal_vector_b",
            "goal_distance",
            "goal_heading",
            "waypoints_b",
            "heading_command",
            "body_height_command",
            "pitch_command",
            "roll_command",
            "gait_command",
            "footswing_height_command",
            "stance_width_command",
            "terrain_style_command",
            "friction_coefficient",
            "restitution",
            "base_velocity_truth",
            "terrain_height_truth",
            "payload_mass_truth",
            "com_offset_truth",
            "motor_strength_scale",
            "external_wrench_truth",
            "terrain_class_truth",
            "recurrent_state",
            "estimator_output",
            "adaptation_latent",
            "visual_latent",
            "traversability_map",
            "gnss_position",
        }
    )

    @classmethod
    def probe(cls) -> BackendCapabilities:
        # MuJoCo chooses its GL backend at import time. A coordinator probe may
        # run before a renderer selects OSMesa, so inspect installation metadata
        # without importing the native module; run() imports it in its worker.
        if find_spec("mujoco") is not None:
            try:
                version = metadata.version("mujoco")
            except metadata.PackageNotFoundError:
                version = "unknown"
            return BackendCapabilities(
                cls.name,
                True,
                version,
                ["box", "sphere", "cylinder", "mesh"],
                deformable=False,
                camera=True,
                contact_forces=True,
                environment_effects=[
                    "linear_velocity_drag",
                    DEPTH_RESISTIVE_EFFECT,
                ],
                observations=cls._OBSERVATIONS,
            )
        return BackendCapabilities(
            cls.name,
            False,
            observations=cls._OBSERVATIONS,
            reason="mujoco is not importable",
        )

    def reset_canonical_state(
        self,
        robot: RobotProfile,
        robot_asset: str,
        robot_format: str,
        velocity_command=(0.0, 0.0, 0.0),
    ) -> Optional[CanonicalState]:
        """Read the robot's exact default pose without stepping physics."""

        if str(robot_format).lower() not in {"mjcf", "xml"}:
            return None
        try:
            import mujoco
        except (ImportError, ModuleNotFoundError):
            return None

        model = mujoco.MjModel.from_xml_path(str(Path(robot_asset).resolve()))
        data = mujoco.MjData(model)
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot.base_link
        )
        if base_id < 0:
            return None

        free_joint_id = next(
            (
                joint_id
                for joint_id in range(model.njnt)
                if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
                and model.jnt_bodyid[joint_id] == base_id
            ),
            None,
        )
        if free_joint_id is None:
            return None
        free_qpos_address = int(model.jnt_qposadr[free_joint_id])
        data.qpos[free_qpos_address : free_qpos_address + 3] = (
            0.0,
            0.0,
            float(robot.standing_height_m),
        )
        data.qpos[free_qpos_address + 3 : free_qpos_address + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )

        joint_ids = []
        for joint_name in robot.controller_joint_order:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            joint_type = model.jnt_type[joint_id] if joint_id >= 0 else None
            if joint_id < 0 or (
                joint_type != mujoco.mjtJoint.mjJNT_HINGE
                and joint_type != mujoco.mjtJoint.mjJNT_SLIDE
            ):
                return None
            joint_ids.append(joint_id)
            data.qpos[model.jnt_qposadr[joint_id]] = (
                robot.default_joint_positions[joint_name]
            )

        mujoco.mj_forward(model, data)
        joint_position = [
            float(data.qpos[model.jnt_qposadr[joint_id]])
            for joint_id in joint_ids
        ]
        joint_velocity = [0.0] * len(joint_ids)
        defaults = [
            float(robot.default_joint_positions[name])
            for name in robot.controller_joint_order
        ]
        quaternion = data.xquat[base_id].tolist()
        return build_canonical_state(
            joint_order=robot.controller_joint_order,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_position_error=[
                position - default
                for position, default in zip(joint_position, defaults)
            ],
            base_position_w=data.xpos[base_id].tolist(),
            base_orientation_wxyz=quaternion,
            base_linear_velocity_w=[0.0, 0.0, 0.0],
            base_angular_velocity_w=[0.0, 0.0, 0.0],
            base_height=float(data.xpos[base_id, 2]),
            velocity_command=list(velocity_command),
            previous_action=[0.0] * len(joint_ids),
            episode_time=0.0,
        )

    def run(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        """Run the flat-ground adapter qualification before scored terrain."""

        compiled_policy = episode.controller.get("type") == "compiled_policy"
        gate_setting = episode.controller.get("flat_ground_gate")
        gate_enabled = (
            bool(gate_setting)
            if gate_setting is not None
            else compiled_policy
        )
        if not gate_enabled:
            return self._run_episode(episode, robot, world)
        locomotion_profile = (
            len(robot.foot_links) >= 2
            and len(robot.controller_joint_order) >= 4
        )
        if gate_setting is None and not locomotion_profile:
            result = self._run_episode(episode, robot, world)
            result.diagnostics["flat_ground_gate"] = {
                "status": "not_applicable",
                "enabled": True,
                "reason": "compiled policy uses a non-locomotion robot profile",
            }
            return result

        duration_s = float(
            episode.controller.get("flat_ground_duration_s", 2.0)
        )
        required_distance_m = float(
            episode.controller.get("flat_ground_required_distance_m", 0.5)
        )
        flat_world = generate_terrain("rs1", robot, 0.0, episode.seed)
        qualification_dir = Path(episode.output_dir).resolve() / "flat_ground_qualification"
        qualification_episode = replace(
            episode,
            episode_id="%s-flat-ground" % episode.episode_id,
            output_dir=str(qualification_dir),
            timeout_s=duration_s,
            record_video=False,
            stressors=[],
        )
        qualification = self._run_episode(
            qualification_episode, robot, flat_world
        )
        distance_m = (
            qualification.samples[-1].base_position_m[0]
            - flat_world.spawn_pose.position[0]
            if qualification.samples
            else 0.0
        )
        elapsed_s = (
            qualification.samples[-1].time_s
            if qualification.samples
            else duration_s
        )
        gate = check_flat_ground(
            distance_m=distance_m,
            duration_s=elapsed_s,
            required_distance_m=required_distance_m,
            fell=qualification.terminal_reason == "fall",
        )
        gate.update(
            {
                "enabled": True,
                "qualification_episode_id": qualification_episode.episode_id,
                "qualification_status": qualification.status,
                "qualification_terminal_reason": qualification.terminal_reason,
                "qualification_scene": qualification.diagnostics.get("scene"),
            }
        )
        if gate["status"] != "ready":
            return EpisodeLog(
                episode_id=episode.episode_id,
                backend=self.name,
                status="skipped",
                terminal_reason="adapter_unverified",
                diagnostics={
                    "reason": gate["reason"],
                    "flat_ground_gate": gate,
                    "qualification_diagnostics": qualification.diagnostics,
                    "policy_interface_modified": False,
                },
            )

        result = self._run_episode(episode, robot, world)
        result.diagnostics["flat_ground_gate"] = gate
        return result

    def _run_episode(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        try:
            import mujoco
            import numpy as np
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailable(str(error))
        environment_support = self.require_world_support(world)
        if episode.robot_format not in {"mjcf", "xml"}:
            raise BackendUnavailable("MuJoCo backend requires an MJCF/XML robot asset")
        supported_stressors = {"push", "push_impulse_ns", "action_latency_s", "perception_latency_s"}
        unsupported_stressors = sorted(
            {
                str(stressor.get("type", ""))
                for stressor in episode.stressors
                if stressor.get("type") not in supported_stressors
                and float(stressor.get("value", 0.0) or 0.0) != 0.0
            }
        )
        if unsupported_stressors:
            raise BackendUnavailable(
                "MuJoCo v1 adapter cannot faithfully apply stressors %s; refusing to run an altered condition"
                % ", ".join(unsupported_stressors)
            )
        if any(stressor.get("gait_phase") is not None for stressor in episode.stressors):
            raise BackendUnavailable("MuJoCo v1 adapter supports timed pushes but not gait-phase-triggered pushes")
        output_dir = Path(episode.output_dir).resolve()
        scene_path = output_dir / "resolved_scene.xml"
        build_mjcf_scene(episode.robot_asset, world, str(scene_path), episode.dt_s)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        for joint_index in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if joint_name in robot.default_joint_positions and model.jnt_type[joint_index] != mujoco.mjtJoint.mjJNT_FREE:
                data.qpos[model.jnt_qposadr[joint_index]] = robot.default_joint_positions[joint_name]
        mujoco.mj_forward(model, data)
        # Spawn drop-to-contact. The scene places the base a fixed
        # standing_height above the spawn point, which only rests the feet on the
        # ground for one particular stance and one flat surface. When the default
        # pose differs (e.g. a policy whose trained nominal pose is more extended
        # than the profile's inferred stance) or the local floor is cambered or
        # curved, that fixed height buries or floats the feet, and the contact
        # impulse on the first control step destabilizes the policy before it has
        # acted. Re-seat the base by measuring each foot against the ACTUAL floor
        # beneath it with a downward ray, then dropping until every foot is in
        # contact. standing_height_m -- and the terrain difficulty scaling that
        # derives from it -- is left untouched.
        free_joint_ids = [
            joint_index
            for joint_index in range(model.njnt)
            if model.jnt_type[joint_index] == mujoco.mjtJoint.mjJNT_FREE
        ]
        foot_geoms = [
            geom_index
            for geom_index in range(model.ngeom)
            if model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_SPHERE
            and (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_index])
                )
                or ""
            )
            in set(robot.foot_links)
        ]
        if free_joint_ids and foot_geoms:
            free_z_address = int(model.jnt_qposadr[free_joint_ids[0]]) + 2
            # Lift clear of any local terrain first so each downward ray reads the
            # top surface, not the underside of a step the foot spawned inside.
            data.qpos[free_z_address] += 0.30
            mujoco.mj_forward(model, data)
            down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            gaps = []
            for geom_index in foot_geoms:
                origin = np.array(data.geom_xpos[geom_index], dtype=np.float64)
                hit_geom = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    down,
                    None,
                    1,
                    int(model.geom_bodyid[geom_index]),
                    hit_geom,
                )
                if hit_geom[0] >= 0 and distance >= 0.0:
                    gaps.append(distance - float(model.geom_size[geom_index][0]))
            if gaps and max(gaps) - min(gaps) <= 0.03:
                # The base sits square to the local floor (feet within 3 cm of
                # coplanar -- only the front/rear stance offset). Drop so the
                # highest foot seats 1 mm in; the lower feet rest a few
                # millimetres deep, enough for the solver to register all four
                # from the first sample.
                data.qpos[free_z_address] -= max(gaps) + 0.001
            elif gaps:
                # The feet are far from coplanar with this floor: the base
                # orientation does not match the surface (a genuine spawn defect,
                # e.g. a level base on a bank). Seat only the lowest foot rather
                # than force the others metres deep -- do not paper over the
                # mismatch; leave it for the orientation logic and its test.
                data.qpos[free_z_address] -= min(gaps) + 0.001
            else:
                data.qpos[free_z_address] -= 0.30
            mujoco.mj_forward(model, data)
        navigation = episode.controller.get("navigation", {})
        central_route = next(
            (
                route
                for route in world.routes
                if route.route_kind in {"central", "graph-route"}
            ),
            None,
        )
        use_route_follower = (
            isinstance(navigation, dict)
            and navigation.get("assistance") == "route_follower"
            and central_route is not None
            and "command" not in episode.controller
        )
        velocity_command_source = "route_follower" if use_route_follower else "constant"
        compiled_policy = episode.controller.get("type") == "compiled_policy"
        compiled_observation_size = None
        policy_source_parameters = {}
        if compiled_policy:
            gate_started = time.perf_counter()
            manifest = load_policy_manifest(episode.controller["manifest"])
            compiled_observation_size = manifest.observation_plan.size

            def compiled_terms(plan: Any):
                for term in getattr(plan, "_terms", ()):
                    yield term
                for _, child in getattr(plan, "_plans", ()):
                    yield from compiled_terms(child)

            for term in compiled_terms(manifest.observation_plan):
                parameters = dict(term.geometry or {})
                parameters.setdefault("size", term.source_size)
                previous = policy_source_parameters.get(term.source)
                if previous is not None and previous != parameters:
                    raise ValueError(
                        "source %r has conflicting runtime parameters" % term.source
                    )
                policy_source_parameters[term.source] = parameters
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
                        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
                        "scene": str(scene_path),
                        "terrain_fidelity_tier": "rigid_contact",
                        "environment_support": environment_support,
                        "policy_interface_modified": False,
                        "velocity_command_source": velocity_command_source,
                        "observation_size": compiled_observation_size,
                        "previous_action_initialised_to_zero": True,
                        "reason": artifact_gate["reason"],
                        "artifact_gate": artifact_gate,
                    },
                )
        controller = make_controller(episode.controller, model.nu)
        controller.reset({"robot": robot.to_dict(), "world": world.to_dict(), "actuator_count": model.nu})
        joint_names = []
        for index in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            joint_names.append(name or "joint_%d" % index)
        actuator_joint_names = []
        for index in range(model.nu):
            joint_id = int(model.actuator_trnid[index, 0])
            actuator_joint_names.append(joint_names[joint_id] if 0 <= joint_id < len(joint_names) else "actuator_%d" % index)
        validate_controller_mapping(episode.controller, actuator_joint_names)
        position_control = compiled_policy or episode.controller.get("type") == "pd_hold" or episode.controller.get("action_mode") == "position"
        default_position_targets = [robot.default_joint_positions.get(name, 0.0) for name in actuator_joint_names]
        foot_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in robot.foot_links}
        foot_ids = {name: body_id for name, body_id in foot_ids.items() if body_id >= 0}
        foot_geom_ids = {
            name: [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
            ]
            for name, body_id in foot_ids.items()
        }
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.base_link)
        if base_id < 0:
            base_id = 1 if model.nbody > 1 else 0
        constant_velocity_command = list(
            episode.controller.get("command", [0.6, 0.0, 0.0])
        )
        velocity_command = constant_velocity_command
        route_follower = None
        if use_route_follower:
            route_follower = RouteFollower(
                central_route.waypoints,
                forward_speed=float(
                    navigation.get(
                        "forward_speed",
                        episode.controller.get("command_velocity_mps", 0.6),
                    )
                ),
                max_yaw_rate=float(navigation.get("max_yaw_rate", 1.0)),
                arrival_radius=float(navigation.get("arrival_radius", 0.15)),
            )
        canonical_action_indices = None
        canonical_joint_ids = None
        canonical_foot_names = None
        sensor_suite = None
        sensor_requests = ()
        stiffness_by_actuator = None
        damping_by_actuator = None
        motor_envelope = None
        if compiled_policy:
            canonical_joint_order = list(robot.controller_joint_order)
            if len(canonical_joint_order) != model.nu or set(canonical_joint_order) != set(actuator_joint_names):
                raise ValueError(
                    "RobotProfile.controller_joint_order does not match MuJoCo actuators; canonical=%s actuators=%s"
                    % (canonical_joint_order, actuator_joint_names)
                )
            canonical_indices = {name: index for index, name in enumerate(canonical_joint_order)}
            canonical_action_indices = [canonical_indices[name] for name in actuator_joint_names]
            motor_envelope = manifest.action.motor
            if manifest.action.gains_declared:
                stiffness_by_actuator = [
                    manifest.action.stiffness[index]
                    for index in canonical_action_indices
                ]
                damping_by_actuator = [
                    manifest.action.damping[index]
                    for index in canonical_action_indices
                ]
            joint_ids_by_name = {name: index for index, name in enumerate(joint_names)}
            canonical_joint_ids = [joint_ids_by_name[name] for name in canonical_joint_order]
            canonical_foot_names = list(robot.foot_links)
            missing_feet = [name for name in canonical_foot_names if name not in foot_ids]
            if missing_feet:
                raise ValueError(
                    "RobotProfile foot links are absent from the MuJoCo model: %s"
                    % ", ".join(missing_feet)
                )
            produced_sensors = {
                "height_scan",
                "height_scan_per_foot",
                "elevation_map",
                "terrain_normal_b",
                "lidar_rays",
                "point_cloud",
                "depth_image",
                "rgb_image",
                "semantic_segmentation",
                "occupancy_grid",
                "signed_distance_field",
                "stereo_disparity",
                "optical_flow",
                "range_sensor",
                "gait_phase",
                "gait_clock",
                "magnetometer",
                "traversability_map",
            }
            unique_requests = {}
            instance_modes = {}
            for request in manifest.observation_plan.sensor_requests:
                source, instance, geometry = request
                if source not in produced_sensors:
                    continue
                mode = instance is not None
                if source in instance_modes and instance_modes[source] != mode:
                    raise ValueError(
                        "sensor %r mixes named and unnamed instance declarations"
                        % source
                    )
                instance_modes[source] = mode
                key = (source, instance)
                if key in unique_requests and unique_requests[key] != geometry:
                    raise ValueError(
                        "sensor %r instance %r has conflicting geometry declarations"
                        % key
                    )
                unique_requests[key] = geometry
            sensor_requests = tuple(
                (source, instance, geometry)
                for (source, instance), geometry in unique_requests.items()
            )
            from .mujoco_sensors import MujocoSensorSuite

            sensor_suite = MujocoSensorSuite(
                model, data, robot, world=world, command=episode.controller
            )

        def current_canonical_state() -> CanonicalState:
            quaternion = data.xquat[base_id].tolist()
            base_position = data.xpos[base_id].tolist()
            linear_velocity_w = data.cvel[base_id, 3:6].tolist()
            angular_velocity_w = data.cvel[base_id, 0:3].tolist()
            commanded_position_targets = controller.decoder.decode(
                controller.previous_action
            )
            foot_contacts = {name: False for name in canonical_foot_names}
            foot_forces_w = {name: [0.0, 0.0, 0.0] for name in canonical_foot_names}
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                bodies = {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                magnitude = float(np.linalg.norm(force[:3]))
                for name in canonical_foot_names:
                    if foot_ids[name] in bodies:
                        foot_contacts[name] = True
                        foot_forces_w[name][2] += magnitude
            state = build_canonical_state(
                joint_order=canonical_joint_order,
                joint_position=[
                    float(data.qpos[model.jnt_qposadr[joint_id]])
                    for joint_id in canonical_joint_ids
                ],
                joint_velocity=[
                    float(data.qvel[model.jnt_dofadr[joint_id]])
                    for joint_id in canonical_joint_ids
                ],
                joint_torque=[
                    float(data.qfrc_actuator[model.jnt_dofadr[joint_id]])
                    for joint_id in canonical_joint_ids
                ],
                joint_acceleration=sensor_suite.joint_acceleration(),
                joint_position_error=sensor_suite.joint_position_error(
                    commanded_position_targets
                ),
                base_position_w=base_position,
                base_orientation_wxyz=quaternion,
                base_linear_velocity_w=linear_velocity_w,
                base_angular_velocity_w=angular_velocity_w,
                base_linear_acceleration_b=(
                    sensor_suite.base_linear_acceleration_b()
                ),
                base_height=sensor_suite.base_height(),
                previous_action=list(controller.previous_action),
                foot_contact=[foot_contacts[name] for name in canonical_foot_names],
                foot_contact_force=[
                    world_to_body(quaternion, foot_forces_w[name])
                    for name in canonical_foot_names
                ],
                foot_position_b=[
                    world_to_body(
                        quaternion,
                        [
                            float(data.xpos[foot_ids[name], axis]) - base_position[axis]
                            for axis in range(3)
                        ],
                    )
                    for name in canonical_foot_names
                ],
                foot_velocity_b=sensor_suite.foot_velocity_b(),
                foot_slip=sensor_suite.foot_slip(),
                self_collision=sensor_suite.self_collision(),
                actuator_saturation=sensor_suite.actuator_saturation(),
                imu_bias=sensor_suite.imu_bias(),
                episode_time=float(data.time),
                time_remaining=sensor_suite.time_remaining(episode.timeout_s),
                gait_phase=sensor_suite.gait_phase(),
                gait_clock=sensor_suite.gait_clock(),
                odometry_position=sensor_suite.odometry_position(),
                odometry_heading=sensor_suite.odometry_heading(),
                magnetometer=sensor_suite.magnetometer(),
                velocity_command=list(velocity_command),
            )
            contextual_sources = {
                "goal_vector_b",
                "goal_distance",
                "goal_heading",
                "waypoints_b",
                "heading_command",
                "body_height_command",
                "pitch_command",
                "roll_command",
                "gait_command",
                "footswing_height_command",
                "stance_width_command",
                "terrain_style_command",
                "friction_coefficient",
                "restitution",
                "base_velocity_truth",
                "terrain_height_truth",
                "payload_mass_truth",
                "com_offset_truth",
                "motor_strength_scale",
                "external_wrench_truth",
                "terrain_class_truth",
                "recurrent_state",
                "estimator_output",
                "adaptation_latent",
                "visual_latent",
                "gnss_position",
            }
            for source in contextual_sources.intersection(policy_source_parameters):
                setattr(
                    state,
                    source,
                    getattr(sensor_suite, source)(policy_source_parameters[source]),
                )
            for source, instance, geometry in sensor_requests:
                values = getattr(sensor_suite, source)(geometry)
                if instance is None:
                    setattr(state, source, values)
                else:
                    instances = getattr(state, source)
                    if not isinstance(instances, dict):
                        instances = {}
                        setattr(state, source, instances)
                    instances[instance] = values
            return state
        samples = []
        action = [0.0] * model.nu
        pending_actions = []
        observation_history = []
        inference_latency = 0.0
        observation_age = 0.0
        started = time.perf_counter()
        status, reason = "failed", "timeout"
        stressor_values = {item.get("type"): item.get("value") for item in episode.stressors if "value" in item}
        perception_latency = float(stressor_values.get("perception_latency_s", 0.0) or 0.0)
        configured_action_latency = float(stressor_values.get("action_latency_s", 0.0) or 0.0)
        marsh_regions = [region for region in world.regions if region.kind == "marsh"]
        marsh_geometry = [
            geometry for geometry in world.geometry if geometry.semantic == "marsh"
        ]
        requested_tier = str(environment_support["requested_tier"])
        marsh_drag = max(
            (
                float(geometry.parameters.get("viscous_drag", 0.0))
                for geometry in marsh_geometry
            ),
            default=0.0,
        )
        mud_patches = []
        if requested_tier == DEPTH_RESISTIVE_TIER:
            for geometry in marsh_geometry:
                mud_patches.append(
                    (
                        geometry,
                        MudParameters(
                            support_stress_pa=float(
                                geometry.parameters["support_stress_pa"]
                            ),
                            depth_exponent=float(
                                geometry.parameters["depth_exponent"]
                            ),
                            reference_depth_m=float(
                                geometry.parameters["reference_depth_m"]
                            ),
                            damping_pa_s_per_m=float(
                                geometry.parameters["damping_pa_s_per_m"]
                            ),
                            yield_stress_pa=float(
                                geometry.parameters["yield_stress_pa"]
                            ),
                            suction_time_constant_s=float(
                                geometry.parameters["suction_time_constant_s"]
                            ),
                            max_force_n=float(geometry.parameters["max_force_n"]),
                        ),
                    )
                )
        initial_mud_parameters = mud_patches[0][1] if mud_patches else None
        mud_contacts = {
            name: MudFootContact(initial_mud_parameters)
            for name in foot_ids
            if initial_mud_parameters is not None
        }
        for contact in mud_contacts.values():
            contact.reset()
        active_mud_patch = {name: None for name in mud_contacts}
        # Project the profile's ellipsoidal/spherical foot onto the horizontal
        # plane.  Go2's 44 mm spherical foot gives about 1.52e-3 m^2.
        foot_contact_area_m2 = math.pi * max(
            0.0, 0.5 * float(robot.foot_size_m[0])
        ) * max(0.0, 0.5 * float(robot.foot_size_m[1]))
        if not math.isfinite(foot_contact_area_m2):
            foot_contact_area_m2 = 0.0
        max_steps = int(math.ceil(episode.timeout_s / episode.dt_s))
        goal_approach = route_final_direction(world)
        video_path = None
        video_renderer = None
        video_writer = None
        video_frame_count = 0
        cv2 = None
        video_width, video_height = 640, 360
        if episode.record_video:
            try:
                import cv2
            except (ImportError, ModuleNotFoundError) as error:
                raise BackendUnavailable(
                    "MuJoCo video encoder requires OpenCV (cv2): %s" % error
                )
            video_path = output_dir / "video" / (episode.episode_id + ".mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_camera = make_mjcf_camera(mujoco, world)
            try:
                video_renderer = mujoco.Renderer(
                    model, height=video_height, width=video_width
                )
            except Exception as error:
                raise BackendUnavailable(
                    "MuJoCo video renderer is unavailable: %s" % error
                )
            video_fps = 1.0 / (episode.dt_s * episode.control_decimation)
            try:
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps,
                    (video_width, video_height),
                )
            except Exception as error:
                video_renderer.close()
                raise BackendUnavailable(
                    "OpenCV MP4 encoder 'mp4v' is unavailable: %s" % error
                )
            if not video_writer.isOpened():
                video_writer.release()
                video_renderer.close()
                if video_path.exists():
                    video_path.unlink()
                raise BackendUnavailable("OpenCV MP4 encoder 'mp4v' is unavailable")
        for step in range(max_steps + 1):
            if step % episode.control_decimation == 0:
                if route_follower is not None:
                    yaw = quaternion_to_euler_rpy(data.xquat[base_id].tolist())[2]
                    velocity_command = route_follower.command(
                        data.xpos[base_id].tolist(), yaw
                    )
                if compiled_policy:
                    current_observation = current_canonical_state()
                else:
                    current_observation = {
                        "qpos": data.qpos.tolist(),
                        "qvel": data.qvel.tolist(),
                        "base_position_m": data.xpos[base_id].tolist(),
                        "base_quaternion_wxyz": data.xquat[base_id].tolist(),
                        "command_velocity_mps": list(velocity_command),
                    }
                observation_history.append((float(data.time), current_observation))
                observation_cutoff = float(data.time) - perception_latency
                eligible_observations = [item for item in observation_history if item[0] <= observation_cutoff]
                observation_time, observation = eligible_observations[-1] if eligible_observations else observation_history[0]
                observation_age = max(0.0, float(data.time) - observation_time)
                observation_history = [item for item in observation_history if item[0] >= observation_time]
                if episode.controller.get("type") == "pd_hold":
                    controller_action = default_position_targets
                    requested_inference_latency = 0.0
                else:
                    controller_action, requested_inference_latency = timed_action(controller, observation)
                if compiled_policy:
                    controller_action = [
                        controller_action[index] for index in canonical_action_indices
                    ]
                if len(controller_action) != model.nu:
                    raise ValueError("controller returned %d actions; MuJoCo model requires %d" % (len(controller_action), model.nu))
                if position_control:
                    if stiffness_by_actuator is None:
                        stiffness_by_actuator = [
                            float(episode.controller.get("stiffness", 25.0))
                        ] * model.nu
                        damping_by_actuator = [
                            float(episode.controller.get("damping", 0.5))
                        ] * model.nu
                    requested_action = []
                    for actuator_index, target in enumerate(controller_action):
                        joint_id = int(model.actuator_trnid[actuator_index, 0])
                        position = float(data.qpos[model.jnt_qposadr[joint_id]])
                        velocity = float(data.qvel[model.jnt_dofadr[joint_id]])
                        torque = (
                            stiffness_by_actuator[actuator_index]
                            * (target - position)
                            - damping_by_actuator[actuator_index] * velocity
                        )
                        if motor_envelope is not None:
                            torque = dc_motor_torque(torque, velocity, motor_envelope)
                        if model.actuator_ctrllimited[actuator_index]:
                            torque = float(np.clip(torque, model.actuator_ctrlrange[actuator_index, 0], model.actuator_ctrlrange[actuator_index, 1]))
                        requested_action.append(torque)
                else:
                    requested_action = controller_action
                pending_actions.append(
                    (float(data.time) + configured_action_latency, requested_action, requested_inference_latency)
                )
            while pending_actions and pending_actions[0][0] <= float(data.time) + 1e-12:
                _, action, inference_latency = pending_actions.pop(0)
            if model.nu:
                data.ctrl[:] = np.asarray(action)
            # Apply configured impulse-like forces to the base.
            data.xfrc_applied[:] = 0.0
            if requested_tier == DEPTH_RESISTIVE_TIER:
                for name, body_id in foot_ids.items():
                    geom_ids = foot_geom_ids[name]
                    lowest_geom_id = min(
                        geom_ids,
                        key=lambda geom_id: float(data.geom_xpos[geom_id, 2]),
                        default=None,
                    )
                    if lowest_geom_id is None:
                        foot_position = data.xpos[body_id]
                    else:
                        foot_position = data.geom_xpos[lowest_geom_id]
                    foot_lowest_z = float(foot_position[2]) - 0.5 * float(
                        robot.foot_size_m[2]
                    )
                    patch_and_parameters = next(
                        (
                            (patch, parameters)
                            for patch, parameters in mud_patches
                            if abs(float(foot_position[0]) - float(patch.pose.position[0]))
                            <= 0.5 * float(patch.size[0])
                            and abs(
                                float(foot_position[1])
                                - float(patch.pose.position[1])
                            )
                            <= 0.5 * float(patch.size[1])
                        ),
                        None,
                    )
                    contact = mud_contacts[name]
                    if patch_and_parameters is None:
                        contact.step(
                            0.0,
                            float(data.cvel[body_id, 5]),
                            foot_contact_area_m2,
                            episode.dt_s,
                        )
                        active_mud_patch[name] = None
                        continue
                    patch, parameters = patch_and_parameters
                    if active_mud_patch[name] != patch.name:
                        contact.reset()
                        contact.parameters = parameters
                        active_mud_patch[name] = patch.name
                    mud_surface_z = float(patch.pose.position[2]) + 0.5 * float(
                        patch.size[2]
                    )
                    depth_m = max(0.0, mud_surface_z - foot_lowest_z)
                    force_n = contact.step(
                        depth_m,
                        float(data.cvel[body_id, 5]),
                        foot_contact_area_m2,
                        episode.dt_s,
                    )
                    data.xfrc_applied[body_id, 2] += force_n
            elif requested_tier == RIGID_VISCOUS_TIER and marsh_drag > 0.0:
                for body_id in sorted(set(foot_ids.values())):
                    position = data.xpos[body_id]
                    inside = any(
                        all(
                            region.minimum[axis]
                            <= float(position[axis])
                            <= region.maximum[axis]
                            for axis in range(3)
                        )
                        for region in marsh_regions
                    ) or any(
                        abs(float(position[0]) - float(patch.pose.position[0]))
                        <= 0.5 * float(patch.size[0])
                        and abs(float(position[1]) - float(patch.pose.position[1]))
                        <= 0.5 * float(patch.size[1])
                        for patch in marsh_geometry
                    )
                    if inside:
                        data.xfrc_applied[body_id, :3] -= (
                            marsh_drag
                            * max(float(model.body_mass[body_id]), 0.1)
                            * data.cvel[body_id, 3:6]
                        )
            disturbance_active = False
            disturbance_magnitude = 0.0
            for stressor in episode.stressors:
                if stressor.get("type") not in {"push", "push_impulse_ns"}:
                    continue
                onset = float(stressor.get("time_s", 2.0))
                duration = float(stressor.get("duration_s", 0.1 if stressor.get("type") == "push_impulse_ns" else episode.dt_s))
                if onset <= data.time < onset + duration:
                    if stressor.get("type") == "push_impulse_ns":
                        impulse = float(stressor.get("value", 0.0))
                        direction = [float(value) for value in stressor.get("direction", [0.0, 1.0, 0.0])]
                        direction_norm = math.sqrt(sum(value * value for value in direction)) or 1.0
                        force = [value / direction_norm * impulse / max(duration, episode.dt_s) for value in direction]
                        disturbance_magnitude = impulse
                    else:
                        force = stressor.get("force_n", [0.0, 0.0, 0.0])
                        disturbance_magnitude = math.sqrt(sum(float(value) ** 2 for value in force)) * duration
                    data.xfrc_applied[base_id, :3] = np.asarray(force)
                    disturbance_active = True
            step_started = time.perf_counter()
            mujoco.mj_step(model, data)
            step_duration = time.perf_counter() - step_started
            if step % episode.control_decimation:
                continue
            joint_position = {}
            joint_velocity = {}
            for joint_id, name in enumerate(joint_names):
                joint_type = model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qpos_address = model.jnt_qposadr[joint_id]
                dof_address = model.jnt_dofadr[joint_id]
                joint_position[name] = float(data.qpos[qpos_address])
                joint_velocity[name] = float(data.qvel[dof_address])
            torque = {name: float(data.qfrc_actuator[model.jnt_dofadr[index]]) for index, name in enumerate(joint_names) if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE}
            foot_positions = {name: data.xpos[body_id].tolist() for name, body_id in foot_ids.items()}
            foot_contacts = {name: False for name in foot_ids}
            foot_forces = {name: [0.0, 0.0, 0.0] for name in foot_ids}
            body_contacts = []
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                magnitude = float(np.linalg.norm(force[:3]))
                matched = False
                for name, body_id in foot_ids.items():
                    if body_id in bodies:
                        foot_contacts[name] = True
                        foot_forces[name][2] += magnitude
                        matched = True
                if base_id in bodies and not matched:
                    body_contacts.append(robot.base_link or "base")
            position = data.xpos[base_id].tolist()
            quaternion = data.xquat[base_id].tolist()
            linear_velocity = data.cvel[base_id, 3:6].tolist()
            angular_velocity = data.cvel[base_id, 0:3].tolist()
            sample = TelemetrySample(
                time_s=float(data.time),
                base_position_m=position,
                base_quaternion_wxyz=quaternion,
                base_linear_velocity_mps=linear_velocity,
                base_angular_velocity_rps=angular_velocity,
                command_velocity_mps=list(velocity_command),
                joint_position_rad=joint_position,
                joint_velocity_rps=joint_velocity,
                joint_torque_nm=torque,
                action=[float(value) for value in action],
                foot_contacts=foot_contacts,
                foot_positions_m=foot_positions,
                foot_forces_n=foot_forces,
                body_contacts=sorted(set(body_contacts)),
                observation_age_s=observation_age,
                inference_latency_s=inference_latency,
                simulator_step_s=step_duration,
                ground_truth={
                    "disturbance_active": disturbance_active,
                    "disturbance_magnitude": disturbance_magnitude,
                    "configured_action_latency_s": configured_action_latency,
                },
            )
            samples.append(sample)
            if video_renderer is not None:
                try:
                    video_renderer.update_scene(data, camera=video_camera)
                    overlay_primitives = video_overlay_primitives(
                        # A route is visualised only when it is the active
                        # source of guidance. This keeps the 3-D recording
                        # truthful for constant-command episodes.
                        central_route if route_follower is not None else None,
                        world.goal_pose.position,
                        position,
                        quaternion_to_euler_rpy(quaternion)[2],
                        velocity_command,
                        route_follower.index if route_follower is not None else None,
                        linear_velocity,
                    )
                    add_video_overlays(mujoco, video_renderer.scene, overlay_primitives)
                    rgb_frame = video_renderer.render()
                    bgr_frame = np.ascontiguousarray(rgb_frame[:, :, ::-1])
                    if route_follower is not None:
                        guidance_text = "WAYPOINT %d/%d" % (
                            min(route_follower.index + 1, len(route_follower.waypoints)),
                            len(route_follower.waypoints),
                        )
                        legend_text = (
                            "cyan route  yellow target  red goal  "
                            "orange command  green motion"
                        )
                    else:
                        guidance_text = "CONSTANT VELOCITY GUIDANCE"
                        legend_text = "red goal  orange command  green motion"
                    # Screen-space legend is written only into the copied video
                    # frame. The coloured route/arrows themselves remain 3-D,
                    # world-space MjvScene geoms.
                    cv2.rectangle(bgr_frame, (8, 8), (632, 56), (8, 12, 20), -1)
                    cv2.putText(
                        bgr_frame,
                        guidance_text,
                        (18, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        bgr_frame,
                        legend_text,
                        (18, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (220, 225, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    video_writer.write(bgr_frame)
                    video_frame_count += 1
                except Exception as error:
                    video_writer.release()
                    video_renderer.close()
                    if video_path.exists():
                        video_path.unlink()
                    raise BackendUnavailable(
                        "MuJoCo video renderer or OpenCV encoder became unavailable: %s"
                        % error
                    )
            route_complete = route_follower is None or route_follower.completed
            if route_complete and reached_goal(
                position,
                world.goal_pose.position,
                world.spawn_pose.position,
                robot.body_length_m,
                goal_approach,
                robot.stance_width_m,
            ):
                status, reason = "completed", "goal"
                break
            if position[2] < 0.2 * robot.standing_height_m:
                status, reason = "failed", "fall"
                break
        if world.terrain_type == "calibration" and reason == "timeout":
            status, reason = "completed", "calibration_complete"
        diagnostics = {"wall_time_s": time.perf_counter() - started, "mujoco_version": getattr(mujoco, "__version__", "unknown"), "scene": str(scene_path), "terrain_fidelity_tier": requested_tier, "environment_support": environment_support, "policy_interface_modified": False, "velocity_command_source": velocity_command_source}
        if video_renderer is not None:
            video_writer.release()
            video_renderer.close()
            if video_frame_count <= 1:
                if video_path.exists():
                    video_path.unlink()
                raise BackendUnavailable(
                    "episode video requires at least two control-step frames; got %d"
                    % video_frame_count
                )
            capture = cv2.VideoCapture(str(video_path))
            decoded_frames = 0
            while decoded_frames < 2:
                decoded, _ = capture.read()
                if not decoded:
                    break
                decoded_frames += 1
            capture.release()
            if decoded_frames < 2:
                if video_path.exists():
                    video_path.unlink()
                raise BackendUnavailable(
                    "OpenCV MP4 encoder 'mp4v' did not produce a multi-frame video"
                )
            diagnostics["video"] = str(video_path)
            diagnostics["video_sha256"] = sha256_file(str(video_path))
            diagnostics["video_overlays"] = {
                "coordinate_frame": "world",
                "guidance_mode": velocity_command_source,
                "route": "cyan" if route_follower is not None else None,
                "waypoints": "cyan" if route_follower is not None else None,
                "next_waypoint": "yellow" if route_follower is not None else None,
                "goal": "red",
                "command_velocity_body_to_world": "orange",
                "measured_velocity_world": "green",
                "scope": "3d_episode_video_only",
                "simulation_coupling": "renderer_only",
            }
        if route_follower is not None:
            diagnostics["route_completion"] = {
                "completed": route_follower.completed,
                "reached_waypoints": min(
                    route_follower.index, len(route_follower.waypoints)
                ),
                "total_waypoints": len(route_follower.waypoints),
                "planned_edge_traversals": world.metadata.get(
                    "canonical_route_edge_traversal_count"
                ),
                "planned_unique_edges": len(
                    world.metadata.get("canonical_route_unique_edges", [])
                ),
                "covers_all_graph_edges": world.metadata.get(
                    "canonical_route_covers_all_edges"
                ),
            }
        if compiled_policy:
            diagnostics.update(
                {
                    "observation_size": compiled_observation_size,
                    "previous_action_initialised_to_zero": True,
                }
            )
        if sensor_suite is not None:
            sensor_suite.close()
        return EpisodeLog(episode.episode_id, self.name, status, reason, samples, diagnostics=diagnostics)
