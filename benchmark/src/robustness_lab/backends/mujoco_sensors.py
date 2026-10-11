"""MuJoCo canonical sensors backed by simulator state and rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..canonical_state import world_to_body
from ..observation.sensor_geometry import (
    height_scan_points,
    validate_camera_geometry,
    validate_lidar_geometry,
)


# Unit direction of magnetic north (+world X) with a roughly 66-degree
# downward inclination, based on the former nominal [20, 0, -45] uT field.
DEFAULT_MAGNETIC_REFERENCE_W = (4.0 / math.sqrt(97.0), 0.0, -9.0 / math.sqrt(97.0))


def _rotation_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("orientation_wxyz must be a finite, non-zero quaternion")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _unit_vector(vector: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("%s must be a finite, non-zero three-vector" % label)
    return result / norm


class MujocoSensorSuite:
    """Produce canonical observations from live MuJoCo data.

    Stateful finite differences and dead reckoning are sampled on calls to the
    corresponding methods.  Repeated calls at the same simulation time return
    the cached value, so asking for multiple observation tensors cannot advance
    an estimator twice during one control step.
    """

    _SELF_HIT_EPSILON = 1e-4
    _MAX_SELF_HITS = 64

    def __init__(
        self,
        model: Any,
        data: Any,
        robot_profile: Any,
        world: Optional[Any] = None,
        command: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.robot_profile = robot_profile
        self.world = world
        self.command = {} if command is None else dict(command)
        self.base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot_profile.base_link
        )
        if self.base_id < 0:
            self.base_id = 1 if model.nbody > 1 else 0
        self._robot_body_ids = self._find_robot_body_ids()
        self._canonical_joint_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in robot_profile.controller_joint_order
        )
        self._foot_body_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in robot_profile.foot_links
        )
        if any(joint_id < 0 for joint_id in self._canonical_joint_ids):
            raise ValueError("canonical joint order contains a joint absent from MuJoCo")
        if any(body_id < 0 for body_id in self._foot_body_ids):
            raise ValueError("canonical foot order contains a body absent from MuJoCo")
        self._actuator_by_joint_id = {
            int(model.actuator_trnid[index, 0]): index
            for index in range(model.nu)
            if int(model.actuator_trnid[index, 0]) >= 0
        }
        self._effort_limit_by_joint = {
            actuator.joint: (
                None
                if actuator.effort_limit is None
                else abs(float(actuator.effort_limit))
            )
            for actuator in robot_profile.actuators
        }
        self._previous_joint_velocity: Optional[np.ndarray] = None
        self._joint_acceleration_time: Optional[float] = None
        self._joint_acceleration_value: Optional[List[float]] = None
        self._previous_base_velocity_w: Optional[np.ndarray] = None
        self._base_acceleration_time: Optional[float] = None
        self._base_acceleration_value: Optional[List[float]] = None
        self._odometry_time: Optional[float] = None
        self._odometry_position_value = np.zeros(3, dtype=float)
        self._odometry_heading_value = 0.0
        self._renderers: Dict[Tuple[int, int], Any] = {}
        self._previous_rgb_frames: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._policy_state: Dict[str, Optional[List[float]]] = {
            "recurrent_state": None,
            "estimator_output": None,
            "adaptation_latent": None,
            "visual_latent": None,
        }

    def reset(self) -> None:
        """Clear every episode-local estimator and policy-state buffer."""

        self._previous_joint_velocity = None
        self._joint_acceleration_time = None
        self._joint_acceleration_value = None
        self._previous_base_velocity_w = None
        self._base_acceleration_time = None
        self._base_acceleration_value = None
        self._odometry_time = None
        self._odometry_position_value[:] = 0.0
        self._odometry_heading_value = 0.0
        self._previous_rgb_frames.clear()
        for name in self._policy_state:
            self._policy_state[name] = None

    def _find_robot_body_ids(self) -> frozenset:
        root_id = self.base_id
        while root_id > 0 and int(self.model.body_parentid[root_id]) != 0:
            root_id = int(self.model.body_parentid[root_id])
        bodies = {root_id}
        changed = True
        while changed:
            changed = False
            for body_id in range(1, self.model.nbody):
                if body_id not in bodies and int(self.model.body_parentid[body_id]) in bodies:
                    bodies.add(body_id)
                    changed = True
        return frozenset(bodies)

    @property
    def _base_position_w(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.base_id], dtype=float)

    @property
    def _body_to_world(self) -> np.ndarray:
        return np.asarray(self.data.xmat[self.base_id], dtype=float).reshape(3, 3)

    def _body_linear_velocity_w(self, body_id: int) -> np.ndarray:
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return velocity[3:6]

    def _base_linear_velocity_w(self) -> np.ndarray:
        return self._body_linear_velocity_w(self.base_id)

    def _base_angular_velocity_b(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_id,
            velocity,
            1,
        )
        return velocity[0:3]

    def _canonical_joint_velocity(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.data.qvel[self.model.jnt_dofadr[joint_id]])
                for joint_id in self._canonical_joint_ids
            ],
            dtype=float,
        )

    def joint_acceleration(self) -> List[float]:
        """Finite-difference canonical joint velocity over a control sample.

        The first sample is explicitly zero because no earlier measurement
        exists.  It is not inferred from MuJoCo's instantaneous generalized
        acceleration, which would violate the requested finite-difference
        observation contract.
        """

        timestamp = float(self.data.time)
        velocity = self._canonical_joint_velocity()
        if self._joint_acceleration_time == timestamp:
            return list(self._joint_acceleration_value or [])
        if (
            self._previous_joint_velocity is None
            or self._joint_acceleration_time is None
            or timestamp <= self._joint_acceleration_time
        ):
            acceleration = np.zeros_like(velocity)
        else:
            acceleration = (
                velocity - self._previous_joint_velocity
            ) / (timestamp - self._joint_acceleration_time)
        self._previous_joint_velocity = velocity.copy()
        self._joint_acceleration_time = timestamp
        self._joint_acceleration_value = acceleration.astype(float).tolist()
        return list(self._joint_acceleration_value)

    def joint_position_error(
        self, commanded_targets: Sequence[float]
    ) -> List[float]:
        """Return commanded position target minus measured joint position."""

        targets = np.asarray(commanded_targets, dtype=float)
        if targets.shape != (len(self._canonical_joint_ids),):
            raise ValueError(
                "commanded joint targets must match the canonical joint count"
            )
        measured = np.asarray(
            [
                float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
                for joint_id in self._canonical_joint_ids
            ],
            dtype=float,
        )
        return (targets - measured).astype(float).tolist()

    def base_linear_acceleration_b(self) -> List[float]:
        """Finite-difference world velocity and express acceleration in body."""

        timestamp = float(self.data.time)
        velocity_w = self._base_linear_velocity_w()
        if self._base_acceleration_time == timestamp:
            return list(self._base_acceleration_value or [])
        if (
            self._previous_base_velocity_w is None
            or self._base_acceleration_time is None
            or timestamp <= self._base_acceleration_time
        ):
            acceleration_b = np.zeros(3, dtype=float)
        else:
            acceleration_w = (
                velocity_w - self._previous_base_velocity_w
            ) / (timestamp - self._base_acceleration_time)
            acceleration_b = self._body_to_world.T @ acceleration_w
        self._previous_base_velocity_w = velocity_w.copy()
        self._base_acceleration_time = timestamp
        self._base_acceleration_value = acceleration_b.astype(float).tolist()
        return list(self._base_acceleration_value)

    def base_height(self) -> float:
        """Return base z clearance above terrain directly below the base."""

        base = self._base_position_w
        return float(base[2] - self._terrain_height(base[:2]))

    def foot_position_b(self) -> List[float]:
        """Return each distal foot body's base-relative position in base frame."""

        quaternion = self.data.xquat[self.base_id]
        base_position_w = self._base_position_w
        values: List[float] = []
        for body_id in self._foot_body_ids:
            position_w = np.asarray(self.data.xpos[body_id], dtype=float)
            values.extend(world_to_body(quaternion, position_w - base_position_w))
        return values

    def foot_velocity_b(self) -> List[float]:
        """Return each foot's base-relative velocity in the base frame."""

        rotation_wb = self._body_to_world
        base_velocity_w = self._base_linear_velocity_w()
        values: List[float] = []
        for body_id in self._foot_body_ids:
            velocity_b = rotation_wb.T @ (
                self._body_linear_velocity_w(body_id) - base_velocity_w
            )
            values.extend(float(value) for value in velocity_b)
        return values

    def _external_foot_contact_normals(self) -> Dict[int, List[np.ndarray]]:
        normals: Dict[int, List[np.ndarray]] = {
            body_id: [] for body_id in self._foot_body_ids
        }
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            for foot_id in self._foot_body_ids:
                other_id = body2 if body1 == foot_id else body1 if body2 == foot_id else None
                if other_id is None or other_id in self._robot_body_ids:
                    continue
                normal_w = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
                normals[foot_id].append(_unit_vector(normal_w, "contact normal"))
        return normals

    def foot_slip(self) -> List[float]:
        """Tangential foot speed for terrain contacts, or zero when airborne."""

        contacts = self._external_foot_contact_normals()
        slips: List[float] = []
        for foot_id in self._foot_body_ids:
            normals = contacts[foot_id]
            if not normals:
                slips.append(0.0)
                continue
            velocity_w = self._body_linear_velocity_w(foot_id)
            tangential_speeds = [
                float(np.linalg.norm(velocity_w - normal * np.dot(velocity_w, normal)))
                for normal in normals
            ]
            slips.append(max(tangential_speeds))
        return slips

    def self_collision(self) -> float:
        """Return 1.0 when a distinct robot-body contact pair is active."""

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if (
                body1 != body2
                and body1 in self._robot_body_ids
                and body2 in self._robot_body_ids
            ):
                return 1.0
        return 0.0

    def actuator_saturation(self) -> List[float]:
        """Return whether canonical joint torque has reached its effort limit."""

        result: List[float] = []
        for joint_name, joint_id in zip(
            self.robot_profile.controller_joint_order,
            self._canonical_joint_ids,
        ):
            dof_id = int(self.model.jnt_dofadr[joint_id])
            applied_torque = abs(float(self.data.qfrc_actuator[dof_id]))
            effort_limit = self._effort_limit_by_joint.get(joint_name)
            if effort_limit is None:
                actuator_id = self._actuator_by_joint_id.get(joint_id)
                if actuator_id is not None and self.model.actuator_forcelimited[actuator_id]:
                    force_range = self.model.actuator_forcerange[actuator_id]
                    effort_limit = max(abs(float(value)) for value in force_range)
            result.append(
                1.0
                if effort_limit is not None
                and math.isfinite(effort_limit)
                and applied_torque >= effort_limit
                else 0.0
            )
        return result

    def imu_bias(self) -> List[float]:
        """Return the constant simulated IMU bias.

        This backend currently models no IMU bias, so the declared constant is
        six zeros: accelerometer xyz followed by gyroscope xyz.
        """

        return [0.0] * 6

    @staticmethod
    def _sensor_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = parameters.get("params")
        if nested is None:
            return parameters
        if not isinstance(nested, Mapping):
            raise ValueError("sensor params must be a mapping")
        merged = dict(parameters)
        merged.update(nested)
        return merged

    @staticmethod
    def _finite_value(value: Any, label: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("%s must be finite" % label)
        return result

    def _parameters_with_command(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        values = dict(self.command)
        if parameters is not None:
            values.update(self._sensor_parameters(parameters))
        return values

    def _goal_w(self, parameters: Mapping[str, Any]) -> np.ndarray:
        values = self._parameters_with_command(parameters)
        goal = values.get("goal")
        if goal is None and self.world is not None:
            goal = self.world.goal_pose.position
        if isinstance(goal, (str, bytes)) or not isinstance(goal, Sequence):
            raise ValueError("goal must be supplied by the command or WorldSpec")
        if len(goal) not in (2, 3):
            raise ValueError("goal must contain two or three coordinates")
        result = np.asarray([self._finite_value(value, "goal") for value in goal])
        if len(result) == 2:
            result = np.asarray([result[0], result[1], self._base_position_w[2]])
        return result

    def goal_vector_b(self, parameters: Mapping[str, Any]) -> List[float]:
        """Return the commanded/world goal relative to the base in body frame."""

        delta_w = self._goal_w(parameters) - self._base_position_w
        return world_to_body(self.data.xquat[self.base_id], delta_w)

    def goal_distance(self, parameters: Mapping[str, Any]) -> List[float]:
        """Return Euclidean distance to the commanded/world goal."""

        return [float(np.linalg.norm(self.goal_vector_b(parameters)))]

    def goal_heading(self, parameters: Mapping[str, Any]) -> List[float]:
        """Return signed body-relative planar heading error to the goal."""

        goal_b = self.goal_vector_b(parameters)
        return [math.atan2(goal_b[1], goal_b[0])]

    def _route_waypoints(self, parameters: Mapping[str, Any]) -> Sequence[Any]:
        values = self._parameters_with_command(parameters)
        waypoints = values.get("waypoints")
        if waypoints is not None:
            return waypoints
        if self.world is None or not self.world.routes:
            raise ValueError("waypoints must be supplied by the command or WorldSpec")
        selector = values.get("route", values.get("route_name", 0))
        if isinstance(selector, str):
            matching = [route for route in self.world.routes if route.name == selector]
            if not matching:
                raise ValueError("WorldSpec contains no route named %r" % selector)
            return matching[0].waypoints
        index = int(selector)
        if index < 0 or index >= len(self.world.routes):
            raise ValueError("route index is outside the WorldSpec route list")
        return self.world.routes[index].waypoints

    def waypoints_b(self, parameters: Mapping[str, Any]) -> List[float]:
        """Return the next lookahead route points, flattened in body frame."""

        values = self._parameters_with_command(parameters)
        waypoints = self._route_waypoints(parameters)
        if isinstance(waypoints, (str, bytes)) or not isinstance(waypoints, Sequence):
            raise ValueError("waypoints must be a sequence")
        lookahead = int(values.get("lookahead", len(waypoints)))
        if lookahead <= 0:
            raise ValueError("waypoint lookahead must be positive")
        normalized: List[np.ndarray] = []
        dimensions: Optional[int] = None
        for index, point in enumerate(waypoints):
            if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
                raise ValueError("waypoint %d must be a coordinate sequence" % index)
            if len(point) not in (2, 3):
                raise ValueError("waypoint %d must contain two or three values" % index)
            if dimensions is None:
                dimensions = len(point)
            elif dimensions != len(point):
                raise ValueError("all waypoints must have the same dimension")
            xyz = [self._finite_value(value, "waypoint") for value in point]
            if len(xyz) == 2:
                xyz.append(float(self._base_position_w[2]))
            normalized.append(np.asarray(xyz, dtype=float))
        if not normalized:
            raise ValueError("waypoints must not be empty")

        # The closest route point is the current progress anchor. If it has
        # already been reached, advance to the following point.
        distances = [float(np.linalg.norm(point[:2] - self._base_position_w[:2])) for point in normalized]
        start = int(np.argmin(distances))
        if distances[start] <= 1e-6 and start + 1 < len(normalized):
            start += 1
        selected = normalized[start : start + lookahead]
        if len(selected) < lookahead:
            selected.extend([normalized[-1]] * (lookahead - len(selected)))

        result: List[float] = []
        for point in selected:
            point_b = world_to_body(
                self.data.xquat[self.base_id], point - self._base_position_w
            )
            result.extend(point_b[: int(dimensions or 3)])
        return result

    def _command_scalar(self, parameters: Mapping[str, Any], name: str) -> List[float]:
        values = self._parameters_with_command(parameters)
        if name not in values:
            raise ValueError("command parameter %r is required" % name)
        return [self._finite_value(values[name], name)]

    def heading_command(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._command_scalar(parameters, "heading")

    def body_height_command(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._command_scalar(parameters, "height")

    def pitch_command(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._command_scalar(parameters, "pitch")

    def roll_command(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._command_scalar(parameters, "roll")

    def footswing_height_command(self, parameters: Mapping[str, Any]) -> List[float]:
        values = self._parameters_with_command(parameters)
        if "footswing_height" in values:
            return [self._finite_value(values["footswing_height"], "footswing_height")]
        return self._command_scalar(parameters, "height")

    def stance_width_command(self, parameters: Mapping[str, Any]) -> List[float]:
        values = self._parameters_with_command(parameters)
        if "stance_width" in values:
            return [self._finite_value(values["stance_width"], "stance_width")]
        return self._command_scalar(parameters, "width")

    def gait_command(self, parameters: Mapping[str, Any]) -> List[float]:
        values = self._parameters_with_command(parameters)
        if "frequency" not in values or "duty" not in values:
            raise ValueError("gait command requires frequency and duty")
        result = [
            self._finite_value(values["frequency"], "frequency"),
            self._finite_value(values["duty"], "duty"),
        ]
        offsets = values.get("offsets", values.get("phase_offsets", []))
        if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence):
            raise ValueError("gait offsets must be a sequence")
        result.extend(self._finite_value(value, "gait offset") for value in offsets)
        return result

    def terrain_style_command(self, parameters: Mapping[str, Any]) -> List[float]:
        values = self._parameters_with_command(parameters)
        if "style" not in values or "classes" not in values:
            raise ValueError("terrain style command requires style and classes")
        classes = values["classes"]
        style = values["style"]
        if isinstance(classes, int) and not isinstance(classes, bool):
            count = classes
            index = int(style)
        elif isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)):
            class_values = list(classes)
            count = len(class_values)
            try:
                index = class_values.index(style)
            except ValueError as error:
                raise ValueError("terrain style is absent from classes") from error
        else:
            raise ValueError("terrain style classes must be a count or sequence")
        if count <= 0 or index < 0 or index >= count:
            raise ValueError("terrain style index is outside the declared classes")
        result = [0.0] * count
        result[index] = 1.0
        return result

    def _policy_values(self, name: str, parameters: Mapping[str, Any]) -> List[float]:
        values = self._sensor_parameters(parameters)
        size = int(values.get("size", 0))
        if size <= 0:
            raise ValueError("%s size must be positive" % name)
        stored = self._policy_state[name]
        if stored is None:
            stored = [0.0] * size
            self._policy_state[name] = stored
        if len(stored) != size:
            raise ValueError(
                "%s contains %d values, but size %d was declared"
                % (name, len(stored), size)
            )
        return list(stored)

    def _set_policy_values(self, name: str, values: Sequence[float]) -> None:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("%s must be a sequence" % name)
        self._policy_state[name] = [
            self._finite_value(value, name) for value in values
        ]

    def recurrent_state(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._policy_values("recurrent_state", parameters)

    def set_recurrent_state(self, values: Sequence[float]) -> None:
        self._set_policy_values("recurrent_state", values)

    def estimator_output(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._policy_values("estimator_output", parameters)

    def set_estimator_output(self, values: Sequence[float]) -> None:
        self._set_policy_values("estimator_output", values)

    def adaptation_latent(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._policy_values("adaptation_latent", parameters)

    def set_adaptation_latent(self, values: Sequence[float]) -> None:
        self._set_policy_values("adaptation_latent", values)

    def visual_latent(self, parameters: Mapping[str, Any]) -> List[float]:
        return self._policy_values("visual_latent", parameters)

    def set_visual_latent(self, values: Sequence[float]) -> None:
        self._set_policy_values("visual_latent", values)

    @staticmethod
    def _gait_frequency(parameters: Mapping[str, Any]) -> float:
        frequency = float(
            parameters.get(
                "frequency_hz",
                parameters.get(
                    "gait_frequency_hz", parameters.get("frequency", 1.0)
                ),
            )
        )
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("gait frequency must be positive and finite")
        return frequency

    def gait_clock(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return raw cycle phase using ``frequency_hz`` (default 1 Hz)."""

        values = self._sensor_parameters({} if parameters is None else parameters)
        offset = float(values.get("phase_offset", 0.0))
        if not math.isfinite(offset):
            raise ValueError("gait phase_offset must be finite")
        phase = (float(self.data.time) * self._gait_frequency(values) + offset) % 1.0
        return [phase]

    def gait_phase(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return sine/cosine phase using ``frequency_hz`` (default 1 Hz)."""

        phase = self.gait_clock(parameters)[0]
        angle = 2.0 * math.pi * phase
        return [math.sin(angle), math.cos(angle)]

    def time_remaining(self, timeout_s: float) -> float:
        """Return non-negative episode timeout minus current episode time."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("episode timeout must be positive and finite")
        return max(0.0, timeout - float(self.data.time))

    def _update_odometry(self) -> None:
        timestamp = float(self.data.time)
        if self._odometry_time == timestamp:
            return
        if self._odometry_time is None or timestamp < self._odometry_time:
            self._odometry_position_value[:] = 0.0
            self._odometry_heading_value = 0.0
        else:
            dt_s = timestamp - self._odometry_time
            velocity_b = self._body_to_world.T @ self._base_linear_velocity_w()
            yaw_rate = float(self._base_angular_velocity_b()[2])
            midpoint_heading = self._odometry_heading_value + 0.5 * yaw_rate * dt_s
            cosine, sine = math.cos(midpoint_heading), math.sin(midpoint_heading)
            self._odometry_position_value += dt_s * np.asarray(
                [
                    cosine * velocity_b[0] - sine * velocity_b[1],
                    sine * velocity_b[0] + cosine * velocity_b[1],
                    velocity_b[2],
                ],
                dtype=float,
            )
            self._odometry_heading_value += yaw_rate * dt_s
        self._odometry_time = timestamp

    def odometry_position(self) -> List[float]:
        """Dead-reckon position from body velocity without ground-truth fixes."""

        self._update_odometry()
        return self._odometry_position_value.astype(float).tolist()

    def odometry_heading(self) -> float:
        """Dead-reckon heading by integrating body yaw rate without corrections."""

        self._update_odometry()
        return float(self._odometry_heading_value)

    def magnetometer(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Project a declared unit magnetic direction into the body frame.

        ``reference_w`` (also accepted as ``magnetic_reference_w``) may have
        any non-zero magnitude and is normalized before projection.  The
        default is magnetic north along +world X with a roughly 66-degree
        downward inclination: ``[4/sqrt(97), 0, -9/sqrt(97)]``.
        """

        values = self._sensor_parameters({} if parameters is None else parameters)
        reference = values.get(
            "reference_w",
            values.get(
                "magnetic_reference_w",
                values.get(
                    "world_magnetic_reference",
                    values.get(
                        "world_reference_w",
                        values.get(
                            "magnetic_reference_vector_w",
                            DEFAULT_MAGNETIC_REFERENCE_W,
                        ),
                    ),
                ),
            ),
        )
        reference_w = _unit_vector(reference, "world magnetic reference")
        return world_to_body(self.data.xquat[self.base_id], reference_w)

    def _cast_ray_result(
        self, origin_w: Sequence[float], direction_w: Sequence[float], max_range: float
    ) -> Tuple[float, bool, int]:
        """Cast a ray and return range, hit status, and external geom id."""

        direction = _unit_vector(direction_w, "ray direction")
        origin = np.asarray(origin_w, dtype=float).copy()
        limit = float(max_range)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("max_range must be positive and finite")
        travelled = 0.0
        geom_id = np.asarray([-1], dtype=np.int32)
        for _ in range(self._MAX_SELF_HITS):
            remaining = limit - travelled
            if remaining <= 0.0:
                return limit, False, -1
            distance = float(
                mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    None,
                    1,
                    -1,
                    geom_id,
                )
            )
            if distance < 0.0 or distance > remaining:
                return limit, False, -1
            body_id = int(self.model.geom_bodyid[int(geom_id[0])])
            if body_id not in self._robot_body_ids:
                return min(limit, travelled + distance), True, int(geom_id[0])
            advance = max(0.0, distance) + self._SELF_HIT_EPSILON
            travelled += advance
            origin += direction * advance
        return limit, False, -1

    def _cast_ray_with_hit(
        self, origin_w: Sequence[float], direction_w: Sequence[float], max_range: float
    ) -> Tuple[float, bool]:
        """Cast a ray and report whether its returned distance is a real hit."""

        distance, hit, _ = self._cast_ray_result(
            origin_w, direction_w, max_range
        )
        return distance, hit

    def _cast_ray(
        self, origin_w: Sequence[float], direction_w: Sequence[float], max_range: float
    ) -> float:
        """Cast a ray, returning the range limit when no geometry is hit."""

        return self._cast_ray_with_hit(origin_w, direction_w, max_range)[0]

    def _grid_world_points(self, geometry: Mapping[str, Any]) -> List[np.ndarray]:
        points = height_scan_points(
            geometry["x_range"], geometry["y_range"], geometry["resolution"]
        )
        base = self._base_position_w
        rotation = self._body_to_world
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return [
            np.asarray(
                [
                    base[0] + cosine * x_b - sine * y_b,
                    base[1] + sine * x_b + cosine * y_b,
                ],
                dtype=float,
            )
            for x_b, y_b in points
        ]

    def _terrain_ray_bounds(self) -> Tuple[float, float]:
        center_z = float(self.model.stat.center[2])
        extent = max(1.0, float(self.model.stat.extent))
        top = max(float(self._base_position_w[2]) + extent, center_z + 2.0 * extent)
        bottom = center_z - 2.0 * extent
        return top, bottom

    def _terrain_sample(self, xy_w: Sequence[float]) -> Tuple[float, bool]:
        top, bottom = self._terrain_ray_bounds()
        origin = np.asarray([float(xy_w[0]), float(xy_w[1]), top], dtype=float)
        distance, hit = self._cast_ray_with_hit(
            origin, [0.0, 0.0, -1.0], top - bottom
        )
        return top - distance, hit

    def _terrain_height(self, xy_w: Sequence[float]) -> float:
        return self._terrain_sample(xy_w)[0]

    def _terrain_geom_beneath_base(self) -> int:
        top, bottom = self._terrain_ray_bounds()
        base = self._base_position_w
        _, hit, geom_id = self._cast_ray_result(
            [float(base[0]), float(base[1]), top],
            [0.0, 0.0, -1.0],
            top - bottom,
        )
        if not hit:
            raise ValueError("no terrain geometry exists beneath the base")
        return geom_id

    def friction_coefficient(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Read the active terrain's static coefficient from MuJoCo metadata."""

        geom_id = self._terrain_geom_beneath_base()
        if self.model.nuser_geom < 1:
            raise ValueError("model does not declare static-friction geom metadata")
        return [float(self.model.geom_user[geom_id, 0])]

    def restitution(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Read the active terrain material's restitution from MuJoCo."""

        geom_id = self._terrain_geom_beneath_base()
        if self.model.nuser_geom < 4:
            raise ValueError("model does not declare restitution geom metadata")
        value = float(self.model.geom_user[geom_id, 3])
        if not 0.0 <= value <= 1.0:
            raise ValueError("model restitution metadata is outside [0, 1]")
        return [value]

    def base_velocity_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return exact MuJoCo base linear velocity in the world frame."""

        return self._base_linear_velocity_w().astype(float).tolist()

    def terrain_height_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return exact ray-intersected terrain height beneath the base."""

        height, hit = self._terrain_sample(self._base_position_w[:2])
        if not hit:
            raise ValueError("no terrain geometry exists beneath the base")
        return [height]

    def payload_mass_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return total mass stored in the MuJoCo model."""

        return [float(np.sum(self.model.body_mass))]

    def com_offset_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return modeled robot center-of-mass offset in the base frame."""

        masses = np.asarray(self.model.body_mass, dtype=float)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            raise ValueError("model has no positive body mass")
        com_w = np.sum(
            masses[:, np.newaxis] * np.asarray(self.data.xipos, dtype=float),
            axis=0,
        ) / total_mass
        offset_w = com_w - self._base_position_w
        return world_to_body(self.data.xquat[self.base_id], offset_w)

    def motor_strength_scale(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return configured per-joint strength scaling, unity by default."""

        values = self._sensor_parameters({} if parameters is None else parameters)
        scaling = values.get("scale", values.get("scales"))
        count = len(self._canonical_joint_ids)
        if scaling is None:
            return [1.0] * count
        if isinstance(scaling, (int, float)) and not isinstance(scaling, bool):
            result = [self._finite_value(scaling, "motor strength scale")] * count
        elif isinstance(scaling, Sequence) and not isinstance(scaling, (str, bytes)):
            result = [self._finite_value(value, "motor strength scale") for value in scaling]
        else:
            raise ValueError("motor strength scale must be a scalar or sequence")
        if len(result) != count or any(value < 0.0 for value in result):
            raise ValueError("motor strength scale must be non-negative per joint")
        return result

    def external_wrench_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return the exact applied base force and torque in the body frame."""

        wrench_w = np.asarray(self.data.xfrc_applied[self.base_id], dtype=float)
        return world_to_body(
            self.data.xquat[self.base_id], wrench_w[:3]
        ) + world_to_body(self.data.xquat[self.base_id], wrench_w[3:])

    def terrain_class_truth(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[float]:
        """Return a one-hot class for the material beneath the base."""

        geom_id = self._terrain_geom_beneath_base()
        if self.model.nuser_geom < 3:
            raise ValueError("model does not declare terrain-class geom metadata")
        index_value = float(self.model.geom_user[geom_id, 1])
        count_value = float(self.model.geom_user[geom_id, 2])
        index, count = int(round(index_value)), int(round(count_value))
        if (
            count <= 0
            or index < 0
            or index >= count
            or not math.isclose(index_value, index)
            or not math.isclose(count_value, count)
        ):
            raise ValueError("model terrain-class metadata is invalid")
        result = [0.0] * count
        result[index] = 1.0
        return result

    def gnss_position(self, parameters: Mapping[str, Any]) -> List[float]:
        """Convert local ENU world coordinates about a geodetic datum.

        This uses a spherical-Earth local tangent-plane approximation: world
        +X is east, +Y north, and +Z up. It is accurate for benchmark-scale
        displacements near the datum, not for long-distance geodesy.
        """

        values = self._sensor_parameters(parameters)
        datum = values.get("datum")
        if isinstance(datum, Mapping):
            datum = [datum.get("latitude"), datum.get("longitude"), datum.get("altitude")]
        if isinstance(datum, (str, bytes)) or not isinstance(datum, Sequence) or len(datum) != 3:
            raise ValueError("GNSS datum must contain latitude, longitude, altitude")
        latitude, longitude, altitude = [self._finite_value(value, "GNSS datum") for value in datum]
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError("GNSS datum latitude/longitude is out of range")
        earth_radius_m = 6378137.0
        latitude_rad = math.radians(latitude)
        cosine = math.cos(latitude_rad)
        if abs(cosine) < 1e-12:
            raise ValueError("local east displacement is undefined at a polar datum")
        east, north, up = (float(value) for value in self._base_position_w)
        result_latitude = latitude + math.degrees(north / earth_radius_m)
        result_longitude = longitude + math.degrees(east / (earth_radius_m * cosine))
        result_longitude = (result_longitude + 180.0) % 360.0 - 180.0
        return [result_latitude, result_longitude, altitude + up]

    def _terrain_samples(
        self, geometry: Mapping[str, Any]
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        points_w = self._grid_world_points(geometry)
        sampled = [self._terrain_sample(point) for point in points_w]
        return (
            points_w,
            [height for height, _ in sampled],
            [hit for _, hit in sampled],
        )

    def _grid_ray_range(self, geometry: Mapping[str, Any]) -> float:
        if "max_range" in geometry:
            max_range = float(geometry["max_range"])
            if not math.isfinite(max_range) or max_range <= 0.0:
                raise ValueError("grid max_range must be positive and finite")
            return max_range
        _, bottom = self._terrain_ray_bounds()
        return max(1.0, float(self._base_position_w[2]) - bottom)

    @staticmethod
    def _clip(values: List[float], geometry: Mapping[str, Any]) -> List[float]:
        bounds = geometry.get("clip")
        if bounds is None:
            return values
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("clip must contain exactly two values")
        lower, upper = float(bounds[0]), float(bounds[1])
        if upper < lower:
            raise ValueError("clip must be ordered from low to high")
        return [min(upper, max(lower, value)) for value in values]

    def height_scan(self, geometry: Mapping[str, Any]) -> List[float]:
        base_z = float(self._base_position_w[2])
        values = [
            base_z - self._terrain_height(point)
            for point in self._grid_world_points(geometry)
        ]
        return self._clip(values, geometry)

    def height_scan_per_foot(self, geometry: Mapping[str, Any]) -> List[float]:
        if geometry.get("kind") == "grid":
            offsets = height_scan_points(
                geometry["x_range"], geometry["y_range"], geometry["resolution"]
            )
        else:
            offsets = geometry.get("offsets", [[0.0, 0.0]])
        if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence):
            raise ValueError("per-foot offsets must be a sequence")

        declared_origins = geometry.get("origins_b")
        rotation = self._body_to_world
        base = self._base_position_w
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        base_z = float(base[2])
        values: List[float] = []
        for foot_index, foot_name in enumerate(self.robot_profile.foot_links):
            if declared_origins is not None and foot_index < len(declared_origins):
                foot_xy = (base + rotation @ np.asarray(declared_origins[foot_index], dtype=float))[:2]
            else:
                foot_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name
                )
                if foot_id < 0:
                    raise ValueError("robot foot body %r is absent from the MuJoCo model" % foot_name)
                foot_xy = np.asarray(self.data.xpos[foot_id, :2], dtype=float)
            for offset_index, offset in enumerate(offsets):
                if not isinstance(offset, Sequence) or len(offset) not in (2, 3):
                    raise ValueError(
                        "per-foot offset %d must contain two or three values"
                        % offset_index
                    )
                x_b, y_b = float(offset[0]), float(offset[1])
                sample_xy = [
                    foot_xy[0] + cosine * x_b - sine * y_b,
                    foot_xy[1] + sine * x_b + cosine * y_b,
                ]
                values.append(base_z - self._terrain_height(sample_xy))
        return self._clip(values, geometry)

    def elevation_map(self, geometry: Mapping[str, Any]) -> List[float]:
        return self._terrain_samples(geometry)[1]

    def traversability_map(self, geometry: Mapping[str, Any]) -> List[float]:
        """Score each elevation cell from local slope and surface roughness.

        Scores are in ``[0, 1]``. A locally level, smooth surface approaches
        one; steeper gradients, height residuals, and missing terrain samples
        reduce the score.
        """

        _, heights, hits = self._terrain_samples(geometry)
        resolution = self._finite_value(geometry.get("resolution"), "resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        x_range = geometry.get("x_range")
        y_range = geometry.get("y_range")
        if (
            isinstance(x_range, (str, bytes))
            or not isinstance(x_range, Sequence)
            or len(x_range) != 2
            or isinstance(y_range, (str, bytes))
            or not isinstance(y_range, Sequence)
            or len(y_range) != 2
        ):
            raise ValueError("traversability grid requires x_range and y_range")
        x_count = int(round((float(x_range[1]) - float(x_range[0])) / resolution)) + 1
        y_count = int(round((float(y_range[1]) - float(y_range[0])) / resolution)) + 1
        if x_count * y_count != len(heights):
            raise ValueError("traversability grid shape does not match sampled heights")

        height_grid = np.asarray(heights, dtype=float).reshape(x_count, y_count)
        hit_grid = np.asarray(hits, dtype=bool).reshape(x_count, y_count)
        scores = np.zeros_like(height_grid)
        for x_index in range(x_count):
            for y_index in range(y_count):
                if not hit_grid[x_index, y_index]:
                    continue
                x_low, x_high = max(0, x_index - 1), min(x_count, x_index + 2)
                y_low, y_high = max(0, y_index - 1), min(y_count, y_index + 2)
                local = height_grid[x_low:x_high, y_low:y_high]
                local_hits = hit_grid[x_low:x_high, y_low:y_high]
                valid = local[local_hits]
                local_indices = np.argwhere(local_hits)
                if len(valid) >= 3:
                    design = np.asarray(
                        [
                            [
                                (x_low + int(index[0])) * resolution,
                                (y_low + int(index[1])) * resolution,
                                1.0,
                            ]
                            for index in local_indices
                        ],
                        dtype=float,
                    )
                    slope_x, slope_y, intercept = np.linalg.lstsq(
                        design, valid, rcond=None
                    )[0]
                    residual = valid - design @ np.asarray(
                        [slope_x, slope_y, intercept]
                    )
                    slope = math.hypot(float(slope_x), float(slope_y))
                    roughness = float(np.sqrt(np.mean(residual**2)))
                else:
                    slope = 0.0
                    roughness = 0.0
                if slope < 1e-12:
                    slope = 0.0
                if roughness < 1e-12:
                    roughness = 0.0
                scores[x_index, y_index] = math.exp(
                    -2.0 * slope - 5.0 * roughness
                )
        return np.clip(scores, 0.0, 1.0).reshape(-1).astype(float).tolist()

    def terrain_normal_b(self, geometry: Mapping[str, Any]) -> List[float]:
        points_w, heights, hits = self._terrain_samples(geometry)
        valid_samples = [
            (point, height)
            for point, height, hit in zip(points_w, heights, hits)
            if hit
        ]
        design = np.asarray(
            [[point[0], point[1], 1.0] for point, _ in valid_samples]
        )
        if len(valid_samples) >= 3 and np.linalg.matrix_rank(design) >= 3:
            slope_x, slope_y, _ = np.linalg.lstsq(
                design,
                np.asarray([height for _, height in valid_samples]),
                rcond=None,
            )[0]
            normal_w = _unit_vector([-slope_x, -slope_y, 1.0], "terrain normal")
        else:
            return [0.0, 0.0, 1.0]
        normal_b = _unit_vector(
            self._body_to_world.T @ normal_w, "body-frame terrain normal"
        )
        return [float(value) for value in normal_b]

    def occupancy_grid(self, geometry: Mapping[str, Any]) -> List[float]:
        """Return downward-ray occupancy at each body-aligned grid point."""

        max_range = self._grid_ray_range(geometry)
        origin_z = float(self._base_position_w[2])
        return [
            1.0
            if self._cast_ray_with_hit(
                [float(point[0]), float(point[1]), origin_z],
                [0.0, 0.0, -1.0],
                max_range,
            )[1]
            else 0.0
            for point in self._grid_world_points(geometry)
        ]

    def signed_distance_field(self, geometry: Mapping[str, Any]) -> List[float]:
        """Return truncated signed vertical clearance to ray-visible surfaces."""

        _, heights, hits = self._terrain_samples(geometry)
        base_z = float(self._base_position_w[2])
        no_hit_clearance = self._grid_ray_range(geometry)
        clearances = [
            base_z - height if hit else no_hit_clearance
            for height, hit in zip(heights, hits)
        ]
        return self._clip(clearances, geometry)

    @staticmethod
    def _directions_b(geometry: Mapping[str, Any]) -> List[np.ndarray]:
        count = int(geometry.get("ray_count", geometry.get("point_count", 0)))
        if count <= 0:
            raise ValueError("ray_count or point_count must be positive")
        pattern = geometry.get("angular_pattern", "uniform")
        if isinstance(pattern, str):
            if pattern not in {"uniform", "horizontal", "uniform_horizontal"}:
                raise ValueError("unsupported angular_pattern %r" % pattern)
            return [
                np.asarray(
                    [math.cos(2.0 * math.pi * index / count), math.sin(2.0 * math.pi * index / count), 0.0]
                )
                for index in range(count)
            ]
        if isinstance(pattern, Mapping):
            azimuth = pattern.get("azimuth_range_deg", pattern.get("azimuth_deg", [-180.0, 180.0]))
            elevation = pattern.get("elevation_range_deg", pattern.get("elevation_deg", [0.0, 0.0]))
            azimuth_values = np.linspace(float(azimuth[0]), float(azimuth[-1]), count, endpoint=False)
            elevation_values = np.linspace(float(elevation[0]), float(elevation[-1]), count)
            return [
                np.asarray(
                    [
                        math.cos(math.radians(el)) * math.cos(math.radians(az)),
                        math.cos(math.radians(el)) * math.sin(math.radians(az)),
                        math.sin(math.radians(el)),
                    ]
                )
                for az, el in zip(azimuth_values, elevation_values)
            ]
        if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
            directions = []
            for index, ray in enumerate(pattern):
                if isinstance(ray, Sequence) and len(ray) == 3:
                    directions.append(_unit_vector(ray, "angular_pattern[%d]" % index))
                elif isinstance(ray, Sequence) and len(ray) == 2:
                    azimuth, elevation = math.radians(float(ray[0])), math.radians(float(ray[1]))
                    directions.append(
                        np.asarray(
                            [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)]
                        )
                    )
                else:
                    azimuth = math.radians(float(ray))
                    directions.append(np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0]))
            return directions
        raise ValueError("angular_pattern must be a name, mapping, or ray sequence")

    def _ray_geometry(self, geometry: Mapping[str, Any]) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        if "ray_count" in geometry:
            normalized = validate_lidar_geometry(geometry)
        else:
            normalized = dict(geometry)
            normalized["ray_count"] = int(geometry["point_count"])
            normalized.setdefault("angular_pattern", "uniform")
            normalized.setdefault("max_range", 10.0)
            normalized.setdefault("origin_b", [0.0, 0.0, 0.0])
        origin_b = normalized.get("origin_b", normalized.get("origin"))
        return (
            np.asarray(origin_b, dtype=float),
            float(normalized["max_range"]),
            self._directions_b(normalized),
        )

    def lidar_rays(self, geometry: Mapping[str, Any]) -> List[float]:
        origin_b, max_range, directions_b = self._ray_geometry(geometry)
        rotation = self._body_to_world
        origin_w = self._base_position_w + rotation @ origin_b
        return [
            self._cast_ray(origin_w, rotation @ direction, max_range)
            for direction in directions_b
        ]

    def point_cloud(self, geometry: Mapping[str, Any]) -> List[float]:
        if "resolution" in geometry and "ray_count" not in geometry:
            return self._camera_point_cloud(geometry)
        origin_b, max_range, directions_b = self._ray_geometry(geometry)
        rotation = self._body_to_world
        origin_w = self._base_position_w + rotation @ origin_b
        cloud: List[float] = []
        for direction_b in directions_b:
            distance = self._cast_ray(origin_w, rotation @ direction_b, max_range)
            cloud.extend(float(value) for value in origin_b + direction_b * distance)
        return cloud

    def range_sensor(self, geometry: Mapping[str, Any]) -> List[float]:
        origin_b = np.asarray(geometry.get("origin_b", geometry.get("origin", [0.0, 0.0, 0.0])), dtype=float)
        direction_b = _unit_vector(geometry["direction_b"], "direction_b")
        max_range = float(geometry["max_range"])
        rotation = self._body_to_world
        return [
            self._cast_ray(
                self._base_position_w + rotation @ origin_b,
                rotation @ direction_b,
                max_range,
            )
        ]

    def _renderer(self, width: int, height: int) -> Any:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), width
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), height
            )
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        return renderer

    @staticmethod
    def _vertical_fov(camera: Mapping[str, Any], width: int, height: int) -> float:
        if "vfov_deg" in camera:
            return float(camera["vfov_deg"])
        if "hfov_deg" in camera:
            return math.degrees(
                2.0
                * math.atan(
                    math.tan(math.radians(float(camera["hfov_deg"])) / 2.0)
                    * height
                    / width
                )
            )
        if "fov_deg" in camera:
            return float(camera["fov_deg"])
        return float(camera["field_of_view_deg"])

    def _prepare_camera(self, geometry: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        camera = validate_camera_geometry(geometry)
        width, height = camera["resolution"]
        renderer = self._renderer(width, height)
        extent = float(self.model.stat.extent)
        self.model.vis.map.znear = camera["near"] / extent
        self.model.vis.map.zfar = camera["far"] / extent
        renderer.update_scene(self.data)

        sensor_rotation_b = _rotation_from_wxyz(camera["orientation_wxyz"])
        sensor_rotation_w = self._body_to_world @ sensor_rotation_b
        position_w = self._base_position_w + self._body_to_world @ np.asarray(camera["position_b"])
        forward_w = sensor_rotation_w @ np.asarray([1.0, 0.0, 0.0])
        up_w = sensor_rotation_w @ np.asarray([0.0, 0.0, 1.0])
        vfov = self._vertical_fov(camera, width, height)
        frustum_top = camera["near"] * math.tan(math.radians(vfov) / 2.0)
        for gl_camera in renderer.scene.camera:
            gl_camera.pos[:] = position_w
            gl_camera.forward[:] = forward_w
            gl_camera.up[:] = up_w
            gl_camera.frustum_near = camera["near"]
            gl_camera.frustum_far = camera["far"]
            gl_camera.frustum_bottom = -frustum_top
            gl_camera.frustum_top = frustum_top
            gl_camera.frustum_center = 0.0
            gl_camera.frustum_width = 0.0
        return renderer, camera

    def depth_image(self, geometry: Mapping[str, Any]) -> List[float]:
        old_near = float(self.model.vis.map.znear)
        old_far = float(self.model.vis.map.zfar)
        try:
            renderer, camera = self._prepare_camera(geometry)
            renderer.enable_depth_rendering()
            image = renderer.render()
            return np.clip(image, camera["near"], camera["far"]).reshape(-1).astype(float).tolist()
        finally:
            self.model.vis.map.znear = old_near
            self.model.vis.map.zfar = old_far

    def rgb_image(self, geometry: Mapping[str, Any]) -> List[float]:
        old_near = float(self.model.vis.map.znear)
        old_far = float(self.model.vis.map.zfar)
        try:
            renderer, _ = self._prepare_camera(geometry)
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            image = renderer.render().astype(float) / 255.0
            return image.reshape(-1).tolist()
        finally:
            self.model.vis.map.znear = old_near
            self.model.vis.map.zfar = old_far

    def semantic_segmentation(self, geometry: Mapping[str, Any]) -> List[float]:
        """Render MuJoCo object IDs, one class identifier per image pixel."""

        old_near = float(self.model.vis.map.znear)
        old_far = float(self.model.vis.map.zfar)
        try:
            renderer, _ = self._prepare_camera(geometry)
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            object_ids = segmentation[..., 0]
            return object_ids.reshape(-1).astype(float).tolist()
        finally:
            self.model.vis.map.znear = old_near
            self.model.vis.map.zfar = old_far

    @staticmethod
    def _focal_length_pixels(camera: Mapping[str, Any]) -> float:
        width, height = camera["resolution"]
        if "hfov_deg" in camera:
            return width / (
                2.0 * math.tan(math.radians(float(camera["hfov_deg"])) / 2.0)
            )
        vfov = MujocoSensorSuite._vertical_fov(camera, width, height)
        return height / (2.0 * math.tan(math.radians(vfov) / 2.0))

    def stereo_disparity(self, geometry: Mapping[str, Any]) -> List[float]:
        """Render a rectified stereo pair and derive left-image disparity."""

        camera = validate_camera_geometry(geometry)
        baseline = float(geometry.get("baseline_m", 0.0))
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("stereo camera baseline_m must be positive and finite")

        left_camera = dict(camera)
        right_camera = dict(camera)
        left_camera["position_b"] = list(camera["position_b"])
        right_camera["position_b"] = list(camera["position_b"])
        left_camera["position_b"][1] -= baseline / 2.0
        right_camera["position_b"][1] += baseline / 2.0
        left_depth = np.asarray(self.depth_image(left_camera), dtype=float)
        # Rendering both views makes the declared stereo rig observable even
        # though rectified disparity is computed from the left depth image.
        self.depth_image(right_camera)
        disparity = baseline * self._focal_length_pixels(camera) / left_depth
        return disparity.astype(float).tolist()

    @staticmethod
    def _camera_cache_key(camera: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            tuple(camera["resolution"]),
            tuple(
                (name, float(camera[name]))
                for name in (
                    "hfov_deg",
                    "vfov_deg",
                    "fov_deg",
                    "field_of_view_deg",
                )
                if name in camera
            ),
            float(camera["near"]),
            float(camera["far"]),
            tuple(camera["position_b"]),
            tuple(camera["orientation_wxyz"]),
            camera["encoding"],
        )

    @staticmethod
    def _dense_lucas_kanade(
        previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        previous_gray = previous.mean(axis=2)
        current_gray = current.mean(axis=2)
        average = 0.5 * (previous_gray + current_gray)
        padded = np.pad(average, 1, mode="edge")
        gradient_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
        gradient_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
        gradient_t = current_gray - previous_gray

        def window_sum(values: np.ndarray) -> np.ndarray:
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(values, 2, mode="edge"), (5, 5)
            )
            return windows.sum(axis=(-2, -1))

        xx = window_sum(gradient_x * gradient_x)
        yy = window_sum(gradient_y * gradient_y)
        xy = window_sum(gradient_x * gradient_y)
        xt = window_sum(gradient_x * gradient_t)
        yt = window_sum(gradient_y * gradient_t)
        determinant = xx * yy - xy * xy
        flow = np.zeros(previous_gray.shape + (2,), dtype=float)
        valid = determinant > 1e-12
        flow[..., 0][valid] = (
            xy[valid] * yt[valid] - yy[valid] * xt[valid]
        ) / determinant[valid]
        flow[..., 1][valid] = (
            xy[valid] * xt[valid] - xx[valid] * yt[valid]
        ) / determinant[valid]
        return flow

    def optical_flow(self, geometry: Mapping[str, Any]) -> List[float]:
        """Estimate dense pixel motion from the prior and current RGB frame."""

        camera = validate_camera_geometry(geometry)
        width, height = camera["resolution"]
        current = np.asarray(self.rgb_image(camera), dtype=float).reshape(
            height, width, 3
        )
        key = self._camera_cache_key(camera)
        previous = self._previous_rgb_frames.get(key)
        self._previous_rgb_frames[key] = current.copy()
        if previous is None:
            return [0.0] * (width * height * 2)
        return self._dense_lucas_kanade(previous, current).reshape(-1).tolist()

    def _camera_point_cloud(self, geometry: Mapping[str, Any]) -> List[float]:
        camera = validate_camera_geometry(geometry)
        width, height = camera["resolution"]
        depths = np.asarray(self.depth_image(camera), dtype=float).reshape(height, width)
        hfov = camera.get("hfov_deg")
        if hfov is None:
            vfov = self._vertical_fov(camera, width, height)
            fy = height / (2.0 * math.tan(math.radians(vfov) / 2.0))
            fx = fy
        else:
            fx = width / (2.0 * math.tan(math.radians(float(hfov)) / 2.0))
            fy = fx
        rotation_b = _rotation_from_wxyz(camera["orientation_wxyz"])
        position_b = np.asarray(camera["position_b"], dtype=float)
        cloud: List[float] = []
        for row in range(height):
            for column in range(width):
                depth = depths[row, column]
                point_camera = np.asarray(
                    [depth, -(column + 0.5 - width / 2.0) * depth / fx, -(row + 0.5 - height / 2.0) * depth / fy]
                )
                cloud.extend(float(value) for value in position_b + rotation_b @ point_camera)
        return cloud

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._previous_rgb_frames.clear()


__all__ = ["MujocoSensorSuite"]
