"""Versioned, backend-independent policy observation state.

Quaternion values in this module always use the ``wxyz`` component order.
Rotations represented by a quaternion map vectors from the body frame into
the world frame.

Availability classes describe how a robot obtains each observation source:

``onboard``
    Direct sensor readings: encoders, IMU, cameras, lidar, contact/force
    sensors, motor current, and joint temperature.
``onboard_derived``
    Values computed on-robot from onboard data or the policy's own memory:
    projected gravity, forward kinematics, elevation/traversability maps,
    height scans built from depth, SLAM odometry, learned state estimators,
    adaptation latents, gait clocks, previous actions, and recurrent state.
``command``
    Values supplied by an operator or high-level planner.
``infrastructure``
    Values needing off-robot systems such as motion capture or an RTK base.
    They are real, but not field-autonomous.
``simulator_only``
    Physics ground truth that no robot can measure.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence


SCHEMA = "canonical_state/1"

AVAILABILITY_CLASSES = frozenset(
    {
        "onboard",
        "onboard_derived",
        "command",
        "infrastructure",
        "simulator_only",
    }
)


# A mapping is used instead of a set so schema order and the unit/frame
# contract are available to observation compilers and manifest validators.
CANONICAL_FIELDS: Mapping[str, Mapping[str, Any]] = OrderedDict(
    (
        ("joint_position", {"unit": "rad", "frame": "canonical joint order", "availability": "onboard", "natural_size": "joint_count"}),
        ("joint_velocity", {"unit": "rad/s", "frame": "canonical joint order", "availability": "onboard", "natural_size": "joint_count"}),
        ("joint_torque", {"unit": "N.m", "frame": "canonical joint order", "availability": "onboard", "natural_size": "joint_count"}),
        ("joint_acceleration", {"unit": "rad/s^2", "frame": "canonical joint order", "availability": "onboard_derived", "natural_size": "joint_count"}),
        ("base_position_w", {"unit": "m", "frame": "world", "availability": "infrastructure", "natural_size": 3}),
        ("base_orientation_wxyz", {"unit": "unit quaternion", "frame": "body to world, wxyz", "availability": "onboard", "natural_size": 4}),
        ("base_euler_rpy", {"unit": "rad", "frame": "world, roll-pitch-yaw", "availability": "onboard_derived", "natural_size": 3}),
        ("projected_gravity_b", {"unit": "unit vector", "frame": "body", "availability": "onboard_derived", "natural_size": 3}),
        ("base_linear_velocity_b", {"unit": "m/s", "frame": "body", "availability": "onboard_derived", "natural_size": 3}),
        ("base_angular_velocity_b", {"unit": "rad/s", "frame": "body", "availability": "onboard", "natural_size": 3}),
        ("base_linear_acceleration_b", {"unit": "m/s^2", "frame": "body", "availability": "onboard", "natural_size": 3}),
        ("base_height", {"unit": "m", "frame": "above local terrain", "availability": "onboard_derived", "natural_size": 1}),
        ("height_scan", {"unit": "m", "frame": "body scan pattern", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("height_scan_per_foot", {"unit": "m", "frame": "body scan pattern per foot", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("elevation_map", {"unit": "m", "frame": "declared map frame", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("traversability_map", {"unit": "unitless", "frame": "declared map frame", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("depth_image", {"unit": "declared encoding", "frame": "camera", "availability": "onboard", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("rgb_image", {"unit": "declared encoding", "frame": "camera", "availability": "onboard", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("lidar_rays", {"unit": "m", "frame": "body ray pattern", "availability": "onboard", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("point_cloud", {"unit": "m", "frame": "declared sensor frame", "availability": "onboard", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("semantic_segmentation", {"unit": "class id", "frame": "camera image plane", "availability": "onboard_derived", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("optical_flow", {"unit": "pixel displacement", "frame": "camera image plane", "availability": "onboard_derived", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("thermal_image", {"unit": "declared encoding", "frame": "thermal camera", "availability": "onboard", "natural_size": "geometry", "multi_instance": True}),
        ("event_camera", {"unit": "events", "frame": "camera image plane", "availability": "onboard", "natural_size": "declared", "multi_instance": True}),
        ("stereo_disparity", {"unit": "pixel disparity", "frame": "stereo camera image plane", "availability": "onboard_derived", "natural_size": "geometry", "multi_instance": True, "perception": True}),
        ("occupancy_grid", {"unit": "occupancy probability", "frame": "declared map frame", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("signed_distance_field", {"unit": "m", "frame": "declared map frame", "availability": "onboard_derived", "natural_size": "geometry", "perception": True}),
        ("range_sensor", {"unit": "m", "frame": "declared sensor frame", "availability": "onboard", "natural_size": 1, "multi_instance": True}),
        ("gnss_position", {"unit": "degrees, degrees, m", "frame": "latitude/longitude/altitude about a declared local tangent-plane datum", "availability": "onboard", "natural_size": 3}),
        ("magnetometer", {"unit": "unit vector", "frame": "body", "availability": "onboard", "natural_size": 3}),
        ("battery_state", {"unit": "declared encoding", "frame": "electrical system", "availability": "onboard", "natural_size": "declared"}),
        ("foot_slip", {"unit": "m/s", "frame": "body, canonical foot order", "availability": "onboard_derived", "natural_size": "foot_count"}),
        ("actuator_saturation", {"unit": "bool", "frame": "canonical joint order", "availability": "onboard_derived", "natural_size": "joint_count"}),
        ("self_collision", {"unit": "bool", "frame": "canonical body link pairs", "availability": "onboard_derived", "natural_size": 1}),
        ("imu_bias", {"unit": "m/s^2, rad/s", "frame": "body", "availability": "onboard_derived", "natural_size": 6}),
        ("previous_action", {"unit": "raw network output", "frame": "canonical joint order, pre-scaling", "availability": "onboard_derived", "natural_size": "joint_count"}),
        ("foot_contact", {"unit": "bool", "frame": "canonical foot order", "availability": "onboard", "natural_size": "foot_count"}),
        ("foot_contact_force", {"unit": "N", "frame": "body, canonical foot order", "availability": "onboard", "natural_size": "foot_count_x3"}),
        ("foot_position_b", {"unit": "m", "frame": "body, canonical foot order", "availability": "onboard_derived", "natural_size": "foot_count_x3"}),
        ("foot_velocity_b", {"unit": "m/s", "frame": "body, canonical foot order", "availability": "onboard_derived", "natural_size": "foot_count_x3"}),
        ("episode_time", {"unit": "s", "frame": "episode clock", "availability": "onboard_derived", "natural_size": 1}),
        ("velocity_command", {"unit": "m/s, m/s, rad/s", "frame": "body x, body y, body yaw", "availability": "command", "natural_size": 3}),
        ("heading_command", {"unit": "rad", "frame": "world heading", "availability": "command", "natural_size": 1}),
        ("goal_vector_b", {"unit": "m", "frame": "body", "availability": "command", "natural_size": 3}),
        ("goal_distance", {"unit": "m", "frame": "frame-invariant scalar", "availability": "command", "natural_size": 1}),
        ("goal_heading", {"unit": "rad", "frame": "body-relative heading", "availability": "command", "natural_size": 1}),
        ("waypoints_b", {"unit": "m", "frame": "body, N x 2 or N x 3", "availability": "command", "natural_size": "declared"}),
        ("body_height_command", {"unit": "m", "frame": "body height above local terrain", "availability": "command", "natural_size": 1}),
        ("joint_temperature", {"unit": "degC", "frame": "canonical joint order", "availability": "onboard", "natural_size": "joint_count"}),
        ("motor_current", {"unit": "A", "frame": "canonical joint order", "availability": "onboard", "natural_size": "joint_count"}),
        ("joint_position_error", {"unit": "rad", "frame": "canonical joint order", "availability": "onboard_derived", "natural_size": "joint_count"}),
        ("time_remaining", {"unit": "s", "frame": "episode clock", "availability": "onboard_derived", "natural_size": 1}),
        ("gait_phase", {"unit": "unitless", "frame": "gait cycle sine/cosine", "availability": "onboard_derived", "natural_size": 2}),
        ("gait_clock", {"unit": "cycle fraction", "frame": "gait cycle [0, 1)", "availability": "onboard_derived", "natural_size": 1}),
        ("recurrent_state", {"unit": "unitless", "frame": "policy internal", "availability": "onboard_derived", "natural_size": "declared"}),
        ("estimator_output", {"unit": "declared encoding", "frame": "declared estimator frame", "availability": "onboard_derived", "natural_size": "declared"}),
        ("adaptation_latent", {"unit": "unitless", "frame": "policy internal", "availability": "onboard_derived", "natural_size": "declared"}),
        ("visual_latent", {"unit": "unitless", "frame": "policy internal", "availability": "onboard_derived", "natural_size": "declared", "perception": True}),
        ("terrain_normal_b", {"unit": "unit vector", "frame": "body", "availability": "onboard_derived", "natural_size": 3, "perception": True}),
        ("odometry_position", {"unit": "m", "frame": "odometry", "availability": "onboard_derived", "natural_size": 3}),
        ("odometry_heading", {"unit": "rad", "frame": "odometry heading", "availability": "onboard_derived", "natural_size": 1}),
        ("pitch_command", {"unit": "rad", "frame": "body pitch", "availability": "command", "natural_size": 1}),
        ("roll_command", {"unit": "rad", "frame": "body roll", "availability": "command", "natural_size": 1}),
        ("gait_command", {"unit": "declared encoding", "frame": "gait selector", "availability": "command", "natural_size": "declared"}),
        ("footswing_height_command", {"unit": "m", "frame": "canonical foot order", "availability": "command", "natural_size": 1}),
        ("stance_width_command", {"unit": "m", "frame": "body lateral", "availability": "command", "natural_size": 1}),
        ("terrain_style_command", {"unit": "declared encoding", "frame": "terrain-style selector", "availability": "command", "natural_size": "declared"}),
        ("friction_coefficient", {"unit": "unitless", "frame": "contact pair", "availability": "simulator_only", "natural_size": 1}),
        ("restitution", {"unit": "unitless", "frame": "contact pair", "availability": "simulator_only", "natural_size": 1}),
        ("terrain_class_truth", {"unit": "declared encoding", "frame": "world terrain", "availability": "simulator_only", "natural_size": "declared"}),
        ("terrain_height_truth", {"unit": "m", "frame": "world terrain", "availability": "simulator_only", "natural_size": 1}),
        ("payload_mass_truth", {"unit": "kg", "frame": "body", "availability": "simulator_only", "natural_size": 1}),
        ("com_offset_truth", {"unit": "m", "frame": "body", "availability": "simulator_only", "natural_size": 3}),
        ("motor_strength_scale", {"unit": "unitless", "frame": "canonical joint order", "availability": "simulator_only", "natural_size": "joint_count"}),
        ("external_wrench_truth", {"unit": "N, N.m", "frame": "body", "availability": "simulator_only", "natural_size": 6}),
        ("base_velocity_truth", {"unit": "m/s", "frame": "world", "availability": "simulator_only", "natural_size": 3}),
    )
)


@dataclass
class CanonicalState(Mapping):
    """Canonical policy state for schema ``canonical_state/1``.

    The canonical joint order is ``RobotProfile.controller_joint_order`` and
    its default pose is ``RobotProfile.default_joint_positions``. The fields
    have the following unit and frame contract:

    ``joint_position``: rad, canonical joint order.
    ``joint_velocity``: rad/s, canonical joint order.
    ``joint_torque``: N.m, canonical joint order.
    ``joint_acceleration``: rad/s^2, canonical joint order.
    ``base_position_w``: m, world frame.
    ``base_orientation_wxyz``: unit body-to-world quaternion, w first.
    ``base_euler_rpy``: rad, world-frame roll, pitch, yaw (derived).
    ``projected_gravity_b``: unit gravity-direction vector, body frame (derived).
    ``base_linear_velocity_b``: m/s, body frame.
    ``base_angular_velocity_b``: rad/s, body frame.
    ``base_linear_acceleration_b``: m/s^2, body frame.
    ``base_height``: m above local terrain.
    ``height_scan``: terrain-clearance samples in m, body-frame scan pattern.
    ``height_scan_per_foot``: terrain-clearance samples in m per foot.
    ``elevation_map``: terrain elevations in m in the declared map frame.
    ``traversability_map``: unitless cells in the declared map frame.
    ``depth_image``: camera depth values in the declared encoding.
    ``rgb_image``: camera colour values in the declared encoding.
    ``lidar_rays``: ray ranges in m in the declared body-frame pattern.
    ``point_cloud``: xyz points in m in the declared sensor frame.
    ``previous_action``: raw network output in canonical joint order, pre-scaling.
    ``foot_contact``: bool per foot, canonical foot order.
    ``foot_contact_force``: N, body frame, canonical foot order.
    ``foot_position_b``: m, body frame, canonical foot order.
    ``foot_velocity_b``: m/s, body frame, canonical foot order.
    ``foot_slip``: tangential speed in m/s per foot, zero while airborne.
    ``actuator_saturation``: 1.0 per joint at its effort limit, else 0.0.
    ``self_collision``: 1.0 when any robot-robot contact pair is active.
    ``imu_bias``: accelerometer xyz then gyroscope xyz constant bias.
    ``episode_time``: s on the episode clock.
    ``time_remaining``: non-negative seconds to the episode timeout.
    ``gait_phase``: sine and cosine of the configured gait phase.
    ``gait_clock``: raw configured gait cycle fraction in [0, 1).
    ``odometry_position``: dead-reckoned position in the odometry frame.
    ``odometry_heading``: dead-reckoned yaw in rad.
    ``magnetometer``: unit magnetic-field direction in the body frame.
    ``velocity_command``: body x/y velocity in m/s and body yaw rate in rad/s.
    ``heading_command``: rad, world-frame heading.
    ``goal_vector_b``: m, body frame.
    ``goal_distance``: m, frame-invariant scalar.
    ``goal_heading``: rad, body-relative heading.
    ``waypoints_b``: N x 2 or N x 3 points in m, body frame.
    ``body_height_command``: m above local terrain.

    Unsupported or unavailable fields are ``None``. They must not be replaced
    by zero-filled arrays because zero is a valid sensor reading.
    """

    schema: ClassVar[str] = SCHEMA

    joint_position: Optional[List[float]] = None
    joint_velocity: Optional[List[float]] = None
    joint_torque: Optional[List[float]] = None
    joint_acceleration: Optional[List[float]] = None
    base_position_w: Optional[List[float]] = None
    base_orientation_wxyz: Optional[List[float]] = None
    base_euler_rpy: Optional[List[float]] = None
    projected_gravity_b: Optional[List[float]] = None
    base_linear_velocity_b: Optional[List[float]] = None
    base_angular_velocity_b: Optional[List[float]] = None
    base_linear_acceleration_b: Optional[List[float]] = None
    base_height: Optional[float] = None
    height_scan: Optional[List[float]] = None
    previous_action: Optional[List[float]] = None
    foot_contact: Optional[List[bool]] = None
    foot_contact_force: Optional[List[List[float]]] = None
    foot_position_b: Optional[List[List[float]]] = None
    foot_velocity_b: Optional[List[float]] = None
    episode_time: Optional[float] = None
    velocity_command: Optional[List[float]] = None
    heading_command: Optional[float] = None
    goal_vector_b: Optional[List[float]] = None
    goal_distance: Optional[float] = None
    goal_heading: Optional[float] = None
    waypoints_b: Optional[List[List[float]]] = None
    body_height_command: Optional[float] = None
    height_scan_per_foot: Optional[List[List[float]]] = None
    elevation_map: Optional[List[List[float]]] = None
    traversability_map: Optional[List[List[float]]] = None
    depth_image: Optional[List[List[float]]] = None
    rgb_image: Optional[List[List[List[float]]]] = None
    lidar_rays: Optional[List[float]] = None
    point_cloud: Optional[List[List[float]]] = None
    semantic_segmentation: Optional[List[List[int]]] = None
    optical_flow: Optional[List[List[List[float]]]] = None
    thermal_image: Optional[List[List[float]]] = None
    event_camera: Optional[List[List[float]]] = None
    stereo_disparity: Optional[List[List[float]]] = None
    occupancy_grid: Optional[List[List[float]]] = None
    signed_distance_field: Optional[List[List[float]]] = None
    range_sensor: Optional[List[float]] = None
    gnss_position: Optional[List[float]] = None
    magnetometer: Optional[List[float]] = None
    battery_state: Optional[List[float]] = None
    foot_slip: Optional[List[float]] = None
    actuator_saturation: Optional[List[float]] = None
    self_collision: Optional[float] = None
    imu_bias: Optional[List[float]] = None
    joint_temperature: Optional[List[float]] = None
    motor_current: Optional[List[float]] = None
    joint_position_error: Optional[List[float]] = None
    time_remaining: Optional[float] = None
    gait_phase: Optional[List[float]] = None
    gait_clock: Optional[List[float]] = None
    recurrent_state: Optional[List[float]] = None
    estimator_output: Optional[List[float]] = None
    adaptation_latent: Optional[List[float]] = None
    visual_latent: Optional[List[float]] = None
    terrain_normal_b: Optional[List[float]] = None
    odometry_position: Optional[List[float]] = None
    odometry_heading: Optional[float] = None
    pitch_command: Optional[float] = None
    roll_command: Optional[float] = None
    gait_command: Optional[List[float]] = None
    footswing_height_command: Optional[float] = None
    stance_width_command: Optional[float] = None
    terrain_style_command: Optional[List[float]] = None
    friction_coefficient: Optional[float] = None
    restitution: Optional[float] = None
    terrain_class_truth: Optional[List[float]] = None
    terrain_height_truth: Optional[List[float]] = None
    payload_mass_truth: Optional[float] = None
    com_offset_truth: Optional[List[float]] = None
    motor_strength_scale: Optional[List[float]] = None
    external_wrench_truth: Optional[List[float]] = None
    base_velocity_truth: Optional[List[float]] = None

    def __getitem__(self, name: str) -> Any:
        if name not in CANONICAL_FIELDS:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(CANONICAL_FIELDS)

    def __len__(self) -> int:
        return len(CANONICAL_FIELDS)


def _normalized_quaternion(wxyz: Sequence[float]) -> List[float]:
    if len(wxyz) != 4:
        raise ValueError("a wxyz quaternion must contain exactly four values")
    values = [float(value) for value in wxyz]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("a quaternion must have a finite, non-zero norm")
    return [value / norm for value in values]


def world_to_body(wxyz: Sequence[float], vector_w: Sequence[float]) -> List[float]:
    """Rotate a world-frame vector into the body frame using a wxyz quaternion."""

    if len(vector_w) != 3:
        raise ValueError("a vector must contain exactly three values")
    w, x, y, z = _normalized_quaternion(wxyz)
    vx, vy, vz = (float(value) for value in vector_w)

    # R_wb maps body vectors into world. Multiplication by R_wb.T maps the
    # supplied world vector into body without constructing a matrix object.
    return [
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y + w * z) * vy
        + 2.0 * (x * z - w * y) * vz,
        2.0 * (x * y - w * z) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z + w * x) * vz,
        2.0 * (x * z + w * y) * vx
        + 2.0 * (y * z - w * x) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    ]


def projected_gravity_from_quaternion(wxyz: Sequence[float]) -> List[float]:
    """Return world gravity direction ``[0, 0, -1]`` in the body frame."""

    return world_to_body(wxyz, [0.0, 0.0, -1.0])


def quaternion_to_euler_rpy(wxyz: Sequence[float]) -> List[float]:
    """Convert a normalized-or-unnormalized wxyz quaternion to roll/pitch/yaw."""

    w, x, y, z = _normalized_quaternion(wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def _float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _optional_float_list(
    values: Optional[Sequence[float]],
) -> Optional[List[float]]:
    return None if values is None else _float_list(values)


def _optional_float_rows(
    values: Optional[Sequence[Sequence[float]]],
) -> Optional[List[List[float]]]:
    return None if values is None else [_float_list(row) for row in values]


def _optional_flat_float_values(
    values: Optional[Sequence[Any]],
) -> Optional[List[float]]:
    if values is None:
        return None
    flattened: List[float] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened.extend(float(item) for item in value)
        else:
            flattened.append(float(value))
    return flattened


def build_canonical_state(
    joint_order: Sequence[str],
    joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    base_position_w: Sequence[float],
    base_orientation_wxyz: Sequence[float],
    base_linear_velocity_w: Sequence[float],
    base_angular_velocity_w: Sequence[float],
    velocity_command: Sequence[float],
    previous_action: Sequence[float],
    episode_time: float,
    joint_torque: Optional[Sequence[float]] = None,
    foot_contact: Optional[Sequence[bool]] = None,
    foot_contact_force: Optional[Sequence[Sequence[float]]] = None,
    foot_position_b: Optional[Sequence[Sequence[float]]] = None,
    foot_velocity_b: Optional[Sequence[Any]] = None,
    base_height: Optional[float] = None,
    joint_acceleration: Optional[Sequence[float]] = None,
    base_linear_acceleration_b: Optional[Sequence[float]] = None,
    height_scan: Optional[Sequence[float]] = None,
    heading_command: Optional[float] = None,
    goal_vector_b: Optional[Sequence[float]] = None,
    goal_distance: Optional[float] = None,
    goal_heading: Optional[float] = None,
    waypoints_b: Optional[Sequence[Sequence[float]]] = None,
    body_height_command: Optional[float] = None,
    joint_position_error: Optional[Sequence[float]] = None,
    foot_slip: Optional[Sequence[float]] = None,
    actuator_saturation: Optional[Sequence[float]] = None,
    self_collision: Optional[float] = None,
    imu_bias: Optional[Sequence[float]] = None,
    time_remaining: Optional[float] = None,
    gait_phase: Optional[Sequence[float]] = None,
    gait_clock: Optional[Sequence[float]] = None,
    odometry_position: Optional[Sequence[float]] = None,
    odometry_heading: Optional[float] = None,
    magnetometer: Optional[Sequence[float]] = None,
) -> CanonicalState:
    """Build canonical policy state from simulator world-frame readings.

    Orientation-derived fields are computed here so every backend uses the
    same quaternion convention and rotation helpers. Unsupported optional
    readings remain ``None``.
    """

    joint_count = len(joint_order)
    joint_values = {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "previous_action": previous_action,
    }
    if joint_torque is not None:
        joint_values["joint_torque"] = joint_torque
    if joint_acceleration is not None:
        joint_values["joint_acceleration"] = joint_acceleration
    if joint_position_error is not None:
        joint_values["joint_position_error"] = joint_position_error
    if actuator_saturation is not None:
        joint_values["actuator_saturation"] = actuator_saturation
    for name, values in joint_values.items():
        if len(values) != joint_count:
            raise ValueError(
                "%s has length %d, but joint_order has length %d"
                % (name, len(values), joint_count)
            )

    quaternion = _float_list(base_orientation_wxyz)
    return CanonicalState(
        joint_position=_float_list(joint_position),
        joint_velocity=_float_list(joint_velocity),
        joint_torque=_optional_float_list(joint_torque),
        joint_acceleration=_optional_float_list(joint_acceleration),
        base_position_w=_float_list(base_position_w),
        base_orientation_wxyz=quaternion,
        base_euler_rpy=quaternion_to_euler_rpy(quaternion),
        projected_gravity_b=projected_gravity_from_quaternion(quaternion),
        base_linear_velocity_b=world_to_body(
            quaternion, base_linear_velocity_w
        ),
        base_angular_velocity_b=world_to_body(
            quaternion, base_angular_velocity_w
        ),
        base_linear_acceleration_b=_optional_float_list(
            base_linear_acceleration_b
        ),
        base_height=None if base_height is None else float(base_height),
        height_scan=_optional_float_list(height_scan),
        previous_action=_float_list(previous_action),
        foot_contact=(
            None if foot_contact is None else [bool(value) for value in foot_contact]
        ),
        foot_contact_force=_optional_float_rows(foot_contact_force),
        foot_position_b=_optional_float_rows(foot_position_b),
        foot_velocity_b=_optional_flat_float_values(foot_velocity_b),
        foot_slip=_optional_float_list(foot_slip),
        actuator_saturation=_optional_float_list(actuator_saturation),
        self_collision=(
            None if self_collision is None else float(self_collision)
        ),
        imu_bias=_optional_float_list(imu_bias),
        episode_time=float(episode_time),
        time_remaining=(
            None if time_remaining is None else float(time_remaining)
        ),
        gait_phase=_optional_float_list(gait_phase),
        gait_clock=_optional_float_list(gait_clock),
        odometry_position=_optional_float_list(odometry_position),
        odometry_heading=(
            None if odometry_heading is None else float(odometry_heading)
        ),
        magnetometer=_optional_float_list(magnetometer),
        joint_position_error=_optional_float_list(joint_position_error),
        velocity_command=_float_list(velocity_command),
        heading_command=(
            None if heading_command is None else float(heading_command)
        ),
        goal_vector_b=_optional_float_list(goal_vector_b),
        goal_distance=None if goal_distance is None else float(goal_distance),
        goal_heading=None if goal_heading is None else float(goal_heading),
        waypoints_b=_optional_float_rows(waypoints_b),
        body_height_command=(
            None
            if body_height_command is None
            else float(body_height_command)
        ),
    )


__all__ = [
    "SCHEMA",
    "AVAILABILITY_CLASSES",
    "CANONICAL_FIELDS",
    "CanonicalState",
    "build_canonical_state",
    "projected_gravity_from_quaternion",
    "quaternion_to_euler_rpy",
    "world_to_body",
]
