"""Canonical, JSON-serializable public data contracts.

The contracts intentionally use stdlib dataclasses so the same records can be
loaded by Isaac Gym's Python 3.8 environment and newer Isaac Lab/MuJoCo
environments without installing a second validation framework.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "1.0"


class ValidationError(ValueError):
    """Raised when a public model violates a schema invariant."""


def _vec3(value: List[float], name: str) -> List[float]:
    if len(value) != 3:
        raise ValidationError("%s must contain exactly three values" % name)
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValidationError("%s must contain finite values" % name)
    return result


@dataclass
class Pose:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def validate(self) -> None:
        self.position = _vec3(self.position, "position")
        self.rpy = _vec3(self.rpy, "rpy")


@dataclass
class JointSpec:
    name: str
    parent: str = ""
    child: str = ""
    joint_type: str = "fixed"
    axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    origin_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None


@dataclass
class ActuatorSpec:
    name: str
    joint: str
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    control_lower: Optional[float] = None
    control_upper: Optional[float] = None


@dataclass
class RobotProfile:
    name: str
    source_path: str
    source_format: str
    source_sha256: str
    schema_version: str = SCHEMA_VERSION
    provenance: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    joints: List[JointSpec] = field(default_factory=list)
    actuators: List[ActuatorSpec] = field(default_factory=list)
    link_inertias: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    base_link: str = ""
    foot_links: List[str] = field(default_factory=list)
    total_mass_kg: float = 0.0
    center_of_mass_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    body_size_m: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.2])
    collision_bounds_m: Dict[str, List[float]] = field(
        default_factory=lambda: {"minimum": [-0.25, -0.1, -0.1], "maximum": [0.25, 0.1, 0.1]}
    )
    standing_height_m: float = 0.4
    stance_length_m: float = 0.4
    stance_width_m: float = 0.25
    foot_size_m: List[float] = field(default_factory=lambda: [0.05, 0.04, 0.03])
    limb_reach_m: float = 0.35
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    default_joint_positions: Dict[str, float] = field(default_factory=dict)
    controller_joint_order: List[str] = field(default_factory=list)
    control_frequency_hz: float = 50.0
    inference_confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def body_length_m(self) -> float:
        return float(self.body_size_m[0])

    @property
    def body_width_m(self) -> float:
        return float(self.body_size_m[1])

    def validate(self) -> None:
        if self.source_format not in {"urdf", "mjcf", "usd", "yaml"}:
            raise ValidationError("unsupported robot format: %s" % self.source_format)
        self.center_of_mass_m = _vec3(self.center_of_mass_m, "center_of_mass_m")
        self.body_size_m = _vec3(self.body_size_m, "body_size_m")
        minimum = _vec3(self.collision_bounds_m.get("minimum", []), "collision_bounds_m.minimum")
        maximum = _vec3(self.collision_bounds_m.get("maximum", []), "collision_bounds_m.maximum")
        if any(lo > hi for lo, hi in zip(minimum, maximum)):
            raise ValidationError("collision bounds are inverted")
        self.collision_bounds_m = {"minimum": minimum, "maximum": maximum}
        self.foot_size_m = _vec3(self.foot_size_m, "foot_size_m")
        if any(value <= 0 for value in self.body_size_m):
            raise ValidationError("body dimensions must be positive")
        if self.standing_height_m <= 0 or self.limb_reach_m <= 0:
            raise ValidationError("standing height and limb reach must be positive")
        normalized_limits = {}
        for name, bounds in self.joint_limits.items():
            if len(bounds) != 2:
                raise ValidationError("joint limit for %s must contain exactly two values" % name)
            lower, upper = float(bounds[0]), float(bounds[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValidationError("joint limit for %s is invalid" % name)
            normalized_limits[str(name)] = (lower, upper)
        self.joint_limits = normalized_limits
        self.default_joint_positions = {
            str(name): float(value) for name, value in self.default_joint_positions.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotProfile":
        data = dict(data)
        data["joints"] = [JointSpec(**item) for item in data.get("joints", [])]
        data["actuators"] = [ActuatorSpec(**item) for item in data.get("actuators", [])]
        value = cls(**data)
        value.validate()
        return value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialSpec:
    name: str
    static_friction: float = 0.8
    dynamic_friction: float = 0.7
    restitution: float = 0.0
    roughness: float = 0.0
    compliance: float = 0.0
    damping: float = 0.0
    color: str = "#5c677d"


@dataclass
class GeometrySpec:
    name: str
    shape: str
    size: List[float]
    pose: Pose = field(default_factory=Pose)
    material: str = "nominal"
    semantic: str = "terrain"
    dynamic: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.shape not in {"box", "sphere", "cylinder", "mesh"}:
            raise ValidationError("unsupported geometry shape: %s" % self.shape)
        if not self.size or any(float(value) <= 0 or not math.isfinite(float(value)) for value in self.size):
            raise ValidationError("geometry %s has invalid size" % self.name)
        self.size = [float(value) for value in self.size]
        self.pose.validate()


@dataclass
class RegionSpec:
    name: str
    kind: str
    minimum: List[float]
    maximum: List[float]
    mandatory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.minimum = _vec3(self.minimum, "%s.minimum" % self.name)
        self.maximum = _vec3(self.maximum, "%s.maximum" % self.name)
        if any(lo > hi for lo, hi in zip(self.minimum, self.maximum)):
            raise ValidationError("region %s has inverted bounds" % self.name)


@dataclass
class RouteSpec:
    name: str
    waypoints: List[List[float]]
    region_names: List[str] = field(default_factory=list)
    route_kind: str = "central"


@dataclass
class WorldSpec:
    name: str
    terrain_type: str
    seed: int
    robot_profile_sha256: str
    schema_version: str = SCHEMA_VERSION
    generator_version: str = "1.1"
    dimensions_m: List[float] = field(default_factory=lambda: [5.0, 3.0, 1.0])
    spawn_pose: Pose = field(default_factory=lambda: Pose([0.5, 0.0, 0.5]))
    goal_pose: Pose = field(default_factory=lambda: Pose([4.5, 0.0, 0.5]))
    materials: Dict[str, MaterialSpec] = field(default_factory=dict)
    geometry: List[GeometrySpec] = field(default_factory=list)
    regions: List[RegionSpec] = field(default_factory=list)
    routes: List[RouteSpec] = field(default_factory=list)
    checkpoints: List[List[float]] = field(default_factory=list)
    normalized_parameters: Dict[str, Any] = field(default_factory=dict)
    resolved_parameters: Dict[str, Any] = field(default_factory=dict)
    clips: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.dimensions_m = _vec3(self.dimensions_m, "dimensions_m")
        if any(value <= 0 for value in self.dimensions_m):
            raise ValidationError("world dimensions must be positive")
        self.spawn_pose.validate()
        self.goal_pose.validate()
        names = set()
        for geometry in self.geometry:
            geometry.validate()
            if geometry.name in names:
                raise ValidationError("duplicate geometry name: %s" % geometry.name)
            names.add(geometry.name)
        region_names = set()
        for region in self.regions:
            region.validate()
            if region.name in region_names:
                raise ValidationError("duplicate region name: %s" % region.name)
            region_names.add(region.name)
        for route in self.routes:
            missing = set(route.region_names) - region_names
            if missing:
                raise ValidationError("route %s references missing regions: %s" % (route.name, sorted(missing)))
            if len(route.waypoints) < 2:
                raise ValidationError("route %s requires at least two waypoints" % route.name)
            route.waypoints = [_vec3(point, "%s.waypoint" % route.name) for point in route.waypoints]
        covered_regions = {name for route in self.routes for name in route.region_names}
        uncovered = {region.name for region in self.regions if region.mandatory} - covered_regions
        if uncovered:
            raise ValidationError("mandatory regions are not reachable from a declared route: %s" % sorted(uncovered))
        for name, pose in (("spawn", self.spawn_pose), ("goal", self.goal_pose)):
            if not 0.0 <= pose.position[0] <= self.dimensions_m[0]:
                raise ValidationError("%s pose lies outside world length" % name)
        if not self.geometry:
            raise ValidationError("world contains no collision geometry")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldSpec":
        data = dict(data)
        data["spawn_pose"] = Pose(**data.get("spawn_pose", {}))
        data["goal_pose"] = Pose(**data.get("goal_pose", {}))
        data["materials"] = {
            name: MaterialSpec(**value) for name, value in data.get("materials", {}).items()
        }
        geometries = []
        for item in data.get("geometry", []):
            item = dict(item)
            item["pose"] = Pose(**item.get("pose", {}))
            geometries.append(GeometrySpec(**item))
        data["geometry"] = geometries
        data["regions"] = [RegionSpec(**item) for item in data.get("regions", [])]
        data["routes"] = [RouteSpec(**item) for item in data.get("routes", [])]
        value = cls(**data)
        value.validate()
        return value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeSpec:
    episode_id: str
    backend: str
    robot_profile: str
    robot_asset: str
    robot_format: str
    world: str
    controller: Dict[str, Any]
    output_dir: str
    seed: int = 0
    dt_s: float = 0.005
    control_decimation: int = 4
    timeout_s: float = 20.0
    headless: bool = True
    record_video: bool = False
    stressors: List[Dict[str, Any]] = field(default_factory=list)
    event_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.backend not in {"fake", "mujoco", "isaaclab", "isaacgym"}:
            raise ValidationError("unsupported backend: %s" % self.backend)
        if self.dt_s <= 0 or self.control_decimation < 1 or self.timeout_s <= 0:
            raise ValidationError("episode timing values must be positive")
        if not self.controller.get("type"):
            raise ValidationError("controller.type is required")
        for stressor in self.stressors:
            if not stressor.get("type"):
                raise ValidationError("every stressor requires a type")
            if "value" in stressor and not isinstance(stressor["value"], (int, float)):
                raise ValidationError("stressor %s value must be numeric" % stressor["type"])
            if float(stressor.get("duration_s", 0.0)) < 0.0:
                raise ValidationError("stressor %s duration must be non-negative" % stressor["type"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeSpec":
        value = cls(**data)
        value.validate()
        return value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetrySample:
    time_s: float
    base_position_m: List[float]
    base_quaternion_wxyz: List[float]
    base_linear_velocity_mps: List[float]
    base_angular_velocity_rps: List[float]
    command_velocity_mps: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_position_rad: Dict[str, float] = field(default_factory=dict)
    joint_velocity_rps: Dict[str, float] = field(default_factory=dict)
    joint_torque_nm: Dict[str, float] = field(default_factory=dict)
    action: List[float] = field(default_factory=list)
    foot_contacts: Dict[str, bool] = field(default_factory=dict)
    foot_positions_m: Dict[str, List[float]] = field(default_factory=dict)
    foot_velocities_mps: Dict[str, List[float]] = field(default_factory=dict)
    foot_forces_n: Dict[str, List[float]] = field(default_factory=dict)
    body_contacts: List[str] = field(default_factory=list)
    observation_age_s: float = 0.0
    inference_latency_s: float = 0.0
    simulator_step_s: float = 0.0
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEvent:
    time_s: float
    event_type: str
    region: str = ""
    severity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    episode_id: str
    backend: str
    status: str
    terminal_reason: str
    samples: List[TelemetrySample] = field(default_factory=list)
    events: List[EpisodeEvent] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeLog":
        data = dict(data)
        data["samples"] = [TelemetrySample(**item) for item in data.get("samples", [])]
        data["events"] = [EpisodeEvent(**item) for item in data.get("events", [])]
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    name: str
    value: Optional[float]
    unit: str
    family: str
    aggregation: str = "episode"
    valid: bool = True
    reason: str = ""
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    threshold_version: str = "events-v1"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    status: str
    episode_spec: Dict[str, Any]
    resolved_paths: Dict[str, str]
    hashes: Dict[str, str]
    runtime: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)
    comparability: Dict[str, Any] = field(default_factory=dict)
    observation_deployability: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
