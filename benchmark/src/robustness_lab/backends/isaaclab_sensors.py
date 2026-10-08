"""Unverified native Isaac Lab exteroceptive sensor implementations.

Isaac Lab and Isaac Sim are intentionally imported only while constructing a
suite.  Coordinator and verification processes can therefore import this
module without loading (or even installing) the Omniverse runtime.

This code has never executed in an episode.  The adapter currently hangs after
``SimulationContext`` setup, before it can produce any canonical observation.
Promoting these sources to verified requires an Isaac Lab episode to pass that
point, read each sensor into the canonical state, and preserve the declared
shape and value semantics in recorded run evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..observation.sensor_geometry import (
    height_scan_points,
    validate_camera_geometry,
    validate_lidar_geometry,
)


SensorKey = Tuple[str, Optional[str]]


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
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if values.shape != (3,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("%s must be a finite, non-zero three-vector" % label)
    return values / norm


def _cpu_values(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class IsaacLabSensorSuite:
    """Produce flat canonical observations from requested native sensors."""

    _DEFAULT_RAY_RANGE = 1000.0

    def __init__(
        self,
        articulation: Any,
        robot_profile: Any,
        sensor_requests: Sequence[Tuple[str, Optional[str], Mapping[str, Any]]],
        update_period: float,
        terrain_prim_path: str = "/World/Terrain",
    ) -> None:
        # These imports must remain local.  Importing Isaac Lab before its app
        # is launched is unsafe, and the package is absent from coordinators.
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import (
            RayCaster,
            RayCasterCamera,
            RayCasterCameraCfg,
            RayCasterCfg,
            TiledCamera,
            TiledCameraCfg,
            patterns,
        )

        self.articulation = articulation
        self.robot_profile = robot_profile
        self.update_period = float(update_period)
        self.terrain_prim_path = terrain_prim_path
        self._sim_utils = sim_utils
        self._RayCaster = RayCaster
        self._RayCasterCfg = RayCasterCfg
        self._RayCasterCamera = RayCasterCamera
        self._RayCasterCameraCfg = RayCasterCameraCfg
        self._TiledCamera = TiledCamera
        self._TiledCameraCfg = TiledCameraCfg
        self._patterns = patterns
        self._link_paths: Dict[str, str] = {}
        self._bindings: Dict[SensorKey, Dict[str, Any]] = {}
        self._geometries: Dict[SensorKey, Mapping[str, Any]] = {}
        self._sensors: List[Any] = []
        self._first_flow: Dict[SensorKey, bool] = {}

        for index, (source, instance, geometry) in enumerate(sensor_requests):
            key = (source, instance)
            self._geometries[key] = geometry
            self._bindings[key] = self._make_binding(source, geometry, index)
            if source == "optical_flow":
                self._first_flow[key] = True

    @property
    def _base_prim_path(self) -> str:
        return self._link_prim_path(self.robot_profile.base_link)

    def _link_prim_path(self, link_name: str) -> str:
        cached = self._link_paths.get(link_name)
        if cached is not None:
            return cached
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prefix = "/World/Robot/"
        candidates = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() == link_name and str(prim.GetPath()).startswith(prefix)
        ]
        if len(candidates) > 1:
            raise ValueError(
                "robot link %r resolves to multiple Isaac Lab prims: %s"
                % (link_name, candidates)
            )
        resolved = candidates[0] if candidates else prefix + link_name
        self._link_paths[link_name] = resolved
        return resolved

    @property
    def _base_position_w(self) -> np.ndarray:
        return np.asarray(
            _cpu_values(self.articulation.data.root_pos_w[0]), dtype=float
        )

    @property
    def _body_to_world(self) -> np.ndarray:
        quaternion = _cpu_values(self.articulation.data.root_quat_w[0])
        return _rotation_from_wxyz(quaternion)

    @staticmethod
    def _clip(values: List[float], geometry: Mapping[str, Any]) -> List[float]:
        bounds = geometry.get("clip")
        if bounds is None:
            return values
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("clip must contain exactly two values")
        lower, upper = float(bounds[0]), float(bounds[1])
        if upper < lower:
            raise ValueError("clip must be ordered from low to high")
        return [min(upper, max(lower, value)) for value in values]

    def _register(self, sensor: Any) -> Any:
        self._sensors.append(sensor)
        return sensor

    def _make_binding(
        self, source: str, geometry: Mapping[str, Any], index: int
    ) -> Dict[str, Any]:
        if source in {
            "height_scan",
            "elevation_map",
            "terrain_normal_b",
            "occupancy_grid",
            "signed_distance_field",
        }:
            return self._make_grid_binding(geometry, self._base_prim_path)
        if source == "height_scan_per_foot":
            return self._make_per_foot_binding(geometry)
        if source in {"lidar_rays", "range_sensor"}:
            return self._make_ray_binding(source, geometry, self._base_prim_path)
        if source == "point_cloud":
            if "resolution" in geometry and "ray_count" not in geometry:
                return self._make_raycast_camera_binding(source, geometry)
            return self._make_ray_binding(source, geometry, self._base_prim_path)
        if source in {"depth_image", "stereo_disparity"}:
            return self._make_raycast_camera_binding(source, geometry)
        if source in {"rgb_image", "semantic_segmentation", "optical_flow"}:
            return self._make_tiled_camera_binding(source, geometry, index)
        raise ValueError("unsupported Isaac Lab exteroceptive source %r" % source)

    def _grid_pattern(self, geometry: Mapping[str, Any]) -> Tuple[Any, Tuple[float, float]]:
        points = height_scan_points(
            geometry["x_range"], geometry["y_range"], geometry["resolution"]
        )
        x_range = [float(value) for value in geometry["x_range"]]
        y_range = [float(value) for value in geometry["y_range"]]
        center = ((x_range[0] + x_range[1]) / 2.0, (y_range[0] + y_range[1]) / 2.0)
        pattern = self._patterns.GridPatternCfg(
            resolution=float(geometry["resolution"]),
            size=(x_range[1] - x_range[0], y_range[1] - y_range[0]),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        )
        if len(points) == 0:
            raise ValueError("grid ray pattern must contain at least one point")
        return pattern, center

    def _ray_caster(
        self,
        prim_path: str,
        pattern: Any,
        max_range: float,
        offset: Sequence[float],
        yaw_only: bool,
    ) -> Any:
        cfg = self._RayCasterCfg(
            prim_path=prim_path,
            update_period=self.update_period,
            offset=self._RayCasterCfg.OffsetCfg(pos=tuple(float(value) for value in offset)),
            attach_yaw_only=yaw_only,
            pattern_cfg=pattern,
            max_distance=max_range,
            debug_vis=False,
            mesh_prim_paths=[self.terrain_prim_path],
        )
        return self._register(self._RayCaster(cfg=cfg))

    def _make_grid_binding(
        self, geometry: Mapping[str, Any], prim_path: str
    ) -> Dict[str, Any]:
        pattern, center = self._grid_pattern(geometry)
        max_range = float(geometry.get("max_range", self._DEFAULT_RAY_RANGE))
        sensor = self._ray_caster(
            prim_path, pattern, max_range, (center[0], center[1], 0.0), True
        )
        return {"kind": "grid", "sensors": [sensor], "max_range": max_range}

    def _offset_pattern(self, offsets: Sequence[Sequence[float]]) -> Any:
        starts = []
        for index, offset in enumerate(offsets):
            if isinstance(offset, (str, bytes)) or not isinstance(offset, Sequence) or len(offset) not in (2, 3):
                raise ValueError(
                    "per-foot offset %d must contain two or three values" % index
                )
            starts.append(
                [float(offset[0]), float(offset[1]), float(offset[2]) if len(offset) == 3 else 0.0]
            )

        def declared_offset_pattern(cfg: Any, device: str) -> Tuple[Any, Any]:
            import torch

            ray_starts = torch.tensor(starts, dtype=torch.float32, device=device)
            ray_directions = torch.zeros_like(ray_starts)
            ray_directions[:, 2] = -1.0
            return ray_starts, ray_directions

        return self._patterns.PatternBaseCfg(func=declared_offset_pattern)

    def _make_per_foot_binding(self, geometry: Mapping[str, Any]) -> Dict[str, Any]:
        if geometry.get("kind") == "grid":
            pattern, center = self._grid_pattern(geometry)
            offset = (center[0], center[1], 0.0)
        else:
            offsets = geometry.get("offsets", [[0.0, 0.0]])
            if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence):
                raise ValueError("per-foot offsets must be a sequence")
            pattern = self._offset_pattern(offsets)
            offset = (0.0, 0.0, 0.0)

        origins = geometry.get("origins_b")
        if origins is None:
            attachments = [
                (self._link_prim_path(foot_name), offset)
                for foot_name in self.robot_profile.foot_links
            ]
        else:
            attachments = []
            for index, origin in enumerate(origins):
                if isinstance(origin, (str, bytes)) or not isinstance(origin, Sequence) or len(origin) != 3:
                    raise ValueError("origins_b[%d] must contain exactly three values" % index)
                attachments.append(
                    (
                        self._base_prim_path,
                        tuple(float(origin[axis]) + offset[axis] for axis in range(3)),
                    )
                )
        max_range = float(geometry.get("max_range", self._DEFAULT_RAY_RANGE))
        sensors = [
            self._ray_caster(path, pattern, max_range, sensor_offset, True)
            for path, sensor_offset in attachments
        ]
        return {"kind": "grid", "sensors": sensors, "max_range": max_range}

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
                    [
                        math.cos(2.0 * math.pi * index / count),
                        math.sin(2.0 * math.pi * index / count),
                        0.0,
                    ],
                    dtype=float,
                )
                for index in range(count)
            ]
        if isinstance(pattern, Mapping):
            azimuth = pattern.get(
                "azimuth_range_deg", pattern.get("azimuth_deg", [-180.0, 180.0])
            )
            elevation = pattern.get(
                "elevation_range_deg", pattern.get("elevation_deg", [0.0, 0.0])
            )
            azimuth_values = np.linspace(
                float(azimuth[0]), float(azimuth[-1]), count, endpoint=False
            )
            elevation_values = np.linspace(
                float(elevation[0]), float(elevation[-1]), count
            )
            return [
                np.asarray(
                    [
                        math.cos(math.radians(el)) * math.cos(math.radians(az)),
                        math.cos(math.radians(el)) * math.sin(math.radians(az)),
                        math.sin(math.radians(el)),
                    ],
                    dtype=float,
                )
                for az, el in zip(azimuth_values, elevation_values)
            ]
        if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
            directions = []
            for index, ray in enumerate(pattern):
                if isinstance(ray, Sequence) and len(ray) == 3:
                    directions.append(_unit_vector(ray, "angular_pattern[%d]" % index))
                elif isinstance(ray, Sequence) and len(ray) == 2:
                    azimuth = math.radians(float(ray[0]))
                    elevation = math.radians(float(ray[1]))
                    directions.append(
                        np.asarray(
                            [
                                math.cos(elevation) * math.cos(azimuth),
                                math.cos(elevation) * math.sin(azimuth),
                                math.sin(elevation),
                            ],
                            dtype=float,
                        )
                    )
                else:
                    azimuth = math.radians(float(ray))
                    directions.append(
                        np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0])
                    )
            return directions
        raise ValueError("angular_pattern must be a name, mapping, or ray sequence")

    def _declared_ray_pattern(self, directions: Sequence[np.ndarray]) -> Any:
        direction_values = [direction.astype(float).tolist() for direction in directions]

        def declared_ray_pattern(cfg: Any, device: str) -> Tuple[Any, Any]:
            import torch

            ray_directions = torch.tensor(
                direction_values, dtype=torch.float32, device=device
            )
            return torch.zeros_like(ray_directions), ray_directions

        return self._patterns.PatternBaseCfg(func=declared_ray_pattern)

    def _make_ray_binding(
        self, source: str, geometry: Mapping[str, Any], prim_path: str
    ) -> Dict[str, Any]:
        if source == "range_sensor":
            origin_b = np.asarray(
                geometry.get("origin_b", geometry.get("origin", [0.0, 0.0, 0.0])),
                dtype=float,
            )
            direction_b = _unit_vector(geometry["direction_b"], "direction_b")
            directions = [direction_b]
            azimuth = math.degrees(math.atan2(direction_b[1], direction_b[0]))
            elevation = math.degrees(math.asin(float(direction_b[2])))
            pattern = self._patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(elevation, elevation),
                horizontal_fov_range=(azimuth, azimuth),
                horizontal_res=1.0,
            )
            max_range = float(geometry["max_range"])
        else:
            if "ray_count" in geometry:
                normalized = validate_lidar_geometry(geometry)
            else:
                normalized = dict(geometry)
                normalized["ray_count"] = int(geometry["point_count"])
                normalized.setdefault("angular_pattern", "uniform")
                normalized.setdefault("max_range", 10.0)
                normalized.setdefault("origin_b", [0.0, 0.0, 0.0])
            origin_b = np.asarray(
                normalized.get("origin_b", normalized.get("origin")), dtype=float
            )
            max_range = float(normalized["max_range"])
            directions = self._directions_b(normalized)
            angular_pattern = normalized["angular_pattern"]
            if isinstance(angular_pattern, str):
                pattern = self._patterns.LidarPatternCfg(
                    channels=1,
                    vertical_fov_range=(0.0, 0.0),
                    horizontal_fov_range=(0.0, 360.0),
                    horizontal_res=360.0 / len(directions),
                )
            else:
                # Explicit and coupled azimuth/elevation declarations cannot be
                # represented by LidarPatternCfg's Cartesian product.  Keep the
                # declared order through Isaac's PatternBaseCfg extension point.
                pattern = self._declared_ray_pattern(directions)
        sensor = self._ray_caster(
            prim_path, pattern, max_range, origin_b.tolist(), False
        )
        return {
            "kind": "ray",
            "sensor": sensor,
            "max_range": max_range,
            "origin_b": origin_b,
            "directions_b": directions,
        }

    @staticmethod
    def _horizontal_fov(camera: Mapping[str, Any]) -> float:
        width, height = camera["resolution"]
        if "hfov_deg" in camera:
            return float(camera["hfov_deg"])
        if "vfov_deg" in camera:
            return math.degrees(
                2.0
                * math.atan(
                    math.tan(math.radians(float(camera["vfov_deg"])) / 2.0)
                    * width
                    / height
                )
            )
        if "fov_deg" in camera:
            return float(camera["fov_deg"])
        return float(camera["field_of_view_deg"])

    @classmethod
    def _focal_length_pixels(cls, camera: Mapping[str, Any]) -> float:
        width = int(camera["resolution"][0])
        return width / (
            2.0 * math.tan(math.radians(cls._horizontal_fov(camera)) / 2.0)
        )

    def _pinhole_parameters(self, camera: Mapping[str, Any]) -> Tuple[float, float]:
        aperture = 20.955
        focal_length = aperture / (
            2.0 * math.tan(math.radians(self._horizontal_fov(camera)) / 2.0)
        )
        return focal_length, aperture

    def _raycast_camera(self, camera: Mapping[str, Any]) -> Any:
        width, height = camera["resolution"]
        focal_length, aperture = self._pinhole_parameters(camera)
        pattern = self._patterns.PinholeCameraPatternCfg(
            focal_length=focal_length,
            horizontal_aperture=aperture,
            width=width,
            height=height,
        )
        cfg = self._RayCasterCameraCfg(
            prim_path=self._base_prim_path,
            update_period=self.update_period,
            offset=self._RayCasterCameraCfg.OffsetCfg(
                pos=tuple(camera["position_b"]),
                rot=tuple(camera["orientation_wxyz"]),
                convention="world",
            ),
            attach_yaw_only=False,
            pattern_cfg=pattern,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            max_distance=float(camera["far"]),
            debug_vis=False,
            mesh_prim_paths=[self.terrain_prim_path],
        )
        return self._register(self._RayCasterCamera(cfg=cfg))

    def _make_raycast_camera_binding(
        self, source: str, geometry: Mapping[str, Any]
    ) -> Dict[str, Any]:
        camera = validate_camera_geometry(geometry)
        if source != "stereo_disparity":
            return {
                "kind": "raycast_camera",
                "sensors": [self._raycast_camera(camera)],
                "camera": camera,
            }
        baseline = float(geometry.get("baseline_m", 0.0))
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("stereo camera baseline_m must be positive and finite")
        left = dict(camera)
        right = dict(camera)
        left["position_b"] = list(camera["position_b"])
        right["position_b"] = list(camera["position_b"])
        left["position_b"][1] -= baseline / 2.0
        right["position_b"][1] += baseline / 2.0
        return {
            "kind": "stereo",
            "sensors": [self._raycast_camera(left), self._raycast_camera(right)],
            "camera": camera,
            "baseline": baseline,
        }

    def _make_tiled_camera_binding(
        self, source: str, geometry: Mapping[str, Any], index: int
    ) -> Dict[str, Any]:
        camera = validate_camera_geometry(geometry)
        data_type = {
            "rgb_image": "rgb",
            "semantic_segmentation": "semantic_segmentation",
            "optical_flow": "motion_vectors",
        }[source]
        focal_length, aperture = self._pinhole_parameters(camera)
        cfg = self._TiledCameraCfg(
            prim_path="%s/RobustnessLabCamera_%04d" % (self._base_prim_path, index),
            update_period=self.update_period,
            offset=self._TiledCameraCfg.OffsetCfg(
                pos=tuple(camera["position_b"]),
                rot=tuple(camera["orientation_wxyz"]),
                convention="world",
            ),
            data_types=[data_type],
            spawn=self._sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=aperture,
                clipping_range=(float(camera["near"]), float(camera["far"])),
            ),
            width=int(camera["resolution"][0]),
            height=int(camera["resolution"][1]),
            depth_clipping_behavior="max",
            colorize_semantic_segmentation=False,
        )
        return {
            "kind": "tiled_camera",
            "sensor": self._register(self._TiledCamera(cfg=cfg)),
            "camera": camera,
            "data_type": data_type,
        }

    def reset(self) -> None:
        for sensor in self._sensors:
            sensor.reset()
        for key in self._first_flow:
            self._first_flow[key] = True

    def update(self, dt_s: float) -> None:
        for sensor in self._sensors:
            sensor.update(float(dt_s))

    def close(self) -> None:
        self._bindings.clear()
        self._geometries.clear()
        self._link_paths.clear()
        self._sensors.clear()
        self._first_flow.clear()

    def _binding(
        self,
        source: str,
        geometry: Mapping[str, Any],
        instance: Optional[str] = None,
    ) -> Tuple[SensorKey, Dict[str, Any]]:
        key = (source, instance)
        if key in self._bindings and self._geometries[key] == geometry:
            return key, self._bindings[key]
        matches = [
            candidate
            for candidate, declared in self._geometries.items()
            if candidate[0] == source and declared == geometry
        ]
        if not matches:
            raise ValueError("sensor %r was not requested before simulation reset" % source)
        selected = matches[0]
        return selected, self._bindings[selected]

    def read(
        self, source: str, instance: Optional[str], geometry: Mapping[str, Any]
    ) -> List[float]:
        key, binding = self._binding(source, geometry, instance)
        reader = getattr(self, "_read_%s" % source)
        return reader(binding, geometry, key)

    @staticmethod
    def _ray_starts_w(sensor: Any) -> np.ndarray:
        data = sensor.data
        declared_starts = getattr(data, "ray_starts_w", None)
        if declared_starts is not None:
            return np.asarray(_cpu_values(declared_starts[0]), dtype=float)
        local_starts = np.asarray(_cpu_values(sensor.ray_starts[0]), dtype=float)
        position_w = np.asarray(_cpu_values(data.pos_w[0]), dtype=float)
        rotation = _rotation_from_wxyz(_cpu_values(data.quat_w[0]))
        if getattr(sensor.cfg, "attach_yaw_only", False):
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
        return position_w + local_starts @ rotation.T

    @classmethod
    def _ray_samples(cls, sensor: Any, max_range: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[float]]:
        starts = cls._ray_starts_w(sensor)
        hits = np.asarray(_cpu_values(sensor.data.ray_hits_w[0]), dtype=float)
        valid = np.isfinite(hits).all(axis=1)
        distances = np.linalg.norm(hits - starts, axis=1)
        distances = np.where(valid, np.clip(distances, 0.0, max_range), max_range)
        return list(starts), list(hits), valid.tolist(), distances.astype(float).tolist()

    def _grid_samples(
        self, binding: Mapping[str, Any]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool]]:
        starts: List[np.ndarray] = []
        hits: List[np.ndarray] = []
        valid: List[bool] = []
        for sensor in binding["sensors"]:
            sensor_starts, sensor_hits, sensor_valid, _ = self._ray_samples(
                sensor, binding["max_range"]
            )
            starts.extend(sensor_starts)
            hits.extend(sensor_hits)
            valid.extend(sensor_valid)
        return starts, hits, valid

    def _read_height_scan(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        starts, hits, valid = self._grid_samples(binding)
        base_z = float(self._base_position_w[2])
        values = [
            base_z - (float(hit[2]) if hit_valid else float(start[2]) - binding["max_range"])
            for start, hit, hit_valid in zip(starts, hits, valid)
        ]
        return self._clip(values, geometry)

    def _read_height_scan_per_foot(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        return self._read_height_scan(binding, geometry, key)

    def _read_elevation_map(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        starts, hits, valid = self._grid_samples(binding)
        return [
            float(hit[2]) if hit_valid else float(start[2]) - binding["max_range"]
            for start, hit, hit_valid in zip(starts, hits, valid)
        ]

    def _read_terrain_normal_b(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        _, hits, valid = self._grid_samples(binding)
        samples = [hit for hit, hit_valid in zip(hits, valid) if hit_valid]
        if len(samples) < 3:
            return [0.0, 0.0, 1.0]
        design = np.asarray([[point[0], point[1], 1.0] for point in samples])
        if np.linalg.matrix_rank(design) < 3:
            return [0.0, 0.0, 1.0]
        slope_x, slope_y, _ = np.linalg.lstsq(
            design, np.asarray([point[2] for point in samples]), rcond=None
        )[0]
        normal_w = _unit_vector([-slope_x, -slope_y, 1.0], "terrain normal")
        normal_b = _unit_vector(
            self._body_to_world.T @ normal_w, "body-frame terrain normal"
        )
        return [float(value) for value in normal_b]

    def _read_occupancy_grid(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        _, _, valid = self._grid_samples(binding)
        return [1.0 if hit_valid else 0.0 for hit_valid in valid]

    def _read_signed_distance_field(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        starts, hits, valid = self._grid_samples(binding)
        base_z = float(self._base_position_w[2])
        values = [
            base_z - float(hit[2]) if hit_valid else binding["max_range"]
            for start, hit, hit_valid in zip(starts, hits, valid)
        ]
        return self._clip(values, geometry)

    def _read_lidar_rays(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        return self._ray_samples(binding["sensor"], binding["max_range"])[3]

    def _read_range_sensor(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        return self._read_lidar_rays(binding, geometry, key)[:1]

    @staticmethod
    def _camera_output(sensor: Any, name: str) -> np.ndarray:
        return np.asarray(_cpu_values(sensor.data.output[name][0]))

    def _depth_values(self, binding: Mapping[str, Any], sensor_index: int = 0) -> np.ndarray:
        camera = binding["camera"]
        image = self._camera_output(
            binding["sensors"][sensor_index], "distance_to_image_plane"
        ).astype(float)
        return np.nan_to_num(
            image, nan=float(camera["far"]), posinf=float(camera["far"]), neginf=float(camera["near"])
        ).clip(float(camera["near"]), float(camera["far"]))

    def _read_depth_image(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        return self._depth_values(binding).reshape(-1).tolist()

    def _read_point_cloud(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        if binding["kind"] == "ray":
            sensor = binding["sensor"]
            starts, hits, valid, _ = self._ray_samples(sensor, binding["max_range"])
            rotation_t = self._body_to_world.T
            base = self._base_position_w
            cloud: List[float] = []
            for index, (start, hit, hit_valid) in enumerate(zip(starts, hits, valid)):
                if hit_valid:
                    point_b = rotation_t @ (hit - base)
                else:
                    point_b = binding["origin_b"] + binding["directions_b"][index] * binding["max_range"]
                cloud.extend(float(value) for value in point_b)
            return cloud

        camera = binding["camera"]
        width, height = camera["resolution"]
        depths = self._depth_values(binding).reshape(height, width)
        focal = self._focal_length_pixels(camera)
        rotation_b = _rotation_from_wxyz(camera["orientation_wxyz"])
        position_b = np.asarray(camera["position_b"], dtype=float)
        cloud = []
        for row in range(height):
            for column in range(width):
                depth = float(depths[row, column])
                point_camera = np.asarray(
                    [
                        depth,
                        -(column + 0.5 - width / 2.0) * depth / focal,
                        -(row + 0.5 - height / 2.0) * depth / focal,
                    ]
                )
                cloud.extend(
                    float(value) for value in position_b + rotation_b @ point_camera
                )
        return cloud

    def _read_rgb_image(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        image = self._camera_output(binding["sensor"], "rgb").astype(float)
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.size and float(np.max(image)) > 1.0:
            image /= 255.0
        return np.clip(image, 0.0, 1.0).reshape(-1).tolist()

    def _read_semantic_segmentation(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        image = self._camera_output(
            binding["sensor"], "semantic_segmentation"
        )
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return image.astype(float).reshape(-1).tolist()

    def _read_stereo_disparity(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        left_depth = self._depth_values(binding, 0).reshape(-1)
        # Access the right image as part of the declared native stereo rig.
        self._depth_values(binding, 1)
        disparity = binding["baseline"] * self._focal_length_pixels(binding["camera"]) / left_depth
        return disparity.astype(float).tolist()

    def _read_optical_flow(self, binding: Mapping[str, Any], geometry: Mapping[str, Any], key: SensorKey) -> List[float]:
        width, height = binding["camera"]["resolution"]
        flow = self._camera_output(binding["sensor"], "motion_vectors").astype(float)
        if flow.shape[-1] > 2:
            flow = flow[..., :2]
        if self._first_flow.get(key, True):
            self._first_flow[key] = False
            return [0.0] * (width * height * 2)
        return flow.reshape(-1).tolist()

    def height_scan(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("height_scan", geometry)
        return self._read_height_scan(binding, geometry, key)

    def height_scan_per_foot(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("height_scan_per_foot", geometry)
        return self._read_height_scan_per_foot(binding, geometry, key)

    def elevation_map(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("elevation_map", geometry)
        return self._read_elevation_map(binding, geometry, key)

    def terrain_normal_b(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("terrain_normal_b", geometry)
        return self._read_terrain_normal_b(binding, geometry, key)

    def lidar_rays(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("lidar_rays", geometry)
        return self._read_lidar_rays(binding, geometry, key)

    def point_cloud(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("point_cloud", geometry)
        return self._read_point_cloud(binding, geometry, key)

    def range_sensor(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("range_sensor", geometry)
        return self._read_range_sensor(binding, geometry, key)

    def occupancy_grid(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("occupancy_grid", geometry)
        return self._read_occupancy_grid(binding, geometry, key)

    def signed_distance_field(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("signed_distance_field", geometry)
        return self._read_signed_distance_field(binding, geometry, key)

    def depth_image(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("depth_image", geometry)
        return self._read_depth_image(binding, geometry, key)

    def rgb_image(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("rgb_image", geometry)
        return self._read_rgb_image(binding, geometry, key)

    def semantic_segmentation(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("semantic_segmentation", geometry)
        return self._read_semantic_segmentation(binding, geometry, key)

    def stereo_disparity(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("stereo_disparity", geometry)
        return self._read_stereo_disparity(binding, geometry, key)

    def optical_flow(self, geometry: Mapping[str, Any]) -> List[float]:
        key, binding = self._binding("optical_flow", geometry)
        return self._read_optical_flow(binding, geometry, key)


__all__ = ["IsaacLabSensorSuite"]
