"""Geometry contracts and sampling helpers for exteroceptive observations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Tuple


Point2D = Tuple[float, float]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % label)
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return int(value)


def _fixed_vector(value: Any, length: int, label: str) -> List[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError("%s must contain exactly %d values" % (label, length))
    return [
        _finite_float(component, "%s[%d]" % (label, index))
        for index, component in enumerate(value)
    ]


def _axis_values(axis_range: Any, resolution: Any, label: str) -> List[float]:
    bounds = _fixed_vector(axis_range, 2, label)
    lower, upper = bounds
    step = _finite_float(resolution, "resolution")
    if step <= 0.0:
        raise ValueError("resolution must be positive")
    if upper < lower:
        raise ValueError("%s must be ordered from low to high" % label)

    span = upper - lower
    interval_count = int(round(span / step))
    tolerance = max(1e-12, abs(step) * 1e-9, abs(span) * 1e-12)
    if not math.isclose(
        span, interval_count * step, rel_tol=1e-12, abs_tol=tolerance
    ):
        raise ValueError(
            "%s span must be an integer multiple of resolution" % label
        )

    # Compute every coordinate from its integer index. This avoids cumulative
    # floating-point drift, while assigning the declared endpoint verbatim.
    values = [lower + index * step for index in range(interval_count + 1)]
    values[-1] = upper
    return values


def height_scan_points(
    x_range: Sequence[float],
    y_range: Sequence[float],
    resolution: float,
) -> List[Point2D]:
    """Return an endpoint-inclusive regular body-frame scan grid.

    Points are ordered with each complete y row for one x coordinate before
    advancing to the next x coordinate.
    """

    x_values = _axis_values(x_range, resolution, "x_range")
    y_values = _axis_values(y_range, resolution, "y_range")
    return [(x_value, y_value) for x_value in x_values for y_value in y_values]


def sample_height_scan(
    base_position: Sequence[float],
    base_yaw: float,
    points: Sequence[Sequence[float]],
    terrain_height: Callable[[float, float], float],
    clip: Optional[Sequence[float]] = None,
) -> List[float]:
    """Sample terrain clearances at body-frame points rotated into the world."""

    position = _fixed_vector(base_position, 3, "base_position")
    yaw = _finite_float(base_yaw, "base_yaw")
    if not callable(terrain_height):
        raise ValueError("terrain_height must be callable")

    clip_bounds = None
    if clip is not None:
        clip_bounds = _fixed_vector(clip, 2, "clip")
        if clip_bounds[1] < clip_bounds[0]:
            raise ValueError("clip must be ordered from low to high")

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    samples = []
    for index, point in enumerate(points):
        point_b = _fixed_vector(point, 2, "points[%d]" % index)
        world_x = position[0] + cosine * point_b[0] - sine * point_b[1]
        world_y = position[1] + sine * point_b[0] + cosine * point_b[1]
        terrain_z = _finite_float(
            terrain_height(world_x, world_y), "terrain height"
        )
        clearance = position[2] - terrain_z
        if clip_bounds is not None:
            clearance = min(clip_bounds[1], max(clip_bounds[0], clearance))
        samples.append(clearance)
    return samples


def _missing_keys(geometry: Mapping[str, Any], keys: Sequence[str]) -> List[str]:
    return [key for key in keys if key not in geometry or geometry[key] is None]


def validate_camera_geometry(geometry: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize camera intrinsics, clipping, and body-frame pose."""

    if not isinstance(geometry, Mapping):
        raise ValueError("camera geometry must be a mapping")
    required = [
        "resolution",
        "near",
        "far",
        "position_b",
        "orientation_wxyz",
        "encoding",
    ]
    missing = _missing_keys(geometry, required)
    fov_keys = [
        key
        for key in ("hfov_deg", "vfov_deg", "fov_deg", "field_of_view_deg")
        if key in geometry and geometry[key] is not None
    ]
    if not fov_keys:
        missing.append("field of view")
    if missing:
        raise ValueError(
            "camera geometry is missing: %s" % ", ".join(missing)
        )

    resolution = geometry["resolution"]
    if (
        isinstance(resolution, (str, bytes))
        or not isinstance(resolution, Sequence)
        or len(resolution) != 2
    ):
        raise ValueError("camera geometry resolution must contain width and height")
    normalized = dict(geometry)
    normalized["resolution"] = [
        _positive_int(value, "camera geometry resolution") for value in resolution
    ]

    for key in fov_keys:
        fov = _finite_float(geometry[key], "camera geometry %s" % key)
        if not 0.0 < fov < 180.0:
            raise ValueError("camera geometry %s must be in (0, 180)" % key)
        normalized[key] = fov

    near = _finite_float(geometry["near"], "camera geometry near")
    far = _finite_float(geometry["far"], "camera geometry far")
    if near <= 0.0 or far <= near:
        raise ValueError("camera geometry requires 0 < near < far")
    normalized["near"] = near
    normalized["far"] = far
    normalized["position_b"] = _fixed_vector(
        geometry["position_b"], 3, "camera geometry position_b"
    )

    quaternion = _fixed_vector(
        geometry["orientation_wxyz"], 4, "camera geometry orientation_wxyz"
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ValueError("camera geometry orientation_wxyz must be non-zero")
    normalized["orientation_wxyz"] = [value / norm for value in quaternion]

    encoding = geometry["encoding"]
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("camera geometry encoding must be a non-empty string")
    normalized["encoding"] = encoding.strip()
    return normalized


def validate_lidar_geometry(geometry: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a lidar ray pattern and body-frame origin."""

    if not isinstance(geometry, Mapping):
        raise ValueError("lidar geometry must be a mapping")
    required = ["ray_count", "angular_pattern", "max_range"]
    missing = _missing_keys(geometry, required)
    origin_key = next(
        (
            key
            for key in ("origin_b", "origin")
            if key in geometry and geometry[key] is not None
        ),
        None,
    )
    if origin_key is None:
        missing.append("origin")
    if missing:
        raise ValueError("lidar geometry is missing: %s" % ", ".join(missing))

    normalized = dict(geometry)
    ray_count = _positive_int(geometry["ray_count"], "lidar geometry ray_count")
    normalized["ray_count"] = ray_count

    pattern = geometry["angular_pattern"]
    if isinstance(pattern, str):
        if not pattern.strip():
            raise ValueError("lidar geometry angular_pattern must not be empty")
        normalized["angular_pattern"] = pattern.strip()
    elif isinstance(pattern, Mapping):
        if not pattern:
            raise ValueError("lidar geometry angular_pattern must not be empty")
        normalized["angular_pattern"] = dict(pattern)
    elif isinstance(pattern, Sequence) and not isinstance(pattern, (bytes, str)):
        if len(pattern) != ray_count:
            raise ValueError(
                "lidar geometry angular_pattern has %d rays, expected %d"
                % (len(pattern), ray_count)
            )
        normalized["angular_pattern"] = list(pattern)
    else:
        raise ValueError(
            "lidar geometry angular_pattern must be a name, mapping, or ray sequence"
        )

    max_range = _finite_float(geometry["max_range"], "lidar geometry max_range")
    if max_range <= 0.0:
        raise ValueError("lidar geometry max_range must be positive")
    normalized["max_range"] = max_range
    normalized[origin_key] = _fixed_vector(
        geometry[origin_key], 3, "lidar geometry %s" % origin_key
    )
    return normalized


def _grid_count(geometry: Mapping[str, Any]) -> int:
    if geometry.get("kind") != "grid":
        raise ValueError("grid geometry kind must be 'grid'")
    missing = _missing_keys(geometry, ["x_range", "y_range", "resolution"])
    if missing:
        raise ValueError("grid geometry is missing: %s" % ", ".join(missing))
    if "max_range" in geometry:
        max_range = _finite_float(geometry["max_range"], "grid geometry max_range")
        if max_range <= 0.0:
            raise ValueError("grid geometry max_range must be positive")
    return len(
        height_scan_points(
            geometry["x_range"], geometry["y_range"], geometry["resolution"]
        )
    )


def geometry_element_count(source: str, geometry: Mapping[str, Any]) -> int:
    """Return the flattened observation width implied by sensor geometry."""

    if not isinstance(geometry, Mapping):
        raise ValueError("%s geometry must be a mapping" % source)
    if source in {
        "height_scan",
        "height_scan_per_foot",
        "elevation_map",
        "traversability_map",
        "occupancy_grid",
        "signed_distance_field",
    }:
        count = _grid_count(geometry)
        if source == "height_scan_per_foot":
            if "foot_count" in geometry:
                count *= _positive_int(
                    geometry["foot_count"], "height_scan_per_foot geometry foot_count"
                )
            elif "origins_b" in geometry:
                origins = geometry["origins_b"]
                if (
                    isinstance(origins, (str, bytes))
                    or not isinstance(origins, Sequence)
                    or not origins
                ):
                    raise ValueError(
                        "height_scan_per_foot geometry origins_b must not be empty"
                    )
                for index, origin in enumerate(origins):
                    _fixed_vector(
                        origin,
                        3,
                        "height_scan_per_foot geometry origins_b[%d]" % index,
                    )
                count *= len(origins)
        return count
    if source in {
        "depth_image",
        "rgb_image",
        "thermal_image",
        "semantic_segmentation",
        "stereo_disparity",
        "optical_flow",
    }:
        camera = validate_camera_geometry(geometry)
        pixel_count = camera["resolution"][0] * camera["resolution"][1]
        if source == "rgb_image":
            return pixel_count * 3
        if source == "optical_flow":
            return pixel_count * 2
        if source == "stereo_disparity":
            if "baseline_m" not in geometry:
                raise ValueError("stereo camera geometry is missing: baseline_m")
            baseline = _finite_float(
                geometry["baseline_m"], "stereo camera geometry baseline_m"
            )
            if baseline <= 0.0:
                raise ValueError("stereo camera geometry baseline_m must be positive")
        return pixel_count
    if source == "lidar_rays":
        return validate_lidar_geometry(geometry)["ray_count"]
    if source == "point_cloud":
        if "point_count" in geometry:
            return _positive_int(
                geometry["point_count"], "point_cloud geometry point_count"
            ) * 3
        if "ray_count" in geometry:
            return validate_lidar_geometry(geometry)["ray_count"] * 3
        if "resolution" in geometry:
            camera = validate_camera_geometry(geometry)
            return camera["resolution"][0] * camera["resolution"][1] * 3
        raise ValueError(
            "point_cloud geometry must declare point_count or a camera/lidar pattern"
        )
    raise ValueError("unsupported exteroceptive source %r" % source)


__all__ = [
    "geometry_element_count",
    "height_scan_points",
    "sample_height_scan",
    "validate_camera_geometry",
    "validate_lidar_geometry",
]
