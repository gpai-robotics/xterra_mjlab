"""Geometry-visible modifiers shared by every simulator-neutral world.

The modifier operates on collision geometry, routes, regions and poses before
backend export. Consequently MuJoCo, Isaac Gym and Isaac Lab receive the same
banked/curved world rather than a visual-only deformation.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Sequence, Tuple

from ..models import GeometrySpec, Pose, RegionSpec, WorldSpec


Matrix = List[List[float]]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][item] * right[item][column] for item in range(3)) for column in range(3)] for row in range(3)]


def _matvec(matrix: Matrix, vector: Sequence[float]) -> List[float]:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _rotation_x(angle: float) -> Matrix:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]


def _rotation_y(angle: float) -> Matrix:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]


def _rotation_z(angle: float) -> Matrix:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]


def _rpy_matrix(rpy: Sequence[float]) -> Matrix:
    return _multiply(_multiply(_rotation_z(rpy[2]), _rotation_y(rpy[1])), _rotation_x(rpy[0]))


def _matrix_rpy(matrix: Matrix) -> List[float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return [roll, pitch, yaw]


def _oriented_bounds(geometry: GeometrySpec) -> Tuple[List[float], List[float]]:
    rotation = _rpy_matrix(geometry.pose.rpy)
    half = [value / 2.0 for value in geometry.size[:3]]
    corners = []
    for x in (-half[0], half[0]):
        for y in (-half[1], half[1]):
            for z in (-half[2], half[2]):
                rotated = _matvec(rotation, [x, y, z])
                corners.append([geometry.pose.position[axis] + rotated[axis] for axis in range(3)])
    return ([min(point[axis] for point in corners) for axis in range(3)], [max(point[axis] for point in corners) for axis in range(3)])


def apply_geometry_modifiers(
    source: WorldSpec,
    camber_deg: float = 0.0,
    curve_angle_deg: float = 0.0,
    curve_segments: int = 24,
) -> WorldSpec:
    """Apply sideways banking and plan-view curvature to a WorldSpec.

    Positive camber raises the +Y/left side and lowers the -Y/right side.
    Positive curvature turns left/anticlockwise when viewed from above;
    negative curvature turns right/clockwise. The transformation is a spatial
    sweep: long boxes are segmented so their collision geometry follows the
    arc instead of merely rotating as one straight slab.
    """

    camber_deg = float(camber_deg)
    curve_angle_deg = float(curve_angle_deg)
    curve_segments = int(curve_segments)
    if not -35.0 <= camber_deg <= 35.0:
        raise ValueError("camber angle must be within -35 to 35 degrees")
    if not -150.0 <= curve_angle_deg <= 150.0:
        raise ValueError("curve angle must be within -150 to 150 degrees")
    if not 4 <= curve_segments <= 200:
        raise ValueError("curve segments must be in [4, 200]")
    if abs(camber_deg) < 1e-12 and abs(curve_angle_deg) < 1e-12:
        return copy.deepcopy(source)

    world = copy.deepcopy(source)
    original_length = float(source.dimensions_m[0])
    camber = math.radians(camber_deg)
    total_turn = math.radians(curve_angle_deg)
    turn_sign = 1.0 if total_turn >= 0.0 else -1.0
    radius = original_length / abs(total_turn) if abs(total_turn) > 1e-12 else math.inf
    maximum_segment_length = original_length / curve_segments

    def path_pose(x: float) -> Tuple[float, float, float]:
        if not math.isfinite(radius):
            return x, 0.0, 0.0
        theta = x / radius
        return radius * math.sin(theta), turn_sign * radius * (1.0 - math.cos(theta)), turn_sign * theta

    def transform_point(point: Sequence[float]) -> List[float]:
        center_x, center_y, yaw = path_pose(float(point[0]))
        lateral = float(point[1])
        projected_lateral = lateral * math.cos(camber)
        return [
            center_x - math.sin(yaw) * projected_lateral,
            center_y + math.cos(yaw) * projected_lateral,
            float(point[2]) + lateral * math.sin(camber),
        ]

    def transform_orientation(rpy: Sequence[float], x: float) -> List[float]:
        _, _, yaw = path_pose(x)
        # Path yaw is global; camber is around the local direction of travel.
        combined = _multiply(_multiply(_rotation_z(yaw), _rotation_x(camber)), _rpy_matrix(rpy))
        return _matrix_rpy(combined)

    geometry: List[GeometrySpec] = []
    for item in source.geometry:
        split_count = 1
        if abs(total_turn) > 1e-12 and item.shape == "box":
            split_count = max(1, int(math.ceil(item.size[0] / maximum_segment_length)))
        original_rotation = _rpy_matrix(item.pose.rpy)
        for index in range(split_count):
            value = copy.deepcopy(item)
            if split_count > 1:
                value.name = "%s-curve-%03d" % (item.name, index)
                segment_pitch = item.size[0] / split_count
                heading_step = abs(total_turn) / split_count
                value.size[0] = segment_pitch + item.size[1] * math.tan(heading_step) + 0.005
                local_offset = -item.size[0] / 2.0 + (index + 0.5) * segment_pitch
                offset = _matvec(original_rotation, [local_offset, 0.0, 0.0])
                original_position = [item.pose.position[axis] + offset[axis] for axis in range(3)]
            else:
                original_position = list(item.pose.position)
            value.pose.position = transform_point(original_position)
            value.pose.rpy = transform_orientation(item.pose.rpy, original_position[0])
            value.parameters["geometry_modifier"] = {
                "camber_deg": camber_deg,
                "curve_angle_deg": curve_angle_deg,
                "source_geometry": item.name,
                "segment_index": index,
                "segment_count": split_count,
            }
            geometry.append(value)
    world.geometry = geometry

    transformed_regions: List[RegionSpec] = []
    for region in source.regions:
        samples = []
        for step in range(5):
            x = region.minimum[0] + step * (region.maximum[0] - region.minimum[0]) / 4.0
            for y in (region.minimum[1], region.maximum[1]):
                for z in (region.minimum[2], region.maximum[2]):
                    samples.append(transform_point([x, y, z]))
        transformed = copy.deepcopy(region)
        transformed.minimum = [min(point[axis] for point in samples) for axis in range(3)]
        transformed.maximum = [max(point[axis] for point in samples) for axis in range(3)]
        transformed.metadata["geometry_modifier"] = {"camber_deg": camber_deg, "curve_angle_deg": curve_angle_deg}
        transformed_regions.append(transformed)
    world.regions = transformed_regions

    for route in world.routes:
        route.waypoints = [transform_point(point) for point in route.waypoints]
    world.checkpoints = [transform_point(point) for point in world.checkpoints]

    def transform_pose(pose: Pose) -> Pose:
        original = copy.deepcopy(pose)
        return Pose(transform_point(original.position), transform_orientation(original.rpy, original.position[0]))

    world.spawn_pose = transform_pose(source.spawn_pose)
    world.goal_pose = transform_pose(source.goal_pose)

    bounds = [_oriented_bounds(item) for item in world.geometry if item.shape == "box"]
    minimum = [min(pair[0][axis] for pair in bounds) for axis in range(3)]
    maximum = [max(pair[1][axis] for pair in bounds) for axis in range(3)]
    world.dimensions_m = [max(maximum[0], world.spawn_pose.position[0], world.goal_pose.position[0]) + 0.01, max(maximum[1] - minimum[1], source.dimensions_m[1], 0.1), max(maximum[2] - minimum[2], source.dimensions_m[2], 0.1)]
    curve_direction = "straight" if abs(curve_angle_deg) < 1e-12 else ("left / anticlockwise" if curve_angle_deg > 0 else "right / clockwise")
    camber_direction = "level" if abs(camber_deg) < 1e-12 else ("left side high / right side low" if camber_deg > 0 else "left side low / right side high")
    modifier = {
        "camber_deg": camber_deg,
        "camber_direction": camber_direction,
        "curve_angle_deg": curve_angle_deg,
        "curve_direction": curve_direction,
        "curve_radius_m": None if not math.isfinite(radius) else radius,
        "curve_segments": curve_segments,
        "source_geometry_count": len(source.geometry),
        "resolved_geometry_count": len(world.geometry),
        "collision_geometry_transformed": True,
    }
    world.name += "-camber%+.1f-curve%+.1f" % (camber_deg, curve_angle_deg)
    world.generator_version = "1.2"
    world.normalized_parameters["camber_deg"] = camber_deg
    world.normalized_parameters["curve_angle_deg"] = curve_angle_deg
    world.resolved_parameters["geometry_modifier"] = modifier
    world.metadata["geometry_modifier"] = modifier
    world.validate()
    return world
