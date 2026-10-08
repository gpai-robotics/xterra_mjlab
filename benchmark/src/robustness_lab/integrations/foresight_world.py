"""Bridge a canonical Robustness Lab WorldSpec into FoReSight's terrain API.

This module has no Isaac imports, so its geometry and waypoint mapping can be
unit-tested on CPU.  The native playback launcher wraps it in FoReSight's
``ParkourSubTerrainBaseCfg`` only after Isaac Sim has started.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import trimesh

from ..io import load_data, sha256_file
from ..environment_physics import assess_environment_support, world_physics_plan
from ..models import WorldSpec


def material_face_tag(name: str, material: Any = None) -> List[int]:
    """Return an opaque RGBA tag preserved through trimesh concatenation."""
    identity = name
    if material is not None:
        identity += ":%.9g:%.9g:%.9g" % (
            float(material.static_friction), float(material.dynamic_friction), float(material.restitution)
        )
    digest = hashlib.sha256(("robustlab-material:" + identity).encode("utf-8")).digest()
    # Avoid very dark values that can be confused with trimesh defaults.
    return [32 + digest[0] % 192, 32 + digest[1] % 192, 32 + digest[2] % 192, 255]


def _box_mesh(item: Any, local_offset: Sequence[float], materials: Dict[str, Any]) -> trimesh.Trimesh:
    if item.shape != "box":
        raise ValueError("FoReSight native bridge currently supports box geometry; got %s" % item.shape)
    if item.dynamic:
        raise ValueError("FoReSight native bridge does not silently convert dynamic geometry to static")
    roll, pitch, yaw = item.pose.rpy
    transform = trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")
    transform[:3, 3] = np.asarray(item.pose.position, dtype=float) + np.asarray(local_offset, dtype=float)
    mesh = trimesh.creation.box(extents=np.asarray(item.size, dtype=float), transform=transform)
    mesh.visual.face_colors = np.repeat(np.asarray(material_face_tag(item.material, materials[item.material]), dtype=np.uint8)[None, :], len(mesh.faces), axis=0)
    return mesh


def _polyline_samples(anchors: List[np.ndarray], count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("FoReSight parkour integration requires at least two goals")
    lengths = [0.0]
    for left, right in zip(anchors, anchors[1:]):
        lengths.append(lengths[-1] + float(np.linalg.norm(right[:2] - left[:2])))
    if lengths[-1] <= 1e-9:
        return np.repeat(anchors[0][None, :], count, axis=0)
    targets = np.linspace(0.0, lengths[-1], count)
    result = []
    segment = 0
    for target in targets:
        while segment + 1 < len(lengths) - 1 and target > lengths[segment + 1]:
            segment += 1
        span = max(lengths[segment + 1] - lengths[segment], 1e-9)
        alpha = (target - lengths[segment]) / span
        result.append(anchors[segment] * (1.0 - alpha) + anchors[segment + 1] * alpha)
    return np.asarray(result)


def foresight_terrain_requirements(
    world_path: str,
    native_spawn_x: float = -7.0,
    minimum_size: Tuple[float, float] = (16.0, 4.0),
    margin_m: float = 0.5,
) -> Dict[str, Any]:
    """Resolve a native cell size and waypoint count for any canonical world.

    FoReSight's source task resets the robot at ``x=-7`` relative to the
    terrain origin.  A fixed 16 x 4 m cell therefore only works for the small
    demonstration worlds.  This calculation keeps that trained reset
    convention while expanding the cell around the canonical spawn so chains
    and graph worlds are not cropped.
    """
    world = WorldSpec.from_dict(load_data(str(Path(world_path).resolve())))
    spawn = np.asarray(world.spawn_pose.position, dtype=float)
    native_offset = np.asarray([native_spawn_x - spawn[0], -spawn[1], 0.0])
    meshes = [_box_mesh(item, native_offset, world.materials) for item in world.geometry]
    bounds = trimesh.util.concatenate(meshes).bounds
    half_x = max(abs(float(bounds[0, 0])), abs(float(bounds[1, 0]))) + margin_m
    half_y = max(abs(float(bounds[0, 1])), abs(float(bounds[1, 1]))) + margin_m
    size = (max(float(minimum_size[0]), 2.0 * half_x), max(float(minimum_size[1]), 2.0 * half_y))

    route = world.routes[0].waypoints if world.routes else world.checkpoints
    anchors = [np.asarray(world.spawn_pose.position, dtype=float)]
    anchors.extend(np.asarray(point, dtype=float) for point in route)
    anchors.append(np.asarray(world.goal_pose.position, dtype=float))
    length_m = sum(float(np.linalg.norm(right[:2] - left[:2])) for left, right in zip(anchors, anchors[1:]))
    # Keep goals no farther than about 2 m apart. Policies still observe only
    # the current and next goal, so this does not alter their input shape.
    num_goals = max(8, min(128, int(math.ceil(length_m / 2.0)) + 1))
    return {
        "size": [round(size[0], 6), round(size[1], 6)],
        "num_goals": num_goals,
        "route_length_m": length_m,
        "unshifted_bounds": bounds.tolist(),
    }


def world_to_foresight_terrain(
    world_path: str,
    num_goals: int,
    size: Tuple[float, float] = (16.0, 4.0),
    native_spawn_x: float = -7.0,
    horizontal_scale: float = 0.08,
    vertical_scale: float = 0.005,
):
    """Return FoReSight-compatible meshes, origin, goals and edge mask.

    FoReSight's source reset event places the Go2 at x=-7 m relative to the
    sub-terrain origin for its standard 16 x 4 m cell. The canonical world's
    declared spawn is aligned with that point; this avoids tuning geometry to
    a particular policy while retaining the policy's native observation stack.
    """
    world_file = Path(world_path).resolve()
    world = WorldSpec.from_dict(load_data(str(world_file)))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("terrain size must be positive")
    spawn = np.asarray(world.spawn_pose.position, dtype=float)
    centered_offset = np.asarray([native_spawn_x - spawn[0], -spawn[1], 0.0])
    generator_offset = centered_offset + np.asarray([size[0] * 0.5, size[1] * 0.5, 0.0])
    meshes = [_box_mesh(item, generator_offset, world.materials) for item in world.geometry]
    combined = trimesh.util.concatenate(meshes)
    bounds = combined.bounds
    if bounds[0, 0] < -1e-6 or bounds[1, 0] > size[0] + 1e-6 or bounds[0, 1] < -1e-6 or bounds[1, 1] > size[1] + 1e-6:
        raise ValueError("world does not fit the FoReSight terrain cell after spawn alignment; bounds=%s size=%s" % (bounds.tolist(), size))

    # Routes are ordered; graph-node checkpoints are not necessarily in
    # traversal order.  Prefer the canonical route so a waypoint-conditioned
    # policy follows the exact course selected by the benchmark.
    checkpoint_values = [np.asarray(value, dtype=float) for value in (world.routes[0].waypoints if world.routes else world.checkpoints)]
    spawn_anchor = spawn.copy()
    goal_anchor = np.asarray(world.goal_pose.position, dtype=float)
    # Spawn/goal pose z is the robot-base height. FoReSight's goal-height array
    # describes the terrain surface used for marker placement, so use the
    # nearest canonical checkpoint height at the two endpoints.
    if checkpoint_values:
        spawn_anchor[2] = checkpoint_values[0][2]
        goal_anchor[2] = checkpoint_values[-1][2]
    anchors = [spawn_anchor]
    anchors.extend(checkpoint_values)
    anchors.append(goal_anchor)
    deduplicated = [anchors[0]]
    for value in anchors[1:]:
        if np.linalg.norm(value[:2] - deduplicated[-1][:2]) > 1e-6:
            deduplicated.append(value)
    sampled = _polyline_samples(deduplicated, num_goals)
    goals = sampled[:, :2] + centered_offset[:2]
    goal_heights = np.rint(np.maximum(sampled[:, 2], 0.0) / vertical_scale).astype(np.int16)
    width_pixels = int(size[0] / horizontal_scale) + 1
    length_pixels = int(size[1] / horizontal_scale) + 1
    x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    origin = np.asarray([size[0] * 0.5, size[1] * 0.5, 0.0])
    return meshes, origin, goals, goal_heights, x_edge_mask


def bridge_manifest(world_path: str, **kwargs: Any) -> Dict[str, Any]:
    world = WorldSpec.from_dict(load_data(str(Path(world_path).resolve())))
    meshes, origin, goals, heights, mask = world_to_foresight_terrain(world_path, **kwargs)
    native_spawn_x = float(kwargs.get("native_spawn_x", -7.0))
    physics_plan = world_physics_plan(
        world,
        (native_spawn_x - float(world.spawn_pose.position[0]), -float(world.spawn_pose.position[1]), 0.0),
    )
    physics_support = assess_environment_support(physics_plan, {"linear_velocity_drag"})
    assignments: Dict[str, int] = {}
    for item in world.geometry:
        assignments[item.material] = assignments.get(item.material, 0) + 1
    per_region_equivalent = len(assignments) <= 1
    return {
        "schema_version": "1.0",
        "world": str(Path(world_path).resolve()),
        "world_sha256": sha256_file(str(Path(world_path).resolve())),
        "geometry_count": len(meshes),
        "triangle_count": int(sum(len(mesh.faces) for mesh in meshes)),
        "origin": origin.tolist(),
        "goals": goals.tolist(),
        "goal_heights_grid_units": heights.tolist(),
        "edge_mask_shape": list(mask.shape),
        "environment_physics": {"plan": physics_plan, "support": physics_support},
        "material_fidelity": {
            "mode": "single_global_material",
            "geometry_material_assignments": assignments,
            "face_tags": {name: material_face_tag(name, world.materials[name]) for name in assignments},
            "per_region_equivalent": per_region_equivalent,
            "capability_claim_eligible": per_region_equivalent,
        },
        "limitations": [
            "single global Isaac terrain material in the current FoReSight importer",
            "box collision geometry only",
            "RS11 follows its recorded physics tier; this FoReSight bridge supports legacy linear drag but not depth-resistive mud",
        ],
    }


def combine_native_result(bridge_path: str, summary_path: str, video_path: str = "") -> Dict[str, Any]:
    """Join geometry provenance and source-native playback output."""
    bridge = load_data(bridge_path)
    summary = load_data(summary_path)
    video = Path(video_path).resolve() if video_path else None
    return {
        "schema_version": "1.0",
        "status": "completed",
        "world": bridge["world"],
        "world_sha256": bridge["world_sha256"],
        "geometry_count": bridge["geometry_count"],
        "policy_checkpoint": summary.get("checkpoint"),
        "task": summary.get("task"),
        "seed": summary.get("seed"),
        "requested_seed": summary.get("requested_seed"),
        "effective_seed_scope": summary.get("effective_seed_scope"),
        "environment_index": summary.get("environment_index"),
        "assigned_world_index": summary.get("assigned_world_index"),
        "final_world_index": summary.get("final_world_index"),
        "world_switch_violations": summary.get("world_switch_violations"),
        "applied_stressors": summary.get("applied_stressors", []),
        "steps": summary.get("steps"),
        "success": bool(summary.get("success")),
        "max_goal_index": summary.get("max_goal_index"),
        "num_goals": summary.get("num_goals"),
        "episode_terminations": summary.get("episode_terminations"),
        "control_frequency_hz": summary.get("control_frequency_hz"),
        "evaluation_duration_s": summary.get("evaluation_duration_s"),
        "aggregate_path_distance_m": summary.get("aggregate_path_distance_m"),
        "mean_base_speed_mps": summary.get("mean_base_speed_mps"),
        "max_base_speed_mps": summary.get("max_base_speed_mps"),
        "max_abs_roll_rad": summary.get("max_abs_roll_rad"),
        "max_abs_pitch_rad": summary.get("max_abs_pitch_rad"),
        "final_base_position_m": summary.get("final_base_position_m"),
        "trajectory": summary.get("trajectory"),
        "telemetry_sample_count": summary.get("telemetry_sample_count"),
        "video": str(video) if video and video.is_file() else None,
        "video_sha256": sha256_file(str(video)) if video and video.is_file() else None,
        "bridge_manifest": str(Path(bridge_path).resolve()),
        "native_summary": str(Path(summary_path).resolve()),
        "claim_scope": "single external-policy integration run on canonical Robustness Lab geometry; not a robust success-rate estimate",
        "limitations": bridge.get("limitations", []),
        "material_fidelity": bridge.get("material_fidelity", {}),
        "material_binding_report": bridge.get("material_binding_report", {}),
        "environment_physics": bridge.get("environment_physics", {}),
    }
