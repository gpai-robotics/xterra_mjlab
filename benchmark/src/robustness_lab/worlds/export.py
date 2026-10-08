"""Simulator-neutral SVG preview and Wavefront OBJ collision export."""

from __future__ import annotations

import json

import html
import math
from pathlib import Path
from typing import Dict, List, Tuple

from ..io import atomic_write_text
from ..models import GeometrySpec, WorldSpec


def preview_svg(world: WorldSpec, output_path: str, width_px: int = 1400) -> None:
    margin = 40
    x_values = []
    y_values = []
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        x, y = geometry.pose.position[:2]
        x_values.extend([x - geometry.size[0] / 2.0, x + geometry.size[0] / 2.0])
        y_values.extend([y - geometry.size[1] / 2.0, y + geometry.size[1] / 2.0])
    minimum_x, maximum_x = min(x_values, default=0.0), max(x_values, default=1.0)
    minimum_y, maximum_y = min(y_values, default=-0.5), max(y_values, default=0.5)
    span_x, span_y = max(maximum_x - minimum_x, 1e-6), max(maximum_y - minimum_y, 1e-6)
    height_px = max(300, int((width_px - 2 * margin) * span_y / span_x) + 2 * margin)
    scale = min((width_px - 2 * margin) / span_x, (height_px - 2 * margin) / span_y)
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width_px, height_px, width_px, height_px), '<rect width="100%%" height="100%%" fill="#f7f8fa"/>']
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        x = margin + (geometry.pose.position[0] - geometry.size[0] / 2.0 - minimum_x) * scale
        y = height_px - margin - (geometry.pose.position[1] + geometry.size[1] / 2.0 - minimum_y) * scale
        display_material = str(geometry.parameters.get("display_material", geometry.material))
        color = world.materials.get(display_material).color if display_material in world.materials else "#777777"
        pieces.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s" stroke="#17202a" stroke-width="0.6"><title>%s (%s)</title></rect>' % (x, y, geometry.size[0] * scale, geometry.size[1] * scale, color, html.escape(geometry.name), html.escape(geometry.semantic)))
    for pose, color, label in ((world.spawn_pose, "#16a34a", "spawn"), (world.goal_pose, "#dc2626", "goal")):
        x = margin + (pose.position[0] - minimum_x) * scale
        y = height_px - margin - (pose.position[1] - minimum_y) * scale
        pieces.append('<circle cx="%.2f" cy="%.2f" r="7" fill="%s"><title>%s</title></circle>' % (x, y, color, label))
    pieces.append('<text x="%d" y="24" font-family="sans-serif" font-size="18">%s</text>' % (margin, html.escape(world.name)))
    pieces.append('</svg>')
    atomic_write_text(output_path, "\n".join(pieces) + "\n")


def _rotation_matrix(rpy: List[float]) -> List[List[float]]:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _box_vertices(geometry: GeometrySpec) -> List[Tuple[float, float, float]]:
    half = [value / 2.0 for value in geometry.size]
    rotation = _rotation_matrix(geometry.pose.rpy)
    vertices = []
    for x, y, z in ((-half[0], -half[1], -half[2]), (half[0], -half[1], -half[2]), (half[0], half[1], -half[2]), (-half[0], half[1], -half[2]), (-half[0], -half[1], half[2]), (half[0], -half[1], half[2]), (half[0], half[1], half[2]), (-half[0], half[1], half[2])):
        local = [x, y, z]
        vertices.append(tuple(geometry.pose.position[row] + sum(rotation[row][column] * local[column] for column in range(3)) for row in range(3)))
    return vertices


def _hex_to_rgb(color: str) -> tuple:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return (0.5, 0.5, 0.5)
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def export_obj(world: WorldSpec, output_path: str) -> None:
    """Write a mesh any engine can load, with the physics it cannot carry beside it.

    OBJ holds shape only. Friction, compliance and "this patch is mud" have no
    place in the format, so every object is tagged with its material name
    (``usemtl``) -- which every engine can assign physics to per group -- and
    two files are written beside the mesh:

    * ``<name>.mtl``            colours per material, so viewers show the terrain
    * ``<name>.materials.json`` the actual numbers per material, plus each
                                object's semantic tag and parameters (the mud
                                model for marsh patches)

    ``.obj`` + ``.materials.json`` is a complete, repository-independent
    description of the terrain. Without the tags a slope at mu = 0.25 and the
    same slope at mu = 0.9 were byte-identical files.
    """

    output = Path(output_path)
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    mtl_name = stem + ".mtl"
    sidecar_path = output.with_name(stem + ".materials.json")

    lines = ["# robustness-lab WorldSpec %s" % world.name, "mtllib %s" % mtl_name]
    vertex_offset = 1
    faces = ((1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (5, 1, 4, 8))
    objects = []
    # MuJoCo loads exactly one shape per OBJ file -- verified: a 35-box world
    # exported as one multi-object file loaded all 280 vertices but only the
    # first object's 12 faces. So alongside the tagged combined mesh, one
    # single-shape file is written per material; each loads in full, and
    # grouping by material is how physics gets assigned after loading anyway.
    per_material: Dict[str, List[str]] = {}
    per_material_offset: Dict[str, int] = {}
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        name = geometry.name.replace(" ", "_")
        lines.append("o %s" % name)
        lines.append("usemtl %s" % geometry.material)
        vertices = list(_box_vertices(geometry))
        for vertex in vertices:
            lines.append("v %.9g %.9g %.9g" % vertex)
        for face in faces:
            lines.append("f " + " ".join(str(vertex_offset + index - 1) for index in face))
        vertex_offset += 8
        bucket = per_material.setdefault(geometry.material, ["# %s -- material %s" % (world.name, geometry.material)])
        base = per_material_offset.get(geometry.material, 1)
        for vertex in vertices:
            bucket.append("v %.9g %.9g %.9g" % vertex)
        for face in faces:
            bucket.append("f " + " ".join(str(base + index - 1) for index in face))
        per_material_offset[geometry.material] = base + 8
        objects.append(
            {
                "name": name,
                "material": geometry.material,
                "semantic": geometry.semantic,
                "parameters": dict(geometry.parameters or {}),
            }
        )
    atomic_write_text(str(output), "\n".join(lines) + "\n")
    mesh_files = {}
    for material, bucket in per_material.items():
        mesh_name = "%s.%s.obj" % (stem, material)
        atomic_write_text(str(output.with_name(mesh_name)), "\n".join(bucket) + "\n")
        mesh_files[material] = mesh_name

    materials = {}
    for key, spec in world.materials.items():
        materials[str(key)] = {
            "static_friction": float(spec.static_friction),
            "dynamic_friction": float(spec.dynamic_friction),
            "restitution": float(spec.restitution),
            "roughness": float(spec.roughness),
            "compliance": float(spec.compliance),
            "damping": float(spec.damping),
            "color": str(spec.color),
            "mesh": mesh_files.get(str(key)),
        }
    mtl = ["# materials for %s" % world.name]
    for key, values in materials.items():
        r, g, b = _hex_to_rgb(values["color"])
        mtl += ["newmtl %s" % key, "Kd %.4f %.4f %.4f" % (r, g, b), "Ka %.4f %.4f %.4f" % (r, g, b), ""]
    atomic_write_text(str(output.with_name(mtl_name)), "\n".join(mtl) + "\n")

    sidecar = {
        "schema_version": "1.0",
        "world": world.name,
        "terrain_type": world.terrain_type,
        "units": {"length": "m", "friction": "coefficient", "compliance": "s (MuJoCo solref timeconst)",
                  "stress": "Pa", "angle": "rad"},
        "spawn_pose": {"position": list(world.spawn_pose.position), "rpy": list(world.spawn_pose.rpy or [0.0, 0.0, 0.0])},
        "goal_pose": {"position": list(world.goal_pose.position), "rpy": list(world.goal_pose.rpy or [0.0, 0.0, 0.0])},
        "geometry_modifier": dict(world.metadata.get("geometry_modifier", {})),
        "materials": materials,
        "objects": objects,
    }
    atomic_write_text(str(sidecar_path), json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
