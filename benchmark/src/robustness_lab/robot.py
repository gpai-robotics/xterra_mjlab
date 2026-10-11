"""Robot-description ingestion and cross-format conformance auditing."""

from __future__ import annotations

import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io import load_data, sha256_file
from .models import ActuatorSpec, JointSpec, RobotProfile


FOOT_PATTERN = re.compile(r"(foot|toe|ankle|calf.*end|lower.*end)", re.IGNORECASE)
BASE_PATTERN = re.compile(r"(base|trunk|torso|body)", re.IGNORECASE)


def _floats(text: Optional[str], default: Iterable[float]) -> List[float]:
    if not text:
        return [float(value) for value in default]
    return [float(value) for value in text.split()]


def _norm_name(name: str) -> str:
    value = name.lower()
    for old, new in (("front", "f"), ("rear", "r"), ("hind", "r"), ("left", "l"), ("right", "r")):
        value = value.replace(old, new)
    return re.sub(r"[^a-z0-9]", "", value.replace("joint", "").replace("link", ""))


def _collision_aabb(size: List[float], xyz: List[float], rpy: List[float]) -> Dict[str, List[float]]:
    """Axis-aligned bounds of a locally centered primitive after its URDF pose."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    half = [abs(value) / 2.0 for value in size]
    extent = [sum(abs(rotation[row][column]) * half[column] for column in range(3)) for row in range(3)]
    return {
        "minimum": [xyz[index] - extent[index] for index in range(3)],
        "maximum": [xyz[index] + extent[index] for index in range(3)],
    }


def _inertia_body_frame(record: Dict[str, Any]) -> Optional[List[List[float]]]:
    inertia = record.get("matrix_kg_m2")
    if not inertia:
        return None
    if "quaternion_wxyz" in record:
        w, x, y, z = [float(value) for value in record["quaternion_wxyz"]]
        norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        rotation = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    else:
        roll, pitch, yaw = [float(value) for value in record.get("origin_rpy_rad", [0.0, 0.0, 0.0])]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    rotated = [[sum(rotation[row][inner] * float(inertia[inner][column]) for inner in range(3)) for column in range(3)] for row in range(3)]
    return [[sum(rotated[row][inner] * rotation[column][inner] for inner in range(3)) for column in range(3)] for row in range(3)]


def _resolve_mesh(filename: str, source: Path, package_paths: List[str]) -> Optional[Path]:
    if filename.startswith("package://"):
        relative = filename[len("package://") :]
        candidates = []
        for root in package_paths:
            candidates.append(Path(root) / relative)
            parts = Path(relative).parts
            if len(parts) > 1:
                candidates.append(Path(root) / Path(*parts[1:]))
    else:
        candidates = [source.parent / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _mesh_size(path: Path, scale: List[float]) -> Optional[List[float]]:
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh", process=False)
        extents = mesh.bounding_box.extents
        return [float(extents[index]) * scale[index] for index in range(3)]
    except (ImportError, ModuleNotFoundError, ValueError, OSError):
        return None


def _collision_size(collision: ET.Element, source: Path, package_paths: List[str]) -> Tuple[Optional[List[float]], Optional[str]]:
    geometry = collision.find("geometry")
    if geometry is None:
        return None, "collision has no geometry"
    box = geometry.find("box")
    if box is not None:
        return _floats(box.get("size"), [0.0, 0.0, 0.0]), None
    sphere = geometry.find("sphere")
    if sphere is not None:
        diameter = 2.0 * float(sphere.get("radius", "0"))
        return [diameter, diameter, diameter], None
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        diameter = 2.0 * float(cylinder.get("radius", "0"))
        return [diameter, diameter, float(cylinder.get("length", "0"))], None
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename", "")
        scale = _floats(mesh.get("scale"), [1.0, 1.0, 1.0])
        path = _resolve_mesh(filename, source, package_paths)
        if path is None:
            return None, "unresolved collision mesh: %s" % filename
        size = _mesh_size(path, scale)
        if size is None:
            return None, "could not measure collision mesh: %s" % path
        return size, None
    return None, "unsupported collision geometry"


def _load_override(path: Optional[str]) -> Dict[str, Any]:
    return load_data(path) if path else {}


def _apply_override(profile: RobotProfile, override: Dict[str, Any]) -> RobotProfile:
    protected = {"source_path", "source_format", "source_sha256", "schema_version"}
    for key, value in override.items():
        if key in protected or not hasattr(profile, key):
            continue
        if key in {"default_joint_positions", "joint_limits"}:
            merged = dict(getattr(profile, key))
            merged.update(value)
            setattr(profile, key, merged)
        else:
            setattr(profile, key, value)
    if "body_size_m" in override and "collision_bounds_m" not in override:
        half = [float(value) / 2.0 for value in profile.body_size_m]
        profile.collision_bounds_m = {"minimum": [-value for value in half], "maximum": half}
    profile.validate()
    return profile


def _position_is_legal(value: float, bounds: Optional[Tuple[float, float]]) -> bool:
    return math.isfinite(value) and (bounds is None or bounds[0] <= value <= bounds[1])


def _derived_default(bounds: Optional[Tuple[float, float]]) -> float:
    if bounds is None or bounds[0] <= 0.0 <= bounds[1]:
        return 0.0
    return (bounds[0] + bounds[1]) / 2.0


def _default_joint_positions(
    controller_joints: List[str],
    joint_limits: Dict[str, Tuple[float, float]],
    explicit: Optional[Dict[str, float]] = None,
    mimic: Optional[Dict[str, Tuple[str, float, float]]] = None,
) -> Dict[str, float]:
    """Choose legal, non-boundary defaults for every controller joint."""
    explicit = explicit or {}
    positions = {}
    for name in controller_joints:
        candidate = explicit.get(name)
        bounds = joint_limits.get(name)
        positions[name] = (
            float(candidate)
            if candidate is not None and _position_is_legal(float(candidate), bounds)
            else _derived_default(bounds)
        )

    unresolved = dict(mimic or {})
    while unresolved:
        progress = False
        for name, (source, multiplier, offset) in list(unresolved.items()):
            if source not in positions or source in unresolved:
                continue
            candidate = multiplier * positions[source] + offset
            if _position_is_legal(candidate, joint_limits.get(name)):
                positions[name] = candidate
            del unresolved[name]
            progress = True
        if not progress:
            break
    return positions


def validate_default_pose(profile: RobotProfile) -> Dict[str, Any]:
    """Report missing, non-finite, or out-of-range controller joint defaults."""
    violations = []
    for joint in profile.controller_joint_order:
        bounds = profile.joint_limits.get(joint)
        value = profile.default_joint_positions.get(joint)
        if value is None:
            violations.append({"joint": joint, "value": None, "bounds": bounds, "reason": "missing default position"})
        elif not math.isfinite(float(value)):
            violations.append({"joint": joint, "value": value, "bounds": bounds, "reason": "default position is not finite"})
        elif bounds is not None and not bounds[0] <= float(value) <= bounds[1]:
            violations.append({"joint": joint, "value": float(value), "bounds": bounds, "reason": "default position is outside joint limits"})
    return {"status": "blocked" if violations else "ready", "violations": violations}


def inspect_urdf(path: str, override_path: Optional[str] = None, package_paths: Optional[List[str]] = None) -> RobotProfile:
    source = Path(path).resolve()
    root = ET.parse(str(source)).getroot()
    package_paths = package_paths or [str(source.parent), str(source.parent.parent)]
    override = _load_override(override_path)
    links = [item.get("name", "") for item in root.findall("link")]
    child_links = set()
    joints = []
    declared_defaults = {}
    mimic_defaults = {}
    origins = {}
    parents = {}
    for element in root.findall("joint"):
        parent_element = element.find("parent")
        child_element = element.find("child")
        parent = parent_element.get("link", "") if parent_element is not None else ""
        child = child_element.get("link", "") if child_element is not None else ""
        child_links.add(child)
        parents[child] = parent
        origin = element.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
        origins[child] = xyz
        axis_element = element.find("axis")
        axis = _floats(axis_element.get("xyz") if axis_element is not None else None, [0.0, 0.0, 1.0])
        limit = element.find("limit")
        joint_name = element.get("name", "")
        declared = element.find("default")
        declared_value = next(
            (
                value
                for value in (
                    element.get("initial_position"),
                    element.get("position"),
                    element.get("value"),
                    declared.get("position") if declared is not None else None,
                    declared.get("value") if declared is not None else None,
                )
                if value is not None
            ),
            None,
        )
        if declared_value is not None:
            declared_defaults[joint_name] = float(declared_value)
        mimic = element.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic_defaults[joint_name] = (
                mimic.get("joint", ""),
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            )
        joints.append(
            JointSpec(
                name=joint_name,
                parent=parent,
                child=child,
                joint_type=element.get("type", "fixed"),
                axis=axis,
                origin_xyz=xyz,
                lower=float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None,
                upper=float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None,
                effort=float(limit.get("effort")) if limit is not None and limit.get("effort") is not None else None,
                velocity=float(limit.get("velocity")) if limit is not None and limit.get("velocity") is not None else None,
            )
        )
    roots = [link for link in links if link not in child_links]
    base_link = override.get("base_link") or next((item for item in roots if BASE_PATTERN.search(item)), roots[0] if roots else "")
    foot_links = override.get("foot_links") or [link for link in links if FOOT_PATTERN.search(link)]
    if not foot_links:
        parent_names = set(parents.values())
        foot_links = [link for link in links if link not in parent_names and link != base_link]

    masses = []
    weighted_com = [0.0, 0.0, 0.0]
    collision_by_link = {}
    collision_bounds_by_link = {}
    link_inertias = {}
    warnings = []
    for link in root.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            mass = float(inertial.find("mass").get("value", "0"))
            inertial_origin = inertial.find("origin")
            com = _floats(inertial_origin.get("xyz") if inertial_origin is not None else None, [0.0, 0.0, 0.0])
            inertia = inertial.find("inertia")
            link_inertias[name] = {
                "mass_kg": mass,
                "origin_xyz_m": com,
                "origin_rpy_rad": _floats(inertial_origin.get("rpy") if inertial_origin is not None else None, [0.0, 0.0, 0.0]),
                "matrix_kg_m2": [
                    [float(inertia.get("ixx", "0")) if inertia is not None else 0.0, float(inertia.get("ixy", "0")) if inertia is not None else 0.0, float(inertia.get("ixz", "0")) if inertia is not None else 0.0],
                    [float(inertia.get("ixy", "0")) if inertia is not None else 0.0, float(inertia.get("iyy", "0")) if inertia is not None else 0.0, float(inertia.get("iyz", "0")) if inertia is not None else 0.0],
                    [float(inertia.get("ixz", "0")) if inertia is not None else 0.0, float(inertia.get("iyz", "0")) if inertia is not None else 0.0, float(inertia.get("izz", "0")) if inertia is not None else 0.0],
                ],
            }
            masses.append(mass)
            for index in range(3):
                weighted_com[index] += mass * com[index]
        bounds = []
        for collision in link.findall("collision"):
            size, warning = _collision_size(collision, source, package_paths)
            if size and all(value > 0 for value in size):
                origin = collision.find("origin")
                xyz = _floats(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
                rpy = _floats(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
                bounds.append(_collision_aabb(size, xyz, rpy))
            if warning:
                warnings.append("%s: %s" % (name, warning))
        if bounds:
            minimum = [min(item["minimum"][index] for item in bounds) for index in range(3)]
            maximum = [max(item["maximum"][index] for item in bounds) for index in range(3)]
            collision_bounds_by_link[name] = {"minimum": minimum, "maximum": maximum}
            collision_by_link[name] = [maximum[index] - minimum[index] for index in range(3)]
    total_mass = sum(masses)
    center_of_mass = [value / total_mass for value in weighted_com] if total_mass > 0 else [0.0, 0.0, 0.0]
    body_size = collision_by_link.get(base_link)
    body_bounds = collision_bounds_by_link.get(base_link)
    if body_size is None and collision_by_link:
        body_link = max(collision_by_link, key=lambda key: math.prod(collision_by_link[key]))
        body_size = collision_by_link[body_link]
        body_bounds = collision_bounds_by_link[body_link]
        warnings.append("base collision missing; used largest collision geometry for body size")
    if body_size is None:
        body_size = [0.5, 0.2, 0.2]
        body_bounds = {"minimum": [-0.25, -0.1, -0.1], "maximum": [0.25, 0.1, 0.1]}
        warnings.append("no measurable collision geometry; using conservative body-size fallback")

    def absolute_origin(link: str) -> List[float]:
        position = [0.0, 0.0, 0.0]
        seen = set()
        current = link
        while current in origins and current not in seen:
            seen.add(current)
            for index in range(3):
                position[index] += origins[current][index]
            current = parents.get(current, "")
        return position

    foot_positions = [absolute_origin(link) for link in foot_links]
    stance_length = max((point[0] for point in foot_positions), default=0.2) - min((point[0] for point in foot_positions), default=-0.2)
    stance_width = max((point[1] for point in foot_positions), default=0.125) - min((point[1] for point in foot_positions), default=-0.125)
    standing_height = max(abs(point[2]) for point in foot_positions) if foot_positions else max(body_size[2] * 2.0, 0.3)

    limb_reaches = []
    for foot in foot_links:
        length = 0.0
        seen = set()
        current = foot
        while current in origins and current not in seen:
            seen.add(current)
            length += math.sqrt(sum(value * value for value in origins[current]))
            current = parents.get(current, "")
            if current == base_link:
                break
        if length > 0:
            limb_reaches.append(length)
    foot_sizes = [collision_by_link[link] for link in foot_links if link in collision_by_link]
    foot_size = [max((size[index] for size in foot_sizes), default=value) for index, value in enumerate([0.05, 0.04, 0.03])]
    actuators = [
        ActuatorSpec(joint=joint.name, name=joint.name, effort_limit=joint.effort, velocity_limit=joint.velocity)
        for joint in joints
        if joint.joint_type not in {"fixed", "floating"}
    ]
    controller_joint_order = [item.name for item in joints if item.joint_type not in {"fixed", "floating"}]
    joint_limits = {
        joint.name: (float(joint.lower), float(joint.upper))
        for joint in joints
        if joint.joint_type != "continuous" and joint.lower is not None and joint.upper is not None
    }
    default_joint_positions = _default_joint_positions(
        controller_joint_order, joint_limits, declared_defaults, mimic_defaults
    )
    profile = RobotProfile(
        name=override.get("name", root.get("name", source.stem)),
        source_path=str(source),
        source_format="urdf",
        source_sha256=sha256_file(str(source)),
        provenance={"original_path": path, "package_paths": package_paths},
        links=links,
        joints=joints,
        actuators=actuators,
        link_inertias=link_inertias,
        base_link=base_link,
        foot_links=foot_links,
        total_mass_kg=total_mass,
        center_of_mass_m=center_of_mass,
        body_size_m=body_size,
        collision_bounds_m=body_bounds,
        standing_height_m=max(standing_height, 0.1),
        stance_length_m=max(stance_length, body_size[0] * 0.5),
        stance_width_m=max(stance_width, body_size[1]),
        foot_size_m=foot_size,
        limb_reach_m=max(limb_reaches, default=standing_height),
        joint_limits=joint_limits,
        default_joint_positions=default_joint_positions,
        controller_joint_order=controller_joint_order,
        inference_confidence={"mass": 0.9 if masses else 0.0, "body_size": 0.9 if collision_by_link else 0.2, "kinematics": 0.7},
        warnings=sorted(set(warnings)),
    )
    if total_mass <= 0:
        profile.warnings.append("URDF contains no positive inertial masses")
    if masses:
        profile.warnings.append("reported COM combines link-local inertial origins; use a nominal-pose dynamics engine for whole-body COM")
    return _apply_override(profile, override)


def _mjcf_range(element: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    values = _floats(element.get("range"), [])
    if len(values) == 2:
        return values[0], values[1]
    return None, None


def _mjcf_defaults(root: ET.Element) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Resolve MJCF nested defaults for the fields used by conformance."""
    resolved: Dict[str, Dict[str, Dict[str, str]]] = {}

    def visit(node: ET.Element, inherited: Dict[str, Dict[str, str]]) -> None:
        current = {tag: dict(values) for tag, values in inherited.items()}
        for child in list(node):
            if child.tag != "default":
                current.setdefault(child.tag, {}).update(child.attrib)
        class_name = node.get("class", "")
        resolved[class_name] = {tag: dict(values) for tag, values in current.items()}
        for child in node.findall("default"):
            visit(child, current)

    root_default = root.find("default")
    if root_default is not None:
        visit(root_default, {})
    return resolved


def inspect_mjcf(path: str, override_path: Optional[str] = None) -> RobotProfile:
    source = Path(path).resolve()
    root = ET.parse(str(source)).getroot()
    override = _load_override(override_path)
    worldbody = root.find("worldbody")
    bodies = list(worldbody.iter("body")) if worldbody is not None else []
    defaults = _mjcf_defaults(root)
    compiler = root.find("compiler")
    angle_unit = compiler.get("angle", "degree").lower() if compiler is not None else "degree"
    angle_scale = math.pi / 180.0 if angle_unit == "degree" else 1.0
    links = [body.get("name", "") for body in bodies if body.get("name")]
    body_by_name = {body.get("name", ""): body for body in bodies}
    base_link = override.get("base_link") or next((name for name in links if BASE_PATTERN.search(name)), links[0] if links else "")
    foot_links = override.get("foot_links") or [name for name in links if FOOT_PATTERN.search(name)]
    if not foot_links:
        # MJCF commonly models a foot as a collision geom on the distal leg
        # body instead of giving the body itself a foot-like name.  Select the
        # owning body of each contact-enabled, foot-marked geom in document
        # order.  This maps (for example) Go2's FL/FR/RL/RR ``class="foot"``
        # geoms to the corresponding distal calf bodies used by MuJoCo APIs.
        for body in bodies:
            for geom in body.findall("geom"):
                class_name = geom.get("class", body.get("childclass", ""))
                attributes = dict(defaults.get(class_name, {}).get("geom", {}))
                attributes.update(geom.attrib)
                marker = " ".join(
                    attributes.get(key, "") for key in ("name", "class", "mesh")
                )
                if (
                    FOOT_PATTERN.search(marker)
                    and int(attributes.get("contype", "1")) != 0
                ):
                    name = body.get("name", "")
                    if name and name not in foot_links:
                        foot_links.append(name)
                    break
    if not foot_links:
        # Match the URDF structural fallback: absent semantic names, terminal
        # bodies are the deepest bodies in their articulated chains.
        parent_names = {
            body.get("name", "") for body in bodies if body.find("body") is not None
        }
        foot_links = [
            name for name in links if name not in parent_names and name != base_link
        ]
    warnings = []
    masses = []
    weighted_com = [0.0, 0.0, 0.0]
    link_inertias = {}
    for body in bodies:
        inertial = body.find("inertial")
        if inertial is None or inertial.get("mass") is None:
            continue
        mass = float(inertial.get("mass", "0"))
        position = _floats(inertial.get("pos"), [0.0, 0.0, 0.0])
        diagonal = _floats(inertial.get("diaginertia"), [])
        full = _floats(inertial.get("fullinertia"), [])
        if len(full) == 6:
            inertia_matrix = [[full[0], full[3], full[4]], [full[3], full[1], full[5]], [full[4], full[5], full[2]]]
        elif len(diagonal) == 3:
            inertia_matrix = [[diagonal[0], 0.0, 0.0], [0.0, diagonal[1], 0.0], [0.0, 0.0, diagonal[2]]]
        else:
            inertia_matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        link_inertias[body.get("name", "")] = {
            "mass_kg": mass,
            "origin_xyz_m": position,
            "quaternion_wxyz": _floats(inertial.get("quat"), [1.0, 0.0, 0.0, 0.0]),
            "matrix_kg_m2": inertia_matrix,
        }
        masses.append(mass)
        for index in range(3):
            weighted_com[index] += mass * position[index]
    total_mass = sum(masses)
    center_of_mass = [value / total_mass for value in weighted_com] if total_mass else [0.0, 0.0, 0.0]
    joints = []
    qpos_layout = []
    qpos0_defaults = {}

    def walk_bodies(parent: ET.Element, parent_name: str = "", inherited_childclass: str = "") -> None:
        for body in parent.findall("body"):
            body_name = body.get("name", "")
            childclass = body.get("childclass", inherited_childclass)
            for element in list(body):
                if element.tag == "freejoint":
                    qpos_layout.append(("", "free", 7))
                    continue
                if element.tag != "joint":
                    continue
                class_name = element.get("class", childclass)
                attributes = dict(defaults.get(class_name, {}).get("joint", {}))
                attributes.update(element.attrib)
                range_values = _floats(attributes.get("range"), [])
                joint_type = attributes.get("type", "hinge")
                if joint_type == "hinge":
                    range_values = [value * angle_scale for value in range_values]
                lower, upper = (range_values[0], range_values[1]) if len(range_values) == 2 else (None, None)
                joint_name = attributes.get("name", "")
                qpos_width = 4 if joint_type == "ball" else 1
                qpos_layout.append((joint_name, joint_type, qpos_width))
                if attributes.get("ref") is not None and joint_type in {"hinge", "slide"}:
                    reference = float(attributes["ref"])
                    qpos0_defaults[joint_name] = reference * angle_scale if joint_type == "hinge" else reference
                joints.append(
                    JointSpec(
                        name=joint_name,
                        parent=parent_name,
                        child=body_name,
                        joint_type=joint_type,
                        axis=_floats(attributes.get("axis"), [0.0, 0.0, 1.0]),
                        origin_xyz=_floats(body.get("pos"), [0.0, 0.0, 0.0]),
                        lower=lower,
                        upper=upper,
                    )
                )
            walk_bodies(body, body_name, childclass)

    if worldbody is not None:
        walk_bodies(worldbody)

    qpos_values = _floats(root.get("qpos0"), [])
    keyframe = root.find("keyframe")
    if keyframe is not None:
        keys = [key for key in keyframe.findall("key") if key.get("qpos")]
        preferred = next((key for key in keys if key.get("name", "").lower() in {"home", "default", "stand", "standing"}), None)
        selected = preferred or (keys[0] if keys else None)
        if selected is not None:
            qpos_values = _floats(selected.get("qpos"), [])
    if qpos_values:
        offset = 0
        for joint_name, joint_type, width in qpos_layout:
            if offset + width > len(qpos_values):
                break
            if width == 1 and joint_name:
                # MJCF keyframe qpos values are already in MuJoCo's runtime
                # coordinates (radians for hinge joints), independent of the
                # compiler angle setting used for XML angular attributes.
                qpos0_defaults[joint_name] = qpos_values[offset]
            offset += width
    actuators = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for element in list(actuator_root):
            class_name = element.get("class", "")
            attributes = dict(defaults.get(class_name, {}).get(element.tag, defaults.get(class_name, {}).get("motor", {})))
            attributes.update(element.attrib)
            control_range = _floats(attributes.get("ctrlrange"), [])
            lower, upper = (control_range[0], control_range[1]) if len(control_range) == 2 else (None, None)
            force_range = _floats(attributes.get("forcerange"), [])
            effort = max(abs(value) for value in (force_range or control_range)) if (force_range or control_range) else None
            actuators.append(
                ActuatorSpec(
                    name=attributes.get("name", attributes.get("joint", "")),
                    joint=attributes.get("joint", ""),
                    effort_limit=effort,
                    control_lower=lower,
                    control_upper=upper,
                )
            )
    controller_joint_order = [actuator.joint for actuator in actuators]
    joint_limits = {
        joint.name: (float(joint.lower), float(joint.upper))
        for joint in joints
        if joint.lower is not None and joint.upper is not None
    }
    default_joint_positions = _default_joint_positions(
        controller_joint_order, joint_limits, qpos0_defaults
    )
    body_size = None
    body_bounds = None
    base = body_by_name.get(base_link)
    if base is not None:
        for geom in base.findall("geom"):
            if geom.get("type", "sphere") == "box":
                half = _floats(geom.get("size"), [])
                if len(half) == 3:
                    candidate = [2.0 * value for value in half]
                    if body_size is None or math.prod(candidate) > math.prod(body_size):
                        body_size = candidate
                        position = _floats(geom.get("pos"), [0.0, 0.0, 0.0])
                        body_bounds = _collision_aabb(candidate, position, _floats(geom.get("euler"), [0.0, 0.0, 0.0]))
    if body_size is None:
        body_size = [0.5, 0.2, 0.2]
        body_bounds = {"minimum": [-0.25, -0.1, -0.1], "maximum": [0.25, 0.1, 0.1]}
        warnings.append("MJCF base has no explicit box geom; body size requires override for high-confidence scaling")
    positions = [_floats(body_by_name[name].get("pos"), [0.0, 0.0, 0.0]) for name in foot_links if name in body_by_name]
    stance_length = max((p[0] for p in positions), default=0.2) - min((p[0] for p in positions), default=-0.2)
    stance_width = max((p[1] for p in positions), default=0.125) - min((p[1] for p in positions), default=-0.125)
    profile = RobotProfile(
        name=override.get("name", root.get("model", source.stem)),
        source_path=str(source),
        source_format="mjcf",
        source_sha256=sha256_file(str(source)),
        provenance={"original_path": path},
        links=links,
        joints=joints,
        actuators=actuators,
        link_inertias=link_inertias,
        base_link=base_link,
        foot_links=foot_links,
        total_mass_kg=total_mass,
        center_of_mass_m=center_of_mass,
        body_size_m=body_size,
        collision_bounds_m=body_bounds,
        standing_height_m=float(override.get("standing_height_m", 0.4)),
        stance_length_m=max(stance_length, body_size[0] * 0.5),
        stance_width_m=max(stance_width, body_size[1]),
        foot_size_m=list(override.get("foot_size_m", [0.05, 0.04, 0.03])),
        limb_reach_m=float(override.get("limb_reach_m", 0.35)),
        joint_limits=joint_limits,
        default_joint_positions=default_joint_positions,
        controller_joint_order=controller_joint_order,
        inference_confidence={"mass": 0.9 if masses else 0.0, "body_size": 0.7 if base is not None else 0.2, "kinematics": 0.5},
        warnings=warnings,
    )
    return _apply_override(profile, override)


def inspect_usd(path: str, override_path: Optional[str] = None) -> RobotProfile:
    source = Path(path).resolve()
    override = _load_override(override_path)
    warnings = ["USD inspection requires explicit morphology overrides unless run inside an Isaac/pxr environment"]
    profile = RobotProfile(
        name=override.get("name", source.stem),
        source_path=str(source),
        source_format="usd",
        source_sha256=sha256_file(str(source)),
        provenance={"original_path": path},
        base_link=override.get("base_link", "base"),
        foot_links=override.get("foot_links", []),
        warnings=warnings,
        inference_confidence={"mass": 0.0, "body_size": 0.1, "kinematics": 0.0},
    )
    return _apply_override(profile, override)


def inspect_robot(path: str, source_format: Optional[str] = None, override_path: Optional[str] = None, package_paths: Optional[List[str]] = None) -> RobotProfile:
    detected = (source_format or Path(path).suffix.lstrip(".")).lower()
    if detected == "xml":
        detected = "mjcf"
    if detected == "urdf":
        return inspect_urdf(path, override_path, package_paths)
    if detected == "mjcf":
        return inspect_mjcf(path, override_path)
    if detected in {"usd", "usda", "usdc"}:
        return inspect_usd(path, override_path)
    raise ValueError("cannot inspect robot format: %s" % detected)


def convert_urdf_to_usd(urdf_path: str, output_path: str, launcher: str = "/usr/local/bin/isaaclab", stiffness: float = 25.0, damping: float = 0.5, timeout_s: float = 300.0) -> Dict[str, Any]:
    source = Path(urdf_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    isaaclab_root = os.environ.get("ROBUSTLAB_ISAACLAB_ROOT")
    if not isaaclab_root:
        raise FileNotFoundError("Isaac Lab URDF converter unavailable: ROBUSTLAB_ISAACLAB_ROOT is not set")
    script = Path(isaaclab_root) / "scripts" / "tools" / "convert_urdf.py"
    if not script.exists():
        raise FileNotFoundError("Isaac Lab URDF converter not found at %s" % script)
    command = [launcher, "-p", str(script), str(source), str(output), "--headless", "--joint-stiffness", str(stiffness), "--joint-damping", str(damping), "--joint-target-type", "position"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        return {"status": "error", "reason": "conversion timeout", "error": str(error), "command": command}
    status = "converted" if result.returncode == 0 and output.exists() else "error"
    return {"status": status, "output": str(output), "command": command, "returncode": result.returncode, "stdout": result.stdout[-10000:], "stderr": result.stderr[-10000:], "sha256": sha256_file(str(output)) if output.exists() else None}


def compare_profiles(profiles: List[RobotProfile], tolerances: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if len(profiles) < 2:
        raise ValueError("at least two robot profiles are required")
    limits = {"mass_relative": 0.02, "dimension_relative": 0.02, "com_m": 0.005, "inertia_relative": 0.05, "joint_origin_m": 0.005, "joint_limit_rad": math.radians(1.0), "joint_axis_rad": math.radians(1.0), "actuator_relative": 0.05}
    limits.update(tolerances or {})
    reference = profiles[0]
    comparisons = []
    comparable = True
    for candidate in profiles[1:]:
        issues = []
        if len(reference.links) != len(candidate.links):
            issues.append({"field": "link_count", "reference": len(reference.links), "candidate": len(candidate.links)})
        if reference.total_mass_kg > 0 and candidate.total_mass_kg > 0:
            mass_error = abs(candidate.total_mass_kg - reference.total_mass_kg) / reference.total_mass_kg
            if mass_error > limits["mass_relative"]:
                issues.append({"field": "total_mass_kg", "error": mass_error, "limit": limits["mass_relative"]})
        else:
            issues.append({"field": "total_mass_kg", "error": "missing"})
        for index, axis in enumerate("xyz"):
            reference_dimension = reference.body_size_m[index]
            error = abs(candidate.body_size_m[index] - reference_dimension) / max(reference_dimension, 1e-9)
            if error > limits["dimension_relative"]:
                issues.append({"field": "body_size_%s" % axis, "error": error, "limit": limits["dimension_relative"]})
            foot_reference = reference.foot_size_m[index]
            foot_error = abs(candidate.foot_size_m[index] - foot_reference) / max(foot_reference, 1e-9)
            if foot_error > limits["dimension_relative"]:
                issues.append({"field": "foot_size_%s" % axis, "error": foot_error, "limit": limits["dimension_relative"]})
            for bound in ("minimum", "maximum"):
                reference_bound = reference.collision_bounds_m[bound][index]
                candidate_bound = candidate.collision_bounds_m[bound][index]
                bound_error = abs(candidate_bound - reference_bound) / max(reference.body_size_m[index], 1e-9)
                if bound_error > limits["dimension_relative"]:
                    issues.append({"field": "collision_%s_%s" % (bound, axis), "error": bound_error, "limit": limits["dimension_relative"]})
        com_error = math.sqrt(sum((left - right) ** 2 for left, right in zip(reference.center_of_mass_m, candidate.center_of_mass_m)))
        if com_error > limits["com_m"]:
            issues.append({"field": "center_of_mass_m", "error": com_error, "limit": limits["com_m"], "blocking": False, "note": "diagnostic only unless both profiles provide nominal-pose whole-body COM"})
        reference_inertia = _inertia_body_frame(reference.link_inertias.get(reference.base_link, {}))
        candidate_inertia = _inertia_body_frame(candidate.link_inertias.get(candidate.base_link, {}))
        if reference_inertia and candidate_inertia:
            reference_norm = math.sqrt(sum(float(value) ** 2 for row in reference_inertia for value in row))
            inertia_error = math.sqrt(sum((float(left) - float(right)) ** 2 for left_row, right_row in zip(reference_inertia, candidate_inertia) for left, right in zip(left_row, right_row))) / max(reference_norm, 1e-12)
            if inertia_error > limits["inertia_relative"]:
                issues.append({"field": "base_inertia", "error": inertia_error, "limit": limits["inertia_relative"]})
        else:
            issues.append({"field": "base_inertia", "error": "missing", "blocking": True})
        reference_joints = {_norm_name(joint.name): joint for joint in reference.joints if joint.joint_type != "fixed"}
        candidate_joints = {_norm_name(joint.name): joint for joint in candidate.joints if joint.joint_type != "fixed"}
        missing = sorted(set(reference_joints) - set(candidate_joints))
        extra = sorted(set(candidate_joints) - set(reference_joints))
        if missing:
            issues.append({"field": "joints", "error": "missing", "names": missing})
        if extra:
            issues.append({"field": "joints", "error": "extra", "names": extra})
        for name in sorted(set(reference_joints) & set(candidate_joints)):
            left, right = reference_joints[name], candidate_joints[name]
            origin_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(left.origin_xyz, right.origin_xyz)))
            if origin_error > limits["joint_origin_m"]:
                issues.append({"field": "joint_origin", "joint": name, "error": origin_error, "limit": limits["joint_origin_m"]})
            left_axis_norm = math.sqrt(sum(value * value for value in left.axis))
            right_axis_norm = math.sqrt(sum(value * value for value in right.axis))
            if left_axis_norm > 0 and right_axis_norm > 0:
                cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(left.axis, right.axis)) / (left_axis_norm * right_axis_norm)))
                axis_error = math.acos(cosine)
                if axis_error > limits["joint_axis_rad"]:
                    issues.append({"field": "joint_axis", "joint": name, "error": axis_error, "limit": limits["joint_axis_rad"]})
            for bound in ("lower", "upper"):
                a, b = getattr(left, bound), getattr(right, bound)
                if a is not None and b is not None and abs(a - b) > limits["joint_limit_rad"]:
                    issues.append({"field": "joint_%s" % bound, "joint": name, "error": abs(a - b), "limit": limits["joint_limit_rad"]})
        reference_actuators = {_norm_name(item.joint): item for item in reference.actuators}
        candidate_actuators = {_norm_name(item.joint): item for item in candidate.actuators}
        for name in sorted(set(reference_actuators) & set(candidate_actuators)):
            left, right = reference_actuators[name], candidate_actuators[name]
            if left.effort_limit is None or right.effort_limit is None:
                issues.append({"field": "actuator_effort", "joint": name, "error": "missing", "blocking": True})
            else:
                error = abs(left.effort_limit - right.effort_limit) / max(abs(left.effort_limit), 1e-9)
                if error > limits["actuator_relative"]:
                    issues.append({"field": "actuator_effort", "joint": name, "error": error, "limit": limits["actuator_relative"]})
            if left.velocity_limit is None or right.velocity_limit is None:
                issues.append({"field": "actuator_velocity", "joint": name, "error": "missing", "blocking": True})
        for name in sorted(set(reference.default_joint_positions) & set(candidate.default_joint_positions)):
            error = abs(reference.default_joint_positions[name] - candidate.default_joint_positions[name])
            if error > limits["joint_limit_rad"]:
                issues.append({"field": "default_joint_position", "joint": name, "error": error, "limit": limits["joint_limit_rad"]})
        reference_order = [_norm_name(name) for name in reference.controller_joint_order]
        candidate_order = [_norm_name(name) for name in candidate.controller_joint_order]
        if reference_order != candidate_order:
            issues.append({"field": "controller_joint_order", "reference": reference_order, "candidate": candidate_order})
        blocking = [issue for issue in issues if issue.get("blocking", True) and not (issue["field"] == "total_mass_kg" and issue.get("error") == "missing")]
        comparisons.append({"reference": reference.name, "candidate": candidate.name, "comparable": not blocking, "issues": issues})
        comparable = comparable and not blocking
    return {"comparable": comparable, "tolerances": limits, "comparisons": comparisons}
