"""AE ordered-pair cells and AEE graph/layout composition."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import GeometrySpec, Pose, RegionSpec, RobotProfile, RouteSpec, WorldSpec
from .catalog import _box, _floor, _region, _scale, default_materials, generate_terrain


AE_CHALLENGES = ("rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs9", "rs10", "rs11")
DISCRETE_CHALLENGES = tuple("rs%d" % index for index in range(1, 12))


def generate_ae_pairs(challenge_codes: Sequence[str] = AE_CHALLENGES, include_self: bool = True) -> List[Tuple[str, str]]:
    codes = tuple(str(value).lower() for value in challenge_codes)
    return [(left, right) for left in codes for right in codes if include_self or left != right]


def _translate_geometry(geometry: GeometrySpec, dx: float, dy: float, dz: float, prefix: str, material_override: str = "") -> GeometrySpec:
    value = copy.deepcopy(geometry)
    value.name = "%s-%s" % (prefix, geometry.name)
    value.pose.position = [geometry.pose.position[0] + dx, geometry.pose.position[1] + dy, geometry.pose.position[2] + dz]
    if material_override and value.semantic not in {"approach", "landing"}:
        value.parameters["source_material"] = value.material
        # Slot colors communicate the AE decision structure. They must not
        # replace the source terrain's friction/compliance physics.
        value.parameters["display_material"] = material_override
    return value


def _translate_region(region: RegionSpec, dx: float, dy: float, dz: float, prefix: str) -> RegionSpec:
    value = copy.deepcopy(region)
    value.name = "%s-%s" % (prefix, region.name)
    value.minimum = [region.minimum[0] + dx, region.minimum[1] + dy, region.minimum[2] + dz]
    value.maximum = [region.maximum[0] + dx, region.maximum[1] + dy, region.maximum[2] + dz]
    return value


def _translate_route(route: RouteSpec, dx: float, dy: float, dz: float, prefix: str) -> RouteSpec:
    return RouteSpec(
        "%s-%s" % (prefix, route.name),
        [[point[0] + dx, point[1] + dy, point[2] + dz] for point in route.waypoints],
        ["%s-%s" % (prefix, name) for name in route.region_names],
        route.route_kind,
    )


def _transform_point(point: Sequence[float], dx: float, dy: float, dz: float, yaw: float) -> List[float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        dx + cosine * point[0] - sine * point[1],
        dy + sine * point[0] + cosine * point[1],
        dz + point[2],
    ]


def _transform_geometry(geometry: GeometrySpec, dx: float, dy: float, dz: float, yaw: float, prefix: str) -> GeometrySpec:
    value = copy.deepcopy(geometry)
    value.name = "%s-%s" % (prefix, geometry.name)
    value.pose.position = _transform_point(geometry.pose.position, dx, dy, dz, yaw)
    value.pose.rpy[2] += yaw
    return value


def _transform_route(route: RouteSpec, dx: float, dy: float, dz: float, yaw: float, prefix: str) -> RouteSpec:
    return RouteSpec(
        "%s-%s" % (prefix, route.name),
        [_transform_point(point, dx, dy, dz, yaw) for point in route.waypoints],
        ["%s-%s" % (prefix, name) for name in route.region_names],
        route.route_kind,
    )


def _box_surface_at(geometry: GeometrySpec, x: float, y: float) -> Optional[float]:
    """Return the declared top of a box containing an x/y point, if present."""

    if geometry.shape != "box":
        return None
    px, py, pz = geometry.pose.position
    sx, sy, sz = geometry.size
    tolerance = 1e-6
    if abs(x - px) <= sx / 2.0 + tolerance and abs(y - py) <= sy / 2.0 + tolerance:
        return pz + sz / 2.0
    return None


def _surface_at(geometry: Sequence[GeometrySpec], x: float, y: float) -> Optional[float]:
    surfaces = [surface for item in geometry if (surface := _box_surface_at(item, x, y)) is not None]
    return max(surfaces) if surfaces else None


def _resolve_composed_heights(world: WorldSpec, standing_height: float) -> None:
    for route in world.routes:
        for point in route.waypoints:
            surface = _surface_at(world.geometry, point[0], point[1])
            if surface is not None:
                point[2] = max(point[2], surface)
    goal = world.goal_pose.position
    goal_surface = _surface_at(world.geometry, goal[0], goal[1])
    if goal_surface is not None:
        goal[2] = max(goal[2], goal_surface + standing_height)


def _transform_region_rotated(region: RegionSpec, dx: float, dy: float, dz: float, yaw: float, prefix: str) -> RegionSpec:
    corners = [
        _transform_point([x, y, z], dx, dy, dz, yaw)
        for x in (region.minimum[0], region.maximum[0])
        for y in (region.minimum[1], region.maximum[1])
        for z in (region.minimum[2], region.maximum[2])
    ]
    return RegionSpec(
        "%s-%s" % (prefix, region.name),
        region.kind,
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
        region.mandatory,
        copy.deepcopy(region.metadata),
    )


def generate_ae(
    profile: RobotProfile,
    purple: str,
    brown: str,
    difficulty: float = 0.5,
    seed: int = 0,
    reversed_direction: bool = False,
    overrides: Dict[str, Any] = None,
    challenge_codes: Sequence[str] = AE_CHALLENGES,
) -> WorldSpec:
    purple, brown = purple.lower(), brown.lower()
    valid_codes = tuple(str(value).lower() for value in challenge_codes)
    if purple not in valid_codes or brown not in valid_codes:
        raise ValueError("AE slots must use %s" % (valid_codes,))
    overrides = overrides or {}
    scale = _scale(profile, [])
    transition_length = float(overrides.get("transition_length_m", max(scale.body_length * 1.2, 0.5)))
    approach = float(overrides.get("approach_m", scale.approach))
    central_width = float(overrides.get("central_width_m", max(scale.body_width * 2.2, 0.7)))
    bypass_width = float(overrides.get("bypass_width_m", max(scale.body_width * 1.3, 0.45)))
    lane_gap = float(overrides.get("lane_gap_m", max(scale.foot_width, 0.03)))
    landing = float(overrides.get("landing_m", scale.landing))
    for name, value in (
        ("transition_length_m", transition_length),
        ("approach_m", approach),
        ("central_width_m", central_width),
        ("bypass_width_m", bypass_width),
        ("lane_gap_m", lane_gap),
        ("landing_m", landing),
    ):
        if value <= 0.0:
            raise ValueError("AE override %s must be positive" % name)
    full_width = central_width + 2.0 * (bypass_width + lane_gap)
    # A body length of floor behind the spawn, and (for the reversed spawn at the
    # far end) beyond the exit. There is no ground plane, and the spawn sits
    # 0.6 body-lengths from the edge, so a robot rocking back as its policy
    # starts would otherwise step off the end of the world. Routes, regions and
    # the goal are unchanged; dimensions_m still describes the nominal course.
    apron = scale.body_length
    geometry: List[GeometrySpec] = [_floor("approach", -apron, approach + apron, full_width)]
    regions: List[RegionSpec] = [_region("approach", "approach", 0.0, approach, full_width)]
    cursor = approach
    central_waypoints = [[0.5 * scale.body_length, 0.0, 0.0], [approach, 0.0, 0.0]]

    def append_central(point: Sequence[float]) -> None:
        value = list(point)
        if any(abs(left - right) > 1e-9 for left, right in zip(value, central_waypoints[-1])):
            central_waypoints.append(value)

    def transition(label: str) -> None:
        nonlocal cursor
        # A continuous rough, low-friction run into and out of the challenge.
        #
        # This was five platforms laid at 0.65 coverage on a 1.0 pitch, so 35%
        # of every transition was a hole: five gaps before the pair and five
        # after it. Both generate_aee and the graph builder compose their cells
        # from generate_ae, so a 12-cell grid carried 24 of these -- a small-gap
        # course bolted onto both ends of a test measuring something else, and a
        # confound for any policy that cannot cross gaps. RS8 and RS9 remain the
        # gap challenges; nothing here is meant to be one.
        geometry.append(_floor("%s-platform" % label, cursor, transition_length, central_width, material="rough_low", semantic="transition"))
        regions.append(_region("%s-transition" % label, "rough_friction", cursor, cursor + transition_length, central_width, True))
        cursor += transition_length

    transition("entry")
    append_central([cursor, 0.0, 0.0])
    slot_regions = []
    slot_parameters: Dict[str, Any] = {}
    clips: List[Dict[str, Any]] = []
    for slot_name, code, color, slot_seed in (("purple", purple, "purple", seed + 101), ("brown", brown, "challenge", seed + 202)):
        nested = overrides.get("%s_overrides" % slot_name, {})
        if nested and not isinstance(nested, dict):
            raise ValueError("%s_overrides must be a JSON object" % slot_name)
        slot_overrides = dict(nested)
        prefix = "%s_" % slot_name
        for key, value in overrides.items():
            if key.startswith(prefix) and key != "%soverrides" % prefix:
                slot_overrides[key[len(prefix) :]] = value
        challenge = generate_terrain(code, profile, difficulty, slot_seed, slot_overrides)
        clips.extend({**item, "parameter": "%s.%s" % (slot_name, item["parameter"])} for item in challenge.clips)
        slot_parameters[slot_name] = {
            "terrain": code,
            "normalized": challenge.normalized_parameters,
            "resolved": challenge.resolved_parameters,
        }
        source_start = challenge.regions[0].maximum[0]
        source_goal = next(region for region in challenge.regions if region.name == "goal")
        source_end = source_goal.minimum[0]
        slot_length = source_end - source_start
        for item in challenge.geometry:
            # A nested terrain's own spawn and goal floors never belong inside a
            # slot: the composer provides the transitions. They end exactly at
            # the challenge boundary and used to slip through on floating-point
            # equality; once the approach grew an apron behind the spawn, the
            # leaked floor reached back far enough to pave over the previous
            # slot's gap.
            if item.name in {"approach", "landing"}:
                continue
            center_x = item.pose.position[0]
            half_x = item.size[0] / 2.0 if item.shape == "box" else 0.0
            if center_x + half_x <= source_start + 1e-9 or center_x - half_x >= source_end - 1e-9:
                continue
            geometry.append(_translate_geometry(item, cursor - source_start, 0.0, 0.0, slot_name, color))
        source_route = next(route for route in challenge.routes if route.route_kind == "central")
        for point in source_route.waypoints:
            if source_start - 1e-9 <= point[0] <= source_end + 1e-9:
                translated = [point[0] + cursor - source_start, point[1], point[2]]
                append_central(translated)
        regions.append(_region("%s-slot" % slot_name, "challenge_slot", cursor, cursor + slot_length, central_width, True, metadata={"terrain": code, "color": slot_name}))
        slot_regions.append("%s-slot" % slot_name)
        cursor += slot_length
    transition("exit")
    append_central([cursor, 0.0, 0.0])
    geometry.append(_floor("exit", cursor, landing + apron, full_width))
    regions.append(_region("goal", "goal", cursor, cursor + landing, full_width, True))
    total_length = cursor + landing
    bypass_y = central_width / 2.0 + lane_gap + bypass_width / 2.0
    for side, y in (("upper", bypass_y), ("lower", -bypass_y)):
        geometry.append(_box("%s-bypass" % side, [(approach + cursor) / 2.0, y, -0.05], [cursor - approach, bypass_width, 0.1], "rough_low", "bypass"))
        regions.append(RegionSpec("%s-bypass" % side, "bypass", [approach, y - bypass_width / 2.0, -0.5], [cursor, y + bypass_width / 2.0, 3.0], False, {"friction": "low", "roughness": True}))
    routes = [
        RouteSpec("central", [*central_waypoints, [total_length, 0.0, 0.0]], ["approach", "entry-transition", *slot_regions, "exit-transition", "goal"], "central"),
        RouteSpec("upper-bypass", [[approach, 0.0, 0.0], [approach, bypass_y, 0.0], [cursor, bypass_y, 0.0], [cursor, 0.0, 0.0]], ["approach", "upper-bypass", "goal"], "bypass"),
        RouteSpec("lower-bypass", [[approach, 0.0, 0.0], [approach, -bypass_y, 0.0], [cursor, -bypass_y, 0.0], [cursor, 0.0, 0.0]], ["approach", "lower-bypass", "goal"], "bypass"),
    ]
    if reversed_direction:
        for item in geometry:
            item.pose.position[0] = total_length - item.pose.position[0]
            item.pose.rpy[2] += 3.141592653589793
        for region in regions:
            lo, hi = region.minimum[0], region.maximum[0]
            region.minimum[0], region.maximum[0] = total_length - hi, total_length - lo
        for route in routes:
            route.waypoints = [[total_length - point[0], point[1], point[2]] for point in reversed(route.waypoints)]
    world = WorldSpec(
        name="AE-%s-%s-d%.3f-s%d%s" % (purple.upper(), brown.upper(), difficulty, seed, "-reversed" if reversed_direction else ""),
        terrain_type="ae",
        seed=seed,
        robot_profile_sha256=profile.source_sha256,
        dimensions_m=[total_length, full_width, 3.0],
        spawn_pose=Pose([scale.body_length * 0.6 if not reversed_direction else total_length - scale.body_length * 0.6, 0.0, scale.standing_height], [0.0, 0.0, 3.141592653589793 if reversed_direction else 0.0]),
        goal_pose=Pose([total_length - scale.body_length * 0.5 if not reversed_direction else scale.body_length * 0.5, 0.0, scale.standing_height], [0.0, 0.0, 3.141592653589793 if reversed_direction else 0.0]),
        materials=default_materials(difficulty),
        geometry=geometry,
        regions=regions,
        routes=routes,
        checkpoints=[[approach, 0.0, 0.0], [cursor, 0.0, 0.0]],
        normalized_parameters={
            "purple": purple,
            "brown": brown,
            "difficulty": difficulty,
            "ordered_pair_index": valid_codes.index(purple) * len(valid_codes) + valid_codes.index(brown),
            "course_body_lengths": total_length / scale.body_length,
            "central_width_body_widths": central_width / scale.body_width,
        },
        resolved_parameters={
            "total_length_m": total_length,
            "approach_m": approach,
            "transition_length_m": transition_length,
            "landing_m": landing,
            "central_width_m": central_width,
            "bypass_width_m": bypass_width,
            "lane_gap_m": lane_gap,
            "reversed": reversed_direction,
            "slots": slot_parameters,
        },
        clips=clips,
        metadata={"pair": [purple, brown], "all_pair_count": len(valid_codes) ** 2, "challenge_codes": list(valid_codes), "bypass_is_optional": True, "overrides": overrides},
    )
    _resolve_composed_heights(world, scale.standing_height)
    world.validate()
    return world


def _generate_aee_grid(profile: RobotProfile, difficulty: float, seed: int, options: Dict[str, Any]) -> WorldSpec:
    rows = int(options.get("rows", 3))
    columns = int(options.get("columns", 3))
    maximum_columns = 50 if str(options.get("topology", "grid")).lower() == "chain" else 10
    if not 1 <= rows <= 10 or not 2 <= columns <= maximum_columns:
        raise ValueError("graph rows must be in [1, 10] and columns in [2, %d]" % maximum_columns)
    pair_mode = str(options.get("pair_mode", "cycle")).lower()
    if pair_mode not in {"cycle", "random", "repeat", "adaptive"}:
        raise ValueError("graph pair_mode must be cycle, random, repeat, or adaptive")
    rng = random.Random(seed)
    topology = str(options.get("topology", "grid")).lower()
    if topology not in {"chain", "ring", "tree", "grid", "maze", "random_connected"}:
        raise ValueError("topology must be chain, ring, tree, grid, maze, or random_connected")
    if topology == "ring" and rows < 2:
        raise ValueError("ring topology requires at least two rows")
    all_edges: List[Dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns - 1):
            all_edges.append(
                {
                    "id": "h-%02d-%02d" % (row, column),
                    "source": "n-%02d-%02d" % (row, column),
                    "target": "n-%02d-%02d" % (row, column + 1),
                    "orientation": "east",
                    "yaw_rad": 0.0,
                    "row": row,
                    "column": column,
                }
            )
    for row in range(rows - 1):
        for column in range(columns):
            all_edges.append(
                {
                    "id": "v-%02d-%02d" % (row, column),
                    "source": "n-%02d-%02d" % (row, column),
                    "target": "n-%02d-%02d" % (row + 1, column),
                    "orientation": "north",
                    "yaw_rad": math.pi / 2.0,
                    "row": row,
                    "column": column,
                }
            )
    def endpoints(edge: Dict[str, Any]) -> Tuple[str, str]:
        return edge["source"], edge["target"]

    if topology == "grid":
        edge_defs = all_edges
    elif topology == "chain":
        wanted = []
        for row in range(rows):
            columns_in_order = range(columns - 1) if row % 2 == 0 else range(columns - 2, -1, -1)
            wanted.extend("h-%02d-%02d" % (row, column) for column in columns_in_order)
            if row + 1 < rows:
                wanted.append("v-%02d-%02d" % (row, columns - 1 if row % 2 == 0 else 0))
        by_id = {edge["id"]: edge for edge in all_edges}
        edge_defs = [by_id[edge_id] for edge_id in wanted]
    elif topology == "ring":
        edge_defs = [
            edge for edge in all_edges
            if (edge["orientation"] == "east" and edge["row"] in {0, rows - 1})
            or (edge["orientation"] == "north" and edge["column"] in {0, columns - 1})
        ]
    elif topology == "tree":
        edge_defs = [
            edge for edge in all_edges
            if edge["orientation"] == "east" or (edge["orientation"] == "north" and edge["column"] == 0)
        ]
    else:
        # Seeded randomized depth-first spanning tree. random_connected adds a
        # bounded number of extra edges without ever compromising connectivity.
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        for edge in all_edges:
            for node in endpoints(edge):
                adjacency.setdefault(node, []).append(edge)
        visited = {"n-00-00"}
        stack = ["n-00-00"]
        edge_defs = []
        while stack:
            node = stack[-1]
            candidates = [edge for edge in adjacency[node] if next(item for item in endpoints(edge) if item != node) not in visited]
            if not candidates:
                stack.pop()
                continue
            edge = rng.choice(candidates)
            other = next(item for item in endpoints(edge) if item != node)
            edge_defs.append(edge)
            visited.add(other)
            stack.append(other)
        if topology == "random_connected":
            remaining = [edge for edge in all_edges if edge not in edge_defs]
            rng.shuffle(remaining)
            edge_defs.extend(remaining[: max(1, len(remaining) // 3)])
    challenge_codes = tuple(options.get("challenge_codes", AE_CHALLENGES))
    include_self = bool(options.get("include_self_pairs", True))
    all_pairs = generate_ae_pairs(challenge_codes, include_self)
    requested_pairs = options.get("allowed_pairs")
    if requested_pairs is not None:
        pairs = [tuple(str(value).lower() for value in item) for item in requested_pairs]
        if not pairs or any(pair not in all_pairs for pair in pairs):
            raise ValueError("allowed_pairs must contain one or more valid ordered AE pairs")
    else:
        pairs = all_pairs
    if pair_mode == "repeat":
        repeated = (str(options.get("purple", "rs11")).lower(), str(options.get("brown", "rs4")).lower())
        if repeated not in pairs:
            raise ValueError("repeated grid pair must be present in allowed_pairs and use valid AE challenges")
        selected = [repeated for _ in edge_defs]
    elif pair_mode == "random":
        selected = [rng.choice(pairs) for _ in edge_defs]
    elif pair_mode == "adaptive":
        failures = [tuple(item) for item in options.get("adaptive_failure_pairs", [])]
        ranked = [item for item in failures if item in pairs]
        ranked.extend(item for item in pairs if item not in ranked)
        selected = [ranked[index % len(ranked)] for index in range(len(edge_defs))]
    else:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        selected = [shuffled[index % len(shuffled)] for index in range(len(edge_defs))]
    cell_overrides = options.get("cell_overrides", {})
    if cell_overrides and not isinstance(cell_overrides, dict):
        raise ValueError("grid cell_overrides must be an object")
    cells = [
        generate_ae(profile, pair[0], pair[1], difficulty, seed + 1009 * (index + 1), False, dict(cell_overrides), challenge_codes)
        for index, pair in enumerate(selected)
    ]
    maximum_length = max(cell.dimensions_m[0] for cell in cells)
    maximum_width = max(cell.dimensions_m[1] for cell in cells)
    junction_size = float(options.get("junction_size_m", max(maximum_width, 2.5 * profile.body_length_m)))
    connector_margin = float(options.get("connector_margin_m", max(profile.body_length_m, 0.35)))
    if junction_size < maximum_width or connector_margin <= 0.0:
        raise ValueError("junction_size_m must cover the widest AE cell and connector_margin_m must be positive")
    pitch = junction_size + maximum_length + 2.0 * connector_margin
    node_positions: Dict[str, List[float]] = {}
    node_ports: Dict[str, Dict[str, str]] = {}
    geometry: List[GeometrySpec] = []
    regions: List[RegionSpec] = []
    routes: List[RouteSpec] = []
    clips: List[Dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            node_id = "n-%02d-%02d" % (row, column)
            position = [junction_size / 2.0 + column * pitch, junction_size / 2.0 + row * pitch, 0.0]
            node_positions[node_id] = position
            node_ports[node_id] = {}
            geometry.append(_box("junction-%s" % node_id, [position[0], position[1], -0.05], [junction_size, junction_size, 0.1], "nominal", "junction"))
            regions.append(
                RegionSpec(
                    "junction-%s" % node_id,
                    "junction",
                    [position[0] - junction_size / 2.0, position[1] - junction_size / 2.0, -0.5],
                    [position[0] + junction_size / 2.0, position[1] + junction_size / 2.0, 3.0],
                    False,
                    {"node": node_id, "row": row, "column": column},
                )
            )

    edge_routes: Dict[str, RouteSpec] = {}
    edge_metadata: List[Dict[str, Any]] = []
    cell_metadata: List[Dict[str, Any]] = []
    for index, (edge, cell, pair) in enumerate(zip(edge_defs, cells, selected)):
        prefix = "edge-%s" % edge["id"]
        source = node_positions[edge["source"]]
        target = node_positions[edge["target"]]
        cell_length, cell_width = cell.dimensions_m[:2]
        available = pitch - junction_size
        connector_length = (available - cell_length) / 2.0
        if connector_length < 0.0:
            raise ValueError("grid pitch cannot contain AE cell %s" % edge["id"])
        if edge["orientation"] == "east":
            origin = [source[0] + junction_size / 2.0 + connector_length, source[1], 0.0]
            source_connector_center = [source[0] + junction_size / 2.0 + connector_length / 2.0, source[1], -0.05]
            target_connector_center = [target[0] - junction_size / 2.0 - connector_length / 2.0, target[1], -0.05]
            connector_size = [connector_length, cell_width, 0.1]
            source_bounds = [
                [source_connector_center[0] - connector_length / 2.0, source[1] - cell_width / 2.0, -0.5],
                [source_connector_center[0] + connector_length / 2.0, source[1] + cell_width / 2.0, 3.0],
            ]
            target_bounds = [
                [target_connector_center[0] - connector_length / 2.0, target[1] - cell_width / 2.0, -0.5],
                [target_connector_center[0] + connector_length / 2.0, target[1] + cell_width / 2.0, 3.0],
            ]
            source_port, target_port = "east", "west"
        else:
            origin = [source[0], source[1] + junction_size / 2.0 + connector_length, 0.0]
            source_connector_center = [source[0], source[1] + junction_size / 2.0 + connector_length / 2.0, -0.05]
            target_connector_center = [target[0], target[1] - junction_size / 2.0 - connector_length / 2.0, -0.05]
            connector_size = [cell_width, connector_length, 0.1]
            source_bounds = [
                [source[0] - cell_width / 2.0, source_connector_center[1] - connector_length / 2.0, -0.5],
                [source[0] + cell_width / 2.0, source_connector_center[1] + connector_length / 2.0, 3.0],
            ]
            target_bounds = [
                [target[0] - cell_width / 2.0, target_connector_center[1] - connector_length / 2.0, -0.5],
                [target[0] + cell_width / 2.0, target_connector_center[1] + connector_length / 2.0, 3.0],
            ]
            source_port, target_port = "north", "south"
        source_connector = "%s-source-connector" % prefix
        target_connector = "%s-target-connector" % prefix
        geometry.append(_box(source_connector, source_connector_center, connector_size, "nominal", "connector"))
        geometry.append(_box(target_connector, target_connector_center, connector_size, "nominal", "connector"))
        regions.append(RegionSpec(source_connector, "connector", source_bounds[0], source_bounds[1], False, {"edge": edge["id"]}))
        regions.append(RegionSpec(target_connector, "connector", target_bounds[0], target_bounds[1], False, {"edge": edge["id"]}))
        geometry.extend(_transform_geometry(item, origin[0], origin[1], 0.0, edge["yaw_rad"], prefix) for item in cell.geometry)
        regions.extend(_transform_region_rotated(item, origin[0], origin[1], 0.0, edge["yaw_rad"], prefix) for item in cell.regions)
        central = next(route for route in cell.routes if route.route_kind == "central")
        transformed_central = _transform_route(central, origin[0], origin[1], 0.0, edge["yaw_rad"], prefix)
        edge_route = RouteSpec(
            "%s-central" % prefix,
            [source, source_connector_center[:2] + [0.0], *transformed_central.waypoints, target_connector_center[:2] + [0.0], target],
            [
                "junction-%s" % edge["source"],
                source_connector,
                *transformed_central.region_names,
                target_connector,
                "junction-%s" % edge["target"],
            ],
            "graph-edge",
        )
        routes.append(edge_route)
        edge_routes[edge["id"]] = edge_route
        node_ports[edge["source"]][source_port] = edge["id"]
        node_ports[edge["target"]][target_port] = edge["id"]
        for item in cell.clips:
            clips.append({**item, "parameter": "%s.%s" % (edge["id"], item["parameter"])})
        edge_record = {
            **edge,
            "pair": list(pair),
            "cell_index": index,
            "origin_m": origin,
            "cell_length_m": cell_length,
            "cell_width_m": cell_width,
            "connector_length_m": connector_length,
            "source_port": source_port,
            "target_port": target_port,
            "bidirectional": True,
        }
        edge_metadata.append(edge_record)
        cell_metadata.append({"index": index, "edge": edge["id"], "pair": list(pair), "origin_m": origin, "yaw_rad": edge["yaw_rad"]})

    edge_definition_by_id = {item["id"]: item for item in edge_defs}

    def graph_route(name: str, edge_ids: Sequence[str], start: str = "n-00-00") -> RouteSpec:
        waypoints: List[List[float]] = []
        region_names: List[str] = []
        current = start
        for edge_id in edge_ids:
            route = edge_routes[edge_id]
            edge = edge_definition_by_id[edge_id]
            reverse = current == edge["target"]
            route_waypoints = list(reversed(route.waypoints)) if reverse else route.waypoints
            route_regions = list(reversed(route.region_names)) if reverse else route.region_names
            waypoints.extend(route_waypoints if not waypoints else route_waypoints[1:])
            for region_name in route_regions:
                if not region_names or region_names[-1] != region_name:
                    region_names.append(region_name)
            current = edge["source"] if reverse else edge["target"]
        return RouteSpec(name, waypoints, region_names, "graph-route")

    adjacency_ids: Dict[str, List[Tuple[str, str]]] = {}
    for edge in edge_defs:
        adjacency_ids.setdefault(edge["source"], []).append((edge["target"], edge["id"]))
        adjacency_ids.setdefault(edge["target"], []).append((edge["source"], edge["id"]))
    start_id, goal_id = "n-00-00", "n-%02d-%02d" % (rows - 1, columns - 1)
    queue = [start_id]
    previous: Dict[str, Tuple[str, str]] = {}
    seen = {start_id}
    while queue:
        node = queue.pop(0)
        if node == goal_id:
            break
        for neighbor, edge_id in sorted(adjacency_ids.get(node, [])):
            if neighbor not in seen:
                seen.add(neighbor)
                previous[neighbor] = (node, edge_id)
                queue.append(neighbor)
    path_edges: List[str] = []
    cursor_node = goal_id
    while cursor_node != start_id:
        parent, edge_id = previous[cursor_node]
        path_edges.append(edge_id)
        cursor_node = parent
    path_edges.reverse()

    # A graph condition claims that every RS pair assigned to an edge was
    # exercised.  A shortest corner-to-corner path cannot support that claim:
    # on the 2x2 ring/random worlds it touched two of four edges, and on a tree
    # it touched two of three.  Build one continuous edge-covering walk instead.
    #
    # Keep the shortest start->goal path for the final approach.  At every node
    # on it, depth-first tour all remaining incident edges and return, then take
    # the next reserved path edge.  Every graph edge is therefore visited at
    # least once, the walk remains continuous, and it still finishes at the
    # declared goal rather than back at spawn.  An off-path excursion crosses
    # its edge in both directions.  If a reserved path edge must be taken
    # target->source, cross it back source->target and return once more; this
    # guarantees the declared directional RS pair is exercised too.
    reserved_path_edges = set(path_edges)
    covered_edge_ids = set()
    coverage_walk: List[str] = []

    def cover_off_path_edges(node: str) -> None:
        for neighbor, edge_id in sorted(adjacency_ids.get(node, [])):
            if edge_id in reserved_path_edges or edge_id in covered_edge_ids:
                continue
            covered_edge_ids.add(edge_id)
            coverage_walk.append(edge_id)
            cover_off_path_edges(neighbor)
            coverage_walk.append(edge_id)

    current_node = start_id
    for edge_id in path_edges:
        cover_off_path_edges(current_node)
        edge = edge_definition_by_id[edge_id]
        if current_node == edge["source"]:
            current_node = edge["target"]
            coverage_walk.append(edge_id)
        elif current_node == edge["target"]:
            current_node = edge["source"]
            coverage_walk.extend([edge_id, edge_id, edge_id])
        else:  # pragma: no cover - protected by the BFS reconstruction above
            raise ValueError("canonical path is not contiguous at %s" % edge_id)
        covered_edge_ids.add(edge_id)
    cover_off_path_edges(current_node)
    if current_node != goal_id or covered_edge_ids != set(edge_definition_by_id):
        raise ValueError("failed to construct a complete start-to-goal graph traversal")

    if topology == "grid":
        canonical_name = "canonical-grid-all-edges"
        east_then_north = [
            "h-%02d-%02d" % (0, column) for column in range(columns - 1)
        ] + [
            "v-%02d-%02d" % (row, columns - 1) for row in range(rows - 1)
        ]
        north_then_east = [
            "v-%02d-%02d" % (row, 0) for row in range(rows - 1)
        ] + [
            "h-%02d-%02d" % (rows - 1, column)
            for column in range(columns - 1)
        ]
        # Retain the two direct alternatives for route-choice consumers, but
        # keep the all-edge route first: routes[0] is the scored traversal.
        routes.insert(0, graph_route("grid-north-then-east", north_then_east))
        routes.insert(0, graph_route("grid-east-then-north", east_then_north))
    else:
        canonical_name = "canonical-%s-all-edges" % topology
    routes.insert(0, graph_route(canonical_name, coverage_walk))
    canonical_route_names = [canonical_name]
    start_node = node_positions["n-00-00"]
    goal_node = node_positions["n-%02d-%02d" % (rows - 1, columns - 1)]
    dimensions = [junction_size + (columns - 1) * pitch, junction_size + (rows - 1) * pitch, 3.0]
    graph_nodes = [
        {
            "id": node_id,
            "position_m": position,
            "ports": node_ports[node_id],
            "degree": len(node_ports[node_id]),
        }
        for node_id, position in sorted(node_positions.items())
    ]
    world = WorldSpec(
        name="AEE-%s-%dx%d-%s-d%.3f-s%d" % (topology, rows, columns, pair_mode, difficulty, seed),
        terrain_type="aee",
        seed=seed,
        robot_profile_sha256=profile.source_sha256,
        generator_version="1.2",
        dimensions_m=dimensions,
        spawn_pose=Pose([start_node[0], start_node[1], profile.standing_height_m]),
        goal_pose=Pose([goal_node[0], goal_node[1], profile.standing_height_m]),
        materials=default_materials(difficulty),
        geometry=geometry,
        regions=regions,
        routes=routes,
        checkpoints=[node["position_m"] for node in graph_nodes],
        normalized_parameters={
            "layout": "graph",
            "topology": topology,
            "difficulty": difficulty,
            "rows": rows,
            "columns": columns,
            "node_count": len(graph_nodes),
            "edge_count": len(edge_metadata),
            "pair_mode": pair_mode,
            "pitch_body_lengths": pitch / profile.body_length_m,
        },
        resolved_parameters={
            "dimensions_m": dimensions,
            "junction_size_m": junction_size,
            "connector_margin_m": connector_margin,
            "pitch_m": pitch,
            "cell_overrides": cell_overrides,
        },
        clips=clips,
        metadata={
            "cells": cell_metadata,
            "graph_nodes": graph_nodes,
            "graph_edges": edge_metadata,
            "topology": topology,
            "connected": True,
            "bidirectional_edges": True,
            "independent_cells": False,
            "canonical_routes": canonical_route_names,
            "alternative_routes": (
                ["grid-east-then-north", "grid-north-then-east"]
                if topology == "grid"
                else []
            ),
            "canonical_route_edges": coverage_walk,
            "canonical_route_unique_edges": sorted(covered_edge_ids),
            "canonical_route_forward_edges": sorted(covered_edge_ids),
            "canonical_route_edge_traversal_count": len(coverage_walk),
            "canonical_route_covers_all_edges": True,
            "port_schema": ["north", "east", "south", "west"],
        },
    )
    _resolve_composed_heights(world, profile.standing_height_m)
    world.validate()
    return world


def generate_aee(
    profile: RobotProfile,
    layout: str = "sparse",
    difficulty: float = 0.5,
    seed: int = 0,
    count: int = 12,
    options: Dict[str, Any] = None,
) -> WorldSpec:
    if layout not in {"sparse", "exhaustive", "random", "long", "grid"}:
        raise ValueError("AEE layout must be sparse, exhaustive, random, long, or grid")
    options = options or {}
    if layout == "grid":
        return _generate_aee_grid(profile, difficulty, seed, options)
    challenge_codes = tuple(options.get("challenge_codes", AE_CHALLENGES))
    include_self = bool(options.get("include_self_pairs", True))
    all_pairs = generate_ae_pairs(challenge_codes, include_self)
    requested_pairs = options.get("allowed_pairs")
    if requested_pairs is not None:
        pairs = [tuple(str(value).lower() for value in item) for item in requested_pairs]
        if not pairs or any(pair not in all_pairs for pair in pairs):
            raise ValueError("allowed_pairs must contain one or more valid ordered AE pairs")
    else:
        pairs = all_pairs
    rng = random.Random(seed)
    if layout == "exhaustive":
        selected = pairs
    elif layout == "random":
        selected = [rng.choice(pairs) for _ in range(max(1, count))]
    elif layout == "long":
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        selected = shuffled[: max(1, min(count, len(shuffled)))]
    else:
        selected = [("rs1", "rs5"), ("rs8" if "rs8" in AE_CHALLENGES else "rs9", "rs10"), ("rs2", "rs7"), ("rs11", "rs4")]
    # The PDF-inspired and long layouts form one traversable chain. The
    # exhaustive layout is a 10x10 test matrix with one spawn/goal per cell in metadata.
    cells = [generate_ae(profile, left, right, difficulty, seed + index * 17, False, challenge_codes=challenge_codes) for index, (left, right) in enumerate(selected)]
    gap = max(profile.body_length_m, 0.5)
    geometry: List[GeometrySpec] = []
    regions: List[RegionSpec] = []
    routes: List[RouteSpec] = []
    cell_placements = []
    if layout == "exhaustive":
        cell_length = max(cell.dimensions_m[0] for cell in cells)
        cell_width = max(cell.dimensions_m[1] for cell in cells)
        for index, cell in enumerate(cells):
            row, column = divmod(index, 10)
            dx, dy = column * (cell_length + gap), row * (cell_width + gap)
            cell_placements.append((dx, dy))
        dimensions = [10 * (cell_length + gap), 10 * (cell_width + gap), 3.0]
    else:
        cursor = 0.0
        for cell in cells:
            cell_placements.append((cursor, 0.0))
            cursor += cell.dimensions_m[0] + gap
        dimensions = [max(cursor - gap, 0.1), max(cell.dimensions_m[1] for cell in cells), 3.0]
    cell_metadata = []
    for index, (cell, placement) in enumerate(zip(cells, cell_placements)):
        prefix = "cell-%03d" % index
        dx, dy = placement
        geometry.extend(_translate_geometry(item, dx, dy, 0.0, prefix) for item in cell.geometry)
        regions.extend(_translate_region(item, dx, dy, 0.0, prefix) for item in cell.regions)
        for route in cell.routes:
            routes.append(_translate_route(route, dx, dy, 0.0, prefix))
        cell_metadata.append({"index": index, "pair": cell.metadata["pair"], "origin_m": [dx, dy, 0.0], "spawn_pose": [cell.spawn_pose.position[0] + dx, cell.spawn_pose.position[1] + dy, cell.spawn_pose.position[2]], "goal_pose": [cell.goal_pose.position[0] + dx, cell.goal_pose.position[1] + dy, cell.goal_pose.position[2]]})
    graph_edges = []
    if layout != "exhaustive":
        global_waypoints = []
        global_regions = []
        for index, (cell, placement) in enumerate(zip(cells, cell_placements)):
            central = next(route for route in cell.routes if route.route_kind == "central")
            translated = _translate_route(central, placement[0], placement[1], 0.0, "cell-%03d" % index)
            global_waypoints.extend(translated.waypoints if not global_waypoints else translated.waypoints[1:])
            global_regions.extend(translated.region_names)
            if index + 1 < len(cells):
                connector_start = placement[0] + cell.dimensions_m[0]
                connector_name = "connector-%03d-%03d" % (index, index + 1)
                connector_width = min(cell.dimensions_m[1], cells[index + 1].dimensions_m[1])
                geometry.append(_floor(connector_name, connector_start, gap, connector_width, material="nominal", semantic="connector"))
                regions.append(_region(connector_name, "connector", connector_start, connector_start + gap, connector_width, True))
                global_regions.append(connector_name)
                global_waypoints.append([connector_start + gap / 2.0, 0.0, 0.0])
                graph_edges.append({"from": index, "to": index + 1, "region": connector_name})
        routes.insert(0, RouteSpec("aee-central", global_waypoints, global_regions, "central"))
    world = WorldSpec(
        name="AEE-%s-d%.3f-s%d" % (layout, difficulty, seed),
        terrain_type="aee",
        seed=seed,
        robot_profile_sha256=profile.source_sha256,
        dimensions_m=dimensions,
        spawn_pose=Pose(cell_metadata[0]["spawn_pose"]),
        goal_pose=Pose(cell_metadata[-1]["goal_pose"]),
        materials=default_materials(difficulty),
        geometry=geometry,
        regions=regions,
        routes=routes,
        checkpoints=[item["goal_pose"] for item in cell_metadata],
        normalized_parameters={"layout": layout, "difficulty": difficulty, "cell_count": len(cells)},
        resolved_parameters={"dimensions_m": dimensions, "gap_m": gap},
        metadata={"cells": cell_metadata, "graph_edges": graph_edges, "exhaustive_pair_coverage": layout == "exhaustive", "independent_cells": layout == "exhaustive"},
    )
    _resolve_composed_heights(world, profile.standing_height_m)
    world.validate()
    return world
