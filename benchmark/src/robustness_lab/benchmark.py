"""Versioned L0--L7 benchmark materialization, previewing, execution and summaries."""

from __future__ import annotations

import copy
import html
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io import atomic_write_text, dump_json, dump_yaml, load_data, sha256_data, sha256_file
from .models import Pose, RobotProfile, WorldSpec
from .runtime import execute_episode
from .sampling import sobol
from .worlds import apply_geometry_modifiers, export_obj, generate_ae, generate_ae_pairs, generate_aee, generate_terrain


BENCHMARK_SCHEMA_VERSION = "2.0"
HORIZONS_BODY_LENGTHS = {"XS": 5, "S": 10, "M": 25, "L": 50, "XL": 100, "Endurance": 200}
TOPOLOGIES = ("chain", "ring", "tree", "grid", "maze", "random_connected")
PAIR_MODES = ("repeat", "cycle", "random", "adaptive")
DIFFICULTIES = (0.2, 0.5, 0.8)


@dataclass
class BenchmarkSuiteSpec:
    benchmark_id: str
    profile: str
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    levels: List[str] = field(default_factory=list)
    horizons: Dict[str, int] = field(default_factory=lambda: dict(HORIZONS_BODY_LENGTHS))
    topologies: List[str] = field(default_factory=lambda: list(TOPOLOGIES))
    difficulty_policy: Dict[str, Any] = field(default_factory=lambda: {"raw": list(DIFFICULTIES), "adaptive_c90_multipliers": [0.5, 0.8, 1.0, 1.1]})
    directions: List[str] = field(default_factory=lambda: ["forward", "reverse"])
    repetitions: int = 1
    controller_seeds: List[Optional[int]] = field(default_factory=lambda: [None])
    stressors: Dict[str, Any] = field(default_factory=dict)
    geometry_modifiers: List[Dict[str, Any]] = field(default_factory=list)
    backend_gates: Dict[str, Any] = field(default_factory=dict)
    locked: bool = False

    def validate(self) -> None:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("BenchmarkSuiteSpec requires schema_version 2.0")
        if self.profile not in {"smoke", "quick", "standard", "exhaustive"}:
            raise ValueError("profile must be smoke, quick, standard, or exhaustive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if any(value not in TOPOLOGIES for value in self.topologies):
            raise ValueError("unsupported topology")
        if any(value not in {"forward", "reverse"} for value in self.directions):
            raise ValueError("directions must be forward/reverse")
        if set(self.horizons) != set(HORIZONS_BODY_LENGTHS) or any(int(value) <= 0 for value in self.horizons.values()):
            raise ValueError("all six positive morphology horizons are required")

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)


def _route_length(world: WorldSpec) -> float:
    route = next((item for item in world.routes if item.route_kind in {"graph-route", "central"}), world.routes[0])
    return sum(math.dist(left, right) for left, right in zip(route.waypoints, route.waypoints[1:]))


def _reverse_world(source: WorldSpec) -> WorldSpec:
    world = copy.deepcopy(source)
    length = world.dimensions_m[0]
    for item in world.geometry:
        item.pose.position[0] = length - item.pose.position[0]
        item.pose.rpy[2] += math.pi
    for region in world.regions:
        low, high = region.minimum[0], region.maximum[0]
        region.minimum[0], region.maximum[0] = length - high, length - low
    for route in world.routes:
        route.waypoints = [[length - point[0], point[1], point[2]] for point in reversed(route.waypoints)]
        route.region_names = list(reversed(route.region_names))
    world.spawn_pose = copy.deepcopy(source.spawn_pose)
    world.goal_pose = copy.deepcopy(source.goal_pose)
    world.spawn_pose.position[0] = length - source.spawn_pose.position[0]
    world.goal_pose.position[0] = length - source.goal_pose.position[0]
    world.spawn_pose.rpy[2] += math.pi
    world.goal_pose.rpy[2] += math.pi
    world.name += "-reverse"
    world.metadata["direction"] = "reverse"
    world.resolved_parameters["reversed"] = True
    world.validate()
    return world


def _preview_payload(world: WorldSpec, robot: RobotProfile) -> Dict[str, Any]:
    data = world.to_dict()
    return {
        "schema": "preview-1.0",
        "robot": {"profile_sha256": robot.source_sha256, "body_size_m": robot.body_size_m, "standing_height_m": robot.standing_height_m},
        "geometry": data["geometry"],
        "materials": data["materials"],
        "spawn_pose": data["spawn_pose"],
        "goal_pose": data["goal_pose"],
        "routes": data["routes"],
        "graph_nodes": data["metadata"].get("graph_nodes", []),
        "graph_edges": data["metadata"].get("graph_edges", []),
    }


def _pair_sequence(count: int, seed: int, mode: str, adaptive_failures: Sequence[Sequence[str]] = ()) -> List[Tuple[str, str]]:
    pairs = generate_ae_pairs()
    if mode == "repeat":
        return [pairs[seed % len(pairs)]] * count
    if mode == "adaptive":
        ranked = [tuple(item) for item in adaptive_failures if tuple(item) in pairs]
        ranked += [item for item in pairs if item not in ranked]
        return [ranked[index % len(ranked)] for index in range(count)]
    if mode == "random":
        import random

        rng = random.Random(seed)
        return [rng.choice(pairs) for _ in range(count)]
    offset = seed % len(pairs)
    return [pairs[(offset + index) % len(pairs)] for index in range(count)]


def _graph_world(
    robot: RobotProfile,
    topology: str,
    horizon: str,
    difficulty: float,
    seed: int,
    pair_mode: str,
    allowed_pairs: Optional[Sequence[Sequence[str]]] = None,
    challenge_codes: Optional[Sequence[str]] = None,
    include_self_pairs: bool = True,
) -> WorldSpec:
    target_bl = HORIZONS_BODY_LENGTHS[horizon]
    pair_options = {"allowed_pairs": [list(item) for item in allowed_pairs]} if allowed_pairs else {}
    if challenge_codes is not None:
        pair_options["challenge_codes"] = [str(value).lower() for value in challenge_codes]
        pair_options["include_self_pairs"] = bool(include_self_pairs)
    probe = generate_aee(robot, "grid", difficulty, seed, options={"rows": 1, "columns": 2, "pair_mode": pair_mode, "topology": "chain", **pair_options})
    edge_bl = max(_route_length(probe) / robot.body_length_m, 1.0)
    edge_target = max(1, int(round(target_bl / edge_bl)))
    if topology == "chain":
        candidate_dimensions = [(1, columns) for columns in range(max(2, edge_target), min(50, edge_target + 2) + 1)]
    else:
        # The canonical corner-to-corner path has roughly 2*(side-1)
        # edges. Choose the nearest realizable square graph.
        side = max(2, int(round(edge_target / 2.0)) + 1)
        candidate_dimensions = [(size, size) for size in range(2, min(10, side + 1) + 1)]
    candidates = [
        generate_aee(robot, "grid", difficulty, seed, options={"rows": rows, "columns": columns, "pair_mode": pair_mode, "topology": topology, **pair_options})
        for rows, columns in candidate_dimensions
    ]
    world = min(candidates, key=lambda item: abs(_route_length(item) / robot.body_length_m - target_bl))
    actual_before_stretch = _route_length(world)
    target_m = target_bl * robot.body_length_m
    if abs(actual_before_stretch - target_m) > 1e-6:
        canonical = world.routes[0]
        traversed_edges = max(1, sum(name.endswith("source-connector") for name in canonical.region_names))
        extra_margin = (target_m - actual_before_stretch) / (2.0 * traversed_edges)
        rows = int(world.normalized_parameters["rows"])
        columns = int(world.normalized_parameters["columns"])
        world = generate_aee(
            robot,
            "grid",
            difficulty,
            seed,
            options={"rows": rows, "columns": columns, "pair_mode": pair_mode, "topology": topology, "connector_margin_m": max(0.02, max(robot.body_length_m, 0.35) + extra_margin), **pair_options},
        )
    actual_m = _route_length(world)
    actual_bl = actual_m / robot.body_length_m
    world.normalized_parameters.update({"horizon_class": horizon, "requested_body_lengths": target_bl, "actual_body_lengths": actual_bl})
    world.resolved_parameters.update({"requested_length_m": target_bl * robot.body_length_m, "actual_length_m": actual_m, "horizon_error_body_lengths": actual_bl - target_bl})
    world.metadata.update({"topology": topology, "pair_assignment_mode": pair_mode})
    return world


def _condition(level: str, key: str, world: WorldSpec, direction: str = "forward", horizon: str = "XS", stressors: Optional[List[Dict[str, Any]]] = None, locked: bool = False) -> Dict[str, Any]:
    if direction == "reverse":
        world = _reverse_world(world)
    return {"level": level, "key": key, "direction": direction, "horizon": horizon, "stressors": stressors or [], "locked": locked, "world_object": world}


def _build_conditions(spec: BenchmarkSuiteSpec, robot: RobotProfile, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    seed = int(raw.get("seed_base", 23000))
    codes = ["rs%d" % index for index in range(1, 12)]
    pairs = generate_ae_pairs()
    # L0 uses a duplicate visible world intentionally: preview hashing proves
    # that calibration/episode metadata does not create redundant images.
    conditions.append(_condition("L0", "calibration-wiring", generate_terrain("rs1", robot, 0.2, seed)))
    if spec.profile == "smoke":
        for index, code in enumerate(codes):
            conditions.append(_condition("L1", "%s-d0.5" % code, generate_terrain(code, robot, 0.5, seed + index)))
        conditions.append(_condition("L2", "ae-rs11-rs4", generate_ae(robot, "rs11", "rs4", 0.5, seed + 20)))
        conditions.append(_condition("L3", "grid-2x2", generate_aee(robot, "grid", 0.5, seed + 30, options={"rows": 2, "columns": 2, "pair_mode": "cycle", "topology": "grid"}), horizon="M"))
        return conditions
    if spec.profile == "quick":
        for difficulty in DIFFICULTIES:
            for index, code in enumerate(codes):
                conditions.append(_condition("L1", "%s-d%.1f" % (code, difficulty), generate_terrain(code, robot, difficulty, seed + index)))
        for index, pair in enumerate(_pair_sequence(10, seed, "cycle")):
            conditions.append(_condition("L2", "ae-%s-%s" % pair, generate_ae(robot, pair[0], pair[1], 0.5, seed + 100 + index)))
        for index, topology in enumerate(("chain", "grid", "ring")):
            conditions.append(_condition("L3", "%s-L" % topology, _graph_world(robot, topology, "L", 0.5, seed + 200 + index, "cycle"), horizon="L"))
        return conditions
    if spec.profile == "standard":
        for difficulty in DIFFICULTIES:
            for code_index, code in enumerate(codes):
                for direction in spec.directions:
                    conditions.append(_condition("L1", "%s-d%.1f-%s" % (code, difficulty, direction), generate_terrain(code, robot, difficulty, seed + code_index), direction))
        selected_pairs = _pair_sequence(30, seed, "cycle")
        for difficulty in (0.5, 0.8):
            for pair_index, pair in enumerate(selected_pairs):
                for direction in spec.directions:
                    conditions.append(_condition("L2", "ae-%s-%s-d%.1f-%s" % (pair[0], pair[1], difficulty, direction), generate_ae(robot, pair[0], pair[1], difficulty, seed + 100 + pair_index), direction))
        for rows, columns in ((2, 2), (2, 3), (3, 3)):
            for mode_index, mode in enumerate(("repeat", "cycle", "random")):
                world = generate_aee(robot, "grid", 0.5, seed + 300 + rows * 10 + columns + mode_index, options={"rows": rows, "columns": columns, "pair_mode": mode, "topology": "grid"})
                natural_horizon = "L" if rows * columns <= 6 else "XL"
                conditions.append(_condition("L3", "grid-%dx%d-%s" % (rows, columns, mode), world, horizon=natural_horizon))
        conditions.append(_condition("L4", "grid-4x4", generate_aee(robot, "grid", 0.8, seed + 400, options={"rows": 4, "columns": 4, "pair_mode": "cycle", "topology": "grid"}), horizon="XL"))
        conditions.append(_condition("L4", "random-connected", _graph_world(robot, "random_connected", "L", 0.8, seed + 401, "random"), horizon="L"))
        conditions.append(_condition("L5", "endurance-100", _graph_world(robot, "chain", "XL", 0.5, seed + 500, "cycle"), horizon="XL"))
        conditions.append(_condition("L5", "endurance-200", _graph_world(robot, "chain", "Endurance", 0.5, seed + 501, "cycle"), horizon="Endurance"))
        bounds = raw.get("stressor_bounds", {"friction_scale": [0.6, 1.1], "payload_fraction": [0.0, 0.3], "action_latency_s": [0.0, 0.06], "push_impulse_ns": [0.0, 20.0]})
        base = generate_ae(robot, "rs4", "rs11", 0.8, seed + 600)
        for name, values in bounds.items():
            for value in values:
                conditions.append(_condition("L6", "%s-%s" % (name, value), copy.deepcopy(base), stressors=[{"type": name, "value": value}]))
        for index, values in enumerate(sobol({name: tuple(value) for name, value in bounds.items()}, int(raw.get("sobol_samples", 8)), seed)):
            conditions.append(_condition("L6", "sobol-%02d" % index, copy.deepcopy(base), stressors=[{"type": name, "value": value} for name, value in values.items()]))
        for index, topology in enumerate(("maze", "random_connected")):
            for difficulty in (0.8, 0.95):
                conditions.append(_condition("L7", "ood-%s-d%.2f" % (topology, difficulty), _graph_world(robot, topology, "L", difficulty, seed + 700 + index, "random"), horizon="L", locked=True))
        return conditions
    for difficulty in (0.2, 0.35, 0.5, 0.65, 0.8):
        for pair_index, pair in enumerate(pairs):
            for direction in spec.directions:
                conditions.append(_condition("L2", "ae-%s-%s-d%.2f-%s" % (pair[0], pair[1], difficulty, direction), generate_ae(robot, pair[0], pair[1], difficulty, seed + pair_index), direction))
    for index, topology in enumerate(TOPOLOGIES):
        conditions.append(_condition("L3", "%s-L" % topology, _graph_world(robot, topology, "L", 0.8, seed + 2000 + index, "cycle"), horizon="L"))
    return conditions


def _load_spec(config_path: str, profile: str) -> Tuple[BenchmarkSuiteSpec, Dict[str, Any]]:
    raw = load_data(config_path)
    defaults = raw.get("profiles", {}).get(profile, {})
    repetitions = int(defaults.get("repetitions", {"smoke": 1, "quick": 10, "standard": 30, "exhaustive": 30}[profile]))
    controller_seeds = defaults.get("controller_seeds", [None])
    spec = BenchmarkSuiteSpec(
        benchmark_id=str(raw.get("benchmark_id", "tiered-legged-robustness")),
        profile=profile,
        levels=list(defaults.get("levels", ["L0", "L1", "L2", "L3"] if profile in {"smoke", "quick"} else ["L%d" % index for index in range(8)])),
        repetitions=repetitions,
        controller_seeds=list(controller_seeds),
        geometry_modifiers=copy.deepcopy(defaults.get("geometry_modifiers", [])),
        backend_gates=copy.deepcopy(raw.get("backend_gates", {})),
        locked=bool(defaults.get("locked", profile == "exhaustive")),
    )
    spec.validate()
    return spec, raw


def materialize_benchmark(config_path: str, profile: str, robot_profile_path: str, backends: Sequence[str], robot_assets: Dict[str, Dict[str, str]], controller_path: str, output_dir: str) -> Dict[str, Any]:
    spec, raw = _load_spec(config_path, profile)
    robot = RobotProfile.from_dict(load_data(robot_profile_path))
    controller = load_data(controller_path)
    root = Path(output_dir).resolve()
    worlds_dir, episodes_dir = root / "worlds", root / "episodes"
    worlds_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    built = _build_conditions(spec, robot, raw)
    if spec.geometry_modifiers:
        additions: List[Dict[str, Any]] = []
        for modifier in spec.geometry_modifiers:
            selected_levels = set(modifier.get("levels", ["L1"]))
            selected_terrains = set(str(value).lower() for value in modifier.get("terrain_codes", ["all"]))
            for item in built:
                world = item["world_object"]
                if item["level"] not in selected_levels or ("all" not in selected_terrains and world.terrain_type not in selected_terrains):
                    continue
                changed = copy.deepcopy(item)
                changed["key"] = "%s-%s" % (item["key"], str(modifier.get("name", "geometry-modifier")))
                changed["world_object"] = apply_geometry_modifiers(world, float(modifier.get("camber_deg", 0.0)), float(modifier.get("curve_angle_deg", 0.0)), int(modifier.get("curve_segments", 24)))
                additions.append(changed)
        built.extend(additions)
    source_hash = sha256_file(config_path)
    conditions: List[Dict[str, Any]] = []
    preview_index: Dict[str, str] = {}
    for index, item in enumerate(built):
        world = item.pop("world_object")
        world_hash = sha256_data({"world": world.to_dict(), "benchmark_schema": spec.schema_version, "source_hash": source_hash})
        preview_hash = sha256_data(_preview_payload(world, robot))
        world_path = worlds_dir / (world_hash + ".json")
        if not world_path.exists():
            dump_json(str(world_path), world)
        actual_m = _route_length(world)
        requested_bl = HORIZONS_BODY_LENGTHS[item["horizon"]]
        actual_bl = actual_m / robot.body_length_m
        tolerance_bl = max(1.0, requested_bl * 0.1)
        record = {**item, "condition_id": "%s-%04d-%s" % (profile, index, item["key"]), "difficulty": float(world.normalized_parameters.get("difficulty", 0.0)), "world": str(world_path), "world_hash": world_hash, "preview_hash": preview_hash, "requested_body_lengths": requested_bl, "actual_body_lengths": actual_bl, "actual_length_m": actual_m, "horizon_tolerance_body_lengths": tolerance_bl, "horizon_within_tolerance": abs(actual_bl - requested_bl) <= tolerance_bl}
        conditions.append(record)
        preview_index.setdefault(preview_hash, str(world_path))
    episodes: List[str] = []
    base_seed = int(raw.get("seed_base", 23000))
    for backend in backends:
        asset = robot_assets[backend]
        for condition_index, condition in enumerate(conditions):
            for controller_seed in spec.controller_seeds:
                for repetition in range(spec.repetitions):
                    episode_id = "%s-%s-%s-c%s-r%03d" % (spec.benchmark_id, profile, condition["condition_id"], controller_seed if controller_seed is not None else "na", repetition)
                    run_dir = root / "runs" / backend / episode_id
                    episode = {
                        "episode_id": episode_id,
                        "backend": backend,
                        "robot_profile": str(Path(robot_profile_path).resolve()),
                        "robot_asset": str(Path(asset["path"]).resolve()),
                        "robot_format": asset["format"],
                        "world": condition["world"],
                        "controller": controller,
                        "output_dir": str(run_dir),
                        "seed": base_seed + condition_index * 10000 + repetition + (0 if controller_seed is None else int(controller_seed) * 1000000),
                        "dt_s": float(raw.get("dt_s", 0.005)),
                        "control_decimation": int(raw.get("control_decimation", 4)),
                        "timeout_s": float(raw.get("timeout_s", 30.0)),
                        "headless": True,
                        "stressors": condition["stressors"],
                        "metadata": {"benchmark_id": spec.benchmark_id, "profile": profile, "level": condition["level"], "condition_id": condition["condition_id"], "severity": condition["difficulty"], "horizon": condition["horizon"], "topology": load_data(condition["world"]).get("metadata", {}).get("topology", "isolated"), "repetition": repetition, "controller_training_seed": controller_seed, "hold_controller_no_capability_claims": controller.get("type") in {"hold", "pd_hold"}, "locked": condition["locked"] or spec.locked},
                    }
                    episode_path = episodes_dir / backend / (episode_id + ".yaml")
                    dump_yaml(str(episode_path), episode)
                    episodes.append(str(episode_path))
    manifest = {
        "schema_version": spec.schema_version,
        "benchmark_id": spec.benchmark_id,
        "profile": profile,
        "suite_spec": spec.to_dict(),
        "source_config": str(Path(config_path).resolve()),
        "source_config_sha256": source_hash,
        "robot_profile": str(Path(robot_profile_path).resolve()),
        "robot_profile_sha256": robot.source_sha256,
        "backends": list(backends),
        "condition_count": len(conditions),
        "unique_world_count": len({item["world_hash"] for item in conditions}),
        "unique_preview_count": len(preview_index),
        "episode_count": len(episodes),
        "conditions": conditions,
        "preview_worlds": preview_index,
        "episodes": episodes,
        "preview_manifest": str(root / "previews" / "preview_manifest.json"),
        "lock_hash": "",
    }
    manifest["lock_hash"] = sha256_data({key: value for key, value in manifest.items() if key != "lock_hash"})
    dump_json(str(root / "benchmark_manifest.json"), manifest)
    return manifest


def _gallery_html(manifest: Dict[str, Any], previews: List[Dict[str, Any]], root: Path) -> str:
    def href(path: Any) -> str:
        return Path(os.path.relpath(str(path), str(root))).as_posix()

    level_info = {
        "L0": ("Calibration and wiring", "Load, reset, timing and simulator/model checks."),
        "L1": ("Individual terrains", "One RS terrain at a time; difficulty and direction sweeps."),
        "L2": ("Two-terrain AE entities", "Two ordered challenges connected inside one test entity."),
        "L3": ("Small connected networks", "2x2, 2x3 and 3x3 graphs with repeat/cycle/random assignment."),
        "L4": ("Complex route selection", "Large grids and random-connected worlds with alternate routes."),
        "L5": ("Endurance", "Long courses targeting 100 and 200 robot body lengths."),
        "L6": ("Robustness stressors", "Friction, payload, latency, pushes and Sobol interactions; some reuse visible geometry."),
        "L7": ("Locked unseen tests", "OOD maze and random-connected layouts unavailable during development."),
    }
    preview_by_hash = {item["preview_hash"]: item for item in previews}
    levels = [level for level in level_info if any(item["level"] == level for item in manifest["conditions"])]
    nav = []
    sections = []
    for level in levels:
        conditions = [item for item in manifest["conditions"] if item["level"] == level]
        ordered_hashes = list(dict.fromkeys(item["preview_hash"] for item in conditions))
        title, description = level_info[level]
        nav.append("<a href='#%s'><strong>%s</strong><span>%s</span><small>%d conditions · %d images</small></a>" % (level.lower(), level, html.escape(title), len(conditions), len(ordered_hashes)))
        cards = []
        for preview_hash in ordered_hashes:
            item = preview_by_hash[preview_hash]
            details = [condition for condition in conditions if condition["preview_hash"] == preview_hash]
            perspective = href(item["perspective_png"])
            topdown = href(item["topdown_png"])
            condition_names = ", ".join(html.escape(value["key"]) for value in details)
            difficulties = ", ".join(sorted({"%.2g" % float(value["difficulty"]) for value in details}))
            directions = ", ".join(dict.fromkeys(value["direction"] for value in details))
            horizons = ", ".join(dict.fromkeys(value["horizon"] for value in details))
            lengths = [float(value["actual_body_lengths"]) for value in details]
            stressors = sorted({stressor["type"] for value in details for stressor in value.get("stressors", [])})
            sections_links = " ".join("<a class='artifact' href='%s'>section %s</a>" % (href(path), Path(path).stem.split("-")[-1]) for path in item["section_pngs"])
            cards.append(
                "<article><div class='badges'><b>%s</b><span>%s</span><span>%s</span></div><h3>%s</h3>"
                "<p class='condition'>%s</p><div class='images'><a href='%s'><img loading='lazy' src='%s' alt='Perspective'></a>"
                "<a href='%s'><img loading='lazy' src='%s' alt='Top-down'></a></div>"
                "<dl><dt>Difficulty</dt><dd>%s</dd><dt>Direction</dt><dd>%s</dd><dt>Horizon</dt><dd>%s</dd>"
                "<dt>Actual length</dt><dd>%.1f%s body lengths</dd><dt>Stressors</dt><dd>%s</dd></dl>"
                "<p class='links'><a class='artifact' href='%s'>World JSON</a><a class='artifact' href='%s'>OBJ</a><a class='artifact' href='%s'>MuJoCo XML</a>%s</p>"
                "<p class='hash'><code>%s</code></p></article>"
                % (
                    level,
                    html.escape(horizons),
                    html.escape(directions),
                    html.escape(item["world_name"]),
                    condition_names,
                    perspective,
                    perspective,
                    topdown,
                    topdown,
                    html.escape(difficulties or "n/a"),
                    html.escape(directions),
                    html.escape(horizons),
                    min(lengths),
                    ("–%.1f" % max(lengths)) if max(lengths) != min(lengths) else "",
                    html.escape(", ".join(stressors) if stressors else "none / geometry-only"),
                    href(item["world_json"]),
                    href(item["obj"]),
                    href(item["mjcf"]),
                    sections_links,
                    preview_hash[:16],
                )
            )
        sheet_name = "contact-sheet-%s.png" % level.lower()
        sections.append("<section id='%s'><div class='level-head'><div><span class='level'>%s</span><h2>%s</h2><p>%s</p></div><a class='sheet' href='%s'>Open %s contact sheet</a></div><div class='grid'>%s</div></section>" % (level.lower(), level, html.escape(title), html.escape(description), sheet_name, level, "".join(cards)))
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>%s previews</title><style>
body{font-family:Inter,system-ui,sans-serif;background:#eef2f6;margin:0;color:#102a43}.wrap{max-width:1500px;margin:auto;padding:28px}.hero{background:#102a43;color:white;padding:28px;border-radius:16px}.hero p{color:#d9e2ec}.nav{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin:20px 0}.nav a{display:flex;flex-direction:column;background:white;padding:12px;border-radius:10px;text-decoration:none;color:#102a43;border-left:5px solid #00a896}.nav span{font-size:13px;margin:3px 0}.nav small{color:#627d98}.level-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:38px;padding-bottom:10px;border-bottom:3px solid #bcccdc}.level-head h2{display:inline;margin-left:10px}.level-head p{margin:7px 0 0}.level{background:#102a43;color:white;font-size:20px;font-weight:800;padding:8px 12px;border-radius:8px}.sheet,.artifact{display:inline-block;text-decoration:none;background:#e4f7f3;color:#00695c;padding:5px 8px;border-radius:6px;margin:2px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;margin-top:18px}article{background:white;padding:14px;border-radius:12px;box-shadow:0 2px 8px #102a4314}h3{margin:10px 0 4px}.badges{display:flex;gap:7px}.badges>*{background:#eaf0f6;padding:4px 8px;border-radius:999px;font-size:12px}.badges b{background:#102a43;color:white}.condition{color:#486581;font-size:13px;min-height:34px}.images{display:flex;gap:6px}.images a{width:50%%}.images img{width:100%%;border-radius:6px;display:block}dl{display:grid;grid-template-columns:110px 1fr;font-size:13px;margin-bottom:8px}dt{font-weight:700}dd{margin:0}.links{margin:5px 0}.hash{color:#829ab1;font-size:11px}@media(max-width:600px){.grid{grid-template-columns:1fr}.level-head{align-items:flex-start;flex-direction:column}.wrap{padding:12px}}
</style></head><body><div class='wrap'><header class='hero'><h1>%s — %s</h1><p>%d benchmark conditions grouped in increasing order from L0 to L7; %d geometry-deduplicated images. Each card shows perspective and top-down views.</p></header><nav class='nav'>%s</nav>%s</div></body></html>""" % (manifest["benchmark_id"], manifest["benchmark_id"], manifest["profile"], manifest["condition_count"], len(previews), "".join(nav), "".join(sections))


def render_benchmark(manifest_path: str, robot_asset: str, gl_backend: str = "osmesa") -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    from .backends.mujoco import build_mjcf_scene
    from .rendering import render_mjcf_scene, render_topdown

    manifest = load_data(manifest_path)
    root = Path(manifest_path).resolve().parent
    preview_root = root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    expected_preview_hashes = set(manifest["preview_worlds"])
    pruned = 0
    for candidate in preview_root.iterdir():
        if candidate.is_dir() and re.fullmatch(r"[0-9a-f]{64}", candidate.name) and candidate.name not in expected_preview_hashes:
            shutil.rmtree(candidate)
            pruned += 1
    previews: List[Dict[str, Any]] = []
    conditions_by_preview: Dict[str, List[Dict[str, Any]]] = {}
    for condition in manifest["conditions"]:
        conditions_by_preview.setdefault(condition["preview_hash"], []).append(condition)
    robot = RobotProfile.from_dict(load_data(manifest["robot_profile"]))
    ordered_preview_hashes = list(dict.fromkeys(condition["preview_hash"] for condition in manifest["conditions"]))
    for preview_hash in ordered_preview_hashes:
        world_path = manifest["preview_worlds"][preview_hash]
        world = WorldSpec.from_dict(load_data(world_path))
        folder = preview_root / preview_hash
        folder.mkdir(parents=True, exist_ok=True)
        scene, obj = folder / "scene.xml", folder / "world.obj"
        perspective, topdown = folder / "perspective.png", folder / "topdown.png"
        build_mjcf_scene(robot_asset, world, str(scene), 0.002)
        export_obj(world, str(obj))
        if not perspective.exists():
            render_mjcf_scene(str(scene), world, str(perspective), gl_backend=gl_backend)
        if not topdown.exists():
            render_topdown(world, str(topdown))
        sections = []
        actual_bl = _route_length(world) / robot.body_length_m
        if actual_bl > 50:
            route = next((item for item in world.routes if item.route_kind in {"graph-route", "central"}), world.routes[0])
            for marker in range(25, int(actual_bl), 25):
                point = route.waypoints[min(len(route.waypoints) - 1, round((len(route.waypoints) - 1) * marker / actual_bl))]
                section = folder / ("section-%03dbl.png" % marker)
                if not section.exists():
                    half = 12.5 * robot.body_length_m
                    render_mjcf_scene(str(scene), world, str(section), gl_backend=gl_backend, focus_bounds=[point[0] - half, point[0] + half, point[1] - half, point[1] + half])
                sections.append(str(section))
        details = conditions_by_preview[preview_hash]
        record = {"preview_hash": preview_hash, "world_hashes": sorted({item["world_hash"] for item in details}), "world_name": world.name, "conditions": [item["condition_id"] for item in details], "levels": list(dict.fromkeys(item["level"] for item in details)), "world_json": world_path, "obj": str(obj), "mjcf": str(scene), "perspective_png": str(perspective), "topdown_png": str(topdown), "section_pngs": sections}
        previews.append(record)
    def make_contact_sheet(items: List[Dict[str, Any]], output: Path, forced_level: str = "") -> None:
        sheet_width, tile_width, tile_height, columns = 1200, 300, 190, 4
        sheet = Image.new("RGB", (sheet_width, max(1, math.ceil(len(items) / columns)) * tile_height), "#e8edf2")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(items):
            image = Image.open(item["perspective_png"]).convert("RGB")
            image.thumbnail((tile_width - 8, tile_height - 28))
            x, y = (index % columns) * tile_width, (index // columns) * tile_height
            sheet.paste(image, (x + 4, y + 22))
            label = forced_level or "/".join(item["levels"])
            draw.text((x + 5, y + 4), "[%s] %s" % (label, item["world_name"][:34]), fill="#102a43")
        sheet.save(output)
    contact_sheet = preview_root / "contact-sheet.png"
    make_contact_sheet(previews, contact_sheet)
    level_contact_sheets = {}
    for level in sorted({item["level"] for item in manifest["conditions"]}):
        hashes = list(dict.fromkeys(item["preview_hash"] for item in manifest["conditions"] if item["level"] == level))
        output = preview_root / ("contact-sheet-%s.png" % level.lower())
        make_contact_sheet([{**next(item for item in previews if item["preview_hash"] == value), "levels": [level]} for value in hashes], output, level)
        level_contact_sheets[level] = str(output)
    preview_manifest = {"schema_version": "1.1", "ordering": "level_then_condition", "benchmark_manifest": str(Path(manifest_path).resolve()), "condition_count": manifest["condition_count"], "unique_preview_count": len(previews), "previews": previews, "level_contact_sheets": level_contact_sheets, "condition_to_preview": {item["condition_id"]: item["preview_hash"] for item in manifest["conditions"]}, "zero_missing_links": all(Path(item[key]).exists() for item in previews for key in ("world_json", "obj", "mjcf", "perspective_png", "topdown_png"))}
    dump_json(str(preview_root / "preview_manifest.json"), preview_manifest)
    atomic_write_text(str(preview_root / "index.html"), _gallery_html(manifest, previews, preview_root))
    return {"status": "ok", "preview_count": len(previews), "pruned_orphan_previews": pruned, "manifest": str(preview_root / "preview_manifest.json"), "gallery": str(preview_root / "index.html"), "contact_sheet": str(contact_sheet), "zero_missing_links": preview_manifest["zero_missing_links"]}


def run_benchmark(manifest_path: str, runtimes_path: str, limit: Optional[int] = None, resume: bool = False) -> Dict[str, Any]:
    manifest = load_data(manifest_path)
    expected = manifest.get("lock_hash")
    actual = sha256_data({key: value for key, value in manifest.items() if key != "lock_hash"})
    if expected != actual:
        raise ValueError("benchmark manifest changed after materialization; refusing execution")
    selected = manifest["episodes"][:limit] if limit else manifest["episodes"]
    statuses: Dict[str, int] = {}
    executed = 0
    for episode_path in selected:
        episode = load_data(episode_path)
        manifest_file = Path(episode["output_dir"]) / "manifest.json"
        if resume and manifest_file.exists() and load_data(str(manifest_file)).get("status") in {"completed", "skipped", "failed"}:
            statuses["already_finished"] = statuses.get("already_finished", 0) + 1
            continue
        result = execute_episode(episode_path, runtimes_path)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        executed += 1
    summary = {"status": "ok", "selected": len(selected), "executed": executed, "statuses": statuses, "resume": resume}
    dump_json(str(Path(manifest_path).resolve().parent / "execution_status.json"), summary)
    return summary


def summarize_benchmark(manifest_path: str, output_dir: str, conformance_path: Optional[str] = None, calibration_path: Optional[str] = None) -> Dict[str, Any]:
    from .aggregate import aggregate_runs

    manifest = load_data(manifest_path)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cube: Dict[str, Dict[str, Any]] = {}
    status_counts: Dict[str, int] = {}
    completed_run_dirs: List[str] = []
    condition_metrics: Dict[str, Dict[str, List[float]]] = {}
    contains_hold_controller = False
    for condition in manifest["conditions"]:
        topology = load_data(condition["world"]).get("metadata", {}).get("topology", "isolated")
        cube[condition["condition_id"]] = {key: condition[key] for key in ("level", "difficulty", "horizon", "direction", "actual_body_lengths", "stressors")}
        cube[condition["condition_id"]]["topology"] = topology
    for episode_path in manifest["episodes"]:
        episode = load_data(episode_path)
        contains_hold_controller = contains_hold_controller or episode.get("controller", {}).get("type") in {"hold", "pd_hold"}
        run_manifest = Path(episode["output_dir"]) / "manifest.json"
        status = load_data(str(run_manifest)).get("status", "unknown") if run_manifest.exists() else "pending"
        status_counts[status] = status_counts.get(status, 0) + 1
        metrics_path = Path(episode["output_dir"]) / "metrics.json"
        if run_manifest.exists() and metrics_path.exists():
            completed_run_dirs.append(episode["output_dir"])
            condition_id = episode.get("metadata", {}).get("condition_id", "")
            for metric in load_data(str(metrics_path)).get("metrics", []):
                if metric.get("valid") and isinstance(metric.get("value"), (int, float)):
                    condition_metrics.setdefault(condition_id, {}).setdefault(metric["name"], []).append(float(metric["value"]))
    for condition_id, metrics in condition_metrics.items():
        if condition_id in cube:
            cube[condition_id]["metrics"] = {name: {"mean": sum(values) / len(values), "count": len(values)} for name, values in metrics.items()}
    aggregate = aggregate_runs(completed_run_dirs, None, conformance_path, calibration_path) if completed_run_dirs else None
    adaptive = {"status": "awaiting_policy_trials", "c90_multipliers": [0.5, 0.8, 1.0, 1.1], "conditions": {}}
    if aggregate:
        for backend, curve in aggregate.get("capability_curves", {}).items():
            c90 = curve.get("c90")
            if isinstance(c90, (int, float)) and math.isfinite(c90):
                adaptive["status"] = "scheduled"
                adaptive["conditions"][backend] = [{"c90_multiplier": factor, "severity": c90 * factor} for factor in adaptive["c90_multipliers"]]
    comparable = bool(aggregate and aggregate.get("comparability", {}).get("status") == "comparable")
    claims_allowed = not contains_hold_controller and comparable
    reason = "ready" if claims_allowed else ("hold-controller runs are orchestration evidence only" if contains_hold_controller else "conformance and calibration gates are incomplete")
    result = {"schema_version": "1.0", "benchmark_id": manifest["benchmark_id"], "profile": manifest["profile"], "capability_claims_allowed": claims_allowed, "claim_gate_reason": reason, "condition_count": len(cube), "episode_status_counts": status_counts, "capability_cube": cube, "aggregate": aggregate, "adaptive_schedule": adaptive}
    dump_json(str(root / "capability_cube.json"), result)
    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%s</td></tr>" % (html.escape(key), value["level"], value["topology"], value["actual_body_lengths"], value["direction"]) for key, value in cube.items())
    page = "<html><head><meta charset='utf-8'><title>Capability cube</title><style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse;width:100%%}td,th{padding:7px;border:1px solid #ccd}</style></head><body><h1>Capability cube: %s</h1><p>Capability claims blocked until a walking controller is supplied.</p><table><tr><th>Condition</th><th>Level</th><th>Topology</th><th>Body lengths</th><th>Direction</th></tr>%s</table></body></html>" % (manifest["profile"], rows)
    atomic_write_text(str(root / "index.html"), page)
    return {"status": "ok", "json": str(root / "capability_cube.json"), "html": str(root / "index.html"), "episode_status_counts": status_counts}
