"""Command-line interface for robustness-lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .assets import fetch_assets, verify_assets
from .benchmark import materialize_benchmark, render_benchmark, run_benchmark, summarize_benchmark
from .aggregate import aggregate_runs
from .calibration import analyze_calibration, generate_calibration_suite
from .doctor import run_doctor
from .examiner import (
    diagnose_examination,
    examination_status,
    generate_capability_passport,
    initialize_examination,
    run_examination,
)
from .gallery import generate_variation_gallery
from .io import dump_json, load_data
from .models import RobotProfile, WorldSpec
from .parameter_diagrams import generate_parameter_diagrams
from .preflight import preflight_examination
from .report import generate_aggregate_report, generate_report
from .robot import compare_profiles, convert_urdf_to_usd, inspect_robot
from .runtime import execute_episode, project_root, recompute_metrics
from .scaffold import scaffold_policy_manifest
from .skill import discover_skills, extract_portable_members, probe_skill, validate_skill
from .suites import materialize_suite
from .worlds import apply_geometry_modifiers, export_obj, generate_ae, generate_aee, generate_terrain, preview_svg


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _robot_profile(path: str) -> RobotProfile:
    return RobotProfile.from_dict(load_data(path))


def _world(path: str) -> WorldSpec:
    return WorldSpec.from_dict(load_data(path))


def _parse_overrides(values: List[str]) -> Dict[str, Any]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError("world override must use KEY=VALUE")
        key, raw = value.split("=", 1)
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def _doctor(arguments: argparse.Namespace) -> int:
    result = run_doctor(arguments.config)
    if arguments.output:
        dump_json(arguments.output, result)
    _json(result)
    return 0


def _preflight(arguments: argparse.Namespace) -> int:
    result = preflight_examination(arguments.config, arguments.output, not arguments.static_only)
    _json(result)
    return 0 if result["ready"] else 2


def _launch(arguments: argparse.Namespace) -> int:
    """Production entry point: gate first, then create/resume one examination."""
    preflight_output = str(Path(arguments.output).resolve() / "preflight.json") if arguments.output else None
    preflight = preflight_examination(arguments.config, preflight_output, not arguments.static_preflight)
    if not preflight["ready"]:
        _json(
            {
                "status": "blocked",
                "reason": "preflight did not pass; no policy or simulator episode was started",
                "preflight": preflight,
            }
        )
        return 2
    state = initialize_examination(arguments.config, arguments.output)
    state_path = Path(state["settings"]["output_dir"]) / "examination_state.json"
    state["preflight"] = {
        "status": preflight["status"],
        "report": preflight["output"],
        "gates": preflight["gates"],
    }
    dump_json(str(state_path), state)
    if arguments.materialize_only:
        _json(
            {
                "status": state["status"],
                "state": str(state_path),
                "preflight": preflight["output"],
                "execution_started": False,
                "next_command": "robustlab examine resume %s" % str(state_path),
            }
        )
        return 0
    result = run_examination(str(state_path), arguments.max_trials)
    result["preflight"] = preflight["output"]
    _json(result)
    return 0 if result.get("status") != "error" else 2


def _assets(arguments: argparse.Namespace) -> int:
    if arguments.assets_action == "fetch":
        result = fetch_assets(arguments.registry, arguments.cache)
    else:
        result = verify_assets(arguments.registry, arguments.cache)
    _json(result)
    return 0 if result["valid"] else 2


def _robot(arguments: argparse.Namespace) -> int:
    if arguments.robot_action == "inspect":
        profile = inspect_robot(arguments.path, arguments.format, arguments.override, arguments.package_path)
        dump_json(arguments.output, profile)
        _json({"status": "ok", "output": str(Path(arguments.output).resolve()), "warnings": profile.warnings, "profile": profile.to_dict()})
        return 0
    if arguments.robot_action == "convert-urdf":
        result = convert_urdf_to_usd(arguments.path, arguments.output, arguments.launcher, arguments.stiffness, arguments.damping, arguments.timeout)
        _json(result)
        return 0 if result["status"] == "converted" else 5
    profiles = [_robot_profile(path) for path in arguments.profiles]
    result = compare_profiles(profiles)
    if arguments.output:
        dump_json(arguments.output, result)
    _json(result)
    return 0 if result["comparable"] else 3


def _policy(arguments: argparse.Namespace) -> int:
    joint_order = arguments.joint_order
    if joint_order is not None:
        joint_order = [
            name
            for value in joint_order
            for name in value.split(",")
            if name
        ]
    result = scaffold_policy_manifest(
        arguments.artifact,
        joint_order=joint_order,
        output=arguments.output,
        robot_profile=arguments.robot_profile,
    )
    _json(result)
    return 0


def _world_command(arguments: argparse.Namespace) -> int:
    if arguments.world_action == "parameter-diagrams":
        result = generate_parameter_diagrams(arguments.output)
        _json({"status": "ok", "diagram_count": result["diagram_count"], "index": result["index"], "contact_sheet": result["contact_sheet"]})
        return 0
    if arguments.world_action == "generate":
        robot = _robot_profile(arguments.robot)
        if arguments.terrain == "ae":
            world = generate_ae(robot, arguments.purple, arguments.brown, arguments.difficulty, arguments.seed, arguments.reversed, _parse_overrides(arguments.set))
        elif arguments.terrain == "aee":
            options = _parse_overrides(arguments.set)
            options.setdefault("rows", arguments.rows)
            options.setdefault("columns", arguments.columns)
            options.setdefault("pair_mode", arguments.pair_mode)
            options.setdefault("topology", arguments.topology)
            options.setdefault("purple", arguments.purple)
            options.setdefault("brown", arguments.brown)
            world = generate_aee(robot, arguments.layout, arguments.difficulty, arguments.seed, arguments.count, options)
        else:
            world = generate_terrain(arguments.terrain, robot, arguments.difficulty, arguments.seed, _parse_overrides(arguments.set))
        world = apply_geometry_modifiers(world, arguments.camber_deg, arguments.curve_angle_deg, arguments.curve_segments)
        dump_json(arguments.output, world)
        _json({"status": "ok", "output": str(Path(arguments.output).resolve()), "name": world.name, "geometry_count": len(world.geometry), "region_count": len(world.regions), "clips": world.clips})
        return 0
    world = _world(arguments.world if arguments.world_action == "render-mjcf" else arguments.path)
    if arguments.world_action == "validate":
        world.validate()
        _json({"valid": True, "name": world.name, "geometry_count": len(world.geometry), "region_count": len(world.regions), "route_count": len(world.routes)})
    elif arguments.world_action == "preview":
        preview_svg(world, arguments.output, arguments.width)
        _json({"status": "ok", "output": str(Path(arguments.output).resolve())})
    elif arguments.world_action == "export-mjcf":
        robot = _robot_profile(arguments.robot)
        if world.robot_profile_sha256 != robot.source_sha256:
            raise ValueError("world was scaled for a different RobotProfile; regenerate it with the supplied profile")
        from .backends.mujoco import build_mjcf_scene

        build_mjcf_scene(arguments.robot_asset, world, arguments.output, arguments.dt)
        _json(
            {
                "status": "ok",
                "output": str(Path(arguments.output).resolve()),
                "world": world.name,
                "robot_asset": str(Path(arguments.robot_asset).resolve()),
            }
        )
    elif arguments.world_action == "render-mjcf":
        from .rendering import render_mjcf_scene

        result = render_mjcf_scene(
            arguments.path,
            world,
            arguments.output,
            arguments.width,
            arguments.height,
            arguments.azimuth,
            arguments.elevation,
            arguments.distance_scale,
            arguments.gl,
        )
        _json({"status": "ok", **result})
    else:
        export_obj(world, arguments.output)
        _json({"status": "ok", "output": str(Path(arguments.output).resolve())})
    return 0


def _run(arguments: argparse.Namespace) -> int:
    manifest = execute_episode(arguments.config, arguments.runtimes, arguments.output)
    _json(manifest.to_dict())
    return 0 if manifest.status in {"completed", "skipped"} else 4


def _metrics(arguments: argparse.Namespace) -> int:
    result = recompute_metrics(arguments.run_dir)
    _json(result)
    return 0


def _report(arguments: argparse.Namespace) -> int:
    source = Path(arguments.run_dir)
    output = generate_aggregate_report(str(source), arguments.output) if source.is_file() else generate_report(str(source), arguments.output)
    _json({"status": "ok", "output": str(Path(output).resolve())})
    return 0


def _replay(arguments: argparse.Namespace) -> int:
    log = load_data(str(Path(arguments.run_dir) / "episode_log.json"))
    samples = log.get("samples", [])
    stride = max(1, arguments.stride)
    trace = [{"time_s": item["time_s"], "base_position_m": item["base_position_m"], "action": item.get("action", [])} for item in samples[::stride]]
    result = {"episode_id": log.get("episode_id"), "backend": log.get("backend"), "status": log.get("status"), "terminal_reason": log.get("terminal_reason"), "trace": trace}
    if arguments.output:
        dump_json(arguments.output, result)
        _json({"status": "ok", "output": str(Path(arguments.output).resolve()), "samples": len(trace)})
    else:
        _json(result)
    return 0


def _calibrate(arguments: argparse.Namespace) -> int:
    if arguments.calibration_action == "generate":
        result = generate_calibration_suite(_robot_profile(arguments.robot), arguments.output)
    else:
        result = analyze_calibration(arguments.run_dirs, arguments.output)
    _json(result)
    return 0


def _suite(arguments: argparse.Namespace) -> int:
    result = materialize_suite(arguments.config, arguments.robot, arguments.backend, arguments.robot_asset, arguments.robot_format, arguments.controller, arguments.output)
    _json(result)
    return 0


def _aggregate(arguments: argparse.Namespace) -> int:
    result = aggregate_runs(arguments.run_dirs, arguments.output, arguments.conformance, arguments.calibration)
    _json(result)
    return 0


def _gallery(arguments: argparse.Namespace) -> int:
    result = generate_variation_gallery(arguments.config, arguments.robot, arguments.robot_asset, arguments.output)
    _json(
        {
            "status": "ok" if result["failed_count"] == 0 else "partial",
            "variation_count": result["variation_count"],
            "rendered_count": result["rendered_count"],
            "failed_count": result["failed_count"],
            "index": result["index"],
            "contact_sheet": result["contact_sheet"],
            "manifest": result["manifest"],
        }
    )
    return 0 if result["failed_count"] == 0 else 6


def _benchmark(arguments: argparse.Namespace) -> int:
    if arguments.benchmark_action == "materialize":
        backends = arguments.backend or ["fake"]
        assets: Dict[str, Dict[str, str]] = {}
        for value in arguments.asset:
            try:
                backend, robot_format, path = value.split("=", 2)
            except ValueError:
                raise ValueError("--asset must use BACKEND=FORMAT=PATH")
            assets[backend] = {"format": robot_format, "path": path}
        if arguments.robot_asset:
            for backend in backends:
                assets.setdefault(backend, {"format": arguments.robot_format, "path": arguments.robot_asset})
        missing = [backend for backend in backends if backend not in assets]
        if missing:
            raise ValueError("robot asset missing for backends: %s" % ", ".join(missing))
        result = materialize_benchmark(arguments.config, arguments.profile, arguments.robot, backends, assets, arguments.controller, arguments.output)
    elif arguments.benchmark_action == "render":
        result = render_benchmark(arguments.manifest, arguments.robot_asset, arguments.gl)
    elif arguments.benchmark_action in {"run", "resume"}:
        result = run_benchmark(arguments.manifest, arguments.runtimes, arguments.limit, arguments.benchmark_action == "resume")
    else:
        result = summarize_benchmark(arguments.manifest, arguments.output, arguments.conformance, arguments.calibration)
    _json(result)
    return 0


def _skill(arguments: argparse.Namespace) -> int:
    if arguments.skill_action == "discover":
        result = discover_skills(arguments.root, arguments.limit)
        if arguments.output:
            dump_json(arguments.output, result)
    elif arguments.skill_action == "validate":
        result = validate_skill(arguments.manifest)
        if arguments.output:
            dump_json(arguments.output, result)
    elif arguments.skill_action == "probe":
        result = probe_skill(arguments.manifest, arguments.warmup, arguments.repetitions)
        if arguments.output:
            dump_json(arguments.output, result)
    else:
        result = extract_portable_members(arguments.bundle, arguments.output)
    _json(result)
    return 0 if result.get("valid", result.get("status") in {"ok", "passed"}) else 2


def _examine(arguments: argparse.Namespace) -> int:
    if arguments.examine_action == "start":
        state = initialize_examination(arguments.config, arguments.output)
        state_path = str(Path(state["settings"]["output_dir"]) / "examination_state.json")
        result = {
            "status": state["status"],
            "stage": state["stage"],
            "state": state_path,
            "compatibility": state.get("compatibility", {}),
        }
        if state.get("block_reason"):
            result["block_reason"] = state["block_reason"]
        if not arguments.materialize_only:
            result = run_examination(state_path, arguments.max_trials)
    elif arguments.examine_action == "resume":
        result = run_examination(arguments.state, arguments.max_trials)
    elif arguments.examine_action == "status":
        result = examination_status(arguments.state)
    elif arguments.examine_action == "diagnose":
        result = diagnose_examination(arguments.state, arguments.output)
    else:
        result = generate_capability_passport(arguments.state, arguments.output)
    _json(result)
    return 0 if result.get("status") not in {"error"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robustlab", description="Cross-simulator legged-robot robustness laboratory")
    parser.add_argument("--version", action="version", version="robustness-lab 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="inspect simulator and hardware capabilities")
    doctor.add_argument("--config", default=str(project_root() / "configs" / "runtimes.yaml"))
    doctor.add_argument("--output", help="write the environment evidence as JSON")
    doctor.set_defaults(handler=_doctor)

    preflight = subparsers.add_parser(
        "preflight",
        help="validate robot, policy, runtime, sensors, navigation, and environment physics without running an episode",
    )
    preflight.add_argument("config", help="complete examiner YAML")
    preflight.add_argument("--output", help="write the standalone readiness report as JSON")
    preflight.add_argument("--static-only", action="store_true", help="check files and contracts without importing/probing simulator runtimes")
    preflight.set_defaults(handler=_preflight)

    launch = subparsers.add_parser(
        "launch",
        help="preflight and launch a complete resumable examination from one YAML file",
    )
    launch.add_argument("config", help="complete examiner YAML")
    launch.add_argument("--output", help="override the examination output directory")
    launch.add_argument("--max-trials", type=int, help="stop cleanly after this many attempts; resume later")
    launch.add_argument("--materialize-only", action="store_true", help="create state and dashboard but execute no episode")
    launch.add_argument("--static-preflight", action="store_true", help="skip live runtime probes during preflight")
    launch.set_defaults(handler=_launch)

    assets = subparsers.add_parser("assets", help="fetch or verify pinned robot/course assets")
    assets_sub = assets.add_subparsers(dest="assets_action", required=True)
    for action in ("fetch", "verify"):
        command = assets_sub.add_parser(action)
        command.add_argument("--registry", default=str(project_root() / "assets" / "sources.yaml"))
        command.add_argument("--cache")
        command.set_defaults(handler=_assets)

    robot = subparsers.add_parser("robot", help="inspect and compare robot descriptions")
    robot_sub = robot.add_subparsers(dest="robot_action", required=True)
    inspect = robot_sub.add_parser("inspect")
    inspect.add_argument("path")
    inspect.add_argument("--format", choices=["urdf", "mjcf", "xml", "usd"])
    inspect.add_argument("--override")
    inspect.add_argument("--package-path", action="append", default=[])
    inspect.add_argument("--output", required=True)
    inspect.set_defaults(handler=_robot)
    convert = robot_sub.add_parser("convert-urdf")
    convert.add_argument("path")
    convert.add_argument("--output", required=True)
    convert.add_argument("--launcher", default="/usr/local/bin/isaaclab")
    convert.add_argument("--stiffness", type=float, default=25.0)
    convert.add_argument("--damping", type=float, default=0.5)
    convert.add_argument("--timeout", type=float, default=300.0)
    convert.set_defaults(handler=_robot)
    compare = robot_sub.add_parser("compare")
    compare.add_argument("profiles", nargs="+")
    compare.add_argument("--output")
    compare.set_defaults(handler=_robot)

    policy = subparsers.add_parser("policy", help="create and inspect policy manifests")
    policy_sub = policy.add_subparsers(dest="policy_action", required=True)
    policy_scaffold = policy_sub.add_parser(
        "scaffold", help="probe TorchScript weights and draft a schema-v2 manifest"
    )
    policy_scaffold.add_argument("artifact", help="TorchScript policy artifact")
    joint_source = policy_scaffold.add_mutually_exclusive_group(required=True)
    joint_source.add_argument(
        "--joint-order",
        nargs="+",
        help="policy joint names, separated by spaces or commas",
    )
    joint_source.add_argument(
        "--robot-profile",
        help="RobotProfile JSON/YAML whose controller_joint_order should be used",
    )
    policy_scaffold.add_argument("--output", required=True)
    policy_scaffold.set_defaults(handler=_policy)

    world = subparsers.add_parser("world", help="generate, validate, and export worlds")
    world_sub = world.add_subparsers(dest="world_action", required=True)
    generate = world_sub.add_parser("generate")
    generate.add_argument("--terrain", required=True, choices=["rs%d" % index for index in range(1, 12)] + ["ae", "aee"])
    generate.add_argument("--robot", required=True)
    generate.add_argument("--difficulty", type=float, default=0.5)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    generate.add_argument("--purple", default="rs1")
    generate.add_argument("--brown", default="rs2")
    generate.add_argument("--reversed", action="store_true")
    generate.add_argument("--camber-deg", type=float, default=0.0, help="sideways bank: positive raises the left side, negative raises the right")
    generate.add_argument("--curve-angle-deg", type=float, default=0.0, help="plan-view turn: positive left/anticlockwise, negative right/clockwise")
    generate.add_argument("--curve-segments", type=int, default=24, help="collision segments used to approximate a curved route")
    generate.add_argument("--layout", choices=["sparse", "exhaustive", "random", "long", "grid"], default="sparse")
    generate.add_argument("--count", type=int, default=12)
    generate.add_argument("--rows", type=int, default=3, help="junction rows for AEE grid layout")
    generate.add_argument("--columns", type=int, default=3, help="junction columns for AEE grid layout")
    generate.add_argument("--pair-mode", choices=["cycle", "random", "repeat", "adaptive"], default="cycle", help="AE pair assignment for AEE graph edges")
    generate.add_argument("--topology", choices=["chain", "ring", "tree", "grid", "maze", "random_connected"], default="grid", help="junction graph topology for AEE grid layout")
    generate.add_argument("--output", required=True)
    generate.set_defaults(handler=_world_command)
    for action in ("validate", "preview", "export"):
        command = world_sub.add_parser(action)
        command.add_argument("path")
        if action != "validate":
            command.add_argument("--output", required=True)
        if action == "preview":
            command.add_argument("--width", type=int, default=1400)
        command.set_defaults(handler=_world_command)
    export_mjcf = world_sub.add_parser("export-mjcf", help="combine a WorldSpec and MJCF robot into a viewable MuJoCo scene")
    export_mjcf.add_argument("path")
    export_mjcf.add_argument("--robot", required=True, help="RobotProfile used to scale the world")
    export_mjcf.add_argument("--robot-asset", required=True, help="source MJCF/XML robot model")
    export_mjcf.add_argument("--dt", type=float, default=0.002)
    export_mjcf.add_argument("--output", required=True)
    export_mjcf.set_defaults(handler=_world_command)
    render_mjcf = world_sub.add_parser("render-mjcf", help="render a headless PNG from a resolved MuJoCo scene")
    render_mjcf.add_argument("path", help="resolved MJCF/XML scene")
    render_mjcf.add_argument("--world", required=True, help="WorldSpec JSON used to build the scene")
    render_mjcf.add_argument("--output", required=True)
    render_mjcf.add_argument("--width", type=int, default=640)
    render_mjcf.add_argument("--height", type=int, default=360)
    render_mjcf.add_argument("--azimuth", type=float, default=138.0)
    render_mjcf.add_argument("--elevation", type=float, default=-30.0)
    render_mjcf.add_argument("--distance-scale", type=float, default=0.78)
    render_mjcf.add_argument("--gl", choices=["osmesa", "egl"], default="osmesa")
    render_mjcf.set_defaults(handler=_world_command)
    parameter_diagrams = world_sub.add_parser("parameter-diagrams", help="generate labelled RS1-RS11 changeable-parameter PNG diagrams")
    parameter_diagrams.add_argument("--output", required=True)
    parameter_diagrams.set_defaults(handler=_world_command)

    run = subparsers.add_parser("run", help="execute an isolated simulator episode")
    run.add_argument("config")
    run.add_argument("--runtimes", default=str(project_root() / "configs" / "runtimes.yaml"))
    run.add_argument("--output")
    run.set_defaults(handler=_run)
    metrics = subparsers.add_parser("metrics", help="recompute metrics from canonical telemetry")
    metrics_sub = metrics.add_subparsers(dest="metrics_action", required=True)
    compute = metrics_sub.add_parser("compute")
    compute.add_argument("run_dir")
    compute.set_defaults(handler=_metrics)
    report = subparsers.add_parser("report", help="generate an episode or aggregate static evidence report")
    report.add_argument("run_dir", help="episode run directory or aggregate JSON file")
    report.add_argument("--output", required=True)
    report.set_defaults(handler=_report)
    replay = subparsers.add_parser("replay", help="export deterministic state/action trace")
    replay.add_argument("run_dir")
    replay.add_argument("--stride", type=int, default=1)
    replay.add_argument("--output")
    replay.set_defaults(handler=_replay)
    calibration = subparsers.add_parser("calibrate", help="generate or analyze backend calibration fixtures")
    calibration_sub = calibration.add_subparsers(dest="calibration_action", required=True)
    calibration_generate = calibration_sub.add_parser("generate")
    calibration_generate.add_argument("--robot", required=True)
    calibration_generate.add_argument("--output", required=True)
    calibration_generate.set_defaults(handler=_calibrate)
    calibration_analyze = calibration_sub.add_parser("analyze")
    calibration_analyze.add_argument("run_dirs", nargs="+")
    calibration_analyze.add_argument("--output", required=True)
    calibration_analyze.set_defaults(handler=_calibrate)
    suite = subparsers.add_parser("suite", help="materialize a deterministic experiment matrix")
    suite.add_argument("config")
    suite.add_argument("--robot", required=True)
    suite.add_argument("--backend", required=True, choices=["fake", "mujoco", "isaaclab", "isaacgym"])
    suite.add_argument("--robot-asset", required=True)
    suite.add_argument("--robot-format", required=True, choices=["urdf", "mjcf", "usd"])
    suite.add_argument("--controller", required=True)
    suite.add_argument("--output", required=True)
    suite.set_defaults(handler=_suite)
    aggregate = subparsers.add_parser("aggregate", help="aggregate repeated trials and cross-backend deltas")
    aggregate.add_argument("run_dirs", nargs="+")
    aggregate.add_argument("--conformance")
    aggregate.add_argument("--calibration")
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=_aggregate)
    gallery = subparsers.add_parser("gallery", help="generate worlds, MuJoCo scenes, PNG renders, and a clickable catalogue")
    gallery.add_argument("config")
    gallery.add_argument("--robot", required=True)
    gallery.add_argument("--robot-asset", required=True)
    gallery.add_argument("--output", required=True)
    gallery.set_defaults(handler=_gallery)
    benchmark = subparsers.add_parser("benchmark", help="materialize, render, execute, resume, and summarize L0-L7 benchmarks")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_action", required=True)
    benchmark_materialize = benchmark_sub.add_parser("materialize", help="expand a BenchmarkSuiteSpec into immutable worlds and episodes")
    benchmark_materialize.add_argument("config")
    benchmark_materialize.add_argument("--profile", choices=["smoke", "quick", "standard", "exhaustive"], default="smoke")
    benchmark_materialize.add_argument("--robot", required=True)
    benchmark_materialize.add_argument("--backend", action="append", choices=["fake", "mujoco", "isaaclab", "isaacgym"])
    benchmark_materialize.add_argument("--asset", action="append", default=[], metavar="BACKEND=FORMAT=PATH", help="backend-specific robot asset; repeat for multiple engines")
    benchmark_materialize.add_argument("--robot-asset", help="single asset used for every selected backend")
    benchmark_materialize.add_argument("--robot-format", choices=["urdf", "mjcf", "usd"], default="mjcf")
    benchmark_materialize.add_argument("--controller", required=True)
    benchmark_materialize.add_argument("--output", required=True)
    benchmark_materialize.set_defaults(handler=_benchmark)
    benchmark_render = benchmark_sub.add_parser("render", help="render every unique preview hash exactly once")
    benchmark_render.add_argument("manifest")
    benchmark_render.add_argument("--robot-asset", required=True, help="MuJoCo robot MJCF used as the canonical scale reference")
    benchmark_render.add_argument("--gl", choices=["osmesa", "egl"], default="osmesa")
    benchmark_render.set_defaults(handler=_benchmark)
    for action in ("run", "resume"):
        benchmark_run = benchmark_sub.add_parser(action)
        benchmark_run.add_argument("manifest")
        benchmark_run.add_argument("--runtimes", default=str(project_root() / "configs" / "runtimes.yaml"))
        benchmark_run.add_argument("--limit", type=int)
        benchmark_run.set_defaults(handler=_benchmark)
    benchmark_summary = benchmark_sub.add_parser("summarize")
    benchmark_summary.add_argument("manifest")
    benchmark_summary.add_argument("--output", required=True)
    benchmark_summary.add_argument("--conformance")
    benchmark_summary.add_argument("--calibration")
    benchmark_summary.set_defaults(handler=_benchmark)

    skill = subparsers.add_parser("skill", help="discover and validate locomotion skill packages")
    skill_sub = skill.add_subparsers(dest="skill_action", required=True)
    skill_discover = skill_sub.add_parser("discover", help="classify policy artifacts without deserializing checkpoints")
    skill_discover.add_argument("root")
    skill_discover.add_argument("--limit", type=int, default=50)
    skill_discover.add_argument("--output")
    skill_discover.set_defaults(handler=_skill)
    skill_validate = skill_sub.add_parser("validate", help="validate sensor, observation, action, and artifact mappings")
    skill_validate.add_argument("manifest")
    skill_validate.add_argument("--output", help="write the validation evidence as JSON")
    skill_validate.set_defaults(handler=_skill)
    skill_probe = skill_sub.add_parser("probe", help="execute a portable artifact on declared zero-shaped inputs")
    skill_probe.add_argument("manifest")
    skill_probe.add_argument("--warmup", type=int, default=3)
    skill_probe.add_argument("--repetitions", type=int, default=20)
    skill_probe.add_argument("--output", help="write the probe evidence as JSON")
    skill_probe.set_defaults(handler=_skill)
    skill_extract = skill_sub.add_parser("extract", help="extract portable policy exports from a trusted local bundle")
    skill_extract.add_argument("bundle")
    skill_extract.add_argument("--output", required=True)
    skill_extract.set_defaults(handler=_skill)

    examine = subparsers.add_parser("examine", help="run a compute-aware adaptive locomotion examination")
    examine_sub = examine.add_subparsers(dest="examine_action", required=True)
    examine_start = examine_sub.add_parser("start", help="initialize and optionally execute an adaptive examination")
    examine_start.add_argument("config")
    examine_start.add_argument("--output")
    examine_start.add_argument("--max-trials", type=int)
    examine_start.add_argument("--materialize-only", action="store_true")
    examine_start.set_defaults(handler=_examine)
    examine_resume = examine_sub.add_parser("resume", help="resume from the persisted scheduler state")
    examine_resume.add_argument("state")
    examine_resume.add_argument("--max-trials", type=int)
    examine_resume.set_defaults(handler=_examine)
    examine_status_parser = examine_sub.add_parser("status")
    examine_status_parser.add_argument("state")
    examine_status_parser.set_defaults(handler=_examine)
    examine_diagnose = examine_sub.add_parser("diagnose", help="cluster failures and attach matched counterfactual evidence")
    examine_diagnose.add_argument("state")
    examine_diagnose.add_argument("--output")
    examine_diagnose.set_defaults(handler=_examine)
    examine_report = examine_sub.add_parser("report", help="generate the measured/inferred capability passport")
    examine_report.add_argument("state")
    examine_report.add_argument("--output")
    examine_report.set_defaults(handler=_examine)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
