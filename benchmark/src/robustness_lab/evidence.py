"""Evidence aggregation, headless snapshots, video summaries, and dashboard."""

from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import html
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .events import quaternion_to_rpy
from .io import atomic_write_text, dump_json, load_data, sha256_file, write_csv
from .statistics import percentile, wilson_interval


FAILURE_EVENTS = {"slip_start", "body_contact", "fall", "stuck"}


def _relative(path: Optional[str], root: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _relative_file(path: Optional[str], root: Path) -> Optional[str]:
    """Return a report link only when its file actually exists."""
    return _relative(path, root) if path and Path(path).is_file() else None


def _metric_map(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    return {
        item["name"]: float(item["value"])
        for item in load_data(str(path)).get("metrics", [])
        if item.get("valid") and isinstance(item.get("value"), (int, float))
    }


def _event_context(log: Dict[str, Any]) -> Tuple[str, float, str]:
    events = log.get("events", [])
    first = next((item for item in events if item.get("event_type") in FAILURE_EVENTS), None)
    if first:
        return str(first.get("event_type")), float(first.get("time_s", 0.0)), str(first.get("region", ""))
    terminal = str(log.get("terminal_reason", "unknown"))
    samples = log.get("samples", [])
    return terminal, float(samples[-1].get("time_s", 0.0)) if samples else 0.0, ""


def _sample_at(samples: Sequence[Dict[str, Any]], time_s: float) -> Dict[str, Any]:
    return min(samples, key=lambda item: abs(float(item.get("time_s", 0.0)) - time_s))


def _episode_or_trajectory(record: Dict[str, Any], examination_root: Path) -> Optional[Dict[str, Any]]:
    """Load canonical episode data, or adapt compact native trajectory samples."""
    episode_path = Path(record["run_dir"]) / "episode_log.json"
    if episode_path.is_file():
        return load_data(str(episode_path))
    trajectory_name = (record.get("telemetry") or {}).get("trajectory_json")
    if not trajectory_name:
        return None
    trajectory_path = Path(trajectory_name)
    if not trajectory_path.is_absolute():
        trajectory_path = examination_root / trajectory_path
    if not trajectory_path.is_file():
        return None
    raw = load_data(str(trajectory_path))
    frequency = max(1.0, float(raw.get("control_frequency_hz", 0.0) or 0.0))
    samples, events = [], []
    for item in raw.get("samples", []):
        time_s = float(item.get("step", 0)) / frequency
        samples.append({
            "time_s": time_s,
            "base_position_m": item.get("position_m", [0.0, 0.0, 0.0]),
            "base_linear_velocity_mps": item.get("linear_velocity_mps", [0.0, 0.0, 0.0]),
            "base_quaternion_wxyz": item.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
        })
        if item.get("episode_end"):
            events.append({"time_s": time_s, "event_type": "episode_end"})
    if samples:
        events.append({"time_s": samples[-1]["time_s"], "event_type": str(record.get("terminal_reason", "terminal"))})
    return {"samples": samples, "events": events, "terminal_reason": record.get("terminal_reason"), "source": "native_compact_trajectory"}


def _bounds(world: Dict[str, Any]) -> Tuple[float, float, float, float]:
    boxes = [item for item in world.get("geometry", []) if item.get("shape") == "box"]
    x0 = min((item["pose"]["position"][0] - item["size"][0] / 2 for item in boxes), default=0.0)
    x1 = max((item["pose"]["position"][0] + item["size"][0] / 2 for item in boxes), default=float(world["dimensions_m"][0]))
    y0 = min((item["pose"]["position"][1] - item["size"][1] / 2 for item in boxes), default=-float(world["dimensions_m"][1]) / 2)
    y1 = max((item["pose"]["position"][1] + item["size"][1] / 2 for item in boxes), default=float(world["dimensions_m"][1]) / 2)
    return float(x0), float(x1), float(y0), float(y1)


def _draw_frame(base_path: Path, world: Dict[str, Any], samples: Sequence[Dict[str, Any]], sample: Dict[str, Any], output: Path, label: str, event: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(str(base_path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin = 48
    x0, x1, y0, y1 = _bounds(world)
    scale = min((width - 2 * margin) / max(x1 - x0, 0.1), (height - 2 * margin) / max(y1 - y0, 0.1))

    def pixel(point: Sequence[float]) -> Tuple[float, float]:
        return margin + (float(point[0]) - x0) * scale, height - margin - (float(point[1]) - y0) * scale

    upto = [item for item in samples if float(item.get("time_s", 0.0)) <= float(sample.get("time_s", 0.0))]
    if len(upto) > 1:
        segment = [upto[0]]
        for item in upto[1:]:
            previous, current = segment[-1]["base_position_m"], item["base_position_m"]
            jump = math.sqrt(sum((float(current[axis]) - float(previous[axis])) ** 2 for axis in range(min(3, len(current), len(previous)))))
            if jump >= 1.0:
                if len(segment) > 1:
                    draw.line([pixel(value["base_position_m"]) for value in segment], fill="#7c3aed", width=4)
                segment = [item]
            else:
                segment.append(item)
        if len(segment) > 1:
            draw.line([pixel(value["base_position_m"]) for value in segment], fill="#7c3aed", width=4)
    px, py = pixel(sample["base_position_m"])
    draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill="#ef4444" if event else "#16a34a", outline="white", width=2)
    roll, pitch, _ = quaternion_to_rpy(sample.get("base_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))
    speed = math.sqrt(sum(float(value) ** 2 for value in sample.get("base_linear_velocity_mps", [0.0, 0.0])[:2]))
    lines = [
        label,
        "t=%.2fs  speed=%.2fm/s  roll=%.1fdeg  pitch=%.1fdeg" % (
            float(sample.get("time_s", 0.0)), speed, math.degrees(roll), math.degrees(pitch)
        ),
        "event=%s" % (event or "none"),
    ]
    draw.rounded_rectangle((12, height - 82, min(width - 12, 650), height - 12), radius=8, fill="#0f172ee6")
    for index, line in enumerate(lines):
        draw.text((24, height - 74 + index * 20), line, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output))


def render_trial_snapshots(record: Dict[str, Any], examination_root: Path) -> List[Dict[str, Any]]:
    """Render auditable top-down state snapshots from canonical telemetry."""
    if not record.get("valid_behavior"):
        return []
    log = _episode_or_trajectory(record, examination_root)
    if not log:
        return _snapshots_from_video(record, examination_root)
    samples = log.get("samples", [])
    if not samples:
        return _snapshots_from_video(record, examination_root)
    world = load_data(record["world"])
    media_dir = examination_root / "evidence" / "snapshots" / record["trial_id"]
    base_path = media_dir / "world-base.png"
    if not base_path.is_file():
        from .models import WorldSpec
        from .rendering import render_topdown

        render_topdown(WorldSpec.from_dict(world), str(base_path), width=900, height=700)
    events = log.get("events", [])
    event_type, event_time, _ = _event_context(log)
    last_time = float(samples[-1].get("time_s", 0.0))
    if record.get("success") is False:
        selections = [("before", max(0.0, event_time - 1.0), ""), ("event", event_time, event_type), ("terminal", last_time, str(log.get("terminal_reason", "failed")))]
    else:
        transition = next((item for item in events if item.get("event_type") == "transition"), None)
        demanding = max(
            samples,
            key=lambda item: max(abs(value) for value in quaternion_to_rpy(item.get("base_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))[:2]),
        )
        selections = [
            ("entry", float(transition.get("time_s", 0.0)) if transition else 0.0, "transition" if transition else "start"),
            ("demanding", float(demanding.get("time_s", 0.0)), "maximum orientation"),
            ("goal", last_time, "goal"),
        ]
    artifacts = []
    for role, time_s, event in selections:
        sample = _sample_at(samples, time_s)
        full = media_dir / (role + ".png")
        thumb = media_dir / (role + "-thumb.png")
        _draw_frame(base_path, world, samples, sample, full, "%s · %s" % (record["trial_id"], role), event)
        from PIL import Image

        image = Image.open(str(full))
        image.thumbnail((360, 280))
        image.save(str(thumb))
        artifacts.append({"role": role, "time_s": float(sample.get("time_s", 0.0)), "event": event, "path": _relative(str(full), examination_root), "thumbnail": _relative(str(thumb), examination_root), "sha256": sha256_file(str(full))})
    return artifacts


def _snapshots_from_video(record: Dict[str, Any], examination_root: Path) -> List[Dict[str, Any]]:
    """Extract source-camera evidence when canonical native telemetry is absent."""
    media = record.get("media", {}).get("video") or {}
    raw_path = media.get("path")
    if not raw_path or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return []
    source = Path(raw_path)
    if not source.is_absolute():
        source = examination_root / source
    if not source.is_file():
        return []
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = max(0.1, float(probe.stdout.strip()))
    except (TypeError, ValueError):
        duration = 1.0
    roles = ("start", "middle", "terminal") if record.get("success") else ("before", "event", "terminal")
    times = (0.05 * duration, 0.50 * duration, 0.95 * duration)
    destination = examination_root / "evidence" / "snapshots" / record["trial_id"]
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    from PIL import Image, ImageDraw

    for role, time_s in zip(roles, times):
        full = destination / (role + ".png")
        completed = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "%.4f" % time_s, "-i", str(source), "-frames:v", "1", str(full)], capture_output=True, text=True, timeout=30)
        if completed.returncode != 0 or not full.is_file():
            continue
        image = Image.open(str(full)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, min(image.width - 8, 620), 58), fill="#0f172ee6")
        draw.text((18, 16), "%s · %s · t=%.2fs · native camera (telemetry unavailable)" % (record["trial_id"], role, time_s), fill="white")
        image.save(str(full))
        thumb = destination / (role + "-thumb.png")
        image.thumbnail((360, 280))
        image.save(str(thumb))
        artifacts.append({"role": role, "time_s": time_s, "event": "native-summary-only", "path": _relative(str(full), examination_root), "thumbnail": _relative(str(thumb), examination_root), "sha256": sha256_file(str(full))})
    return artifacts


def render_telemetry_video(record: Dict[str, Any], examination_root: Path) -> Optional[Dict[str, Any]]:
    """Create a compact trajectory video when a backend did not provide one."""
    native = record.get("native_summary", {}) if isinstance(record.get("native_summary"), dict) else {}
    native_video = native.get("video")
    if native_video and Path(native_video).is_file():
        return {"path": _relative(native_video, examination_root), "sha256": sha256_file(native_video), "source": "simulator"}
    log = _episode_or_trajectory(record, examination_root)
    if not log or not shutil.which("ffmpeg"):
        return None
    samples = log.get("samples", [])
    if not samples:
        return None
    world = load_data(record["world"])
    destination = examination_root / "evidence" / "videos" / (record["trial_id"] + ".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return {"path": _relative(str(destination), examination_root), "sha256": sha256_file(str(destination)), "source": "telemetry"}
    base_path = examination_root / "evidence" / "snapshots" / record["trial_id"] / "world-base.png"
    if not base_path.is_file():
        render_trial_snapshots(record, examination_root)
    count = min(120, len(samples))
    indices = sorted(set(round(index * (len(samples) - 1) / max(count - 1, 1)) for index in range(count)))
    temporary = Path(tempfile.mkdtemp(prefix="robustlab-video-"))
    try:
        event_type, event_time, _ = _event_context(log)
        for frame_index, sample_index in enumerate(indices):
            sample = samples[sample_index]
            near_event = event_type if abs(float(sample.get("time_s", 0.0)) - event_time) <= 0.15 else ""
            _draw_frame(base_path, world, samples, sample, temporary / ("frame-%05d.png" % frame_index), record["trial_id"], near_event)
        command = [
            "ffmpeg", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(temporary / "frame-%05d.png"),
            "-vf", "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            return None
    finally:
        shutil.rmtree(str(temporary), ignore_errors=True)
    return {"path": _relative(str(destination), examination_root), "sha256": sha256_file(str(destination)), "source": "telemetry"}


def _aggregate_condition(condition: Dict[str, Any], trials_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    trials = [trials_by_id[value] for value in condition.get("attempts", []) if value in trials_by_id]
    valid = [item for item in trials if item.get("valid_behavior") and isinstance(item.get("success"), bool)]
    passes = [item for item in valid if item["success"]]
    failures = [item for item in valid if not item["success"]]
    lower, upper = wilson_interval(len(passes), len(valid)) if valid else (None, None)
    failure_types = Counter(item.get("diagnosis", {}).get("failure_category", item.get("diagnosis", {}).get("classification", "behavioral/unknown")) for item in failures)
    infrastructure = Counter(item.get("terminal_reason", item.get("status", "unknown")) for item in trials if item not in valid)
    metric_values: Dict[str, List[float]] = {}
    for item in valid:
        for name, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(name, []).append(float(value))
    selected = {}
    for name in ("completion_time", "progress_fraction", "estimated_route_distance_m", "remaining_route_distance_m", "route_length_m", "goal_checkpoint_reached", "goal_checkpoint_count", "evaluation_steps", "episode_terminations", "aggregate_path_distance_m", "mean_base_speed_mps", "max_base_speed_mps", "max_abs_roll_rad", "max_abs_pitch_rad", "telemetry_sample_count", "velocity_tracking_rmse", "slip_ratio", "orientation_max", "absolute_mechanical_cot", "torque_saturation_fraction", "contact_impulse", "inference_latency_p95"):
        values = metric_values.get(name, [])
        if values:
            selected[name] = {"median": percentile(values, 0.5), "p95": percentile(values, 0.95), "minimum": min(values), "maximum": max(values)}
    return {
        **{key: value for key, value in condition.items() if key != "attempts"},
        "attempts": list(condition.get("attempts", [])),
        "attempt_count": len(trials),
        "valid_attempts": len(valid),
        "passes": len(passes),
        "failures": len(failures),
        "success_rate": len(passes) / len(valid) if valid else None,
        "lower_ci": lower,
        "upper_ci": upper,
        "failure_types": dict(failure_types),
        "infrastructure_outcomes": dict(infrastructure),
        "metrics": selected,
    }


def _route_length_m(world_path: Optional[str], cache: Dict[str, float]) -> float:
    if not world_path:
        return 0.0
    key = str(Path(world_path).resolve())
    if key in cache:
        return cache[key]
    try:
        world = load_data(key)
        routes = world.get("routes", [])
        waypoints = routes[0].get("waypoints", []) if routes else []
        length = sum(
            math.sqrt(sum((float(right[axis]) - float(left[axis])) ** 2 for axis in range(min(3, len(left), len(right)))))
            for left, right in zip(waypoints, waypoints[1:])
        )
        if length <= 0.0:
            spawn = world.get("spawn_pose", {}).get("position", [0.0, 0.0, 0.0])
            goal = world.get("goal_pose", {}).get("position", [0.0, 0.0, 0.0])
            length = math.sqrt(sum((float(goal[axis]) - float(spawn[axis])) ** 2 for axis in range(min(3, len(spawn), len(goal)))))
    except (OSError, ValueError, TypeError, KeyError):
        length = 0.0
    cache[key] = float(length)
    return float(length)


def _enrich_trial_performance(item: Dict[str, Any], route_cache: Dict[str, float]) -> Dict[str, Any]:
    """Recover auditable progress estimates from native goal-checkpoint summaries."""
    record = dict(item)
    metrics = dict(item.get("metrics", {}))
    native = item.get("native_summary") if isinstance(item.get("native_summary"), dict) else {}
    route_length = _route_length_m(item.get("world"), route_cache)
    goals = int(native.get("num_goals", 0) or 0)
    reached_index = int(native.get("max_goal_index", 0) or 0)
    existing_progress = metrics.get("progress_fraction")
    if goals > 1:
        # Native goal indices are zero-based: index 7 of 8 means the final goal.
        progress = max(0.0, min(1.0, reached_index / float(goals - 1)))
        if item.get("success") is True:
            progress = 1.0
        checkpoint_reached = max(1, min(goals, reached_index + 1))
        source = "highest native navigation checkpoint, interpolated along the canonical route"
    elif isinstance(existing_progress, (int, float)):
        progress = max(0.0, min(1.0, float(existing_progress)))
        checkpoint_reached = None
        source = "saved progress metric"
    else:
        progress = None
        checkpoint_reached = None
        source = "unavailable"
    if progress is not None:
        metrics["progress_fraction"] = progress
        if route_length > 0.0:
            metrics["route_length_m"] = route_length
            metrics["estimated_route_distance_m"] = route_length * progress
            metrics["remaining_route_distance_m"] = route_length * (1.0 - progress)
    if checkpoint_reached is not None:
        metrics["goal_checkpoint_reached"] = float(checkpoint_reached)
        metrics["goal_checkpoint_count"] = float(goals)
    for source_key, metric_key in (("steps", "evaluation_steps"), ("episode_terminations", "episode_terminations")):
        value = native.get(source_key)
        if isinstance(value, (int, float)):
            metrics[metric_key] = float(value)
    record["metrics"] = metrics
    record["performance_evidence"] = {
        "progress_source": source,
        "distance_is_estimated": bool(progress is not None and route_length > 0.0 and goals > 1),
        "true_path_distance_available": bool("path_distance" in metrics),
        "checkpoint_reached": checkpoint_reached,
        "checkpoint_count": goals or None,
    }
    return record


def build_dashboard_data(state: Dict[str, Any], examination_root: Path) -> Dict[str, Any]:
    route_cache: Dict[str, float] = {}
    trial_records = [_enrich_trial_performance(item, route_cache) for item in state.get("trials", [])]
    trials_by_id = {item["trial_id"]: item for item in trial_records}
    condition_map = state.get("conditions", {})
    ordered_ids = [value for value in state.get("condition_order", []) if value in condition_map]
    ordered_ids.extend(value for value in condition_map if value not in set(ordered_ids))
    conditions = [_aggregate_condition(condition_map[value], trials_by_id) for value in ordered_ids]
    level_counts: Dict[str, Counter] = {}
    for item in conditions:
        level_counts.setdefault(item["level"], Counter())[item.get("status", "pending")] += 1
    for level in ["L%d" % index for index in range(8)]:
        level_counts.setdefault(level, Counter())
    valid_trials = [item for item in trial_records if item.get("valid_behavior") and isinstance(item.get("success"), bool)]
    failures = [item for item in valid_trials if not item["success"]]
    infrastructure = [item for item in trial_records if item not in valid_trials]
    terminal_statuses = {"passed", "failed", "pruned", "unsupported", "blocked"}
    unfinished = [item for item in conditions if item.get("status") not in terminal_statuses]
    active = None if state.get("status") in {"completed", "stopped", "blocked"} else next(
        (item for item in unfinished if item.get("status") in {"running", "confirming"}),
        unfinished[0] if unfinished else None,
    )
    minimum_remaining_attempts = sum(max(0, int(item.get("target_repetitions", 3)) - int(item.get("valid_attempts", 0))) for item in unfinished)
    pass_evidence = []
    for condition in state.get("conditions", {}).values():
        trial_id = condition.get("representative_pass")
        if not trial_id or trial_id not in trials_by_id:
            continue
        item = trials_by_id[trial_id]
        pass_evidence.append({"condition_id": condition["id"], "level": condition["level"], "trial_id": trial_id, "metrics": item.get("metrics", {}), "media": item.get("media", {}), "seed": item.get("seed"), "world": _relative(item.get("world"), examination_root), "run_dir": _relative(item.get("run_dir"), examination_root)})
    failure_clusters: Dict[str, Dict[str, Any]] = {}
    for item in failures:
        category = item.get("diagnosis", {}).get("failure_category", "unknown")
        cluster = failure_clusters.setdefault(category, {"category": category, "count": 0, "conditions": [], "terminal_reasons": {}})
        cluster["count"] += 1
        if item.get("condition_id") not in cluster["conditions"]:
            cluster["conditions"].append(item.get("condition_id"))
        terminal = str(item.get("terminal_reason", "unknown"))
        cluster["terminal_reasons"][terminal] = cluster["terminal_reasons"].get(terminal, 0) + 1
    condition_status_counts = Counter(item.get("status", "pending") for item in conditions)
    videos = sum(bool((item.get("media", {}).get("video") or {}).get("path")) for item in trial_records)
    snapshots = sum(len(item.get("media", {}).get("snapshots", [])) for item in trial_records)
    native_logs = {
        _relative(item.get("native_log"), examination_root)
        for item in trial_records
        if item.get("native_log") and Path(item["native_log"]).is_file()
    }
    native_logs.discard(None)
    result_json = sum(
        bool(item.get("native_result") and Path(item["native_result"]).is_file())
        for item in trial_records
    )
    summary_json = sum(
        bool(item.get("run_dir") and (Path(item["run_dir"]) / "summary.json").is_file())
        for item in trial_records
    )
    return {
        "schema_version": "evidence-dashboard-2.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "examiner_id": state.get("examiner_id"),
        "status": state.get("status"),
        "stage": state.get("stage"),
        "updated_at": state.get("updated_at"),
        "scheduler": state.get("scheduler", {}),
        "policy": state.get("skill", {}),
        "backend": state.get("settings", {}).get("backend"),
        "progress": {
            "active_condition": active.get("id") if active else None,
            "completed_conditions": len(conditions) - len(unfinished),
            "remaining_materialized_conditions": len(unfinished),
            "minimum_remaining_attempts": minimum_remaining_attempts,
        },
        "trial_counts": {"all": len(state.get("trials", [])), "valid": len(valid_trials), "passes": len(valid_trials) - len(failures), "behavioral_failures": len(failures), "infrastructure": len(infrastructure)},
        "condition_counts": dict(condition_status_counts),
        "level_counts": {level: dict(counts) for level, counts in sorted(level_counts.items())},
        "level_results": state.get("level_results", {}),
        "variants": variant_summary(state.get("conditions", {}), state.get("trials", [])),
        "level_details": level_details(state.get("conditions", {}), state.get("trials", [])),
        "pair_details": pair_details(state.get("conditions", {}), state.get("trials", [])),
        "composed_levels": composed_levels(state.get("conditions", {}), state.get("trials", [])),
        "later_levels": later_levels_summary(
            {level: dict(counts) for level, counts in level_counts.items()},
            state.get("level_results", {}) or {},
            not any(
                item.get("kind") == "pair" and item.get("status") == "passed"
                for item in state.get("conditions", {}).values()
            ),
        ),
        "progression_pruning": {"description": "Each composed level is built only from the terrain pairs that passed, and a pair is dropped if its counterfactual later fails. There is no aggregate pass-rate gate: a level is absent only when nothing passed to build it from."},
        "evidence_inventory": {
            "videos": videos,
            "snapshots": snapshots,
            "native_runner_logs": len(native_logs),
            "result_json": result_json,
            "summary_json": summary_json,
            "canonical_telemetry_files": sum(bool(item.get("telemetry")) for item in trial_records),
        },
        "performance_availability": {
            "progress_attempts": sum(isinstance(item.get("metrics", {}).get("progress_fraction"), (int, float)) for item in valid_trials),
            "estimated_distance_attempts": sum(isinstance(item.get("metrics", {}).get("estimated_route_distance_m"), (int, float)) for item in valid_trials),
            "true_path_distance_attempts": sum(isinstance(item.get("metrics", {}).get("path_distance"), (int, float)) for item in valid_trials),
            "method": "The completed native run retained its highest reached navigation checkpoint. Distance is estimated by placing that checkpoint fraction on the canonical route; it is not odometry or integrated path length.",
        },
        "conditions": conditions,
        "pass_evidence": pass_evidence,
        "failure_clusters": sorted(failure_clusters.values(), key=lambda item: (-item["count"], item["category"])),
        "failures": [
            {
                "trial_id": item["trial_id"], "condition_id": item.get("condition_id"), "terminal_reason": item.get("terminal_reason"),
                "diagnosis": item.get("diagnosis", {}), "metrics": item.get("metrics", {}), "media": item.get("media", {}), "performance_evidence": item.get("performance_evidence", {}), "telemetry": item.get("telemetry", {}),
                "seed": item.get("seed"), "world": _relative(item.get("world"), examination_root), "run_dir": _relative(item.get("run_dir"), examination_root),
                "native_log": _relative_file(item.get("native_log"), examination_root), "native_result": _relative_file(item.get("native_result"), examination_root),
                "summary": _relative_file(str(Path(item["run_dir"]) / "summary.json"), examination_root) if item.get("run_dir") else None,
            }
            for item in failures
        ],
        "trials": [
            {
                "trial_id": item["trial_id"], "condition_id": item.get("condition_id"), "status": item.get("status"), "success": item.get("success"),
                "valid_behavior": item.get("valid_behavior"), "terminal_reason": item.get("terminal_reason"), "seed": item.get("seed"),
                "diagnosis": item.get("diagnosis", {}), "metrics": item.get("metrics", {}), "media": item.get("media", {}), "performance_evidence": item.get("performance_evidence", {}), "telemetry": item.get("telemetry", {}),
                "world": _relative(item.get("world"), examination_root), "run_dir": _relative(item.get("run_dir"), examination_root),
                "native_log": _relative_file(item.get("native_log"), examination_root), "native_result": _relative_file(item.get("native_result"), examination_root),
                "summary": _relative_file(str(Path(item["run_dir"]) / "summary.json"), examination_root) if item.get("run_dir") else None,
            }
            for item in trial_records
        ],
    }


def _render_world_gallery(data: Dict[str, Any], examination_root: Path) -> List[Dict[str, Any]]:
    """Render a small, representative geometry gallery for an evidence dashboard."""
    from .models import WorldSpec
    from .rendering import render_topdown

    conditions = data.get("conditions", [])
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: Optional[Dict[str, Any]]) -> None:
        if item and item["id"] not in seen:
            seen.add(item["id"])
            selected.append(item)

    # One readable example of every primitive terrain family.
    for terrain in [f"rs{index}" for index in range(1, 12)]:
        candidates = [
            item for item in conditions
            if item.get("kind") == "primitive" and str(item.get("terrain", "")).lower() == terrain
        ]
        candidates.sort(key=lambda item: (
            item.get("geometry_label") != "medium",
            item.get("friction_label") != "moderate",
            item.get("status") != "passed",
            item["id"],
        ))
        add(candidates[0] if candidates else None)

    # Show both successful and unsuccessful compositions, without flooding the page.
    for level in ("L2", "L3"):
        for status in ("passed", "failed"):
            candidates = [
                item for item in conditions
                if item.get("level") == level and item.get("status") == status
            ]
            for item in candidates[:3]:
                add(item)

    output_dir = examination_root / "dashboard" / "world_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery: List[Dict[str, Any]] = []
    for condition in selected:
        world_path = examination_root / "worlds" / f"{condition['id']}.json"
        if not world_path.is_file():
            continue
        output_path = output_dir / f"{condition['id']}.png"
        try:
            if not output_path.is_file() or output_path.stat().st_mtime < world_path.stat().st_mtime:
                world = WorldSpec.from_dict(load_data(str(world_path)))
                render_topdown(world, str(output_path), width=760, height=460)
        except Exception as exc:  # A broken preview must not hide the run evidence.
            gallery.append({
                "condition_id": condition["id"], "level": condition.get("level"),
                "status": condition.get("status"), "error": str(exc),
                "world": _relative(str(world_path), examination_root),
            })
            continue
        gallery.append({
            "condition_id": condition["id"],
            "level": condition.get("level"),
            "kind": condition.get("kind"),
            "status": condition.get("status"),
            "terrain": condition.get("terrain"),
            "pair": condition.get("pair"),
            "topology": condition.get("topology"),
            "geometry_label": condition.get("geometry_label"),
            "passes": condition.get("passes", 0),
            "valid_attempts": condition.get("valid_attempts", 0),
            "metrics": condition.get("metrics", {}),
            "preview": _relative(str(output_path), examination_root),
            # IDE preview servers often rewrite URLs containing workspace paths.
            # Embedding this small catalogue makes the report portable everywhere.
            "preview_data_uri": "data:image/png;base64," + base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "world": _relative(str(world_path), examination_root),
        })
    return gallery


def _dashboard_html_legacy(data: Dict[str, Any], running: bool) -> str:
    payload = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    refresh = "setTimeout(()=>location.reload(),10000);" if running else ""
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Robustness Lab Evidence Dashboard</title><style>
:root{--ink:#102a43;--muted:#627d98;--bg:#eef3f7;--pass:#16855b;--fail:#c2413b;--warn:#b7791f;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Inter,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}.hero{background:#102a43;color:white;padding:24px;border-radius:16px}.hero p{color:#d9e2ec}.cards,.levels{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}.card,.panel{background:white;border-radius:12px;padding:14px;box-shadow:0 2px 8px #102a4312}.big{font-size:28px;font-weight:800}.muted{color:var(--muted)}.pass{color:var(--pass)}.fail{color:var(--fail)}.warn{color:var(--warn)}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.filters select,.filters input{padding:8px;border:1px solid #bcccdc;border-radius:8px;background:white}table{width:100%;border-collapse:collapse;background:white}th,td{padding:8px;border-bottom:1px solid #d9e2ec;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#dfe7ef}.table-wrap{max-height:610px;overflow:auto;border-radius:12px}.badge{padding:3px 8px;border-radius:999px;background:#e5e7eb;font-weight:700}.badge.passed{background:#d1fae5;color:#065f46}.badge.failed{background:#fee2e2;color:#991b1b}.badge.pruned,.badge.unsupported,.badge.blocked{background:#fef3c7;color:#92400e}details{background:white;border-radius:10px;padding:10px;margin:8px 0}img{max-width:240px;border-radius:8px}video{max-width:520px;width:100%}code{font-size:12px}.failure{border-left:5px solid var(--fail);margin:10px 0}.media{display:flex;gap:8px;flex-wrap:wrap}.matrix{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:6px}.cell{padding:8px;border-radius:8px;background:white;border:1px solid #d9e2ec}.cell.passed{border-color:#6ee7b7;background:#ecfdf5}.cell.failed{border-color:#fca5a5;background:#fef2f2}.cell.pruned,.cell.unsupported,.cell.blocked{background:#fffbeb}</style></head>
<body><main class='wrap'><section class='hero'><h1>Discrete Pruned Benchmark</h1><p id='subtitle'></p><p>Standalone evidence report · raw JSON/CSV and telemetry remain the source of truth.</p></section><section id='summary' class='cards'></section><section><h2>L0–L7 funnel</h2><div id='levels' class='levels'></div></section>
<section><h2>RS1–RS11 discrete matrix</h2><div id='rs-matrix' class='matrix'></div></section>
<section class='panel'><h2>Condition explorer</h2><div class='filters'><select id='policy'></select><select id='backend'></select><select id='level'><option value=''>All levels</option></select><select id='kind'><option value=''>All kinds</option></select><select id='terrain'><option value=''>All terrains</option></select><select id='geometry_label'><option value=''>All geometry</option></select><select id='friction_label'><option value=''>All friction</option></select><select id='topology'><option value=''>All topology</option></select><select id='status'><option value=''>All outcomes</option></select><select id='failure'><option value=''>All failure types</option></select><input id='search' placeholder='RS, pair, topology…'></div><div class='table-wrap'><table><thead><tr><th>Condition</th><th>Level</th><th>Parameters</th><th>Status</th><th>Pass / valid</th><th>95% CI</th><th>Key performance</th><th>Failures</th><th>Evidence</th></tr></thead><tbody id='conditions'></tbody></table></div></section>
<section><h2>Directed pair matrix</h2><div id='pairs' class='matrix'></div></section><section><h2>L3–L7 composed and endurance worlds</h2><div id='larger' class='matrix'></div></section><section><h2>Failure heatmap</h2><div id='failure-map' class='matrix'></div></section><section><h2>Passed-condition evidence</h2><div id='passes'></div></section><section><h2>Failure clusters</h2><div id='clusters' class='cards'></div></section><section><h2>Failure explorer</h2><div id='failures'></div></section>
<details><summary>Embedded dashboard evidence JSON</summary><pre id='raw'></pre></details></main><script id='evidence' type='application/json'>""" + payload + """</script><script>
const D=JSON.parse(document.getElementById('evidence').textContent);const esc=x=>String(x??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('subtitle').textContent=`${D.examiner_id} · ${D.policy.skill_id||'policy'} · ${D.backend} · ${D.status} / ${D.stage}`;
const S=D.trial_counts,P=D.progress;document.getElementById('summary').innerHTML=[['Trials',S.all],['Valid',S.valid],['Passed',S.passes],['Behavioral failures',S.behavioral_failures],['Infrastructure',S.infrastructure],['Conditions finished',P.completed_conditions],['Conditions remaining',P.remaining_materialized_conditions],['Minimum attempts left',P.minimum_remaining_attempts],['Active',P.active_condition||'none']].map(x=>`<div class='card'><div class='big'>${esc(x[1])}</div><div class='muted'>${x[0]}</div></div>`).join('');
document.getElementById('levels').innerHTML=Object.entries(D.level_counts).map(([l,c])=>`<div class='card'><div class='big'>${l}</div>${Object.entries(c).map(([s,n])=>`<div>${esc(s)}: <b>${n}</b></div>`).join('')}</div>`).join('');
document.getElementById('policy').innerHTML=`<option>${esc(D.policy.skill_id||'policy')}</option>`;document.getElementById('backend').innerHTML=`<option>${esc(D.backend)}</option>`;
for(const [id,key] of [['level','level'],['kind','kind'],['geometry_label','geometry_label'],['friction_label','friction_label'],['topology','topology'],['status','status']]){const e=document.getElementById(id);[...new Set(D.conditions.map(x=>x[key]).filter(Boolean))].sort().forEach(v=>e.insertAdjacentHTML('beforeend',`<option>${esc(v)}</option>`));e.onchange=render;}const terrainValues=[...new Set(D.conditions.flatMap(x=>x.terrain?[x.terrain]:(x.pair||[])))].sort();terrainValues.forEach(v=>document.getElementById('terrain').insertAdjacentHTML('beforeend',`<option>${esc(v)}</option>`));document.getElementById('terrain').onchange=render;const failureValues=[...new Set(D.conditions.flatMap(x=>Object.keys(x.failure_types||{})))].sort();failureValues.forEach(v=>document.getElementById('failure').insertAdjacentHTML('beforeend',`<option>${esc(v)}</option>`));document.getElementById('failure').onchange=render;document.getElementById('search').oninput=render;
function links(c){const ts=c.attempts.map(id=>D.trials.find(t=>t.trial_id===id)).filter(Boolean);const t=ts.find(x=>x.media?.video)||ts.find(x=>x.media?.snapshots?.length)||ts[0];if(!t)return '—';let a=[];if(t.media?.video?.path)a.push(`<a href='../${esc(t.media.video.path)}'>video</a>`);if(t.media?.snapshots?.[0])a.push(`<a href='../${esc(t.media.snapshots[0].path)}'>snapshots</a>`);if(t.run_dir)a.push(`<a href='../${esc(t.run_dir)}'>raw</a>`);return a.join(' · ')||'—';}
function render(){const q=document.getElementById('search').value.toLowerCase(),lv=document.getElementById('level').value,k=document.getElementById('kind').value,t=document.getElementById('terrain').value,g=document.getElementById('geometry_label').value,fr=document.getElementById('friction_label').value,tp=document.getElementById('topology').value,s=document.getElementById('status').value,fail=document.getElementById('failure').value;const rows=D.conditions.filter(c=>(!lv||c.level===lv)&&(!k||c.kind===k)&&(!t||c.terrain===t||(c.pair||[]).includes(t))&&(!g||c.geometry_label===g)&&(!fr||c.friction_label===fr)&&(!tp||c.topology===tp)&&(!s||c.status===s)&&(!fail||Object.hasOwn(c.failure_types||{},fail))&&(!q||JSON.stringify(c).toLowerCase().includes(q)));document.getElementById('conditions').innerHTML=rows.map(c=>`<tr><td><code>${esc(c.id)}</code></td><td>${esc(c.level)}</td><td>${esc(c.terrain||c.pair?.join('→')||c.topology||'')}</td><td><span class='badge ${esc(c.status)}'>${esc(c.status)}</span></td><td>${c.passes}/${c.valid_attempts}</td><td>${c.lower_ci==null?'—':(100*c.lower_ci).toFixed(1)+'–'+(100*c.upper_ci).toFixed(1)+'%'}</td><td>${Object.entries(c.metrics||{}).slice(0,3).map(([n,v])=>`${esc(n)}: ${Number(v.median).toFixed(3)}`).join('<br>')||'—'}</td><td>${esc(JSON.stringify(c.failure_types))}</td><td>${links(c)}</td></tr>`).join('');}
const primitives=D.conditions.filter(c=>c.kind==='primitive');document.getElementById('rs-matrix').innerHTML=primitives.map(c=>`<div class='cell ${esc(c.status)}'><b>${esc(c.terrain?.toUpperCase())}</b><br>${esc(c.geometry_label)} geometry · ${esc(c.friction_label)} friction<br><span class='badge ${esc(c.status)}'>${esc(c.status)}</span> ${c.passes}/${c.valid_attempts}</div>`).join('');
const pairs=D.conditions.filter(c=>c.kind==='pair');document.getElementById('pairs').innerHTML=pairs.map(c=>`<div class='cell'><b>${esc(c.pair?.join('→'))}</b><br>${esc(c.geometry_label||'')}<br><span class='badge ${esc(c.status)}'>${esc(c.status)}</span> ${c.passes}/${c.valid_attempts}</div>`).join('')||'<div class="card">Pair conditions have not been materialized yet.</div>';
const larger=D.conditions.filter(c=>/^L[3-7]$/.test(c.level));document.getElementById('larger').innerHTML=larger.map(c=>`<div class='cell ${esc(c.status)}'><b>${esc(c.level)} · ${esc(c.topology||c.kind)}</b><br>${esc(c.horizon||'')} · ${esc(c.geometry_label||'')}<br><span class='badge ${esc(c.status)}'>${esc(c.status)}</span> ${c.passes}/${c.valid_attempts}</div>`).join('')||'<div class="card">Larger worlds unlock only from passed ordered pairs.</div>';const withFailures=D.conditions.filter(c=>c.failures||Object.keys(c.infrastructure_outcomes||{}).length);document.getElementById('failure-map').innerHTML=withFailures.map(c=>`<div class='cell ${c.failures?'failed':'unsupported'}'><b>${esc(c.id)}</b><br>behavioral: ${c.failures} · infrastructure: ${Object.values(c.infrastructure_outcomes||{}).reduce((a,b)=>a+b,0)}<br>${esc(Object.keys(c.failure_types||{}).join(', ')||Object.keys(c.infrastructure_outcomes||{}).join(', '))}</div>`).join('')||'<div class="card">No failures or infrastructure issues recorded.</div>';
document.getElementById('passes').innerHTML=D.pass_evidence.map(p=>`<article class='panel'><h3>${esc(p.condition_id)} · representative pass</h3><p>Seed ${p.seed} · ${Object.entries(p.metrics||{}).filter(([n])=>['completion_time','progress_fraction','velocity_tracking_rmse','slip_ratio','orientation_max','absolute_mechanical_cot'].includes(n)).map(([n,v])=>`${esc(n)}=${Number(v).toFixed(4)}`).join(' · ')}</p><div class='media'>${(p.media.snapshots||[]).map(x=>`<a href='../${esc(x.path)}'><img src='../${esc(x.thumbnail)}'></a>`).join('')}</div>${p.media.video?.path?`<video controls preload='metadata' src='../${esc(p.media.video.path)}'></video>`:''}</article>`).join('')||'<div class="card">No condition has completed with a passing representative yet.</div>';
document.getElementById('clusters').innerHTML=D.failure_clusters.map(c=>`<div class='card'><div class='big fail'>${c.count}</div><b>${esc(c.category)}</b><p class='muted'>${esc(c.conditions.join(', '))}</p></div>`).join('')||'<div class="card">No failure clusters recorded.</div>';
document.getElementById('failures').innerHTML=D.failures.map(f=>`<article class='panel failure'><h3>${esc(f.condition_id)} · ${esc(f.terminal_reason)}</h3><p><b>${esc(f.diagnosis.failure_category||f.diagnosis.classification)}</b> · ${esc((f.diagnosis.event_chain||[]).join(' → '))}</p><p>${esc((f.diagnosis.likely_factors||[]).map(x=>x.factor+' ('+x.strength+')').join(', '))}</p><p>${f.diagnosis.trigger?`Trigger: ${esc(f.diagnosis.trigger.event)} at ${Number(f.diagnosis.trigger.time_s).toFixed(2)} s · ${esc(f.diagnosis.trigger.region)} · ${esc(f.diagnosis.trigger.pair_component)}`:''}</p><div class='media'>${(f.media.snapshots||[]).map(x=>`<a href='../${esc(x.path)}'><img src='../${esc(x.thumbnail)}'></a>`).join('')}</div>${f.media.video?.path?`<video controls preload='metadata' src='../${esc(f.media.video.path)}'></video>`:''}</article>`).join('')||'<div class="card">No behavioral failures recorded.</div>';
document.getElementById('raw').textContent=JSON.stringify(D,null,2);render();""" + refresh + """
</script></body></html>"""


def _dashboard_html(data: Dict[str, Any], running: bool) -> str:
    """Build a readable, offline-first report while retaining all raw evidence."""
    payload = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    refresh = "setTimeout(()=>location.reload(),10000);" if running else ""
    return r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobustLab completed-run report</title>
<style>
:root{--navy:#102a43;--blue:#2563eb;--ink:#18324a;--muted:#60758a;--line:#d9e2ec;--bg:#f3f6f9;--card:#fff;--green:#087f5b;--greenbg:#e8f7f1;--red:#b42318;--redbg:#fff0ee;--amber:#9a6700;--amberbg:#fff7db;--gray:#667085;--graybg:#eef1f4}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.wrap{max-width:1260px;margin:auto;padding:28px 22px 80px}.hero{padding:30px;border-radius:20px;background:linear-gradient(135deg,#102a43,#164e63);color:#fff;box-shadow:0 12px 35px #102a4325}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-weight:800;font-size:11px;color:#b9d9ef}.hero h1{font-size:clamp(28px,4vw,48px);line-height:1.05;margin:8px 0 12px}.hero p{max-width:900px;color:#dbeafe;font-size:16px;margin:8px 0}.hero-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}.hero a,.button{display:inline-block;background:#fff;color:var(--navy);padding:9px 13px;border-radius:9px;text-decoration:none;font-weight:750}.notice{margin:18px 0;padding:16px 18px;border-radius:12px;border-left:5px solid var(--amber);background:var(--amberbg)}.notice.red{border-color:var(--red);background:var(--redbg)}h2{font-size:24px;margin:34px 0 6px}h3{margin:4px 0 8px}.lead{color:var(--muted);margin:0 0 16px;max-width:930px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:16px 0}.card,.panel{background:var(--card);border:1px solid #e5ebf0;border-radius:14px;padding:16px;box-shadow:0 3px 12px #102a430b}.big{font-size:28px;line-height:1.1;font-weight:850}.label{color:var(--muted);margin-top:5px}.sub{font-size:12px;color:var(--muted)}.pass{color:var(--green)}.fail{color:var(--red)}.warn{color:var(--amber)}.badge{display:inline-block;padding:3px 9px;border-radius:99px;font-size:12px;font-weight:800;background:var(--graybg);color:var(--gray)}.badge.passed,.badge.success{background:var(--greenbg);color:var(--green)}.badge.failed,.badge.failure{background:var(--redbg);color:var(--red)}.badge.pruned,.badge.unsupported,.badge.blocked{background:var(--amberbg);color:var(--amber)}.progression{display:grid;grid-template-columns:repeat(8,minmax(105px,1fr));gap:8px;overflow:auto;padding:4px 0}.level{min-height:145px;position:relative}.level:not(:last-child):after{content:'→';position:absolute;right:-10px;top:54px;color:#9fb3c8;font-size:18px;z-index:2}.level h3{font-size:20px}.level .rate{font-size:21px;font-weight:850}.section-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}.world{padding:0;overflow:hidden}.world img{width:100%;height:190px;object-fit:cover;background:#e8eef4;display:block}.world .body{padding:13px}.world a{text-decoration:none;color:var(--blue)}.route-progress{position:relative;height:13px;background:#dde5ec;border-radius:9px;margin:20px 8px 10px}.route-progress .done{height:100%;background:linear-gradient(90deg,#16a085,#22c55e);border-radius:9px}.route-progress .dog{position:absolute;top:50%;transform:translate(-50%,-54%);font-size:24px;filter:drop-shadow(0 1px 1px white)}.route-labels{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}.metric-pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.metric-pair b{font-size:18px}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.filters select,.filters input{min-height:38px;padding:8px 10px;border:1px solid #b8c5d1;border-radius:9px;background:#fff;color:var(--ink)}.filters input{min-width:240px;flex:1}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:12px;background:#fff}table{width:100%;border-collapse:collapse;min-width:1050px}th,td{padding:9px 10px;border-bottom:1px solid #e8edf2;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eaf0f5;z-index:1;font-size:12px}tr:hover td{background:#f8fafc}code{font-size:11px;word-break:break-all}.artifact a{white-space:nowrap;margin-right:8px;color:var(--blue)}details{background:#fff;border:1px solid var(--line);border-radius:11px;padding:11px 13px;margin:8px 0}summary{font-weight:750;cursor:pointer}.bar{height:8px;border-radius:8px;background:#e5eaf0;overflow:hidden;margin:8px 0}.bar>span{height:100%;display:block;background:var(--green)}.legend{display:flex;gap:12px;flex-wrap:wrap}.legend span{display:flex;align-items:center;gap:5px}.dot{width:10px;height:10px;border-radius:50%;background:var(--gray)}.dot.green{background:var(--green)}.dot.red{background:var(--red)}.dot.amber{background:var(--amber)}.empty{padding:18px;color:var(--muted);background:#fff;border:1px dashed #b8c5d1;border-radius:12px}.countline{color:var(--muted);margin:6px 0 12px}.footer{margin-top:35px;color:var(--muted);font-size:12px}@media(max-width:900px){.progression{grid-template-columns:repeat(4,minmax(145px,1fr))}.level:after{display:none}.wrap{padding:14px}.hero{padding:22px}}
.variant-table{border-collapse:collapse;margin-top:8px}.variant-table td,.variant-table th{padding:4px 10px;border-bottom:1px solid #d9e2ec;text-align:center}.variant-table td:first-child{text-align:left}.variant-table td.pass{color:#0b7a3b;font-weight:600}.variant-table td.fail{color:#b42318;font-weight:600}.variant-table td.muted{color:#9aa5b1}.variant-table td.takeaway{text-align:left;color:#334e68;font-style:italic}#tip{position:fixed;z-index:9999;max-width:420px;background:#102a43;color:#f0f4f8;font-size:12px;line-height:1.45;padding:8px 10px;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.25);white-space:pre-line;pointer-events:none}[data-tip]{cursor:help}.pair-popup{position:fixed;z-index:10000;background:#fff;color:#102a43;border:1px solid #bcccdc;border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,.25);padding:14px 16px 12px;max-height:80vh;overflow:auto}.pair-popup h3{margin:0 0 4px;font-size:18px}.pair-popup .sub{color:#486581;font-size:12px;margin:2px 0 8px}.pair-popup .note{margin-top:10px;border-top:1px solid #d9e2ec;padding-top:8px}.pair-popup-close{position:absolute;top:6px;right:8px;border:0;background:transparent;font-size:20px;line-height:1;cursor:pointer;color:#486581}.pair-popup-table{border-collapse:collapse;width:100%;min-width:0;max-width:100%;table-layout:fixed;font-size:13px}.pair-popup-table th:nth-child(1),.pair-popup-table td:nth-child(1){width:15%}.pair-popup-table th:nth-child(2),.pair-popup-table td:nth-child(2){width:22%}.pair-popup .pair-parents span,.pair-popup td span{white-space:normal}.pair-popup-table th,.pair-popup-table td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid #e6edf3;white-space:normal;overflow-wrap:anywhere}.pair-popup-table td:nth-child(3){width:62%}.pair-popup .ok{color:#0b7a3b;font-weight:600}.pair-popup .bad{color:#b42318;font-weight:600}.pair-popup .pruned{color:#a15c07;font-weight:600}.pair-popup .pair-parents{color:#486581;font-size:12px;margin-top:4px}.pair-cell[data-tip]{cursor:pointer}.composed-table{border-collapse:collapse;width:100%;min-width:0;table-layout:fixed;font-size:13px;margin:6px 0 18px}.composed-table th,.composed-table td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid #e6edf3;white-space:normal;overflow-wrap:anywhere}.composed-table th:nth-child(1),.composed-table td:nth-child(1){width:26%}.composed-table th:nth-child(2),.composed-table td:nth-child(2){width:9%}.composed-table th:nth-child(3),.composed-table td:nth-child(3){width:17%}.composed-table th:nth-child(4),.composed-table td:nth-child(4){width:20%}.composed-table tr[data-world]{cursor:pointer}.composed-table tr[data-world]:hover{background:#f0f4f8}.composed-table .ok{color:#0b7a3b;font-weight:600}.composed-table .bad{color:#b42318;font-weight:600}.composed-table .muted{color:#9aa5b1}.pair-chip{display:inline-block;padding:1px 6px;border:1px solid #bcccdc;border-radius:999px;font-size:11px;margin:1px 2px 1px 0;background:#fff}#composed-panel h3{margin:18px 0 2px}#composed-panel h3 small{font-weight:400;margin-left:8px}</style><style>
.glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:8px}.term{background:#fff;border:1px solid var(--line);border-radius:11px;padding:11px}.term b{display:block;font-size:15px}.story-shell{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 14px #102a430d}.story-nav{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}.story-nav button{border:1px solid #b8c5d1;background:#f7f9fb;color:var(--ink);border-radius:11px;padding:12px;cursor:pointer;text-align:left;font-weight:800}.story-nav button.active{background:var(--navy);border-color:var(--navy);color:#fff}.story-panel{min-height:360px}.story-title{font-size:27px;margin:4px 0}.story-question{font-size:17px;color:var(--muted);margin:0 0 14px}.story-numbers{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;margin:12px 0}.story-number{border-radius:11px;padding:12px;background:#f3f6f9}.story-number b{display:block;font-size:24px}.result-chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:7px;margin:12px 0}.result-chip{border:1px solid var(--line);border-left:5px solid var(--gray);border-radius:9px;padding:9px;background:#fff}.result-chip.passed{border-left-color:var(--green);background:var(--greenbg)}.result-chip.failed,.result-chip.mixed{border-left-color:var(--red);background:var(--redbg)}.result-chip.unsupported,.result-chip.pruned{border-left-color:var(--amber);background:var(--amberbg)}.decision{margin-top:14px;padding:14px;border-radius:11px;background:#eef6ff;border-left:5px solid var(--blue);font-size:15px}.story-controls{display:flex;justify-content:space-between;gap:8px;margin-top:16px}.story-controls button{border:0;border-radius:9px;padding:10px 14px;background:var(--navy);color:#fff;font-weight:800;cursor:pointer}.story-controls button:disabled{opacity:.35;cursor:default}.coverage-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}.coverage-step{position:relative;background:#fff;border:1px solid var(--line);border-radius:13px;padding:14px;min-height:125px}.coverage-step:not(:last-child):after{content:'→';position:absolute;right:-11px;top:48px;z-index:2;color:#91a5b7;font-size:20px}.coverage-step .total{font-size:24px;font-weight:850}.optional-detail{margin-top:24px}.optional-detail>summary{font-size:20px;padding:5px}.pair-map-shell{overflow:auto;background:#f8fafc;border:1px solid var(--line);border-radius:14px;padding:14px}.pair-map{--pair-cell:74px;display:grid;grid-template-columns:96px repeat(11,var(--pair-cell));gap:5px;width:max-content}.pair-map.large{--pair-cell:94px}.pair-head,.pair-row-head{display:flex;align-items:center;justify-content:center;font-weight:850;font-size:11px;min-height:48px;text-align:center}.pair-axis{font-size:10px;line-height:1.15;color:var(--muted)}.pair-cell{border:2px solid #cbd5df;border-radius:9px;background:#fff;min-height:58px;padding:6px;cursor:pointer;text-align:center}.pair-cell:hover,.pair-cell.selected{border-color:var(--blue);box-shadow:0 0 0 3px #2563eb25}.pair-cell.self{cursor:default;background:#edf1f4;color:#98a2b3;border-color:#e1e6eb}.pair-dots{display:flex;justify-content:center;gap:3px;margin:5px 0}.mini-dot{width:11px;height:11px;border-radius:50%;background:#98a2b3}.mini-dot.passed{background:var(--green)}.mini-dot.failed{background:var(--red)}.mini-dot.pruned,.mini-dot.unsupported,.mini-dot.blocked,.mini-dot.not-run{background:var(--amber)}.pair-cell small{display:block;color:var(--muted);font-size:9px}.tree-shell{overflow:auto;padding:14px;background:#f8fafc;border:1px solid var(--line);border-radius:14px}.tree-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.tree-toolbar button{border:1px solid #b7c5d1;background:#fff;color:var(--ink);padding:7px 10px;border-radius:8px;cursor:pointer;font-weight:700}.tree-toolbar button.active{background:var(--navy);color:#fff}.dot.mixed{background:linear-gradient(135deg,var(--green) 0 48%,var(--red) 52%)}.terrain-branches{display:grid;grid-template-columns:repeat(11,minmax(100px,1fr));gap:7px;min-width:1120px}.tree-node{border:2px solid #cbd5df;background:#fff;border-radius:11px;padding:10px;cursor:pointer;min-height:74px}.tree-node:hover,.tree-node.selected{box-shadow:0 0 0 3px #2563eb25;border-color:var(--blue)}.tree-node.passed{border-color:#55b997;background:var(--greenbg)}.tree-node.failed{border-color:#ef8c84;background:var(--redbg)}.tree-node.pruned,.tree-node.unsupported,.tree-node.not-run{border-color:#e4bd57;background:var(--amberbg)}.tree-node.mixed{border-color:#8b9db0;background:linear-gradient(135deg,var(--greenbg) 0 49%,var(--redbg) 51%)}.tree-node small{display:block;color:var(--muted);margin-top:4px}.selected-banner{background:#eaf2ff;border:1px solid #b8d0ff;border-radius:11px;padding:11px 13px;margin:12px 0;font-size:15px}.branch-flow{display:grid;grid-template-columns:minmax(210px,.8fr) 30px minmax(300px,1.2fr) 30px minmax(290px,1.15fr) 30px minmax(220px,.85fr);gap:6px;align-items:start;min-width:1120px;margin-top:18px}.tree-column{display:grid;gap:7px}.tree-column h3{position:sticky;top:0;background:#f8fafc;padding:6px 0;z-index:1}.column-help{min-height:42px;color:var(--muted);font-size:11px;margin-top:-5px}.tree-arrow{font-size:28px;color:#8fa5b8;text-align:center;padding-top:95px}.tree-group{border:1px solid var(--line);border-radius:10px;background:#fff;padding:8px}.tree-group>summary{font-size:13px}.tree-list{display:grid;gap:6px;margin-top:7px}.tree-compact .tree-node{min-height:58px;padding:7px;font-size:12px}.tree-comfortable .tree-node{min-height:90px;padding:12px}.prune-chain{font-size:11px;margin-top:5px;padding:5px;background:#fff8;border-radius:6px;color:#7c4d00}.tree-explanation{margin-top:14px;padding:13px;background:#eef6ff;border-left:4px solid var(--blue);border-radius:9px}@media(max-width:900px){.story-nav{grid-template-columns:1fr 1fr}.coverage-strip{grid-template-columns:1fr}.coverage-step:after{display:none}.branch-flow{grid-template-columns:220px 24px 300px 24px 300px 24px 220px}.terrain-branches{grid-template-columns:repeat(11,110px)}}
</style></head><body><main class="wrap">
<section class="hero"><div class="eyebrow">Completed benchmark evidence</div><h1>What the policy could—and could not—do</h1><p id="hero-story"></p><div class="hero-links"><a href="dashboard_data.json">Download dashboard JSON</a><a href="conditions.csv">Download condition CSV</a><a href="../examination_state.json">Full run state</a></div></section>
<div id="stop-notice" class="notice"></div><section id="summary" class="cards"></section>
<section><h2>Short forms used on this page</h2><p class="lead">Read this once; the walkthrough below then uses ordinary language.</p><div class="glossary"><div class="term"><b>RS = Robustness Scenario</b>One terrain family. RS1 through RS11 are names, not levels.</div><div class="term"><b>L = Benchmark Level</b>L1 tests one terrain, L2 joins two, and L3 builds larger worlds.</div><div class="term"><b>RS1 → RS3</b>The robot meets RS1 first and RS3 second. RS3 → RS1 is a different test.</div><div class="term"><b>P / F / X</b>Passed / Failed / Not executed. X means pruned or environment unsupported; neither is a policy failure.</div><div class="term"><b>Environment unsupported</b>The simulator adapter could not reproduce the requested physics. The policy was not evaluated.</div><div class="term"><b>Condition</b>One fixed world and difficulty setting.</div><div class="term"><b>Attempt</b>One seeded run of a condition. Important conditions receive repeated attempts.</div></div></section>
<section><h2>Follow the run in order</h2><p class="lead">Use Next and Previous. This starts at L1 and explains what the policy handled, what it did not handle, and exactly what was removed from the next stage.</p><div class="story-shell"><div id="story-nav" class="story-nav"><button data-story="0" class="active">1 · L1<br>Single terrains</button><button data-story="1">2 · L2<br>Terrain pairs</button><button data-story="2">3 · L3<br>Larger worlds</button><button data-story="3">4 · L4–L7<br>Later levels</button></div><div id="story-panel" class="story-panel"></div><div class="story-controls"><button id="story-prev">← Previous</button><button id="story-next">Next →</button></div></div></section>
<section><h2>Whole-run accounting</h2><p class="lead">Before L1, the flat-ground setup check (L0) passed. Every planned case after that is counted below. Pruned tests were deliberately skipped and are not counted as policy failures.</p><div id="coverage-strip" class="coverage-strip"></div><div id="levels" class="progression"></div><div id="variant-panel" class="feature"></div><div id="tip" hidden></div><div id="pair-popup" class="pair-popup" hidden><button class="pair-popup-close" aria-label="close">×</button><div id="pair-popup-body"></div></div></section>
<section><h2>All directional two-terrain combinations (L2)</h2><p class="lead"><b>Read across a row:</b> the row terrain comes first and the column terrain comes second. For example, row RS1 and column RS3 means <b>RS1 → RS3</b>. Each cell contains three dots for easy, medium, and hard. The reverse order has its own separate cell.</p><div class="tree-toolbar"><button id="pair-fit" class="active">Fit map</button><button id="pair-large">Larger cells</button><span class="legend"><span><i class="dot green"></i>passed</span><span><i class="dot red"></i>failed</span><span><i class="dot amber"></i>pruned / unsupported</span></span><span class="sub">Click any cell to inspect exactly why it passed, failed, or was pruned.</span></div><div class="pair-map-shell"><div id="pair-map" class="pair-map"></div></div><div id="pair-map-summary" class="tree-explanation"></div></section><section id="composed-panel"></section>
<div hidden aria-hidden="true"><button id="tree-compact"></button><button id="tree-comfortable"></button><div id="capability-tree" class="tree-shell tree-compact"><div id="selected-banner" class="selected-banner"></div><div id="terrain-branches" class="terrain-branches"></div><div id="branch-flow" class="branch-flow"></div><div id="tree-explanation" class="tree-explanation"></div></div></div>
<section><h2>What was tested at each stage</h2><div id="stage-cards" class="section-grid"></div></section>
<section><h2>How the robot performed on each terrain</h2><p class="lead">These cards combine all measured single-terrain settings. They show success and the recovered median route progress, so a high pass rate cannot hide attempts that stopped early.</p><div id="terrain-performance" class="gallery"></div></section>
<section><h2>Representative test results</h2><div id="distance-notice" class="notice"></div><p class="lead">The dog marker shows the median furthest checkpoint for attempts in that condition. The coloured map underneath is the tested world layout—not a camera frame.</p><div id="world-gallery" class="gallery"></div></section>
<section><h2>Evidence available</h2><div id="inventory" class="cards"></div><div id="media-notice" class="notice red"></div></section>
<section class="panel"><h2>Condition explorer</h2><p class="lead">Filter the measured and pruned conditions. Each row reports how far the robot reached and links to available per-attempt evidence.</p><div class="filters"><select id="level-filter"><option value="">All levels</option></select><select id="kind-filter"><option value="">All test types</option></select><select id="status-filter"><option value="">All outcomes</option></select><input id="condition-search" placeholder="Search condition, terrain, pair, topology…"></div><div id="condition-count" class="countline"></div><div class="table-wrap"><table><thead><tr><th>Condition</th><th>Stage</th><th>Test</th><th>Outcome</th><th>Attempts</th><th>Success</th><th>How far it reached</th><th>Failures / reason</th><th>Evidence</th></tr></thead><tbody id="condition-rows"></tbody></table></div></section>
<section class="panel"><h2>Attempt and log explorer</h2><p class="lead">Every saved execution is searchable here. Open the result or summary JSON for metrics, or the runner log for simulator output.</p><div class="filters"><select id="attempt-outcome"><option value="">All attempt outcomes</option><option value="pass">Pass</option><option value="behavioral_failure">Behavioral failure</option><option value="infrastructure">Environment or infrastructure issue</option></select><input id="attempt-search" placeholder="Search trial, condition, seed, reason…"></div><div id="attempt-count" class="countline"></div><div class="table-wrap"><table><thead><tr><th>Trial</th><th>Condition</th><th>Seed</th><th>Classification</th><th>Reason</th><th>How far it reached</th><th>Artifacts</th></tr></thead><tbody id="attempt-rows"></tbody></table></div></section>
<section><h2>Failure diagnosis</h2><p id="failure-explanation" class="lead"></p><div id="failure-clusters" class="cards"></div><details><summary>Failure explorer (sampled rows)</summary><div id="failure-list"></div></details></section>
<details><summary>Embedded dashboard evidence JSON</summary><pre id="raw-json"></pre></details><div class="footer">Generated <span id="generated"></span>. All percentages state their denominator; skipped tests are not counted as policy failures.</div>
</main><script id="evidence" type="application/json">""" + payload + r"""</script><script>
const D=JSON.parse(document.getElementById('evidence').textContent);const $=x=>document.getElementById(x);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const statusLabel=s=>s==='unsupported'?'environment unsupported':String(s??'').replaceAll('_',' ');const pct=(n,d)=>d?`${(100*n/d).toFixed(1)}%`:'—';const cc=D.condition_counts||{},tc=D.trial_counts||{};const measured=(cc.passed||0)+(cc.failed||0);const trialValid=tc.valid||0;const labels={L0:'Setup check',L1:'Single terrains',L2:'Ordered pairs',L3:'Connected worlds',L4:'Route selection',L5:'Endurance',L6:'Stress tests',L7:'Held-out tests'};const highestMeasured=[0,1,2,3,4,5,6,7].filter(i=>{const c=D.level_counts?.['L'+i]||{};return(c.passed||0)+(c.failed||0)>0}).pop();
$('hero-story').innerHTML=`The run is <b>${esc(D.status)}</b>. It measured <b>${measured}</b> conditions through Level ${highestMeasured} (${esc(labels['L'+highestMeasured])}) and completed <b>${trialValid}</b> valid attempts. ${cc.pruned||0} conditions were deliberately not executed after earlier results failed a progression rule.`;
const l3=D.level_counts?.L3||{},l3m=(l3.passed||0)+(l3.failed||0);$('stop-notice').innerHTML=l3m?`<b>Why the run ended:</b> Level 3 passed ${l3.passed||0} of ${l3m} measured connected-world conditions (${pct(l3.passed||0,l3m)}). Later levels are built only from the terrain pairs that passed, so they are narrower than the full matrix rather than skipped. “Completed” means the scheduler ran everything its evidence supported.`:'<b>Run status:</b> no connected-world conditions were measured.';
$('summary').innerHTML=[['Measured conditions',measured,'passed or failed'],['Condition pass rate',pct(cc.passed||0,measured),`${cc.passed||0} passed / ${cc.failed||0} failed`],['Valid attempts',trialValid,'seeded executions'],['Attempt pass rate',pct(tc.passes||0,trialValid),`${tc.passes||0} pass / ${tc.behavioral_failures||0} fail`],['Pruned conditions',cc.pruned||0,'not counted as failures'],['Infrastructure outcomes',tc.infrastructure||0,'kept separate']].map(([v,n,s])=>`<div class="card"><div class="big">${esc(n)}</div><div class="label">${esc(v)}</div><div class="sub">${esc(s)}</div></div>`).join('');
const levels=[0,1,2,3,4,5,6,7].map(i=>'L'+i);$('levels').innerHTML=levels.map(l=>{const c=D.level_counts?.[l]||{},m=(c.passed||0)+(c.failed||0),notrun=(c.pruned||0)+(c.unsupported||0)+(c.blocked||0);const ld=D.level_details?.[l];return `<div class="card level"${ld&&ld.hover?` data-tip="${esc(ld.hover)}"`:''}><h3>${l}</h3><b>${labels[l]}</b><div class="rate ${m&&c.passed/m>=.7?'pass':m?'fail':'warn'}">${m?pct(c.passed||0,m):'Not run'}</div><div class="sub">${c.passed||0} pass · ${c.failed||0} fail${notrun?' · '+notrun+' other':''}</div></div>`}).join('');
$('stage-cards').innerHTML=levels.slice(0,4).map(l=>{const c=D.level_counts?.[l]||{},m=(c.passed||0)+(c.failed||0);return `<div class="card"><h3>${l}: ${labels[l]}</h3><div class="bar"><span style="width:${m?100*(c.passed||0)/m:0}%"></span></div><b>${c.passed||0} of ${m} measured conditions passed</b><p class="sub">${c.pruned||0} pruned · ${c.unsupported||0} environment unsupported · ${c.blocked||0} blocked</p></div>`}).join('');
const treeConditions=D.conditions;const terrainNames={rs1:'Slippery flat ground',rs2:'Rough slippery ground',rs3:'Slippery slope',rs4:'Rough slippery slope',rs5:'Stairs',rs6:'Repeated waves',rs7:'Curved hill',rs8:'Small repeated gaps',rs9:'Large gap',rs10:'High step',rs11:'Marsh approximation'};const treeStatus=items=>{const s=new Set(items.map(x=>x.status));if(!items.length)return'not-run';if(s.size===1)return[...s][0];if(s.has('passed')&&s.has('failed'))return'mixed';if(s.has('passed'))return'passed';if(s.has('failed'))return'failed';if(s.has('pruned'))return'pruned';if(s.has('unsupported'))return'unsupported';return'not-run'};const orderedStatuses=['passed','failed','pruned','unsupported','blocked'];const countStatuses=items=>Object.fromEntries(orderedStatuses.map(s=>[s,items.filter(x=>x.status===s).length]));const statusCounts=items=>{const c=countStatuses(items);return orderedStatuses.filter(s=>c[s]).map(s=>`${c[s]} ${s}`).join(' · ')||'not created'};const node=(title,sub,status,extra='',attrs='')=>`<div class="tree-node ${esc(status)}" ${attrs}><b>${esc(title)}</b><small>${esc(sub)}</small>${extra}</div>`;const pairKey=c=>(c.pair||[]).join('→');const primitives=treeConditions.filter(c=>c.kind==='primitive');const pairConditions=treeConditions.filter(c=>c.kind==='pair');const terrains=[...new Set(primitives.map(c=>c.terrain))].sort((a,b)=>Number(a.slice(2))-Number(b.slice(2)));const allPairGroups={};pairConditions.forEach(c=>(allPairGroups[pairKey(c)]??=[]).push(c));let selectedTerrain=terrains.includes('rs1')?'rs1':terrains[0];let selectedPair=allPairGroups['rs1→rs2']?'rs1→rs2':Object.keys(allPairGroups)[0]||null;
function stageCounts(levelsWanted){const items=treeConditions.filter(c=>levelsWanted.includes(c.level)),c=countStatuses(items);return{items,total:items.length,executed:c.passed+c.failed,...c}}const l1Coverage=stageCounts(['L1']),l2Coverage=stageCounts(['L2']),l3Coverage=stageCounts(['L3']),laterCoverage=stageCounts(['L4','L5','L6','L7']);$('coverage-strip').innerHTML=[['1. Individual terrains (L1)',l1Coverage,`${(D.variants?.plain?.executed)??l1Coverage.executed} flat executed · ${(D.variants?.variants?.executed)??0} banked/curved executed · ${l1Coverage.unsupported} unsupported`],['2. Directional pairs (L2)',l2Coverage,`${l2Coverage.executed} executed · ${l2Coverage.pruned} pruned`],['3. Connected worlds (L3)',l3Coverage,`${l3Coverage.executed} executed · ${l3Coverage.pruned} pruned`],['4. Later tests (L4–L7)',laterCoverage,laterCoverage.total?`${laterCoverage.executed} executed · ${laterCoverage.pruned} pruned`:`not created: no terrain pair passed, so there was nothing to build them from`]].map(([title,c,detail])=>`<div class="coverage-step"><b>${title}</b><div class="total">${c.total||0} cases</div><div>${esc(detail)}</div><div class="sub">${c.passed} passed · ${c.failed} failed</div></div>`).join('');(()=>{const tip=document.getElementById('tip');if(!tip)return;const show=(el,e)=>{tip.textContent=el.getAttribute('data-tip')||'';tip.hidden=false;move(e)};const move=(e)=>{const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;let x=e.clientX+pad,y=e.clientY+pad;if(x+w>window.innerWidth-8)x=e.clientX-w-pad;if(y+h>window.innerHeight-8)y=e.clientY-h-pad;tip.style.left=x+'px';tip.style.top=y+'px'};document.addEventListener('mouseover',e=>{const el=e.target.closest('[data-tip]');if(el)show(el,e)});document.addEventListener('mousemove',e=>{if(!tip.hidden)move(e)});document.addEventListener('mouseout',e=>{const el=e.target.closest('[data-tip]');if(el&&!el.contains(e.relatedTarget))tip.hidden=true});})();const CL=D.composed_levels||[];const cpanel=document.getElementById('composed-panel');if(cpanel){const words={fall:'fell',stuck:'got stuck',timeout:'timed out'};const how=r=>Object.entries(r||{}).filter(([k])=>k!=='goal').map(([k,n])=>`${words[k]||k} ×${n}`).join(', ');const retestCell=r=>r.retest?`<span class="${r.retest.status==='passed'?'ok':'bad'}">${esc(r.retest.pair.toUpperCase())} alone: ${esc(r.retest.status)} ${r.retest.successes}/${r.retest.valid}</span><br><small>${esc(r.retest.verdict)}</small>`:(r.status==='passed'?'<span class="muted">—</span>':'<span class="muted">no retest</span>');cpanel.innerHTML=CL.map(L=>`<h3>${esc(L.level)} — ${esc(L.title)} <small class="sub">${L.executed?`${L.passed} passed · ${L.failed} failed${L.pruned?` · ${L.pruned} pruned`:''}`:esc(L.empty_reason)}</small></h3><p class="sub">${esc(L.blurb)}</p>${L.rows.length?`<table class="composed-table"><thead><tr><th>world</th><th>difficulty</th><th>outcome</th><th>built from</th><th>retest of one pair alone</th></tr></thead><tbody>${L.rows.map(r=>`<tr class="${r.status==='passed'?'pass':'fail'}" data-world="${esc(r.id)}" data-tip="${esc(r.hover)}"><td><b>${esc(r.name)}</b></td><td>${esc(r.difficulty)}</td><td><span class="${r.status==='passed'?'ok':'bad'}">${esc(r.status)} ${r.successes}/${r.valid}</span>${r.status!=='passed'&&how(r.reasons)?`<br><small>${esc(how(r.reasons))}</small>`:''}</td><td>${r.pairs.map(pk=>`<span class="pair-chip">${esc(pk.toUpperCase())}</span>`).join(' ')}</td><td>${retestCell(r)}</td></tr>`).join('')}</tbody></table>`:''}`).join('');cpanel.querySelectorAll('[data-world]').forEach(tr=>tr.onclick=(ev)=>{const id=tr.dataset.world;const r=CL.flatMap(L=>L.rows).find(x=>x.id===id);if(!r)return;const box=$('pair-popup'),body=$('pair-popup-body');body.innerHTML=`<h3>${esc(r.name)}</h3><p class="sub">${esc(r.id)} · ${esc(r.difficulty)} · <span class="${r.status==='passed'?'ok':'bad'}">${esc(r.status)} ${r.successes}/${r.valid}</span>${r.status!=='passed'&&how(r.reasons)?' — '+esc(how(r.reasons)):''}</p>${r.event_chain&&r.event_chain.length&&r.status!=='passed'?`<p class="sub">commonest failure: ${esc(r.event_chain.join(' → '))}</p>`:''}<p><b>Built from</b> ${r.pairs.length} pair${r.pairs.length===1?'':'s'}: ${r.pairs.map(pk=>`<span class="pair-chip">${esc(pk.toUpperCase())}</span>`).join(' ')}</p>${r.retest?`<p><b>Retest of ${esc(r.retest.pair.toUpperCase())} alone:</b> <span class="${r.retest.status==='passed'?'ok':'bad'}">${esc(r.retest.status)} ${r.retest.successes}/${r.retest.valid}</span>${how(r.retest.reasons)?' — '+esc(how(r.retest.reasons)):''}<br><small>${esc(r.retest.verdict)}</small></p>`:''}<p class="sub note">Composed worlds are assembled only from pairs that passed at L2. When a world fails, one of its pairs is re-run on its own to tell a composition failure from a pair failure.</p>`;placePopup(box,ev)})}const VS=D.variants||{terrains:{},variant_names:[],plain:{},variants:{}};const vcell=(v)=>v&&v.executed?`<td class="${v.pass_rate>=.5?'pass':'fail'}" data-tip="${esc(v.hover||'')}">${v.passed}/${v.executed}</td>`:`<td class="muted">—</td>`;const vrows=(VS.terrains||[]).map(row=>`<tr><td><b>${esc(String(row.code).toUpperCase())}</b> <small>${esc(row.name||'')}</small></td>${vcell(row.plain)}${VS.variant_names.map(n=>vcell(row[n])).join('')}<td class="takeaway">${esc(row.takeaway||'')}</td></tr>`).join('');const vpanel=document.getElementById('variant-panel');if(vpanel){vpanel.innerHTML=VS.variants&&VS.variants.executed?`<h3>Banked and curved variants (L1)</h3><p class="sub">Each terrain the policy passed flat was re-run banked 12° and curved 65° each way. Variants are diagnostic: pairs and later levels are still built from the flat terrains only.</p><table class="variant-table"><thead><tr><th>terrain</th><th>flat</th>${VS.variant_names.map(n=>`<th>${esc(n)}</th>`).join('')}<th>what it means</th></tr></thead><tbody>${vrows}</tbody></table><p class="sub">Flat: ${VS.plain.passed}/${VS.plain.executed} passed · banked/curved: ${VS.variants.passed}/${VS.variants.executed} passed. A terrain that failed flat has no variants.</p>`:`<h3>Banked and curved variants (L1)</h3><p class="sub">None executed in this run.</p>`}
const pairFamilyEntries=Object.entries(allPairGroups),pairFamiliesWithPass=pairFamilyEntries.filter(([,v])=>v.some(x=>x.status==='passed')),pairFamiliesAllPass=pairFamilyEntries.filter(([,v])=>v.length&&v.every(x=>x.status==='passed')),pairFamiliesNeverRun=pairFamilyEntries.filter(([,v])=>v.every(x=>!['passed','failed'].includes(x.status))),pairFamiliesNoPass=pairFamilyEntries.filter(([,v])=>v.some(x=>['passed','failed'].includes(x.status))&&!v.some(x=>x.status==='passed'));const hardBlocked=terrains.filter(t=>primitives.some(c=>c.terrain===t&&Number(c.geometry_level)===.75&&Number(c.static_friction)===.5&&c.status==='failed')),unsupportedTerrains=terrains.filter(t=>primitives.filter(c=>c.terrain===t).every(c=>c.status==='unsupported'));const l3Progression=treeConditions.filter(c=>c.level==='L3'&&c.kind==='progression'),l3Counterfactual=treeConditions.filter(c=>c.level==='L3'&&c.kind==='counterfactual');const topoSummary=topology=>{const items=l3Progression.filter(c=>c.topology===topology),c=countStatuses(items);return{total:items.length,...c}};const chainSummary=topoSummary('chain'),gridSummary=topoSummary('grid'),counterSummary=countStatuses(l3Counterfactual);let storyIndex=0;
function storyMetric(value,label,detail=''){return`<div class="story-number"><b>${esc(value)}</b>${esc(label)}${detail?`<div class="sub">${esc(detail)}</div>`:''}</div>`}function terrainStoryChips(){return terrains.map(t=>{const items=primitives.filter(c=>c.terrain===t),c=countStatuses(items),executed=c.passed+c.failed,status=c.unsupported===items.length?'unsupported':c.failed?'mixed':'passed';return`<div class="result-chip ${status}"><b>${t.toUpperCase()} · ${esc(terrainNames[t])}</b><br>${c.passed}/${executed||items.length} passed${c.unsupported?` · ${c.unsupported} unsupported`:''}</div>`}).join('')}function renderStory(){document.querySelectorAll('[data-story]').forEach((b,i)=>b.classList.toggle('active',i===storyIndex));$('story-prev').disabled=storyIndex===0;$('story-next').disabled=storyIndex===3;const policy=esc(D.policy?.skill_id||'the selected policy');let html='';if(storyIndex===0){html=`<div class="eyebrow" style="color:var(--blue)">Start here</div><h3 class="story-title">L1 — Can the policy cross each terrain by itself?</h3><p class="story-question">Policy tested: <b>${policy}</b>. Each of the 11 Robustness Scenarios was tried at three shape difficulties and three friction settings: nine conditions per terrain.</p><div class="story-numbers">${storyMetric((D.variants?.plain?.executed)??l1Coverage.total,'flat single-terrain conditions')}${(D.variants?.variants?.executed)?storyMetric(D.variants.variants.executed,'banked/curved variants'):''}${storyMetric((D.variants?.plain?.passed)??l1Coverage.passed,'flat passed')}${storyMetric(l1Coverage.failed,'failed')}${storyMetric(l1Coverage.unsupported,'unsupported')}</div><div class="result-chips">${terrainStoryChips()}</div><div class="decision"><b>What was removed before L2:</b> At the moderate friction used for combinations, the hard versions of <b>${esc(hardBlocked.map(x=>x.toUpperCase()).join(', '))}</b> failed, so pair tests requiring those exact hard versions were pruned.${unsupportedTerrains.length?` <b>${esc(unsupportedTerrains.map(x=>x.toUpperCase()).join(', '))}</b> ${unsupportedTerrains.length>1?'were':'was'} unsupported, so every pair containing ${unsupportedTerrains.length>1?'them':'it'} was left unmeasured.`:''} Easier versions that passed were still allowed forward.</div>`}else if(storyIndex===1){html=`<div class="eyebrow" style="color:var(--blue)">Next</div><h3 class="story-title">L2 — What happens when two surviving terrains are joined?</h3><p class="story-question">Order matters: RS1 → RS3 and RS3 → RS1 are different. There are 110 non-repeating directional families, each with easy, medium, and hard versions.</p><div class="story-numbers">${storyMetric(pairFamilyEntries.length,'directional pair families')}${storyMetric(pairConditions.length,'planned pair conditions')}${storyMetric(l2Coverage.executed,'actually executed')}${storyMetric(l2Coverage.pruned,'pruned before execution')}</div><div class="result-chips"><div class="result-chip passed"><b>${pairFamiliesWithPass.length} families</b><br>passed at least one difficulty</div><div class="result-chip passed"><b>${pairFamiliesAllPass.length} families</b><br>passed easy, medium, and hard</div><div class="result-chip failed"><b>${pairFamiliesNoPass.length} families</b><br>were executed but passed no difficulty</div><div class="result-chip pruned"><b>${pairFamiliesNeverRun.length} families</b><br>were never run, mainly because RS11 was unsupported</div></div><p><b>Pairs that passed all three versions:</b> ${esc(pairFamiliesAllPass.map(([k])=>k.toUpperCase()).join(', '))}.</p><div class="decision"><b>What moved into L3:</b> Only the ${l2Coverage.passed} pair conditions that passed were used as building blocks for larger connected worlds. Failed and pruned pair variants did not advance.</div>`}else if(storyIndex===2){html=`<div class="eyebrow" style="color:var(--blue)">Then</div><h3 class="story-title">L3 — Do the passed pairs still work inside larger worlds?</h3><p class="story-question">The benchmark assembled passed L2 building blocks into longer chains and grids. When a larger world failed, an isolated counterfactual retest checked whether the pair itself was really responsible.</p><div class="story-numbers">${storyMetric(chainSummary.total,'chain worlds',`${chainSummary.passed} passed · ${chainSummary.failed} failed`)}${storyMetric(gridSummary.total,'grid worlds',`${gridSummary.passed} passed · ${gridSummary.failed} failed`)}${storyMetric(l3Counterfactual.length,'diagnostic retests',`${counterSummary.passed} passed · ${counterSummary.failed} failed`)}${storyMetric(`${l3Coverage.passed}/${l3Coverage.executed}`,'all L3 tests passed',pct(l3Coverage.passed,l3Coverage.executed))}</div><div class="result-chips"><div class="result-chip ${chainSummary.failed?'mixed':'passed'}"><b>Chains</b><br>${chainSummary.passed} passed and ${chainSummary.failed} failed</div><div class="result-chip ${gridSummary.failed?'failed':'passed'}"><b>Grids</b><br>${gridSummary.passed} passed and ${gridSummary.failed} failed</div><div class="result-chip ${counterSummary.failed?'mixed':'passed'}"><b>Diagnostic retests</b><br>${counterSummary.passed} passed and ${counterSummary.failed} failed</div></div><div class="decision"><b>L3 result:</b> The policy handled many straight chains, but none of the ${gridSummary.total} grid conditions passed. Including diagnostic retests, L3 passed ${pct(l3Coverage.passed,l3Coverage.executed)}.</div>`}else{const LL=D.later_levels||{status:'not_created',executed:0,headline:'L4-L7 -- no data',conclusion:'',levels:{}};const chip=(lvl,v)=>{const cls=v.status==='executed'?(v.pass_rate>=.5?'pass':'fail'):'pruned';const body=v.status==='executed'?`${v.passed} passed · ${v.failed} failed${v.pruned?` · ${v.pruned} pruned`:''} (${pct(v.passed,v.executed)})`:(v.status==='pruned'?`${v.pruned} pruned, none executed`:'not created — no building blocks passed');return`<div class="result-chip ${cls}"><b>${esc(lvl)} · ${esc(v.title)}</b><br>${esc(v.blurb)} — ${esc(body)}</div>`};html=`<div class="eyebrow" style="color:${LL.status==='executed'?'var(--blue)':'var(--amber)'}">${LL.status==='executed'?'Finally':'Stopping decision'}</div><h3 class="story-title">${esc(LL.headline)}</h3><p class="story-question">Later levels are assembled only from terrain pairs that already passed, so their breadth reflects what the policy earned at L2.</p><div class="result-chips">${Object.entries(LL.levels).map(([l,v])=>chip(l,v)).join('')}</div><div class="decision"><b>Final conclusion:</b> ${esc(LL.conclusion)}</div>`}$('story-panel').innerHTML=html}document.querySelectorAll('[data-story]').forEach(b=>b.onclick=()=>{storyIndex=Number(b.dataset.story);renderStory()});$('story-prev').onclick=()=>{storyIndex=Math.max(0,storyIndex-1);renderStory()};$('story-next').onclick=()=>{storyIndex=Math.min(3,storyIndex+1);renderStory()};renderStory();
function placePopup(box,ev){box.hidden=false;const pad=12,w=Math.min(700,window.innerWidth-2*pad);box.style.width=w+'px';let x=ev?ev.clientX+pad:pad,y=ev?ev.clientY+pad:pad;const h=box.offsetHeight;if(x+w>window.innerWidth-pad)x=Math.max(pad,(ev?ev.clientX:0)-w-pad);if(y+h>window.innerHeight-pad)y=Math.max(pad,window.innerHeight-h-pad);box.style.left=x+'px';box.style.top=y+'px'}function openPairPopup(key,ev){const pd=D.pair_details?.[key];const box=$('pair-popup'),body=$('pair-popup-body');if(!pd||!box){return}const parentLine=(d)=>{const t=String(d.terrain||'?').toUpperCase();if(d.status==='passed')return`<span class="ok">${t} passed flat ${d.successes}/${d.valid}</span>`;const how=Object.entries(d.reasons||{}).filter(([r])=>r!=='goal').map(([r,n])=>`${({fall:'fell',stuck:'got stuck',timeout:'timed out'})[r]||r} ×${n}`).join(', ');return`<span class="bad">${t} failed flat ${d.successes}/${d.valid}${how?' — '+esc(how):''}</span>`};const rows=pd.levels.map(r=>{let outcome,why='';if(r.status==='pruned'){outcome=`<span class="pruned">pruned</span>`;why=(r.pruned_by||[]).length?`because ${r.pruned_by.map(parentLine).join(' and ')}`:esc(r.prune_reason||'')}else if(r.status==='passed'||r.status==='failed'){const how=Object.entries(r.reasons||{}).filter(([x])=>x!=='goal').map(([x,n])=>`${({fall:'fell',stuck:'got stuck',timeout:'timed out'})[x]||x} ×${n}`).join(', ');outcome=`<span class="${r.status==='passed'?'ok':'bad'}">${r.status} ${r.successes}/${r.valid}</span>`;why=(r.status!=='passed'&&how)?esc(how):'';if(r.event_chain&&r.event_chain.length&&r.status!=='passed')why+=(why?'<br>':'')+'<small>'+esc(r.event_chain.join(' → '))+'</small>'}else{outcome=esc(r.status)}const parents=(r.parents||[]).length?`<div class="pair-parents">built from: ${r.parents.map(parentLine).join(' · ')}</div>`:'';return`<tr><td><b>${esc(r.difficulty)}</b></td><td>${outcome}</td><td>${why}${parents}</td></tr>`}).join('');body.innerHTML=`<h3>${esc(pd.first.toUpperCase())} → ${esc(pd.second.toUpperCase())}</h3><p class="sub">${esc(pd.first_name)} first, then ${esc(pd.second_name)}.</p><table class="pair-popup-table"><thead><tr><th>difficulty</th><th>outcome</th><th>why / what fed it</th></tr></thead><tbody>${rows}</tbody></table><p class="sub note">${esc(pd.note)}</p>`;placePopup(box,ev)}(()=>{const box=document.getElementById('pair-popup');if(!box)return;const close=()=>{box.hidden=true};box.querySelector('.pair-popup-close').onclick=close;document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});document.addEventListener('mousedown',e=>{if(!box.hidden&&!box.contains(e.target)&&!e.target.closest('[data-map-pair]')&&!e.target.closest('[data-world]'))close()})})();function pairDot(item,label){const status=item?.status||'not-run';return`<i class="mini-dot ${esc(status)}" title="${esc(label)}: ${esc(status)}"></i>`}function renderPairMap(){let html='<div class="pair-head pair-axis">FIRST ↓<br>SECOND →</div>'+terrains.map(t=>`<div class="pair-head" title="${esc(terrainNames[t])}">${t.toUpperCase()}</div>`).join('');for(const first of terrains){html+=`<div class="pair-row-head" title="${esc(terrainNames[first])}">${first.toUpperCase()} first</div>`;for(const second of terrains){if(first===second){html+='<div class="pair-cell self">same<br>terrain</div>';continue}const key=`${first}→${second}`,items=(allPairGroups[key]||[]).slice().sort((a,b)=>(a.geometry_level||0)-(b.geometry_level||0)),c=countStatuses(items);const pd=D.pair_details?.[key];html+=`<div class="pair-cell ${key===selectedPair?'selected':''}" data-map-pair="${esc(key)}"${pd&&pd.hover?` data-tip="${esc(pd.hover)}"`:` title="${esc(first.toUpperCase()+' then '+second.toUpperCase()+': '+statusCounts(items))}"`}><b>${first.toUpperCase()}→${second.toUpperCase()}</b><div class="pair-dots">${['easy','medium','hard'].map((g,i)=>pairDot(items[i],g)).join('')}</div><small>P${c.passed} F${c.failed} X${c.pruned+c.unsupported+c.blocked}</small></div>`}}$('pair-map').innerHTML=html;document.querySelectorAll('[data-map-pair]').forEach(el=>el.onclick=(ev)=>{selectedPair=el.dataset.mapPair;selectedTerrain=selectedPair.split('→')[0];renderPairMap();renderTree();openPairPopup(selectedPair,ev)});const selected=allPairGroups[selectedPair]||[],c=countStatuses(selected);$('pair-map-summary').innerHTML=`<b>Full pair coverage:</b> ${Object.keys(allPairGroups).length} directional pair families and ${pairConditions.length} easy/medium/hard variants are present. ${l2Coverage.executed} variants were executed and ${l2Coverage.pruned+l2Coverage.unsupported} were not executed. <b>Currently selected:</b> ${esc((selectedPair||'none').toUpperCase())} — ${c.passed} passed, ${c.failed} failed, and ${c.pruned+c.unsupported+c.blocked} pruned or unsupported.`}
function renderTree(){const singles=primitives.filter(c=>c.terrain===selectedTerrain).sort((a,b)=>(a.geometry_level-b.geometry_level)||(a.static_friction-b.static_friction));const pairs=pairConditions.filter(c=>(c.pair||[]).includes(selectedTerrain));const grouped={};pairs.forEach(c=>(grouped[pairKey(c)]??=[]).push(c));const pairGroups=Object.entries(grouped).sort(([a],[b])=>a.localeCompare(b,undefined,{numeric:true}));if(!selectedPair||!grouped[selectedPair])selectedPair=pairGroups.find(([,v])=>treeStatus(v)==='passed')?.[0]||pairGroups[0]?.[0]||null;$('terrain-branches').innerHTML=terrains.map(t=>{const items=primitives.filter(c=>c.terrain===t);return node(`${t.toUpperCase()} — ${terrainNames[t]}`,statusCounts(items),treeStatus(items),'',`data-terrain="${esc(t)}"`)}).join('');[...$('terrain-branches').children].forEach(el=>{if(el.dataset.terrain===selectedTerrain)el.classList.add('selected');el.onclick=()=>{selectedTerrain=el.dataset.terrain;selectedPair=null;renderTree();renderPairMap()}});const selected=grouped[selectedPair]||[];const related=treeConditions.filter(c=>c.level==='L3'&&(c.allowed_pairs||c.covered_pairs||[]).some(p=>p.join('→')===selectedPair));$('selected-banner').innerHTML=`<b>Selected branch:</b> ${esc(selectedTerrain.toUpperCase())} (${esc(terrainNames[selectedTerrain])}) → <b>${esc((selectedPair||'no pair').toUpperCase())}</b>. This is one inspected path; the 11×11 map above contains every directional pair.`;const singleHtml=`<div class="tree-column"><h3>Step 1 · ${selectedTerrain.toUpperCase()} alone</h3><div class="column-help">Nine settings: three terrain shapes × three friction settings.</div>${singles.map(c=>node(`${c.geometry_label} shape · ${c.friction_label} friction`,`${c.passes}/${c.valid_attempts} attempts passed`,c.status,c.prune_reason?`<div class="prune-chain">${esc(c.prune_reason)}</div>`:'' )).join('')}</div>`;const groupsHtml=`<div class="tree-column"><h3>Step 2 · Pairs involving ${selectedTerrain.toUpperCase()}</h3><div class="column-help">Both orders are separate. Click one pair to inspect it.</div>${pairGroups.map(([key,items])=>node(key.toUpperCase(),statusCounts(items),treeStatus(items),'',`data-pair="${esc(key)}"`)).join('')}</div>`;const detailHtml=`<div class="tree-column"><h3>Step 3 · Selected ${esc((selectedPair||'No pair').toUpperCase())}</h3><div class="column-help">The three cards are easy, medium, and hard versions of this exact order.</div>${selected.map(c=>node(`${c.geometry_label} pair`,`${c.passes}/${c.valid_attempts} attempts passed`,c.status,c.pruned_by?.length?`<div class="prune-chain">Not run because:<br>${c.pruned_by.map(esc).join('<br>')}</div>`:(c.prune_reason?`<div class="prune-chain">${esc(c.prune_reason)}</div>`:''))).join('')||'<div class="empty">No pair selected.</div>'}</div>`;const relatedNodes=related.slice(0,14).map(c=>node(`${c.topology} · ${c.geometry_label}`,`${c.passes}/${c.valid_attempts} attempts passed`,c.status)).join('');const laterHtml=`<div class="tree-column"><h3>Step 4 · Larger worlds</h3><div class="column-help">Connected worlds containing the selected pair, followed by the global L4–L7 decision.</div>${relatedNodes||node('No connected world used this pair','It did not produce an eligible L3 child','not-run')}${related.length>14?`<details><summary>${related.length-14} more L3 worlds</summary><div class="tree-list">${related.slice(14).map(c=>node(`${c.topology} · ${c.geometry_label}`,`${c.passes}/${c.valid_attempts}`,c.status)).join('')}</div></details>`:''}${node('Levels 4–7 were not created',`Overall L3 pass rate ${pct(l3Coverage.passed,l3Coverage.executed)} passed at L3`,'not-run')}</div>`;$('branch-flow').innerHTML=singleHtml+'<div class="tree-arrow">→</div>'+groupsHtml+'<div class="tree-arrow">→</div>'+detailHtml+'<div class="tree-arrow">→</div>'+laterHtml;document.querySelectorAll('[data-pair]').forEach(el=>{if(el.dataset.pair===selectedPair)el.classList.add('selected');el.onclick=()=>{selectedPair=el.dataset.pair;renderTree();renderPairMap()}});const sc=countStatuses(singles),pc=countStatuses(selected);const stopReason=pc.pruned+pc.unsupported+pc.blocked?` ${pc.pruned+pc.unsupported+pc.blocked} selected-pair variants were not run; their cards name the exact earlier condition responsible.`:'';$('tree-explanation').innerHTML=`<b>Plain-language result for this path:</b> ${esc(selectedTerrain.toUpperCase())} alone passed ${sc.passed} settings and failed ${sc.failed}; ${sc.unsupported} were unsupported. Then ${esc((selectedPair||'no pair').toUpperCase())} passed ${pc.passed} difficulty variants and failed ${pc.failed}. It was used in ${related.length} connected worlds.${stopReason}`;}
$('pair-fit').onclick=()=>{$('pair-map').className='pair-map';$('pair-fit').classList.add('active');$('pair-large').classList.remove('active')};$('pair-large').onclick=()=>{$('pair-map').className='pair-map large';$('pair-large').classList.add('active');$('pair-fit').classList.remove('active')};$('tree-compact').onclick=()=>{$('capability-tree').className='tree-shell tree-compact';$('tree-compact').classList.add('active');$('tree-comfortable').classList.remove('active')};$('tree-comfortable').onclick=()=>{$('capability-tree').className='tree-shell tree-comfortable';$('tree-comfortable').classList.add('active');$('tree-compact').classList.remove('active')};renderPairMap();renderTree();
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;const med=(c,n)=>c?.metrics?.[n]?.median;function routeMeter(progress){const p=Number.isFinite(progress)?Math.max(0,Math.min(1,progress)):0;return `<div class="route-progress"><span class="done" style="width:${100*p}%"></span><span class="dog" style="left:${Math.max(2,Math.min(98,100*p))}%" title="estimated robot progress">🐕</span></div><div class="route-labels"><span>start</span><b>${Number.isFinite(progress)?pct(progress,1):'unknown'}</b><span>goal</span></div>`}function distanceText(metrics){const d=metrics?.estimated_route_distance_m,r=metrics?.route_length_m,cp=metrics?.goal_checkpoint_reached,cn=metrics?.goal_checkpoint_count;if(Number.isFinite(d)&&Number.isFinite(r))return `<b>${d.toFixed(2)} m</b> of ${r.toFixed(2)} m<br><span class="sub">${Number.isFinite(cp)&&Number.isFinite(cn)?`checkpoint ${Math.round(cp)} of ${Math.round(cn)} · `:''}estimated from saved checkpoints</span>`;return 'not recoverable'}
const primitive=D.conditions.filter(c=>c.kind==='primitive');$('terrain-performance').innerHTML=[...new Set(primitive.map(c=>c.terrain))].sort((a,b)=>Number(a.slice(2))-Number(b.slice(2))).map(t=>{const cs=primitive.filter(c=>c.terrain===t),measured=cs.filter(c=>['passed','failed'].includes(c.status)),passed=measured.filter(c=>c.status==='passed'),ids=new Set(cs.flatMap(c=>c.attempts||[])),trials=D.trials.filter(x=>ids.has(x.trial_id)&&x.valid_behavior),progress=mean(trials.map(x=>x.metrics?.progress_fraction).filter(Number.isFinite)),dist=mean(trials.map(x=>x.metrics?.estimated_route_distance_m).filter(Number.isFinite)),route=mean(trials.map(x=>x.metrics?.route_length_m).filter(Number.isFinite)),hard=Math.max(...passed.map(c=>Number(c.geometry_level)).filter(Number.isFinite),-1),low=Math.min(...passed.map(c=>Number(c.static_friction)).filter(Number.isFinite),Infinity);return `<article class="card"><span class="badge ${passed.length===measured.length?'passed':passed.length?'unsupported':'failed'}">${esc(t.toUpperCase())}</span><h3>${esc(terrainNames[t]||t)}</h3><div class="metric-pair"><div><b>${passed.length}/${measured.length}</b><div class="sub">conditions passed</div></div><div><b>${Number.isFinite(progress)?pct(progress,1):'—'}</b><div class="sub">mean attempt progress</div></div></div>${routeMeter(progress)}<p>${Number.isFinite(dist)&&Number.isFinite(route)?`Mean estimated reach: <b>${dist.toFixed(2)} m</b> on ${route.toFixed(2)} m routes.`:'Distance unavailable.'}</p><p class="sub">Hardest passed shape: ${hard>=0?hard.toFixed(2):'none'} · Lowest passed static friction: ${Number.isFinite(low)?low.toFixed(2):'none'}</p></article>`}).join('');
const pa=D.performance_availability||{};$('distance-notice').innerHTML=`<b>Recovered from the completed run:</b> ${pa.progress_attempts||0} valid attempts contain checkpoint progress, and ${pa.estimated_distance_attempts||0} can be converted to approximate route distance. ${esc(pa.method||'')} True travelled path is available for ${pa.true_path_distance_attempts||0} attempts.`;
$('world-gallery').innerHTML=(D.world_gallery||[]).map(w=>{const progress=med(w,'progress_fraction'),distance=med(w,'estimated_route_distance_m'),route=med(w,'route_length_m'),checkpoint=med(w,'goal_checkpoint_reached'),checkpointCount=med(w,'goal_checkpoint_count');return `<article class="card world">${w.preview?`<a href="../${esc(w.preview)}"><img loading="lazy" src="${esc(w.preview_data_uri||'../'+w.preview)}" alt="Top-down test layout for ${esc(w.condition_id)}"></a>`:''}<div class="body"><span class="badge ${esc(w.status)}">${esc(w.status)}</span> <b>${esc(w.level)} · ${esc(w.terrain||(w.pair||[]).join(' → ')||w.topology||w.kind)}</b>${routeMeter(progress)}<div class="metric-pair"><div><b>${Number.isFinite(distance)?distance.toFixed(2)+' m':'—'}</b><div class="sub">median estimated reach</div></div><div><b>${Number.isFinite(route)?route.toFixed(2)+' m':'—'}</b><div class="sub">route length</div></div></div><p>${Number.isFinite(checkpoint)&&Number.isFinite(checkpointCount)?`Median checkpoint: <b>${Math.round(checkpoint)} of ${Math.round(checkpointCount)}</b><br>`:''}${w.valid_attempts!=null?`${w.passes}/${w.valid_attempts} successful attempts · `:''}<a href="../${esc(w.world)}">world JSON</a></p><code>${esc(w.condition_id)}</code></div></article>`}).join('')||'<div class="empty">No representative results could be generated.</div>';
const inv=D.evidence_inventory||{};$('inventory').innerHTML=[['Runner logs',inv.native_runner_logs||0],['Result JSON files',inv.result_json||0],['Summary JSON files',inv.summary_json||0],['Videos',inv.videos||0],['Snapshots',inv.snapshots||0],['Canonical telemetry',inv.canonical_telemetry_files||0]].map(([x,n])=>`<div class="card"><div class="big">${n}</div><div class="label">${x}</div></div>`).join('');$('media-notice').innerHTML=inv.videos?`<b>${inv.videos} videos are available.</b> Open them from the evidence links.`:'<b>No episode videos or snapshots exist for this run.</b> Video capture was disabled in the native batch configuration, and no frame-level episode log was retained from which motion could be reconstructed. The world images above are layout previews. The result JSON, summary JSON, and runner logs are the available execution evidence.';
function artifact(t){let a=[];for(const [k,n] of [['native_result','result'],['summary','summary'],['native_log','runner log'],['world','world'],['run_dir','folder']])if(t?.[k])a.push(`<a href='../${esc(t[k])}'>${n}</a>`);if(t?.telemetry?.trajectory_json)a.push(`<a href='../${esc(t.telemetry.trajectory_json)}'>trajectory</a>`);if(t?.media?.video?.path)a.push(`<a href='../${esc(t.media.video.path)}'>video</a>`);return a.join('')||'—'}
for(const [id,key] of [['level-filter','level'],['kind-filter','kind'],['status-filter','status']]){[...new Set(D.conditions.map(x=>x[key]).filter(Boolean))].sort().forEach(v=>$(id).insertAdjacentHTML('beforeend',`<option value="${esc(v)}">${esc(v)}</option>`));$(id).onchange=renderConditions}$('condition-search').oninput=renderConditions;
function renderConditions(){const lv=$('level-filter').value,k=$('kind-filter').value,s=$('status-filter').value,q=$('condition-search').value.toLowerCase();const rows=D.conditions.filter(c=>(!lv||c.level===lv)&&(!k||c.kind===k)&&(!s||c.status===s)&&(!q||JSON.stringify(c).toLowerCase().includes(q)));$('condition-count').textContent=`Showing ${Math.min(rows.length,300)} of ${rows.length} matching conditions. Refine filters to narrow the table.`;$('condition-rows').innerHTML=rows.slice(0,300).map(c=>{const first=(c.attempts||[]).map(id=>D.trials.find(t=>t.trial_id===id)).find(Boolean);const test=c.terrain||(c.pair||[]).join(' → ')||c.topology||c.kind;const ci=c.lower_ci==null?'—':`${(100*c.lower_ci).toFixed(1)}–${(100*c.upper_ci).toFixed(1)}%`;return `<tr><td><code>${esc(c.id)}</code></td><td>${esc(c.level)}<br><span class="sub">${esc(c.kind)}</span></td><td>${esc(test)}<br><span class="sub">${esc(c.geometry_label||'')} ${esc(c.friction_label||'')}</span></td><td><span class="badge ${esc(c.status)}">${esc(c.status)}</span></td><td>${c.passes||0} pass / ${c.valid_attempts||0} valid</td><td>${c.success_rate==null?'—':pct(c.passes,c.valid_attempts)}<br><span class="sub">${ci}</span></td><td>${distanceText(Object.fromEntries(Object.entries(c.metrics||{}).map(([n,v])=>[n,v.median])))}</td><td>${esc(Object.keys(c.failure_types||{}).join(', ')||c.prune_reason||Object.keys(c.infrastructure_outcomes||{}).join(', ')||'—')}</td><td class="artifact">${artifact(first)}</td></tr>`}).join('')||'<tr><td colspan="9">No matching conditions.</td></tr>'}
$('attempt-outcome').onchange=renderAttempts;$('attempt-search').oninput=renderAttempts;function classOf(t){if(t.valid_behavior&&t.success===true)return'pass';if(t.valid_behavior&&t.success===false)return'behavioral_failure';return'infrastructure'}function renderAttempts(){const o=$('attempt-outcome').value,q=$('attempt-search').value.toLowerCase();const rows=D.trials.filter(t=>(!o||classOf(t)===o)&&(!q||JSON.stringify(t).toLowerCase().includes(q)));$('attempt-count').textContent=`Showing ${Math.min(rows.length,250)} of ${rows.length} matching attempts. All ${D.trials.length} remain in dashboard_data.json.`;$('attempt-rows').innerHTML=rows.slice(0,250).map(t=>{const cl=classOf(t),progress=t.metrics?.progress_fraction;return `<tr><td><code>${esc(t.trial_id)}</code></td><td><code>${esc(t.condition_id)}</code></td><td>${esc(t.seed)}</td><td><span class="badge ${cl==='pass'?'passed':cl==='behavioral_failure'?'failed':'unsupported'}">${esc(cl.replaceAll('_',' '))}</span></td><td>${esc(t.terminal_reason||t.status||'—')}</td><td>${distanceText(t.metrics)}${Number.isFinite(progress)?`<br><b>${pct(progress,1)}</b> progress`:''}</td><td class="artifact">${artifact(t)}</td></tr>`}).join('')||'<tr><td colspan="7">No matching attempts.</td></tr>'}
const allReasons=Object.fromEntries((D.failure_clusters||[]).map(c=>[c.category,c.count]));$('failure-explanation').innerHTML=`There were <b>${tc.behavioral_failures||0}</b> behavioral failures. In this run, ${allReasons.route_incomplete||0} were classified as <b>route incomplete</b>. Because no frame-level canonical telemetry or video was retained, this report cannot reliably separate slip, fall, collision, or control causes; those require a new instrumented run.`;$('failure-clusters').innerHTML=(D.failure_clusters||[]).map(c=>`<div class="card"><div class="big fail">${c.count}</div><b>${esc(c.category.replaceAll('_',' '))}</b><p class="sub">Across ${c.conditions.length} conditions</p></div>`).join('')||'<div class="empty">No behavioral failures recorded.</div>';$('failure-list').innerHTML=(D.failures||[]).slice(0,80).map(f=>`<div class="panel"><b>${esc(f.condition_id)}</b> · seed ${esc(f.seed)} · <span class="badge failed">${esc(f.terminal_reason)}</span><div class="artifact">${artifact(f)}</div></div>`).join('')||'<div class="empty">No failure attempts.</div>';
$('raw-json').textContent=JSON.stringify(D,null,2);$('generated').textContent=D.generated_at;renderConditions();renderAttempts();""" + refresh + r"""
</script></body></html>"""


def _render_trial_media(record: Dict[str, Any], examination_root: Path) -> Dict[str, Any]:
    """Render one trial's missing media without mutating shared state."""
    rendered: Dict[str, Any] = {}
    if not record.get("valid_behavior"):
        return rendered
    media = record.get("media", {})
    if not media.get("snapshots"):
        rendered["snapshots"] = render_trial_snapshots(record, examination_root)
    if record.get("success") is False and not media.get("video"):
        rendered["video"] = render_telemetry_video(record, examination_root)
    return rendered



LATER_LEVEL_TITLES = {
    "L4": "Route selection",
    "L5": "Endurance",
    "L6": "Stress tests",
    "L7": "Held-out tests",
}
LATER_LEVEL_BLURBS = {
    "L4": "branching worlds requiring navigation choices",
    "L5": "100-200 body-length courses",
    "L6": "pushes and control/perception latency",
    "L7": "locked unseen combinations, banked and curved ground",
}



DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
FRICTION_ORDER = {"high": 0, "moderate": 1, "low": 2}
REASON_WORDS = {"goal": "reached goal", "fall": "fell", "stuck": "got stuck", "timeout": "timed out",
                "adapter_unverified": "setup unverified", "stalled": "stalled"}


def _trials_by_condition(trials: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials or ():
        grouped.setdefault(str(trial.get("condition_id")), []).append(trial)
    return grouped


def _condition_detail(condition: Dict[str, Any], trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One condition, with how its attempts ended and the commonest failure chain."""

    reasons: Dict[str, int] = {}
    chains: Dict[str, int] = {}
    successes = 0
    valid = 0
    for trial in trials:
        if trial.get("valid_behavior") is False:
            continue
        valid += 1
        if trial.get("success") is True:
            successes += 1
        reason = str(trial.get("terminal_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        if trial.get("success") is not True:
            chain = tuple((trial.get("diagnosis") or {}).get("event_chain") or ())
            if chain:
                key = " → ".join(str(step) for step in chain)
                chains[key] = chains.get(key, 0) + 1
    summary = condition.get("summary") or {}
    if not trials and summary:
        successes = int(summary.get("successes", 0))
        valid = int(summary.get("valid", 0))
    top_chain = max(chains.items(), key=lambda item: item[1])[0].split(" → ") if chains else []
    return {
        "id": condition.get("id"),
        "status": condition.get("status"),
        "difficulty": condition.get("geometry_label"),
        "friction": condition.get("friction_label"),
        "successes": successes,
        "valid": valid,
        "reasons": reasons,
        "event_chain": top_chain,
    }


def _reason_phrase(reasons: Dict[str, int], exclude_goal: bool = True) -> str:
    parts = []
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        if exclude_goal and reason == "goal":
            continue
        parts.append("%s ×%d" % (REASON_WORDS.get(reason, reason), count))
    return ", ".join(parts)


def _cell_hover(conditions: List[Dict[str, Any]]) -> str:
    """The text behind a variant cell: one line per difficulty/friction combination."""

    if not conditions:
        return "not run"
    lines = []
    for item in conditions:
        head = "%s / %s: %s %d/%d" % (item["difficulty"], item["friction"],
                                       "passed" if item["status"] == "passed" else "failed",
                                       item["successes"], item["valid"])
        how = _reason_phrase(item["reasons"])
        if item["status"] != "passed" and how:
            head += " — " + how
        lines.append(head)
    return "\n".join(lines)


def level_details(conditions: Dict[str, Any], trials: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Per level: every executed condition, how it went, and why the failures failed."""

    grouped = _trials_by_condition(trials)
    out: Dict[str, Any] = {}
    for condition in conditions.values():
        level = str(condition.get("level"))
        entry = out.setdefault(level, {"executed": 0, "passed": 0, "failed": 0, "pruned": 0,
                                       "reasons": {}, "conditions": []})
        status = condition.get("status")
        if status == "pruned":
            entry["pruned"] += 1
            continue
        if status not in {"passed", "failed"}:
            continue
        detail = _condition_detail(condition, grouped.get(str(condition.get("id")), []))
        if condition.get("kind") in {"primitive", "variant"}:
            label = "%s %s / %s%s" % (str(condition.get("terrain", "")).upper(), detail["difficulty"],
                                      detail["friction"], " (banked/curved)" if condition.get("kind") == "variant" else "")
        elif condition.get("kind") in {"pair", "counterfactual"}:
            pair = condition.get("pair") or []
            label = "%s → %s (%s)%s" % (str(pair[0]).upper() if pair else "?", str(pair[1]).upper() if len(pair) > 1 else "?",
                                        detail["difficulty"], " retest" if condition.get("kind") == "counterfactual" else "")
        else:
            label = "%s · %s (%s)" % (condition.get("topology", condition.get("kind")), condition.get("horizon", ""), detail["difficulty"])
            stressors = [str(item.get("type")) for item in (condition.get("stressors") or []) if item.get("type")]
            if stressors:
                label += " · " + ", ".join(stressors)
            modifier = condition.get("modifier") or {}
            if modifier.get("camber_deg") or modifier.get("curve_angle_deg"):
                label += " · banked/curved"
        detail["label"] = label
        entry["conditions"].append(detail)
        entry["executed"] += 1
        entry["passed" if status == "passed" else "failed"] += 1
        for reason, count in detail["reasons"].items():
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + count
    for level, entry in out.items():
        failures = [c for c in entry["conditions"] if c["status"] != "passed"]
        chains: Dict[str, int] = {}
        for c in failures:
            if c["event_chain"]:
                key = " → ".join(c["event_chain"])
                chains[key] = chains.get(key, 0) + 1
        top = max(chains.items(), key=lambda item: item[1])[0] if chains else ""
        lines = ["%d conditions · %d passed · %d failed%s" % (
            entry["executed"], entry["passed"], entry["failed"],
            (" · %d pruned" % entry["pruned"]) if entry["pruned"] else "")]
        how = _reason_phrase(entry["reasons"])
        if how:
            lines.append("attempts that did not reach the goal: " + how)
        if top:
            lines.append("commonest failure: " + top)
        worst = sorted(failures, key=lambda c: (c["successes"] / c["valid"]) if c["valid"] else 0.0)[:3]
        for c in worst:
            lines.append("  %s — %d/%d, %s" % (c["label"], c["successes"], c["valid"], _reason_phrase(c["reasons"]) or "no attempts recorded"))
        entry["hover"] = "\n".join(lines)
    return out


def pair_details(conditions: Dict[str, Any], trials: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Every directional pair: what fed it, and how each difficulty went.

    A pair is built from the two FLAT terrains at the qualification friction,
    one pair condition per difficulty. Banked/curved variants never feed a pair.
    When a difficulty is pruned, `pruned_by` names the parent that failed flat.
    """

    grouped = _trials_by_condition(trials)
    detail_cache: Dict[str, Dict[str, Any]] = {}

    def detail(cid: str) -> Optional[Dict[str, Any]]:
        if cid in detail_cache:
            return detail_cache[cid]
        cond = conditions.get(cid)
        if cond is None:
            return None
        d = _condition_detail(cond, grouped.get(cid, []))
        d["terrain"] = cond.get("terrain")
        d["kind"] = cond.get("kind")
        detail_cache[cid] = d
        return d

    out: Dict[str, Any] = {}
    for cond in conditions.values():
        if cond.get("kind") != "pair":
            continue
        pair = list(cond.get("pair") or [])
        if len(pair) != 2:
            continue
        key = "%s→%s" % (pair[0], pair[1])
        entry = out.setdefault(key, {
            "first": pair[0], "second": pair[1],
            "first_name": TERRAIN_DISPLAY_NAMES.get(pair[0], pair[0]),
            "second_name": TERRAIN_DISPLAY_NAMES.get(pair[1], pair[1]),
            "levels": [],
            "note": "Pairs are built only from the FLAT single-terrain results at the qualification friction. "
                    "Banked/curved variants never feed a pair, and a difficulty is pruned when either terrain failed flat at that difficulty.",
        })
        row = _condition_detail(cond, grouped.get(str(cond.get("id")), []))
        row["difficulty"] = cond.get("geometry_label")
        row["parents"] = [d for d in (detail(pid) for pid in cond.get("parents") or []) if d]
        row["pruned_by"] = [d for d in (detail(pid) for pid in cond.get("pruned_by") or []) if d]
        row["prune_reason"] = cond.get("prune_reason") or ""
        entry["levels"].append(row)

    def label(d: Dict[str, Any]) -> str:
        t = str(d.get("terrain", "?")).upper()
        if d["status"] == "passed":
            return "%s passed flat %d/%d" % (t, d["successes"], d["valid"])
        how = _reason_phrase(d["reasons"])
        return "%s failed flat %d/%d%s" % (t, d["successes"], d["valid"], (" — " + how) if how else "")

    for key, entry in out.items():
        entry["levels"].sort(key=lambda r: DIFFICULTY_ORDER.get(r["difficulty"], 9))
        lines = ["%s → %s" % (entry["first"].upper(), entry["second"].upper())]
        for r in entry["levels"]:
            if r["status"] == "pruned":
                who = ", ".join(label(d) for d in r["pruned_by"]) or "a parent failed flat"
                lines.append("%s: pruned — %s" % (r["difficulty"], who))
            elif r["status"] in {"passed", "failed"}:
                how = _reason_phrase(r["reasons"])
                lines.append("%s: %s %d/%d%s" % (r["difficulty"], r["status"], r["successes"], r["valid"],
                                                 (" — " + how) if (r["status"] != "passed" and how) else ""))
            else:
                lines.append("%s: %s" % (r["difficulty"], r["status"]))
        entry["hover"] = "\n".join(lines)
    return out


COMPOSED_LEVEL_TITLES = {
    "L3": ("Connected worlds", "Passed pairs assembled into chains and grids at medium horizon."),
    "L4": ("Route selection", "Ring, tree and randomly connected worlds at long horizon, where the route branches."),
    "L5": ("Endurance", "Chains of 100-200 body lengths."),
    "L6": ("Stress tests", "Long chains with a push impulse, control latency, or perception latency applied."),
    "L7": ("Held-out tests", "Locked unseen combinations, including banked and curved ground."),
}


def composed_levels(conditions: Dict[str, Any], trials: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """L3-L7, one section each: every composed world, what it was built from, how it went, and its retest.

    A failed composed world spawns a counterfactual: one covered pair run on its
    own. If the pair alone passes, the failure came from the composition; if it
    fails too, the pair is dropped from later levels. That verdict is attached
    to the world's row.
    """

    grouped = _trials_by_condition(trials)
    retests: Dict[str, Dict[str, Any]] = {}
    for cond in conditions.values():
        if cond.get("kind") == "counterfactual" and cond.get("parent_world"):
            retests[str(cond["parent_world"])] = cond

    def pair_key(pair: Any) -> str:
        pair = list(pair or [])
        return "%s→%s" % (pair[0], pair[1]) if len(pair) == 2 else "?"

    out = []
    for level in ("L3", "L4", "L5", "L6", "L7"):
        title, blurb = COMPOSED_LEVEL_TITLES[level]
        rows = []
        for cond in conditions.values():
            if cond.get("level") != level or cond.get("kind") != "progression":
                continue
            if cond.get("status") not in {"passed", "failed"}:
                continue
            d = _condition_detail(cond, grouped.get(str(cond.get("id")), []))
            name = "%s · %s" % (cond.get("topology", "?"), cond.get("horizon", "?"))
            stressors = [str(x.get("type")) for x in (cond.get("stressors") or []) if x.get("type")]
            if stressors:
                name += " · " + ", ".join(stressors)
            modifier = cond.get("modifier") or {}
            if modifier.get("camber_deg") or modifier.get("curve_angle_deg"):
                name += " · banked/curved"
            pairs = [pair_key(pr) for pr in (cond.get("covered_pairs") or [])]
            retest = None
            cf = retests.get(str(cond.get("id")))
            if cf is not None:
                rd = _condition_detail(cf, grouped.get(str(cf.get("id")), []))
                if rd["status"] == "passed":
                    verdict = "the pair alone still passes, so the failure came from the composition (length, turns, stressor), not the pair"
                elif rd["status"] == "failed":
                    verdict = "the pair alone fails too, so it is dropped from later levels"
                else:
                    verdict = "retest %s" % rd["status"]
                retest = {"pair": pair_key(cf.get("pair")), "status": rd["status"], "successes": rd["successes"],
                          "valid": rd["valid"], "reasons": rd["reasons"], "verdict": verdict}
            how = _reason_phrase(d["reasons"])
            lines = ["%s (%s): %s %d/%d%s" % (name, d["difficulty"], d["status"], d["successes"], d["valid"],
                                             (" — " + how) if (d["status"] != "passed" and how) else "")]
            if d["event_chain"] and d["status"] != "passed":
                lines.append("commonest failure: " + " → ".join(d["event_chain"]))
            lines.append("built from: " + (", ".join(pairs) or "—"))
            if retest:
                lines.append("retest %s alone: %s %d/%d — %s" % (retest["pair"], retest["status"], retest["successes"], retest["valid"], retest["verdict"]))
            rows.append({
                "id": cond.get("id"), "name": name, "difficulty": d["difficulty"], "status": d["status"],
                "successes": d["successes"], "valid": d["valid"], "reasons": d["reasons"],
                "event_chain": d["event_chain"], "pairs": pairs, "retest": retest, "hover": "\n".join(lines),
            })
        rows.sort(key=lambda r: (DIFFICULTY_ORDER.get(r["difficulty"], 9), r["name"], str(r["id"])))
        executed = len(rows)
        passed = sum(1 for r in rows if r["status"] == "passed")
        pruned = sum(1 for c in conditions.values() if c.get("level") == level and c.get("status") == "pruned")
        empty_reason = ""
        if not executed:
            empty_reason = ("no composed world reached execution at this level"
                            + (" (%d pruned)" % pruned if pruned else "; no passing pairs to build from"))
        out.append({"level": level, "title": title, "blurb": blurb, "rows": rows, "executed": executed,
                    "passed": passed, "failed": executed - passed, "pruned": pruned, "empty_reason": empty_reason})
    return out


VARIANT_NAMES = ("left-curve-left-low", "right-curve-right-low")


def _variant_name(condition: Dict[str, Any]) -> Optional[str]:
    explicit = condition.get("variant")
    if explicit in VARIANT_NAMES:
        return explicit
    identifier = str(condition.get("id", ""))
    for name in VARIANT_NAMES:
        if identifier.endswith("-" + name):
            return name
    modifier = condition.get("modifier") or {}
    camber = float(modifier.get("camber_deg", 0.0) or 0.0)
    if camber < 0:
        return "left-curve-left-low"
    if camber > 0:
        return "right-curve-right-low"
    return None


def variant_summary(conditions: Dict[str, Any], trials: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Flat versus banked/curved, per terrain, from the L1 conditions.

    Variants ran but the page had no concept of them: the L1 tile counted by
    level and lumped them into one number, and the terrain matrix filtered to
    plain primitives, so 72 results were invisible. The comparison the variants
    exist to make -- does banking and curvature break a terrain the policy
    walks flat? -- is computed here.
    """

    def bucket() -> Dict[str, Any]:
        return {"passed": 0, "failed": 0, "executed": 0, "status": "not_created", "conditions": []}

    grouped = _trials_by_condition(trials)
    terrains: Dict[str, Dict[str, Any]] = {}
    plain_total, variant_total = bucket(), bucket()
    for condition in conditions.values():
        if condition.get("level") != "L1":
            continue
        kind = condition.get("kind")
        if kind not in {"primitive", "variant"}:
            continue
        terrain = str(condition.get("terrain", "?"))
        row = terrains.setdefault(terrain, {"plain": bucket(), **{name: bucket() for name in VARIANT_NAMES}})
        key = "plain" if kind == "primitive" else _variant_name(condition)
        if key is None:
            continue
        status = condition.get("status")
        if status in {"passed", "failed"}:
            row[key]["conditions"].append(
                _condition_detail(condition, grouped.get(str(condition.get("id")), []))
            )
        targets = [row[key], plain_total if kind == "primitive" else variant_total]
        for target in targets:
            if status == "passed":
                target["passed"] += 1
                target["executed"] += 1
            elif status == "failed":
                target["failed"] += 1
                target["executed"] += 1
            target["status"] = "executed" if target["executed"] else target["status"]
    for row in terrains.values():
        for values in row.values():
            values["pass_rate"] = (values["passed"] / values["executed"]) if values["executed"] else None
            values["conditions"].sort(key=lambda c: (DIFFICULTY_ORDER.get(c["difficulty"], 9),
                                                     FRICTION_ORDER.get(c["friction"], 9)))
            values["hover"] = _cell_hover(values["conditions"])
    for total in (plain_total, variant_total):
        total["pass_rate"] = (total["passed"] / total["executed"]) if total["executed"] else None
        total.pop("conditions", None)

    def order(code: str) -> int:
        try:
            return int(code[2:])
        except ValueError:
            return 999

    # A list, not a dict: the page payload is json.dumps(sort_keys=True), which
    # re-sorted dict keys to RS1, RS10, RS11, RS2, ...
    rows = []
    for code in sorted(terrains, key=order):
        row = dict(terrains[code])
        row["code"] = code
        row["name"] = TERRAIN_DISPLAY_NAMES.get(code, code)
        row["takeaway"] = _variant_takeaway(row)
        rows.append(row)
    return {
        "plain": plain_total,
        "variants": variant_total,
        "variant_names": list(VARIANT_NAMES),
        "terrains": rows,
    }


TERRAIN_DISPLAY_NAMES = {
    "rs1": "Slippery flat ground", "rs2": "Rough slippery ground", "rs3": "Slippery slope",
    "rs4": "Rough slippery slope", "rs5": "Stairs", "rs6": "Repeated waves", "rs7": "Curved hill",
    "rs8": "Small repeated gaps", "rs9": "Large gap", "rs10": "High step", "rs11": "Marsh approximation",
}


def _variant_takeaway(row: Dict[str, Any]) -> str:
    """One line a reader can act on, in the words a person would use."""

    flat = row["plain"]
    left = row["left-curve-left-low"]
    right = row["right-curve-right-low"]
    if not flat["executed"]:
        return "not measured flat"
    if flat["passed"] == 0:
        return "failed flat — no variants run"
    if not left["executed"] and not right["executed"]:
        return "passed flat; banked variants not run"
    f = flat["pass_rate"] or 0.0
    l = left["pass_rate"] if left["executed"] else None
    r = right["pass_rate"] if right["executed"] else None
    rates = [v for v in (l, r) if v is not None]
    good = lambda v: v is not None and v >= max(0.8, f - 0.15)
    poor = lambda v: v is not None and v <= 0.34
    if all(good(v) for v in rates):
        return "holds up when banked and curved"
    if good(l) and poor(r):
        return "curving left is fine, curving right isn't"
    if good(r) and poor(l):
        return "curving right is fine, curving left isn't"
    if all(poor(v) for v in rates) and f >= 0.6:
        return "fine flat, collapses when curved"
    if all(v < f for v in rates):
        return "worse when banked and curved"
    return "mixed: %s left, %s right" % (
        "—" if l is None else "%d%%" % round(100 * l),
        "—" if r is None else "%d%%" % round(100 * r),
    )

def later_levels_summary(level_counts: Dict[str, Any], level_results: Dict[str, Any], pair_library_empty: bool) -> Dict[str, Any]:
    """What L4-L7 actually did, computed from the same counts the page uses.

    This used to be an unconditional story step announcing "Not run, not
    failed" with four "not measured" chips -- written for the 70% gate that
    once cancelled L4-L7, and left in place after the gate was removed, so a run
    that executed dozens of L4-L7 conditions still told the reader they never
    ran. Computed here so it can be tested; the JS renders what it is handed.
    """

    levels = {}
    executed_total = 0
    for level in ("L4", "L5", "L6", "L7"):
        counts = dict(level_counts.get(level, {}) or {})
        passed = int(counts.get("passed", 0))
        failed = int(counts.get("failed", 0))
        pruned = int(counts.get("pruned", 0))
        executed = passed + failed
        executed_total += executed
        levels[level] = {
            "title": LATER_LEVEL_TITLES[level],
            "blurb": LATER_LEVEL_BLURBS[level],
            "passed": passed,
            "failed": failed,
            "pruned": pruned,
            "executed": executed,
            "pass_rate": (passed / executed) if executed else None,
            "status": "executed" if executed else ("pruned" if pruned else "not_created"),
            "stage_result": level_results.get(level),
        }
    if executed_total:
        measured = [
            "%s (%s) %d of %d passed" % (level, values["title"].lower(), values["passed"], values["executed"])
            for level, values in levels.items()
            if values["executed"]
        ]
        headline = "L4-L7 -- %d composed conditions executed" % executed_total
        conclusion = (
            "Later levels were built only from the terrain pairs that passed, so they cover the slice "
            "this policy earned rather than the full matrix. Measured: " + "; ".join(measured) + "."
        )
        status = "executed"
    else:
        reason = (
            "no terrain pair passed, so there was nothing to build them from"
            if pair_library_empty
            else "no composed condition reached execution"
        )
        headline = "L4-L7 -- not created: %s" % reason
        conclusion = (
            "Route selection, endurance, stress and held-out behaviour were not measured in this run "
            "because " + reason + ". They are absent for lack of building blocks, not by any pass-rate threshold."
        )
        status = "not_created"
    return {"status": status, "executed": executed_total, "headline": headline, "conclusion": conclusion, "levels": levels}

def refresh_evidence(state_path: str, workers: int = 1) -> Dict[str, Any]:
    """Rebuild media selections, aggregates, CSV, JSON, and standalone HTML."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive int")
    state_file = Path(state_path).resolve()
    root = state_file.parent
    state = load_data(str(state_file))
    trials = state.get("trials", [])
    by_id = {item["trial_id"]: item for item in trials}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_render_trial_media, record, root) for record in trials]
        for record, future in zip(trials, futures):
            rendered = future.result()
            if rendered:
                record.setdefault("media", {}).update(rendered)
    for condition in state.get("conditions", {}).values():
        if condition.get("status") not in {"passed", "failed", "unsupported", "blocked", "pruned"}:
            continue
        passes = [by_id[value] for value in condition.get("attempts", []) if value in by_id and by_id[value].get("success") is True]
        if passes:
            ranked = sorted(passes, key=lambda item: item.get("metrics", {}).get("completion_time", -item.get("metrics", {}).get("progress_fraction", 0.0)))
            representative = ranked[len(ranked) // 2]
            condition["representative_pass"] = representative["trial_id"]
            if not representative.get("media", {}).get("video"):
                representative.setdefault("media", {})["video"] = render_telemetry_video(representative, root)
            for item in passes:
                if item is representative:
                    continue
                video = item.get("media", {}).get("video") or {}
                if video.get("source") != "simulator" or not video.get("path"):
                    continue
                candidate = Path(video["path"])
                candidate = candidate if candidate.is_absolute() else root / candidate
                try:
                    candidate.resolve().relative_to(root.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    candidate.unlink()
                item["media"].pop("video", None)
                item["media"]["video_discarded"] = "non-representative passing attempt"
    dump_json(str(state_file), state)
    data = build_dashboard_data(state, root)
    data["world_gallery"] = _render_world_gallery(data, root)
    dashboard = root / "dashboard"
    dashboard.mkdir(parents=True, exist_ok=True)
    dump_json(str(dashboard / "dashboard_data.json"), data)
    rows = []
    for item in data["conditions"]:
        row = {key: item.get(key) for key in ("id", "level", "kind", "status", "terrain", "geometry_level", "geometry_label", "static_friction", "dynamic_friction", "attempt_count", "valid_attempts", "passes", "failures", "success_rate", "lower_ci", "upper_ci", "prune_reason")}
        for metric_name in ("progress_fraction", "estimated_route_distance_m", "remaining_route_distance_m", "route_length_m", "goal_checkpoint_reached", "goal_checkpoint_count"):
            aggregate = item.get("metrics", {}).get(metric_name, {})
            row[metric_name + "_median"] = aggregate.get("median")
            row[metric_name + "_minimum"] = aggregate.get("minimum")
            row[metric_name + "_maximum"] = aggregate.get("maximum")
        rows.append(row)
    write_csv(str(dashboard / "conditions.csv"), rows)
    atomic_write_text(str(dashboard / "index.html"), _dashboard_html(data, state.get("status") not in {"completed", "stopped", "blocked"}))
    return {"status": "ok", "dashboard": str(dashboard / "index.html"), "data": str(dashboard / "dashboard_data.json"), "conditions": len(data["conditions"]), "trials": len(data["trials"])}
