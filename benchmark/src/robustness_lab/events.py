"""Versioned canonical event state machine."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .models import EpisodeEvent, EpisodeLog, RobotProfile, TelemetrySample, WorldSpec


DEFAULT_THRESHOLDS = {
    "fall_height_fraction": 0.45,
    "fall_angle_deg": 60.0,
    "fall_hold_s": 0.2,
    "stuck_window_s": 2.0,
    "stuck_progress_body_lengths": 0.05,
    "commanded_motion_mps": 0.2,
    "slip_absolute_mps": 0.05,
    "slip_command_fraction": 0.1,
    "recovery_velocity_fraction": 0.1,
    "recovery_angle_deg": 5.0,
    "recovery_hold_s": 0.5,
    "goal_body_lengths": 0.25,
}


def quaternion_to_rpy(quaternion: List[float]) -> Tuple[float, float, float]:
    w, x, y, z = quaternion
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny_cosp, cosy_cosp)


def _inside(sample: TelemetrySample, minimum: List[float], maximum: List[float]) -> bool:
    return all(minimum[index] <= sample.base_position_m[index] <= maximum[index] for index in range(3))


def detect_events(log: EpisodeLog, world: WorldSpec, robot: RobotProfile, overrides: Optional[Dict[str, float]] = None) -> EpisodeLog:
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(overrides or {})
    events: List[EpisodeEvent] = []
    active_regions: Set[str] = set()
    slipping: Set[str] = set()
    fall_started: Optional[float] = None
    recovery_started: Optional[float] = None
    recovery_stable_started: Optional[float] = None
    recovery_trigger: Dict[str, object] = {}
    stuck_emitted = False
    forbidden_contact_emitted = False
    disturbance_active = False
    for index, sample in enumerate(log.samples):
        current_regions = {region.name for region in world.regions if _inside(sample, region.minimum, region.maximum)}
        for name in sorted(current_regions - active_regions):
            kind = next(region.kind for region in world.regions if region.name == name)
            events.append(EpisodeEvent(sample.time_s, "region_enter", name, details={"kind": kind}))
            if kind not in {"approach", "goal", "bypass"}:
                recovery_started = sample.time_s
                recovery_stable_started = None
                recovery_trigger = {"trigger": "transition", "region": name}
                events.append(EpisodeEvent(sample.time_s, "transition", name, details={"kind": kind}))
        for name in sorted(active_regions - current_regions):
            events.append(EpisodeEvent(sample.time_s, "region_exit", name))
        active_regions = current_regions
        command_speed = math.sqrt(sum(value * value for value in sample.command_velocity_mps[:2]))
        slip_threshold = max(thresholds["slip_absolute_mps"], thresholds["slip_command_fraction"] * command_speed)
        for foot, contact in sample.foot_contacts.items():
            velocity = sample.foot_velocities_mps.get(foot, [0.0, 0.0, 0.0])
            tangential = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            is_slipping = bool(contact and tangential > slip_threshold)
            if is_slipping and foot not in slipping:
                events.append(EpisodeEvent(sample.time_s, "slip_start", details={"foot": foot, "speed_mps": tangential, "threshold_mps": slip_threshold}))
                slipping.add(foot)
            elif not is_slipping and foot in slipping:
                events.append(EpisodeEvent(sample.time_s, "slip_end", details={"foot": foot}))
                slipping.remove(foot)
        roll, pitch, _ = quaternion_to_rpy(sample.base_quaternion_wxyz)
        angle_limit = math.radians(thresholds["fall_angle_deg"])
        falling = sample.base_position_m[2] < thresholds["fall_height_fraction"] * robot.standing_height_m or abs(roll) > angle_limit or abs(pitch) > angle_limit or bool(sample.body_contacts)
        if sample.body_contacts and not forbidden_contact_emitted:
            events.append(EpisodeEvent(sample.time_s, "body_contact", severity=1.0, details={"bodies": sample.body_contacts}))
            forbidden_contact_emitted = True
        if falling:
            fall_started = sample.time_s if fall_started is None else fall_started
            if sample.time_s - fall_started >= thresholds["fall_hold_s"] and not any(event.event_type == "fall" for event in events):
                events.append(EpisodeEvent(sample.time_s, "fall", severity=1.0, details={"roll_rad": roll, "pitch_rad": pitch, "base_height_m": sample.base_position_m[2]}))
        else:
            fall_started = None
        disturbance = bool(sample.ground_truth.get("disturbance_active", False))
        if disturbance and not disturbance_active:
            events.append(EpisodeEvent(sample.time_s, "disturbance_start", severity=float(sample.ground_truth.get("disturbance_magnitude", 0.0))))
            recovery_started = sample.time_s
            recovery_stable_started = None
            recovery_trigger = {"trigger": "disturbance", "disturbance_magnitude": float(sample.ground_truth.get("disturbance_magnitude", 0.0))}
        if not disturbance and disturbance_active:
            events.append(EpisodeEvent(sample.time_s, "disturbance_end"))
        disturbance_active = disturbance
        if recovery_started is not None:
            velocity_error = math.sqrt(sum((sample.base_linear_velocity_mps[i] - sample.command_velocity_mps[i]) ** 2 for i in range(2)))
            tolerance = max(0.05, thresholds["recovery_velocity_fraction"] * max(command_speed, 0.1))
            stable = velocity_error <= tolerance and abs(roll) <= math.radians(thresholds["recovery_angle_deg"]) and abs(pitch) <= math.radians(thresholds["recovery_angle_deg"])
            if stable:
                recovery_stable_started = sample.time_s if recovery_stable_started is None else recovery_stable_started
                if sample.time_s - recovery_stable_started >= thresholds["recovery_hold_s"]:
                    events.append(EpisodeEvent(sample.time_s, "recovered", details={"recovery_time_s": sample.time_s - recovery_started, **recovery_trigger}))
                    recovery_started = None
                    recovery_stable_started = None
                    recovery_trigger = {}
            else:
                recovery_stable_started = None
        if not stuck_emitted and command_speed > thresholds["commanded_motion_mps"]:
            window_start = sample.time_s - thresholds["stuck_window_s"]
            earlier = None
            for previous in reversed(log.samples[: index + 1]):
                if previous.time_s <= window_start:
                    earlier = previous
                    break
            if earlier is not None:
                progress = math.sqrt(sum((sample.base_position_m[i] - earlier.base_position_m[i]) ** 2 for i in range(2)))
                if progress < thresholds["stuck_progress_body_lengths"] * robot.body_length_m:
                    events.append(EpisodeEvent(sample.time_s, "stuck", severity=1.0, details={"window_s": thresholds["stuck_window_s"], "progress_m": progress}))
                    stuck_emitted = True
        goal_distance = math.sqrt(sum((sample.base_position_m[index] - world.goal_pose.position[index]) ** 2 for index in range(2)))
        if goal_distance <= thresholds["goal_body_lengths"] * robot.body_length_m and not any(event.event_type == "goal" for event in events):
            events.append(EpisodeEvent(sample.time_s, "goal", details={"distance_m": goal_distance}))
    if log.samples:
        final_time = log.samples[-1].time_s
        for foot in sorted(slipping):
            events.append(EpisodeEvent(final_time, "slip_end", details={"foot": foot, "censored": True}))
    events.sort(key=lambda event: (event.time_s, event.event_type, event.region))
    log.events = events
    event_types = {event.event_type for event in events}
    entered_regions = {event.region for event in events if event.event_type == "region_enter"}
    completed_routes = []
    if world.terrain_type == "ae":
        completed_routes = [route.name for route in world.routes if set(route.region_names).issubset(entered_regions)]
        traversal_complete = bool(completed_routes)
    elif world.terrain_type == "aee" and any(route.name == "aee-central" for route in world.routes):
        course_route = next(route for route in world.routes if route.name == "aee-central")
        traversal_complete = set(course_route.region_names).issubset(entered_regions)
        completed_routes = [course_route.name] if traversal_complete else []
    else:
        mandatory_regions = {region.name for region in world.regions if region.mandatory and region.kind != "goal"}
        traversal_complete = mandatory_regions.issubset(entered_regions)
        completed_routes = [route.name for route in world.routes if set(route.region_names).issubset(entered_regions)]
    if completed_routes and log.samples:
        events.append(EpisodeEvent(log.samples[-1].time_s, "route_complete", details={"routes": completed_routes}))
        events.sort(key=lambda event: (event.time_s, event.event_type, event.region))
    if "fall" in event_types:
        log.status, log.terminal_reason = "failed", "fall"
    elif "stuck" in event_types and "goal" not in event_types:
        log.status, log.terminal_reason = "failed", "stuck"
    elif "goal" in event_types and traversal_complete:
        log.status, log.terminal_reason = "completed", "goal"
    elif "goal" in event_types:
        log.status, log.terminal_reason = "failed", "route_incomplete"
    return log
