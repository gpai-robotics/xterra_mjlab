"""Pure post-processing metrics over canonical EpisodeLog records."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .events import quaternion_to_rpy
from .models import EpisodeLog, MetricRecord, RobotProfile, WorldSpec
from .statistics import percentile


def _metric(name: str, value: Optional[float], unit: str, family: str, valid: bool = True, reason: str = "", provenance: Optional[Dict[str, object]] = None) -> MetricRecord:
    return MetricRecord(name, value, unit, family, valid=valid, reason=reason, provenance=provenance or {})


def _invalid(name: str, unit: str, family: str, reason: str) -> MetricRecord:
    return _metric(name, None, unit, family, False, reason)


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _point_segment_distance(point: List[float], start: List[float], end: List[float]) -> float:
    delta = [end[index] - start[index] for index in range(2)]
    length_squared = sum(value * value for value in delta)
    if length_squared <= 1e-12:
        return _norm([point[index] - start[index] for index in range(2)])
    projection = sum((point[index] - start[index]) * delta[index] for index in range(2)) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = [start[index] + projection * delta[index] for index in range(2)]
    return _norm([point[index] - closest[index] for index in range(2)])


def compute_episode_metrics(log: EpisodeLog, world: WorldSpec, robot: RobotProfile) -> List[MetricRecord]:
    samples = log.samples
    if not samples:
        return [_invalid("episode_has_samples", "bool", "outcome", "episode log is empty")]
    metrics = []
    duration = max(samples[-1].time_s - samples[0].time_s, 0.0)
    force_metadata = {"sample_rate_hz": (len(samples) - 1) / duration if duration > 0 and len(samples) > 1 else None, "filter": "none"}
    start, finish = samples[0].base_position_m, samples[-1].base_position_m
    displacement = _norm([finish[index] - start[index] for index in range(2)])
    path_distance = sum(_norm([right.base_position_m[index] - left.base_position_m[index] for index in range(2)]) for left, right in zip(samples, samples[1:]))
    goal_distance = _norm([finish[index] - world.goal_pose.position[index] for index in range(2)])
    contact_transitions = sum(
        left.foot_contacts.get(foot, False) != right.foot_contacts.get(foot, False)
        for left, right in zip(samples, samples[1:])
        for foot in set(left.foot_contacts) | set(right.foot_contacts)
    )
    direction = 1.0 if world.goal_pose.position[0] >= world.spawn_pose.position[0] else -1.0
    expected_progress = max(abs(world.goal_pose.position[0] - world.spawn_pose.position[0]), 1e-9)
    progress_fraction = max(0.0, min(1.0, direction * (finish[0] - world.spawn_pose.position[0]) / expected_progress))
    events = [event.event_type for event in log.events]
    metrics.extend([
        _metric("success", 1.0 if log.status == "completed" and log.terminal_reason == "goal" else 0.0, "bool", "outcome"),
        _metric("completion_time", duration, "s", "outcome"),
        _metric("displacement", displacement, "m", "outcome"),
        _metric("path_distance", path_distance, "m", "outcome"),
        _metric("goal_distance", goal_distance, "m", "outcome"),
        _metric("progress_fraction", progress_fraction, "ratio", "outcome"),
        _metric("path_efficiency", displacement / path_distance if path_distance else None, "ratio", "outcome", path_distance > 0, "no traveled path" if not path_distance else ""),
        _metric("fall", 1.0 if "fall" in events else 0.0, "bool", "outcome"),
        _metric("stuck", 1.0 if "stuck" in events else 0.0, "bool", "outcome"),
        _metric("retry_count", float(events.count("retry")), "count", "outcome"),
        _metric("reset_count", float(events.count("reset")), "count", "outcome"),
        _metric("intervention_count", float(events.count("intervention")), "count", "outcome"),
        _metric("skip_count", float(events.count("skip")), "count", "outcome"),
        _metric("body_contact_count", float(events.count("body_contact")), "count", "contact"),
        _metric("contact_transition_count", float(contact_transitions), "count", "contact"),
    ])
    entered_regions = {event.region for event in log.events if event.event_type == "region_enter"}
    mandatory_regions = [region for region in world.regions if region.mandatory and region.kind != "goal"]
    completed_regions = sum(region.name in entered_regions for region in mandatory_regions)
    metrics.append(_metric("mandatory_region_completion", completed_regions / len(mandatory_regions) if mandatory_regions else 1.0, "ratio", "outcome"))
    for region in mandatory_regions:
        metrics.append(_metric("obstacle_success.%s" % region.name, 1.0 if region.name in entered_regions else 0.0, "bool", "outcome"))
    velocity_errors = [_norm([sample.base_linear_velocity_mps[index] - sample.command_velocity_mps[index] for index in range(2)]) for sample in samples]
    command_speeds = [_norm(sample.command_velocity_mps[:2]) for sample in samples]
    measured_speeds = [_norm(sample.base_linear_velocity_mps[:2]) for sample in samples]
    signed_speed_errors = [measured - commanded for measured, commanded in zip(measured_speeds, command_speeds)]
    angular_velocity_errors = [sample.base_angular_velocity_rps[2] - sample.command_velocity_mps[2] for sample in samples]
    steady_count = max(1, len(samples) // 5)
    heading_errors = []
    roll_values, pitch_values, angular_rates, heights = [], [], [], []
    for sample in samples:
        roll, pitch, yaw = quaternion_to_rpy(sample.base_quaternion_wxyz)
        command_heading = math.atan2(sample.command_velocity_mps[1], sample.command_velocity_mps[0]) if _norm(sample.command_velocity_mps[:2]) > 1e-9 else yaw
        heading_errors.append(math.atan2(math.sin(yaw - command_heading), math.cos(yaw - command_heading)))
        roll_values.append(roll)
        pitch_values.append(pitch)
        angular_rates.append(_norm(sample.base_angular_velocity_rps))
        heights.append(sample.base_position_m[2])
    metrics.extend([
        _metric("velocity_tracking_rmse", math.sqrt(sum(value * value for value in velocity_errors) / len(velocity_errors)), "m/s", "tracking"),
        _metric("velocity_tracking_p95", percentile(velocity_errors, 0.95), "m/s", "tracking"),
        _metric("steady_state_speed_error", sum(signed_speed_errors[-steady_count:]) / steady_count, "m/s", "tracking"),
        _metric("speed_overshoot", max([0.0] + signed_speed_errors), "m/s", "tracking"),
        _metric("angular_velocity_tracking_rmse", math.sqrt(sum(value * value for value in angular_velocity_errors) / len(angular_velocity_errors)), "rad/s", "tracking"),
        _metric("heading_rmse", math.sqrt(sum(value * value for value in heading_errors) / len(heading_errors)), "rad", "tracking"),
        _metric("roll_rms", math.sqrt(sum(value * value for value in roll_values) / len(roll_values)), "rad", "stability"),
        _metric("pitch_rms", math.sqrt(sum(value * value for value in pitch_values) / len(pitch_values)), "rad", "stability"),
        _metric("orientation_max", max(max(abs(value) for value in roll_values), max(abs(value) for value in pitch_values)), "rad", "stability"),
        _metric("angular_rate_p95", percentile(angular_rates, 0.95), "rad/s", "stability"),
        _metric("body_height_range", max(heights) - min(heights), "m", "stability"),
    ])
    central_route = next((route for route in world.routes if route.route_kind == "central"), world.routes[0] if world.routes else None)
    if central_route and len(central_route.waypoints) >= 2:
        cross_track = [min(_point_segment_distance(sample.base_position_m, left, right) for left, right in zip(central_route.waypoints, central_route.waypoints[1:])) for sample in samples]
        metrics.extend([_metric("cross_track_rmse", math.sqrt(sum(value * value for value in cross_track) / len(cross_track)), "m", "tracking"), _metric("cross_track_max", max(cross_track), "m", "tracking")])
    else:
        metrics.append(_invalid("cross_track_rmse", "m", "tracking", "world has no route polyline"))
    waypoint_threshold = 0.25 * robot.body_length_m
    reached_waypoints = sum(
        any(_norm([sample.base_position_m[index] - waypoint[index] for index in range(2)]) <= waypoint_threshold for sample in samples)
        for waypoint in world.checkpoints
    )
    metrics.append(_metric("waypoint_completion", reached_waypoints / len(world.checkpoints) if world.checkpoints else None, "ratio", "tracking", bool(world.checkpoints), "world has no checkpoints" if not world.checkpoints else ""))
    slip_time = 0.0
    stance_time = 0.0
    slip_distance = 0.0
    total_impulse = 0.0
    peak_force = 0.0
    absolute_work = 0.0
    positive_work = 0.0
    thermal_proxy = 0.0
    saturation_time = 0.0
    total_actuator_time = 0.0
    torque_samples = []
    joint_speed_samples = []
    power_samples = []
    force_samples = []
    loading_rates = []
    duty_counts: Dict[str, List[int]] = {}
    joint_limit_margins = []
    effort_utilization = []
    support_margins = []
    effort_limits = {actuator.joint: actuator.effort_limit for actuator in robot.actuators if actuator.effort_limit}
    velocity_limits = {actuator.joint: actuator.velocity_limit for actuator in robot.actuators if actuator.velocity_limit}
    joint_limits = {joint.name: (joint.lower, joint.upper) for joint in robot.joints if joint.lower is not None and joint.upper is not None}
    previous_force: Dict[str, float] = {}
    for left, right in zip(samples, samples[1:]):
        dt = max(right.time_s - left.time_s, 0.0)
        command_speed = _norm(left.command_velocity_mps[:2])
        slip_threshold = max(0.05, 0.1 * command_speed)
        for foot, contact in left.foot_contacts.items():
            duty_counts.setdefault(foot, [0, 0])
            duty_counts[foot][1] += 1
            if not contact:
                continue
            duty_counts[foot][0] += 1
            stance_time += dt
            velocity = left.foot_velocities_mps.get(foot, [0.0, 0.0, 0.0])
            tangential = _norm(velocity[:2])
            if tangential > slip_threshold:
                slip_time += dt
                slip_distance += tangential * dt
            force = _norm(left.foot_forces_n.get(foot, [0.0, 0.0, 0.0]))
            total_impulse += force * dt
            peak_force = max(peak_force, force)
            force_samples.append(force)
            if foot in previous_force and dt > 0:
                loading_rates.append(abs(force - previous_force[foot]) / dt)
            previous_force[foot] = force
        for joint, torque in left.joint_torque_nm.items():
            velocity = left.joint_velocity_rps.get(joint, 0.0)
            power = torque * velocity
            torque_samples.append(abs(torque))
            power_samples.append(abs(power))
            if joint in velocity_limits:
                joint_speed_samples.append(abs(velocity) / velocity_limits[joint])
            absolute_work += abs(power) * dt
            positive_work += max(power, 0.0) * dt
            thermal_proxy += torque * torque * dt
            if joint in effort_limits:
                effort_utilization.append(abs(torque) / effort_limits[joint])
                total_actuator_time += dt
                if abs(torque) >= 0.98 * effort_limits[joint]:
                    saturation_time += dt
        for joint, position in left.joint_position_rad.items():
            if joint in joint_limits:
                lower, upper = joint_limits[joint]
                span = upper - lower
                if span > 0:
                    joint_limit_margins.append(min(position - lower, upper - position) / span)
        support_points = [left.foot_positions_m[name] for name, contact in left.foot_contacts.items() if contact and name in left.foot_positions_m]
        if len(support_points) >= 3:
            minimum_x, maximum_x = min(point[0] for point in support_points), max(point[0] for point in support_points)
            minimum_y, maximum_y = min(point[1] for point in support_points), max(point[1] for point in support_points)
            base_x, base_y = left.base_position_m[:2]
            support_margins.append(min(base_x - minimum_x, maximum_x - base_x, base_y - minimum_y, maximum_y - base_y))
    metrics.extend([
        _metric("slip_ratio", slip_time / stance_time if stance_time else 0.0, "ratio", "contact", stance_time > 0, "no stance contacts" if not stance_time else ""),
        _metric("slip_distance", slip_distance, "m", "contact", stance_time > 0, "no stance contacts" if not stance_time else ""),
        _metric("contact_impulse", total_impulse, "N*s", "contact", stance_time > 0, "no contact forces" if not stance_time else "", force_metadata),
        _metric("peak_contact_force", peak_force, "N", "contact", stance_time > 0, "no contact forces" if not stance_time else "", force_metadata),
        _metric("contact_force_p95", percentile(force_samples, 0.95) if force_samples else None, "N", "contact", bool(force_samples), "no contact forces" if not force_samples else "", force_metadata),
        _metric("loading_rate_p95", percentile(loading_rates, 0.95) if loading_rates else None, "N/s", "contact", bool(loading_rates), "insufficient contact-force samples" if not loading_rates else "", force_metadata),
        _metric("absolute_mechanical_work", absolute_work, "J", "efficiency", bool(absolute_work), "joint torque/velocity unavailable" if not absolute_work else ""),
        _metric("positive_mechanical_work", positive_work, "J", "efficiency", bool(absolute_work), "joint torque/velocity unavailable" if not absolute_work else ""),
        _metric("thermal_proxy", thermal_proxy, "N^2*m^2*s", "efficiency", bool(thermal_proxy), "joint torque unavailable" if not thermal_proxy else ""),
        _metric("torque_saturation_ratio", saturation_time / total_actuator_time if total_actuator_time else None, "ratio", "efficiency", total_actuator_time > 0, "actuator effort limits unavailable" if not total_actuator_time else ""),
        _metric("torque_peak", max(torque_samples) if torque_samples else None, "N*m", "efficiency", bool(torque_samples), "joint torque unavailable" if not torque_samples else ""),
        _metric("torque_rms", math.sqrt(sum(value * value for value in torque_samples) / len(torque_samples)) if torque_samples else None, "N*m", "efficiency", bool(torque_samples), "joint torque unavailable" if not torque_samples else ""),
        _metric("power_p95", percentile(power_samples, 0.95) if power_samples else None, "W", "efficiency", bool(power_samples), "joint power unavailable" if not power_samples else ""),
        _metric("joint_velocity_utilization_p95", percentile(joint_speed_samples, 0.95) if joint_speed_samples else None, "ratio", "efficiency", bool(joint_speed_samples), "velocity limits unavailable" if not joint_speed_samples else ""),
        _metric("joint_effort_utilization_p95", percentile(effort_utilization, 0.95) if effort_utilization else None, "ratio", "efficiency", bool(effort_utilization), "effort limits unavailable" if not effort_utilization else ""),
        _metric("minimum_joint_limit_margin", min(joint_limit_margins) if joint_limit_margins else None, "ratio", "stability", bool(joint_limit_margins), "joint limits or positions unavailable" if not joint_limit_margins else ""),
        _metric("support_bbox_margin_min", min(support_margins) if support_margins else None, "m", "stability", bool(support_margins), "requires at least three positioned stance feet" if not support_margins else ""),
    ])
    for foot, counts in sorted(duty_counts.items()):
        metrics.append(_metric("duty_factor.%s" % foot, counts[0] / counts[1] if counts[1] else None, "ratio", "contact", counts[1] > 0, "no foot samples" if not counts[1] else ""))
    denominator = robot.total_mass_kg * 9.80665 * displacement
    metrics.append(_metric("absolute_mechanical_cot", absolute_work / denominator if denominator > 0 and absolute_work else None, "ratio", "efficiency", denominator > 0 and absolute_work > 0, "requires mass, displacement, torque, and velocity" if denominator <= 0 or not absolute_work else ""))
    metrics.append(_metric("positive_work_cot", positive_work / denominator if denominator > 0 and absolute_work else None, "ratio", "efficiency", denominator > 0 and absolute_work > 0, "requires mass, displacement, torque, and velocity" if denominator <= 0 or not absolute_work else ""))
    latencies = [sample.inference_latency_s for sample in samples]
    step_times = [sample.simulator_step_s for sample in samples if sample.simulator_step_s > 0]
    deadline = samples[1].time_s - samples[0].time_s if len(samples) > 1 else 0.02
    metrics.extend([
        _metric("inference_latency_p50", percentile(latencies, 0.5), "s", "systems"),
        _metric("inference_latency_p95", percentile(latencies, 0.95), "s", "systems"),
        _metric("inference_latency_p99", percentile(latencies, 0.99), "s", "systems"),
        _metric("deadline_miss_ratio", sum(value > deadline for value in latencies) / len(latencies), "ratio", "systems"),
        _metric("simulator_step_p95", percentile(step_times, 0.95) if step_times else None, "s", "systems", bool(step_times), "simulator step timings unavailable" if not step_times else ""),
    ])
    wall_time = float(log.diagnostics.get("wall_time_s", 0.0))
    metrics.append(_metric("real_time_factor", duration / wall_time if wall_time > 0 else None, "ratio", "systems", wall_time > 0, "wall time unavailable" if not wall_time else ""))
    metrics.extend(
        [
            _metric("process_user_cpu_time", float(log.diagnostics["process_user_cpu_s"]), "s", "systems") if "process_user_cpu_s" in log.diagnostics else _invalid("process_user_cpu_time", "s", "systems", "worker resource usage unavailable"),
            _metric("process_system_cpu_time", float(log.diagnostics["process_system_cpu_s"]), "s", "systems") if "process_system_cpu_s" in log.diagnostics else _invalid("process_system_cpu_time", "s", "systems", "worker resource usage unavailable"),
            _metric("process_peak_rss", float(log.diagnostics["process_peak_rss_mib"]), "MiB", "systems") if "process_peak_rss_mib" in log.diagnostics else _invalid("process_peak_rss", "MiB", "systems", "worker resource usage unavailable"),
            _metric("process_cpu_utilization", 100.0 * (float(log.diagnostics.get("process_user_cpu_s", 0.0)) + float(log.diagnostics.get("process_system_cpu_s", 0.0))) / float(log.diagnostics.get("worker_wall_time_s", wall_time)), "% of one core", "systems") if float(log.diagnostics.get("worker_wall_time_s", wall_time)) > 0 and "process_user_cpu_s" in log.diagnostics else _invalid("process_cpu_utilization", "% of one core", "systems", "worker CPU or wall time unavailable"),
            _metric("gpu_memory_peak", float(log.diagnostics["gpu_memory_peak_mib"]), "MiB", "systems") if "gpu_memory_peak_mib" in log.diagnostics else _invalid("gpu_memory_peak", "MiB", "systems", "backend did not expose GPU memory telemetry"),
            _metric("gpu_utilization_mean", float(log.diagnostics["gpu_utilization_mean_percent"]), "%", "systems") if "gpu_utilization_mean_percent" in log.diagnostics else _invalid("gpu_utilization_mean", "%", "systems", "backend did not expose GPU utilization telemetry"),
        ]
    )
    recoveries = [event.details.get("recovery_time_s") for event in log.events if event.event_type == "recovered" and event.details.get("recovery_time_s") is not None]
    disturbances = [event for event in log.events if event.event_type == "disturbance_start"]
    disturbance_recoveries = [event for event in log.events if event.event_type == "recovered" and event.details.get("trigger") == "disturbance"]
    retention_windows = []
    current_retention = []
    for sample in samples:
        if sample.ground_truth.get("disturbance_active", False):
            commanded = _norm(sample.command_velocity_mps[:2])
            if commanded > 1e-9:
                current_retention.append(_norm(sample.base_linear_velocity_mps[:2]) / commanded)
        elif current_retention:
            retention_windows.append(min(current_retention))
            current_retention = []
    if current_retention:
        retention_windows.append(min(current_retention))
    metrics.append(_metric("recovery_count", float(len(recoveries)), "count", "recovery"))
    metrics.append(_metric("recovery_time_mean", sum(recoveries) / len(recoveries) if recoveries else None, "s", "recovery", bool(recoveries), "no completed recovery event" if not recoveries else ""))
    metrics.append(_metric("disturbance_recovery_probability", min(len(disturbance_recoveries), len(disturbances)) / len(disturbances) if disturbances else None, "ratio", "recovery", bool(disturbances), "no disturbance event" if not disturbances else ""))
    metrics.append(_metric("maximum_recovered_disturbance", max((float(event.details.get("disturbance_magnitude", 0.0)) for event in disturbance_recoveries), default=None), "N*s", "recovery", bool(disturbance_recoveries), "no recovered disturbance" if not disturbance_recoveries else ""))
    metrics.append(_metric("minimum_disturbance_speed_retention", min(retention_windows) if retention_windows else None, "ratio", "recovery", bool(retention_windows), "no commanded-speed disturbance window" if not retention_windows else ""))
    bypass_regions = [region for region in world.regions if region.kind == "bypass"]
    used_bypass = any(any(region.minimum[1] <= sample.base_position_m[1] <= region.maximum[1] and region.minimum[0] <= sample.base_position_m[0] <= region.maximum[0] for region in bypass_regions) for sample in samples)
    metrics.append(_metric("bypass_selected", 1.0 if used_bypass else 0.0, "bool", "affordance", bool(bypass_regions), "world has no bypass route" if not bypass_regions else ""))
    first_bypass = next((sample for sample in samples if any(region.minimum[1] <= sample.base_position_m[1] <= region.maximum[1] and region.minimum[0] <= sample.base_position_m[0] <= region.maximum[0] for region in bypass_regions)), None)
    challenge_starts = [region.minimum[0] for region in world.regions if region.kind == "challenge_slot"]
    metrics.append(_metric("route_decision_time", first_bypass.time_s - samples[0].time_s if first_bypass else None, "s", "affordance", bool(first_bypass), "no bypass decision observed" if not first_bypass else ""))
    metrics.append(_metric("route_decision_distance", min(challenge_starts) - first_bypass.base_position_m[0] if first_bypass and challenge_starts else None, "m", "affordance", bool(first_bypass and challenge_starts), "requires an AE bypass decision" if not first_bypass or not challenge_starts else ""))
    observation_ages = [sample.observation_age_s for sample in samples]
    metrics.append(_metric("observation_age_p95", percentile(observation_ages, 0.95), "s", "perception"))
    terrain_errors = [abs(float(sample.observation["terrain_height_m"]) - float(sample.ground_truth["terrain_height_m"])) for sample in samples if "terrain_height_m" in sample.observation and "terrain_height_m" in sample.ground_truth]
    metrics.append(_metric("terrain_estimate_rmse", math.sqrt(sum(value * value for value in terrain_errors) / len(terrain_errors)) if terrain_errors else None, "m", "perception", bool(terrain_errors), "controller did not expose terrain estimates" if not terrain_errors else ""))
    dropped = [bool(sample.observation.get("dropped", False)) for sample in samples if "dropped" in sample.observation]
    metrics.append(_metric("observation_dropout_ratio", sum(dropped) / len(dropped) if dropped else None, "ratio", "perception", bool(dropped), "dropout flags unavailable" if not dropped else ""))
    affordance = [(float(sample.observation["affordance_probability"]), 1.0 if sample.ground_truth["affordance_safe"] else 0.0) for sample in samples if "affordance_probability" in sample.observation and "affordance_safe" in sample.ground_truth]
    if affordance:
        brier = sum((probability - target) ** 2 for probability, target in affordance) / len(affordance)
        bins = [[] for _ in range(10)]
        for probability, target in affordance:
            bins[min(9, int(probability * 10))].append((probability, target))
        ece = sum(len(group) / len(affordance) * abs(sum(item[0] for item in group) / len(group) - sum(item[1] for item in group) / len(group)) for group in bins if group)
        unsafe = [item for item in affordance if item[1] == 0.0]
        safe = [item for item in affordance if item[1] == 1.0]
        metrics.extend([_metric("affordance_brier_score", brier, "score", "affordance"), _metric("affordance_expected_calibration_error", ece, "score", "affordance"), _metric("false_safe_rate", sum(item[0] >= 0.5 for item in unsafe) / len(unsafe) if unsafe else None, "ratio", "affordance", bool(unsafe), "no unsafe labels exposed" if not unsafe else ""), _metric("false_conservative_rate", sum(item[0] < 0.5 for item in safe) / len(safe) if safe else None, "ratio", "affordance", bool(safe), "no safe labels exposed" if not safe else "")])
    else:
        metrics.extend([_invalid("affordance_brier_score", "score", "affordance", "controller did not expose calibrated probabilities"), _invalid("affordance_expected_calibration_error", "score", "affordance", "controller did not expose calibrated probabilities"), _invalid("false_safe_rate", "ratio", "affordance", "controller did not expose affordance labels"), _invalid("false_conservative_rate", "ratio", "affordance", "controller did not expose affordance labels")])
    metrics.extend([_invalid("route_regret", "s", "affordance", "requires a measured capability profile and counterfactual route costs"), _invalid("electrical_cot", "ratio", "efficiency", "requires voltage/current or electrical power telemetry"), _invalid("capture_margin", "m", "stability", "requires a gait-model-specific capture-point estimator")])
    return metrics


def metric_map(metrics: List[MetricRecord]) -> Dict[str, Optional[float]]:
    return {metric.name: metric.value for metric in metrics if metric.valid}
