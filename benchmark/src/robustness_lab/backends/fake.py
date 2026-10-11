"""Deterministic analytical backend for orchestration and metric tests."""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List

from .base import Backend, BackendCapabilities, BackendUnavailable
from ..models import EpisodeLog, EpisodeSpec, RobotProfile, TelemetrySample, WorldSpec


class FakeBackend(Backend):
    name = "fake"
    _OBSERVATIONS = frozenset(
        {
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "base_position_w",
            "base_orientation_wxyz",
            "base_euler_rpy",
            "projected_gravity_b",
            "base_linear_velocity_b",
            "base_angular_velocity_b",
            "foot_contact",
            "foot_contact_force",
            "foot_velocity_b",
            "episode_time",
            "velocity_command",
        }
    )

    @classmethod
    def probe(cls) -> BackendCapabilities:
        return BackendCapabilities(
            cls.name,
            True,
            "analytical-v1",
            contact_forces=True,
            environment_effects=[],
            observations=cls._OBSERVATIONS,
        )

    def run(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        if episode.record_video:
            raise BackendUnavailable("analytical fake backend cannot record video")
        rng = random.Random(episode.seed)
        control_dt = episode.dt_s * episode.control_decimation
        target_speed = float(episode.controller.get("command_velocity_mps", 0.6))
        difficulty = float(world.metadata.get("difficulty", world.normalized_parameters.get("difficulty", 0.5)))
        samples: List[TelemetrySample] = []
        start_wall = time.perf_counter()
        x, y = world.spawn_pose.position[0], world.spawn_pose.position[1]
        last_x = x
        direction = 1.0 if world.goal_pose.position[0] >= x else -1.0
        command = [direction * target_speed, 0.0, 0.0]
        supported_stressors = {"friction_scale", "payload_fraction", "com_offset_m", "action_latency_s", "perception_latency_s", "depth_dropout", "action_dropout", "push_impulse_ns"}
        unsupported = sorted({str(item.get("type", "")) for item in episode.stressors if item.get("type") not in supported_stressors and float(item.get("value", 0.0) or 0.0) != 0.0})
        if unsupported:
            raise BackendUnavailable("analytical fake backend does not model stressors %s" % ", ".join(unsupported))
        push_config = next((item for item in episode.stressors if item.get("type") == "push_impulse_ns"), {})
        if push_config.get("gait_phase") is not None:
            raise BackendUnavailable("analytical fake backend does not model gait-phase-triggered pushes")
        stressor_values = {item.get("type"): item.get("value") for item in episode.stressors if "value" in item}
        perception_latency = float(stressor_values.get("perception_latency_s", 0.0) or 0.0)
        action_latency = float(stressor_values.get("action_latency_s", 0.0) or 0.0)
        push_impulse = float(stressor_values.get("push_impulse_ns", 0.0) or 0.0)
        push_onset = float(push_config.get("time_s", 2.0))
        push_duration = float(push_config.get("duration_s", 0.1))
        friction_scale = max(float(stressor_values.get("friction_scale", 1.0) or 1.0), 0.05)
        payload_fraction = max(float(stressor_values.get("payload_fraction", 0.0) or 0.0), 0.0)
        com_offset = float(stressor_values.get("com_offset_m", 0.0) or 0.0)
        depth_dropout = min(max(float(stressor_values.get("depth_dropout", 0.0) or 0.0), 0.0), 1.0)
        action_dropout = min(max(float(stressor_values.get("action_dropout", 0.0) or 0.0), 0.0), 1.0)
        step = 0
        status, reason = "completed", "goal"
        while step * control_dt <= episode.timeout_s:
            t = step * control_dt
            in_challenge = any(region.minimum[0] <= x <= region.maximum[0] and region.kind not in {"approach", "goal", "bypass"} for region in world.regions)
            effective_difficulty = min(1.0, difficulty / friction_scale)
            drag = (0.05 + 0.35 * effective_difficulty) if in_challenge else 0.0
            slip = in_challenge and (math.sin(t * 9.0 + episode.seed) > 0.45 + 0.3 * (1.0 - effective_difficulty))
            dropped_action = rng.random() < action_dropout
            dropped_observation = rng.random() < depth_dropout
            speed = max(0.0, target_speed * (1.0 - drag) / (1.0 + payload_fraction) + rng.gauss(0.0, 0.01))
            if dropped_action:
                speed *= 0.5
            disturbance_active = bool(push_impulse and push_onset <= t < push_onset + push_duration)
            if disturbance_active:
                speed *= max(0.1, 1.0 - push_impulse / 25.0)
            if slip:
                speed *= 0.75
            x += direction * speed * control_dt
            y = 0.03 * math.sin(t * 1.7) + (0.03 * push_impulse / 20.0 if disturbance_active else 0.0)
            roll = 0.035 * difficulty * math.sin(t * 4.0) + com_offset / max(robot.body_width_m, 1e-6) * 0.05 + (0.15 * push_impulse / 20.0 if disturbance_active else 0.0)
            pitch = 0.05 * difficulty * math.sin(t * 3.1)
            cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
            cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
            quaternion = [cr * cp, sr * cp, cr * sp, -sr * sp]
            foot_contacts: Dict[str, bool] = {}
            foot_velocities: Dict[str, List[float]] = {}
            foot_forces: Dict[str, List[float]] = {}
            feet = robot.foot_links or ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            for index, foot in enumerate(feet):
                contact = math.sin(t * 8.0 + index * math.pi / 2.0) >= -0.2
                foot_contacts[foot] = contact
                foot_velocities[foot] = [0.12 if slip and contact else 0.01, 0.0, 0.0]
                foot_forces[foot] = [0.0, 0.0, robot.total_mass_kg * 9.81 / max(len(feet), 1) * (1.0 + 0.15 * math.sin(t * 8.0 + index))] if contact else [0.0, 0.0, 0.0]
            joint_positions = {name: value + 0.1 * math.sin(t * 8.0) for name, value in robot.default_joint_positions.items()}
            joint_velocities = {name: 0.8 * math.cos(t * 8.0) for name in joint_positions}
            joint_torques = {name: 2.0 + difficulty * math.sin(t * 8.0) for name in joint_positions}
            samples.append(
                TelemetrySample(
                    time_s=t,
                    base_position_m=[x, y, robot.standing_height_m + 0.01 * math.sin(t * 8.0)],
                    base_quaternion_wxyz=quaternion,
                    base_linear_velocity_mps=[(x - last_x) / control_dt if step else 0.0, 0.0, 0.0],
                    base_angular_velocity_rps=[roll * 4.0, pitch * 3.1, 0.0],
                    command_velocity_mps=command,
                    joint_position_rad=joint_positions,
                    joint_velocity_rps=joint_velocities,
                    joint_torque_nm=joint_torques,
                    action=list(joint_positions.values()),
                    foot_contacts=foot_contacts,
                    foot_velocities_mps=foot_velocities,
                    foot_forces_n=foot_forces,
                    observation_age_s=perception_latency,
                    inference_latency_s=0.0008 + rng.random() * 0.0004 + action_latency,
                    simulator_step_s=0.0002 + rng.random() * 0.0001,
                    ground_truth={"in_challenge": in_challenge, "disturbance_active": disturbance_active, "disturbance_magnitude": push_impulse},
                    observation={"dropped": dropped_observation, "action_dropped": dropped_action},
                )
            )
            last_x = x
            if abs(x - world.goal_pose.position[0]) <= 0.25 * robot.body_length_m or direction * (x - world.goal_pose.position[0]) > 0:
                break
            step += 1
        else:
            status, reason = "failed", "timeout"
        if not samples or direction * (samples[-1].base_position_m[0] - world.goal_pose.position[0]) < -0.25 * robot.body_length_m:
            status, reason = "failed", "timeout"
        return EpisodeLog(episode.episode_id, self.name, status, reason, samples, diagnostics={"wall_time_s": time.perf_counter() - start_wall, "seed": episode.seed, "analytical_backend": True, "terrain_fidelity_tier": "analytical_proxy"})
