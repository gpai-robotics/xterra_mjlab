# MoE-CTS (RoboGauge paper's own policy) — Go2

Mixture-of-Experts policy from *"Toward Reliable Sim-to-Real Predictability for
MoE-based Robust Quadrupedal Locomotion"* (RSS 2026). This is the policy the
RoboGauge suite was built around. Blind / proprioception-only.

**Readiness: ready to run.** ONNX ships, and its graph confirms the dims.

## Weights

| File | Format | Notes |
|---|---|---|
| `weights/go2_moe_cts_176k_0.6984.onnx` | ONNX | **preferred** — RobustLab loads ONNX natively |
| `weights/go2_moe_cts_176k_0.6984.pt` | TorchScript | same policy |
| `weights/go2_moe_cts_137k_0.6739.pt` | TorchScript | RoboGauge's shipped checkpoint |

Verified from the ONNX graph:
```
input :  obs      [1, 450]
output:  actions  [1, 12]
```

## Observation — 450 = 45 × 10 frames

Per-frame 45 terms, in order (all clip ±100):

| # | term | dim | scale |
|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 0.25 |
| 2 | `projected_gravity` | 3 | 1.0 |
| 3 | `velocity_commands` | 3 | 1.0 (Isaac) / [2.0, 2.0, 0.25] (RoboGauge deploy) |
| 4 | `joint_pos_rel` | 12 | 1.0 |
| 5 | `joint_vel_rel` | 12 | 0.05 |
| 6 | `last_action` | 12 | 1.0 |

`history_length = 10` (set in `tasks/go2/env_cfg.py`, inherited by the MoE-CTS task).
Single flattened `obs` tensor — **not** a separate obs + history pair.

> Flatten order (oldest vs newest first, term- vs frame-major) is **not recorded**
> anywhere in the bundle. Start `oldest_first` and A/B it. See README.

## Action

- space `joint_position_offset`, **scale 0.25**, offset = default pose, clip ±100
- 12 outputs, joint order = canonical per-leg (below)

## Robot constants

```
joint order   FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
default pose  [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
control       50 Hz (control_dt 0.02; sim dt 0.005, decimation 4)
PD            stiffness 25.0 / damping 0.5 (Isaac training)
              p 20.0 / d 0.5 (RoboGauge MuJoCo deployment)
effort limit  23.5 N·m
commands      lin_vel_x ±1.0, lin_vel_y ±1.0, ang_vel_yaw ±2.0 flat / ±1.5 terrain
```

## Source

`go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/env_cfg.py`
(base config; MoE-CTS uses the base `history_length = 10`)
