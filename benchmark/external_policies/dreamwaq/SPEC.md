# DreamWaQ — Go2

*"DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain
Imagination via Deep Reinforcement Learning"* (ICRA 2023, arXiv:2301.10602).
Blind / proprioception-only, with a context-aided estimator that implicitly
infers terrain properties.

**Readiness: ready to run.** ONNX ships, and its graph confirms the dims.

## Weights

| File | Format | Notes |
|---|---|---|
| `weights/go2_dreamwaq_300k_resume_135500_policy.onnx` | ONNX | **preferred** |
| `weights/go2_dreamwaq_300k_resume_135500_policy.pt` | TorchScript | same policy |

Verified from the ONNX graph:
```
input :  obs      [1, 225]
output:  actions  [1, 12]
```

## Observation — 225 = 45 × 5 frames

⚠️ **Config/graph discrepancy — trust the graph.** `dreamwaq_env_cfg.py` sets
`PolicyCfg.history_length = 1`, which would imply a 45-wide input. The exported
ONNX is **225 wide**, i.e. 5 stacked frames. Use **225 / history 5**; the
`history_length = 1` in the task config does not describe the exported artifact.

Per-frame 45 terms, in order (all clip ±100):

| # | term | dim | scale |
|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 0.25 |
| 2 | `projected_gravity` | 3 | 1.0 |
| 3 | `velocity_commands` | 3 | 1.0 (Isaac) / [2.0, 2.0, 0.25] (RoboGauge deploy) |
| 4 | `joint_pos_rel` | 12 | 1.0 |
| 5 | `joint_vel_rel` | 12 | 0.05 |
| 6 | `last_action` | 12 | 1.0 |

Single flattened `obs` tensor. Flatten order not recorded — start `oldest_first`
and A/B it (see README).

### Note on the estimator
DreamWaQ's `VelocityCfg` observation group carries `base_lin_vel` (clip ±100,
**scale 2.0**). This is the *estimator target* used during training (the CENet
estimates base linear velocity), **not** an input to the exported policy — the
ONNX takes only the 225-wide proprioceptive history. Do **not** feed base_lin_vel
in; the policy is blind by construction and RobustLab should declare it
`sensor_track: blind`.

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

`go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/dreamwaq_env_cfg.py`
Reference implementation: https://github.com/Manaro-Alpha/DreamWaQ
