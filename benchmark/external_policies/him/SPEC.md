# HIM — Hybrid Internal Model — Go2

*"Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated
Robot Response"* (ICLR 2024). Uses joint encoders + IMU only; explicitly
estimates velocity and implicitly models system response as a latent embedding.

**Readiness: NOT runnable — no weights released.**

## Weights

**None.** No HIM checkpoint ships in any of the cloned repos:

- `go2_rl_robotlab/deploy/pre_train/go2/` — contains DreamWaQ, MoE-CTS and
  `unilab_robotlab_aligned` checkpoints, but **no HIM**
- `RoboGauge/resources/models/go2/` — CTS and MoE-CTS only
- `OpenRobotLab/HIMLoco` — training code only, no released checkpoints

To benchmark HIM you would have to **train it** with the shipped config
(`him_env_cfg.py` + `him_rsl_rl_cfg.py`), or obtain weights from the authors.

## ⚠️ HIM reorders the observation terms

This is the one place HIM diverges from the other three, and it will silently
produce garbage if copied from them. HIM puts **`velocity_commands` first**:

| # | term | dim | scale |
|---|---|---|---|
| 1 | **`velocity_commands`** | 3 | 1.0 |
| 2 | `base_ang_vel` | 3 | 0.25 |
| 3 | `projected_gravity` | 3 | 1.0 |
| 4 | `joint_pos_rel` | 12 | 1.0 |
| 5 | `joint_vel_rel` | 12 | 0.05 |
| 6 | `last_action` | 12 | 1.0 |

Compare to the other three, which lead with `base_ang_vel` and put
`velocity_commands` third. All terms clip ±100.

- `history_length = 1` → 45-wide observation (no stacking)
- The critic group additionally carries privileged terms; a perceptive variant
  uses a height scan with `clip=(-1, 1), scale=2.5`. The policy itself is
  proprioceptive → `sensor_track: blind`.

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

`go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/him_env_cfg.py`
Upstream: https://github.com/OpenRobotLab/HIMLoco — project page
https://junfeng-long.github.io/HIMLoco/
