# CTS — Concurrent Teacher-Student — Go2

*"CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion"*
(IEEE RA-L 2024, arXiv:2405.10830). Teacher and student policies trained
concurrently under RL rather than in two stages. Blind locomotion.

**Readiness: probe the input width before wiring up.** No ONNX ships, and the two
sources disagree on the observation size (see below).

## Weights

| File | Format | Notes |
|---|---|---|
| `weights/go2_cts_max2_100k.pt` | TorchScript | RoboGauge's shipped CTS checkpoint |

No ONNX export is provided for CTS, so the input width could not be read off a
graph the way it was for MoE-CTS and DreamWaQ.

## ⚠️ Observation width is ambiguous — resolve this first

| Source | Says |
|---|---|
| `RoboGauge/robogauge/tasks/robots/go2/go2_config.py` | `num_observations = 45` (and this is the config that *loads this exact checkpoint*) |
| `go2_rl_robotlab/tasks/go2/cts_env_cfg.py` | `history_length = 5`, `flatten_history_dim = True` → 225 |

Because RoboGauge's `go2_config.py` names `go2_cts_max2_100k.pt` as its
`model_path` while declaring 45 observations, **45 is the more likely width for
this specific file** — but it has not been verified. Confirm by loading the
TorchScript module and probing with zeros of each width (torch is needed; it is
not installed in the current shell):

```python
import torch
m = torch.jit.load("weights/go2_cts_max2_100k.pt")
for n in (45, 225):
    try:
        print(n, m(torch.zeros(1, n)).shape)
    except Exception as e:
        print(n, "rejected")
```

The width that returns a `[1, 12]` tensor is the right one.

## Observation — 45 per frame

Per-frame 45 terms, in order (all clip ±100):

| # | term | dim | scale |
|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 0.25 |
| 2 | `projected_gravity` | 3 | 1.0 |
| 3 | `velocity_commands` | 3 | 1.0 (Isaac) / [2.0, 2.0, 0.25] (RoboGauge deploy) |
| 4 | `joint_pos_rel` | 12 | 1.0 |
| 5 | `joint_vel_rel` | 12 | 0.05 |
| 6 | `last_action` | 12 | 1.0 |

If the probe says 225, history is 5 and the flatten order is unrecorded — start
`oldest_first` and A/B it (see README).

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

`go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/cts_env_cfg.py`
`RoboGauge/robogauge/tasks/robots/go2/go2_config.py`
Paper: arXiv:2405.10830 — no official code release found (project page:
https://clearlab-sustech.github.io/concurrentTS/)
