# External locomotion policies for RobustLab

Four Unitree Go2 locomotion policies gathered for benchmarking, with the exact
specs RobustLab needs to build a schema-v2 `manifest.yaml` for each.

All four come from the same research ecosystem (the RoboGauge / MoE-CTS line of
work), which is why their observation and action contracts are nearly identical.

## Status

| Policy | Weights shipped | Obs dim (verified) | History | Ready to run? |
|---|---|---|---|---|
| **MoE-CTS** (the RoboGauge paper's own policy) | ONNX + TorchScript | **450** (ONNX graph) | 10 | **Yes** |
| **DreamWaQ** | ONNX + TorchScript | **225** (ONNX graph) | 5 | **Yes** |
| **CTS** | TorchScript only | 45 per RoboGauge cfg / 225 per Isaac cfg — **conflicting, must probe** | 5 | Probe first |
| **HIM** | **None released** | 45 per frame | 1 | **No — config only** |

"Verified" means read directly off the ONNX graph, not from a config file.

## Shared contract (identical across all four)

- **Robot:** Unitree Go2
- **Joint order** (per-leg) — this *matches RobustLab's canonical order exactly*,
  so unlike `go2_blind` **no permutation is needed**:
  ```
  FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf,
  RL_hip, RL_thigh, RL_calf,  RR_hip, RR_thigh, RR_calf
  ```
- **Default joint pose** (same as the corrected `go2_blind` pose, so the robot
  profile fix already in the repo applies unchanged):
  ```
  [0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,  0.1, 1.0, -1.5,  -0.1, 1.0, -1.5]
  ```
  i.e. left hips +0.1 / right hips -0.1, front thighs 0.8, rear thighs 1.0, all calves -1.5.
- **Action:** `joint_position_offset`, scale **0.25**, offset = default pose, clip ±100
- **Control rate:** **50 Hz** (`control_dt = 0.02`; Isaac side: `sim.dt = 0.005`, `decimation = 4`)
- **PD gains:** Isaac training used **stiffness 25.0 / damping 0.5**;
  RoboGauge's MuJoCo *deployment* config uses **p = 20.0 / d = 0.5**.
  Effort limit 23.5 N·m. The 25-vs-20 difference is a real sim-to-sim choice — pick deliberately.
- **Per-frame observation (45)**, in this order for all except HIM:

  | # | term | dim | scale |
  |---|---|---|---|
  | 1 | `base_ang_vel` | 3 | **0.25** |
  | 2 | `projected_gravity` | 3 | 1.0 |
  | 3 | `velocity_commands` | 3 | 1.0 (Isaac) / **[2.0, 2.0, 0.25]** (RoboGauge deploy) |
  | 4 | `joint_pos_rel` | 12 | 1.0 |
  | 5 | `joint_vel_rel` | 12 | **0.05** |
  | 6 | `last_action` | 12 | 1.0 |

  All terms clipped to ±100. `joint_pos_rel` is measured against the default pose above.
- **Command ranges** (RoboGauge): `lin_vel_x ±1.0`, `lin_vel_y ±1.0`,
  `ang_vel_yaw ±2.0` on flat / **±1.5 on terrain**. Keep the route follower inside these.

## ⚠️ The one thing that will silently break these

The observation is a **flattened history** (one `obs` tensor, e.g. 450 = 45 × 10).
Neither the ONNX graph nor the configs record **which end is the newest frame**, nor
whether the flatten is term-major or frame-major. This is exactly the failure that
cost us time on `go2_blind`: both directions are the same width, so nothing downstream
can catch a mistake.

Isaac Lab's `CircularBuffer` is documented as *"most recent entry at the end and oldest
entry at the beginning"* → start with **oldest_first**, and verify by A/B: run flat
ground both ways and keep the one that doesn't collapse. (On `go2_blind`, the wrong
order dropped the robot from a completed run to a failed one at 0.15 m base height.)

## Layout

```
external_policies/
├── README.md          <- this file
├── cts/      SPEC.md + weights/
├── moe_cts/  SPEC.md + weights/
├── dreamwaq/ SPEC.md + weights/
└── him/      SPEC.md  (no weights)
```

Paths cited in each SPEC.md (`go2_rl_robotlab/...`, `RoboGauge/...`) are
relative to the upstream repos listed under Sources.

## Sources

- RoboGauge (eval suite): https://github.com/wty-yy/RoboGauge
- go2_rl_robotlab (training + all four task configs + checkpoints): https://github.com/APLaS-Plus/go2_rl_robotlab
- Project page: https://robogauge.github.io/complete/
- HIMLoco: https://github.com/OpenRobotLab/HIMLoco
- DreamWaQ (ref impl): https://github.com/Manaro-Alpha/DreamWaQ — paper arXiv:2301.10602
- CTS: arXiv:2405.10830 — project page https://clearlab-sustech.github.io/concurrentTS/ (no official code release found)
