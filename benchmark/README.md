# RobustLab

Robustness benchmark for legged-robot locomotion policies. It runs a policy over
11 terrain types at fixed difficulty and friction levels in MuJoCo, then reports
which terrains it can and cannot cross, with videos and logs. There is no single score.

Tested on Python 3.10 and Ubuntu 22.04.

## Setup

Run everything from this folder (`benchmark/`):

```bash
python -m pip install -U pip                  # Ubuntu's default pip can't do editable installs
pip install -e '.[mujoco,onnx,analytics]'
export MUJOCO_GL=egl                          # headless rendering; add to ~/.bashrc
```

Only for TorchScript (`.pt`) policies: `pip install torch`.

## Run

```bash
robustlab preflight submissions/go2_blind/examiner.yaml   # checks setup, runs no episodes
robustlab launch    submissions/go2_blind/examiner.yaml   # runs the benchmark
```

Stop early and resume later:

```bash
robustlab launch submissions/go2_blind/examiner.yaml --max-trials 5
robustlab examine resume runs/submissions/go2_blind/examination_state.json
```

Results: open `runs/submissions/go2_blind/dashboard/index.html` for pass/fail per
condition, diagnostics and per-trial videos.

### Troubleshooting

- **Every trial ends in `simulator_error`:** the video renderer crashed, usually
  on a busy GPU. Render on another GPU: `export MUJOCO_EGL_DEVICE_ID=1`
  (check free GPUs with `nvidia-smi`).
- **A `.pt` policy fails to load:** install torch with `pip install torch`.

## Terrains

| Code | Terrain | Code | Terrain |
|---|---|---|---|
| RS1 | Slippery flat ground | RS7 | Curved hill |
| RS2 | Rough slippery ground | RS8 | Repeated small gaps |
| RS3 | Slippery slope | RS9 | One large gap |
| RS4 | Rough slippery slope | RS10 | High step |
| RS5 | Stairs | RS11 | Mud |
| RS6 | Repeated waves | | |

Order: flat-ground qualification, then each terrain at 3 difficulties × 3 friction
levels, then directed pairs of terrains that passed, then longer chains. Each
condition gets 3 seeds; a mixed result is retried 10 times and needs 7 successes.

## Benchmark your own policy

Copy `submissions/go2_blind/`, replace `policy.onnx`, and update `manifest.yaml`
and `skill.yaml` to match your network. `preflight` catches any mismatch before a
run starts. Full guide: [`docs/bring_your_own_policy.md`](docs/bring_your_own_policy.md).

## Docs

- Terrain parameters and why they were chosen: [`docs/terrain_difficulty_scaling.pdf`](docs/terrain_difficulty_scaling.pdf)
- Design rationale: [`docs/robustness_benchmark_design.pdf`](docs/robustness_benchmark_design.pdf)
- Schedule and pass gates: [`docs/discrete_pruned_benchmark.md`](docs/discrete_pruned_benchmark.md)
- Metric definitions: [`docs/metrics.md`](docs/metrics.md)
- Isaac Lab / Isaac Gym setup (not needed for MuJoCo): [`docs/isaac_lab_setup.md`](docs/isaac_lab_setup.md)
- Other Go2 policies (MoE-CTS, DreamWaQ, CTS, HIM): [`external_policies/README.md`](external_policies/README.md)

## License

MIT. Vendored robot models (BSD-3-Clause) and policy weights (MIT / Apache-2.0)
are listed in [`../licenses/README.md`](../licenses/README.md).
