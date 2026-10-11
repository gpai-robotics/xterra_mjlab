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

### After the run

The `go2_blind` run is already set up, so these work on it as-is. All paths are relative to `benchmark/`.

- **Dashboard (HTML):** written during the run. Open `runs/submissions/go2_blind/dashboard/index.html` for pass/fail per terrain, diagnostics and videos.
- **Export videos** into one folder, named by level, result and terrain, with an `INDEX.md`:
  ```bash
  python3 scripts/export_unique_scenario_videos.py --submission-dir runs/submissions/go2_blind --dest-dir unique_scenario_videos
  ```
  Output: `benchmark/unique_scenario_videos/`. Add `--move` to move files instead of copying.
- **One-trial HTML report** (optional):
  ```bash
  robustlab report runs/submissions/go2_blind/trials/L0-flat-a01 --output runs/submissions/go2_blind/reports/L0-flat-a01.html
  ```

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

`go2_blind` is the reference. To test your own policy, copy it and give the copy its own output folder:

```bash
cp -r submissions/go2_blind submissions/my_policy
sed -i 's#output_dir: ../../runs/submissions/go2_blind#output_dir: ../../runs/submissions/my_policy#' submissions/my_policy/examiner.yaml
```

Then:

1. Replace `submissions/my_policy/policy.onnx` with your network (same file name, or update `path:` in `submissions/my_policy/manifest.yaml`).
2. Update `submissions/my_policy/manifest.yaml` and `skill.yaml` to match your network: observation layout, action scale, PD gains and joint order.
3. Run the same commands with `my_policy` in place of `go2_blind`:

```bash
robustlab preflight submissions/my_policy/examiner.yaml
robustlab launch submissions/my_policy/examiner.yaml
```

Results go to `runs/submissions/my_policy/`, and the export and report commands above work with that path. `preflight` rejects any mismatch before a run starts. Full guide: [`docs/bring_your_own_policy.md`](docs/bring_your_own_policy.md).

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
