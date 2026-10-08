# Tiered benchmark workflow

The schema-v2 benchmark layer builds on the existing simulator-neutral `WorldSpec`. It keeps legacy `robustlab suite` inputs working while adding content-addressed L0–L7 suites, graph courses, and deduplicated PNG catalogues.

## Profiles

- `smoke`: L0 wiring, RS1–RS11 once, one AE pair, and one 2×2 grid.
- `quick`: three RS difficulties, ten balanced AE pairs, and representative chain/grid/ring courses with ten repetitions.
- `standard`: L0–L7, both traversal directions, route-choice/endurance worlds, one-factor and Sobol stressors, locked OOD cases, and 30 repetitions.
- `exhaustive`: all 100 ordered AE pairs in two directions at five difficulties. It is only expanded when explicitly requested.

Horizon labels are morphology-normalized: XS/S/M/L/XL/Endurance request 5/10/25/50/100/200 robot body lengths. The manifest always records the requested and generated metric/body-length route lengths.

## Materialize and render

From `benchmarking_automation/`:

```bash
PYTHONPATH=src python -m robustness_lab.cli benchmark materialize \
  configs/benchmarks/tiered_go2.yaml --profile smoke \
  --robot assets/cache/go2_mjcf_profile.json --backend mujoco \
  --robot-asset assets/cache/mujoco_menagerie/unitree_go2/go2.xml \
  --robot-format mjcf --controller configs/controllers/pd_hold.yaml \
  --output runs/benchmark-smoke

PYTHONPATH=src python -m robustness_lab.cli benchmark render \
  runs/benchmark-smoke/benchmark_manifest.json \
  --robot-asset assets/cache/mujoco_menagerie/unitree_go2/go2.xml
```

Open `runs/benchmark-smoke/previews/index.html` or `contact-sheet.png`. Every preview directory contains perspective/top-down PNGs, World JSON, collision OBJ, and a loadable MuJoCo scene. Courses over 50 body lengths also contain 25-body-length section PNGs.

## Execute, recover, summarize

```bash
PYTHONPATH=src python -m robustness_lab.cli benchmark run runs/benchmark-smoke/benchmark_manifest.json --limit 2
PYTHONPATH=src python -m robustness_lab.cli benchmark resume runs/benchmark-smoke/benchmark_manifest.json
PYTHONPATH=src python -m robustness_lab.cli benchmark summarize runs/benchmark-smoke/benchmark_manifest.json --output reports/benchmark-smoke
```

The supplied PD hold controller is useful only for compilation/orchestration checks. Reports explicitly block locomotion capability claims until a valid walking-controller contract and required simulator gates are present.

## Universal camber and curvature

Every isolated RS or AE world can be physically banked sideways and swept along a left/right arc before export. Positive curvature turns left/anticlockwise; negative curvature turns right/clockwise. Positive camber raises the left side and lowers the right; negative camber does the opposite.

```bash
robustlab world generate --terrain rs5 \
  --robot assets/cache/go2_mjcf_profile.json \
  --difficulty 0.6 --camber-deg -12 --curve-angle-deg 65 \
  --curve-segments 28 --output runs/curved-banked-stairs.json
```

This is not a visual shader. Long collision boxes are subdivided along the arc, each collision segment is oriented to the local route tangent and camber, and routes, regions, checkpoints, spawn and goal poses are transformed with them. Friction/material parameters remain independent. A profile may add the same transformations through its `geometry_modifiers` cases; see `configs/benchmarks/camber_curvature_go2.yaml`.

For multiple engines, repeat `--backend` and provide explicit assets such as `--asset mujoco=mjcf=/path/go2.xml --asset isaaclab=usd=/path/go2.usd`. Each backend remains isolated in its configured runtime.
