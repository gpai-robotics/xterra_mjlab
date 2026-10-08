# Architecture

## Separation of concerns

The system has four strict layers:

1. The core parses robot descriptions and produces canonical `RobotProfile` and `WorldSpec` JSON.
2. The coordinator resolves an `EpisodeSpec`, hashes all inputs, and launches exactly one simulator worker.
3. The worker converts the canonical scene into backend objects and writes an `EpisodeLog` plus raw telemetry.
4. Offline event, metric, statistics, aggregation, and reporting code consumes the canonical log.

Metric code never imports a simulator. A simulator crash therefore cannot corrupt an entire suite, and new metric definitions can be applied to old raw logs.

## Runtime boundary

`configs/runtimes.yaml` selects an executable per backend. This is required on the reference workstation because legacy Isaac Gym uses Python 3.8 while Isaac Lab and MuJoCo use Python 3.11. The worker protocol consists of:

- `resolved_episode.json` as input.
- `worker_status.json`, `episode_log.json`, and `telemetry.jsonl` as required output.
- `telemetry.parquet` when `pyarrow` is installed.
- Process return code and captured stdout/stderr in `manifest.json`.

The coordinator treats missing GPU/runtime support as `skipped` and simulator exceptions as `error`. Neither is a behavioral failure.

## Connected AEE graph worlds

An AE entity is a reusable directed tile containing two ordered RS challenges,
its entry/exit transitions, and optional bypass lanes. The `grid` AEE layout
places square junction pads as graph nodes and rotated AE entities as horizontal
or vertical graph edges. Every node declares `north`, `east`, `south`, and
`west` ports where present; every edge records its source/target nodes, pair,
orientation, connector lengths, and bidirectional traversal capability.

This representation supports repetitions, mixed seeded pairs, branches, loops,
left/right/front/back travel, and alternate route evaluation. The generated
`WorldSpec` includes both edge-level routes and two canonical corner-to-corner
routes, while `metadata.graph_nodes` and `metadata.graph_edges` preserve the
full navigation graph for future planners.

## Backend contract

A backend must:

- Build supported collision geometry and apply the closest available physics material.
- Spawn the robot at the canonical pose and preserve the declared joint order.
- Reject unsupported robot formats or action contracts.
- Emit ground-truth base/joint states and disclose unavailable sensors.
- Apply stressors it claims to support.
- Report wall time, engine version, fidelity tier, and unsupported capabilities.

MuJoCo accepts MJCF, Isaac Gym accepts URDF, and the Isaac Lab v1 adapter accepts USD. Conversion artifacts must retain their source hash.

Learned/Python controllers must declare `joint_order` and `action_mode`
(`position` or `effort`); TorchScript and ONNX controllers must also declare
`observation_keys`. Workers compare the declared joint order with the loaded
articulation and abort on a mismatch. Position actions use backend PD actuation,
while effort actions are sent as actuator efforts.

### v1 backend fidelity matrix

| Capability | Fake | MuJoCo | Isaac Gym P4 | Isaac Lab |
|---|---:|---:|---:|---:|
| RS/AE/AEE box collision worlds | analytical | yes | yes | yes |
| Ground-truth base/joint telemetry | synthetic | yes | yes | yes |
| Per-foot contact telemetry | synthetic | yes | yes | contact sensor |
| Push impulse | synthetic | yes | rejected | rejected |
| Action/perception latency | synthetic | yes | rejected | rejected |
| Full Sobol OOD stressor vector | synthetic | rejected if unsupported | rejected | rejected |
| Camera/depth observations | declared synthetic dropout | no | no | no |

“Rejected” is intentional: the worker emits a structured `skipped/backend_unavailable`
record instead of claiming a stressor was applied. Backend-specific implementations can
be added behind the same contract without changing suites, metrics, or reports.

## Comparability gate

An episode can be valid by itself without being comparable across engines. Cross-engine labels require:

1. `robot compare` to pass model tolerances.
2. Every engine to complete all calibration fixtures.
3. Matching controller observation/action semantics and condition IDs.

`aggregate` computes cross-backend deltas even when the gate is blocked, but marks them as invalid comparisons so discrepancies can still be diagnosed.

## Adding a backend

Implement `Backend.probe()` and `Backend.run()` under `robustness_lab.backends`, add the backend to the worker dispatch table, and add its interpreter/launcher to `runtimes.yaml`. Tests must cover unavailable-runtime behavior, scene loading, state conventions, one calibration fixture, and a deterministic action-trace replay.
