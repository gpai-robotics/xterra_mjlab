# Standalone operation

RobustLab runs from one examination YAML. The YAML points to four external
inputs and keeps everything else inside the run directory:

1. a `RobotProfile` generated from URDF, MJCF, or USD;
2. the simulator-specific robot asset;
3. a skill manifest describing the supplied policy interface;
4. `runtimes.yaml`, which declares isolated simulator environments.

The tested policy is not edited. Its manifest only states what the policy
already expects: sensor inputs, tensor shapes, joint order, action meaning,
control rate, navigation input, and artifact path. Geometry, friction,
compliance approximations, viscous drag, pushes, and other world effects belong
to simulator adapters, not policy wrappers.

## One-command lifecycle

From the repository root:

```bash
python -m pip install -e .

# Readiness only. No policy inference or physics episode is executed.
robustlab preflight configs/examiners/my_robot.yaml

# Readiness gate, state creation, and execution.
robustlab launch configs/examiners/my_robot.yaml
```

`launch` refuses to start when a required file, sensor bridge, action mapping,
runtime, or environment effect is missing. It writes `preflight.json` before
creating trials. A passed preflight means the declared interfaces exist; it
does not claim the robot can walk.

For review before allocating compute:

```bash
robustlab launch configs/examiners/my_robot.yaml --materialize-only
```

That command creates immutable scheduler state and the dashboard but executes
no episode. Start later with the exact command printed by RobustLab:

```bash
robustlab examine resume runs/my_robot/examination_state.json
```

Interruption is safe. The state is updated atomically, completed attempt IDs
are not reused, and the same resume command continues from the next pending
attempt. `--max-trials N` provides a clean batch boundary for Slurm or other
job schedulers.

## What is checked before a run

The preflight report has independent gates:

- **Robot contract:** profile, asset, format, and hashes are present.
- **Policy contract:** artifact, observations, sensors, action order, and
  navigation mode are explicit.
- **Observation/action bridge:** the selected backend can supply every required
  input without guessing.
- **Simulator runtime:** the configured executable and required GPU are
  available.
- **Environment physics:** every requested terrain effect has an implementation
  in the simulator adapter. RS11, for example, uses the declared rigid-viscous
  tier with simulator-owned drag. A missing implementation blocks before the
  policy runs and is never counted as policy failure.

Policies that take waypoints, goal poses, velocity commands, or no navigation
input use separate declared adapters. A route follower may be used for a
locomotion-only claim, while route-choice claims require a policy that can
actually choose a route. The dashboard records that distinction.

## Outputs

Each run directory is self-contained:

- `preflight.json`: readiness and ownership of every interface;
- `examination_state.json`: resumable scheduler state and hashes;
- `worlds/`: canonical simulator-neutral worlds;
- `trials/` and `episodes/`: outcomes, telemetry, metrics, logs, and media;
- `dashboard/index.html`: standalone live/final report;
- `dashboard/dashboard_data.json` and CSV files: machine-readable summaries.

Infrastructure errors, missing integrations, unsupported fidelity, and policy
behavioral failures remain separate throughout scheduling and reporting.

## Portability boundary

RobustLab can evaluate any policy for which its real input/output contract can
be declared and supplied. It cannot make an unknown observation layout,
missing perception encoder, proprietary simulator plugin, or incompatible
robot asset correct automatically. Those cases fail preflight with a concrete
missing requirement instead of producing misleading benchmark numbers.
