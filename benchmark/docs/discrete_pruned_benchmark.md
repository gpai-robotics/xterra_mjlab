# Discrete pruned benchmark

`discrete_pruned_v2` is the fixed-grid alternative to the original adaptive
boundary examiner. It never invents intermediate severity values.

## Exact order

1. Three flat qualification attempts.
2. Medium-geometry RS1–RS11 at static/dynamic friction `0.75/0.60`, then
   `0.50/0.40`, then `0.25/0.20`.
3. Easy and hard geometry at the same three friction points. This completes
   the 99-condition primitive catalogue.
4. Every distinct directed pair whose two constituent geometry levels passed
   `0.50/0.40`. RS8 is included and self-pairs are excluded. Pair challenges
   use fixed `0.50/0.40` friction and three matched geometry levels.
5. Balanced M-length chains/grids, L-length ring/tree/random graphs,
   100/200-body-length endurance, stress, and locked OOD worlds.

Screening uses three seeds. A mixed result or a condition that unlocks a later
stage expands to ten attempts; confirmation requires seven successes. A failed
large world creates an isolated pair counterfactual before any pair is removed.

## Run and inspect

```bash
cd "${ROBUSTLAB_ROOT}"
export PYTHONPATH="$PWD/src"

# Complete readiness report without a simulator episode.
python -m robustness_lab preflight configs/examiners/discrete_go2_native_baseline.yaml

# Production entry point: preflight first, then execute or resume.
python -m robustness_lab launch configs/examiners/discrete_go2_native_baseline.yaml

# Fast orchestration/evidence validation.
python -m robustness_lab examine start configs/examiners/discrete_go2_fake.yaml \
  --max-trials 3

# Continue exactly where it stopped.
python -m robustness_lab examine resume \
  runs/discrete_examiner/go2_fake/examination_state.json --max-trials 25

# Rebuild diagnostics and the dashboard without rerunning simulation.
python -m robustness_lab examine report \
  runs/discrete_examiner/go2_fake/examination_state.json
```

Open `runs/discrete_examiner/go2_fake/dashboard/index.html`. While an
examination is active it refreshes every ten seconds. It contains the L0–L7
funnel, RS and directed-pair results, valid pass/failure counts, Wilson
intervals, infrastructure outcomes, diagnostic chains, snapshots, videos, and
links to raw worlds, telemetry and metrics.

## Parallel trials

Parallel execution is opt-in. Set the worker count in the examiner config:

```yaml
discrete:
  parallel_trials: 24
```

Start with no more workers than physical CPU cores for CPU simulation, then
reduce the count if memory pressure, thermal throttling, or simulator license
limits appear. GPU backends are usually bounded by GPU memory and measured
throughput instead; increase the count gradually while watching utilization and
out-of-memory failures. `parallel_trials: 1` retains serial execution.

The schedule cannot run every trial at once. L0 qualification gates the entire
examination. A condition's screening repetitions must finish before the result
can promote it to confirmation, and L2 pairs are not eligible until their L1
parents have been decided because failed parents prune L2. These dependency
barriers bound achievable speedup even when many workers are available.

Recording every long trial is expensive. MuJoCo renders and encodes a frame at
every control step; a 180-second episode at 50 Hz produces about 9,000 frames.
That adds rendering and encoding load to every worker and can make GPU, CPU, and
storage bandwidth the bottleneck. Prefer the standard retained failure and
representative-pass videos unless every-trial footage is required.

Real native policy configurations are:

- `configs/examiners/discrete_go2_native_baseline.yaml`
- `configs/examiners/discrete_go2_native_foresight.yaml`

The source-native bridge binds separate physics materials to collision regions.
RS11 uses a simulator-owned rigid-viscous effect. Preflight verifies both
features before policy execution. A missing backend feature is reported as an
environment/integration problem and never as a policy failure.

## Evidence retention

- Every valid canonical episode receives three top-down trajectory snapshots.
- Every behavioral failure retains a video.
- Each completed condition retains one median representative pass video.
- Native runs without canonical per-foot telemetry use labelled source-camera
  frames and disclose that limitation.
- `dashboard/dashboard_data.json` and `dashboard/conditions.csv` are the
  machine-readable dashboard sources; episode JSONL/Parquet remains canonical.
