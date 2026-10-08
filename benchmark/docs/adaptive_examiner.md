# Adaptive locomotion examiner

The adaptive examiner measures one declared robot–policy–sensor–simulator
combination. It does not assume that loading a neural-network file proves that
the robot can walk.

## Evidence gates

1. `skill discover` finds likely policy artifacts without unpickling them.
2. `skill validate` checks the explicit sensor, tensor, joint-order, action, and
   artifact contract.
3. `skill probe` loads a portable TorchScript/ONNX export, supplies tensors with
   the declared shapes, and checks finite 12-joint output and inference timing.
4. `examine start` checks whether the selected backend actually provides every
   required sensor. Unsupported depth, lidar, or privileged observations block
   execution instead of being replaced with guessed values.
5. A flat qualification trial must pass before terrain capability is measured.
6. RS branches are screened at sparse severities. The scheduler refines only
   the pass/fail boundary, confirms it statistically, and audits a deterministic
   sample of pruned conditions.
7. Ordered AE interactions are chosen only after primitive boundaries are known.

The capability passport keeps measured episodes separate from monotonicity-based
inferences. A failure explanation is descriptive until a matched, same-terrain
counterfactual is available.

## Go2 policy artifacts found locally

The extracted baseline bundle is represented by two manifests:

- `configs/skills/go2_extreme_parkour_teacher.yaml`: a privileged teacher with
  one 753-value input. This is useful as an oracle/reference but is not a
  deployable sensor track.
- `configs/skills/go2_extreme_parkour_student_policy.yaml`: the deploy policy
  with a 753-value policy tensor and 32-value terrain latent. The latent must be
  produced from the bundled depth encoder and the exact training preprocessing.

Both artifacts can be shape-probed on CPU. The native teacher is also supported
through the FoReSight bridge, which keeps the source observation provider and
replaces only its terrain with a canonical WorldSpec. The depth student still
requires its exact encoder/camera integration.

## External policy on a Robustness Lab world

`scripts/play_foresight_on_world.py` forwards checkpoint/task arguments to the
source repository's native playback runner and injects one canonical WorldSpec
as its terrain. The bridge is limited to box geometry and one global terrain
material in its current version.

The verified example is stored under
`runs/external_policy_demo/ae-rs9-rs10/`. Its summary reports waypoint 7/7
reached in a fresh Isaac recording, with seven terminations during the
1,000-step window. The video is therefore an integration success, not yet a
robust success-rate claim.

Run another world/checkpoint with:

```bash
GPU_INDEX=1 scripts/run_foresight_world.sh \
  runs/external_policy_demo/ae-rs9-rs10/world.json \
  "/absolute/path/to/model_N.pt" \
  runs/external_policy_demo/my-run
```

The output directory receives the video, source-native summary, geometry bridge
manifest and a combined `result.json` containing both policy and world hashes.

## Commands

From `benchmarking_automation/`:

```bash
export PYTHONPATH="$PWD/src"

python -m robustness_lab skill discover \
  "../foresight_go2_parkour" \
  --output reports/foresight-skill-discovery.json

python -m robustness_lab skill validate \
  configs/skills/go2_extreme_parkour_student_policy.yaml \
  --output reports/go2-student-skill-validation.json

python -m robustness_lab skill probe \
  configs/skills/go2_extreme_parkour_student_policy.yaml \
  --warmup 5 --repetitions 50 \
  --output reports/go2-student-policy-probe.json

python -m robustness_lab examine start \
  configs/examiners/adaptive_go2_fake.yaml

python -m robustness_lab examine status \
  runs/examinations/adaptive-go2-fake/examination_state.json

python -m robustness_lab examine diagnose \
  runs/examinations/adaptive-go2-fake/examination_state.json

python -m robustness_lab examine report \
  runs/examinations/adaptive-go2-fake/examination_state.json
```

The fake run validates orchestration and reporting only. It is deliberately
marked as incapable of supporting a locomotion claim.

## Adding another policy

Create a skill YAML that declares every required sensor, each network input
tensor and its shape/source order, the exact 12-joint action order and meaning,
control rate, robot-profile hash, and artifact provenance. Do not describe a
depth policy as blind, and do not map privileged terrain values to physical
sensors. The examiner rejects those mismatches before allocating simulation
time.
