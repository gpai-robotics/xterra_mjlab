# Bring your own policy

Submit a trained locomotion policy and get it evaluated. You supply two things:
an artifact and a manifest. The manifest is drafted for you from the artifact;
you only correct it.

## Quick start

```bash
# 1. Draft the manifest from your weights
robustlab policy scaffold my_policy.pt \
    --robot-profile runs/go2_profile.json \
    --output my_policy.yaml

# 2. Correct the items it flags under needs_review, then check readiness
robustlab preflight configs/examiners/my_examination.yaml

# 3. Run
robustlab launch configs/examiners/my_examination.yaml
```

A complete worked example, runnable as-is, is in
`examples/bring_your_own_policy/run_example.py`.

## What you supply

**The artifact.** TorchScript (`.pt`) or ONNX. It must load standalone — a
training-framework checkpoint that needs your repository to reconstruct is not
enough. Export it first.

**The manifest.** Describes what your policy already expects. RobustLab never
modifies your policy; the manifest only states its interface.

```yaml
schema_version: "2.0"
policy_id: my-go2-policy
runtime:
  format: torchscript          # torchscript | onnx
  path: my_policy.pt
  control_frequency_hz: 50
observation:
  terms:                       # in the exact order your network expects
    - {source: base_angular_velocity_b, transform: [{scale: 0.25}], size: 3}
    - {source: projected_gravity_b, size: 3}
    - {source: velocity_command, transform: [{scale: [2.0, 2.0, 0.25]}], size: 3}
    - source: joint_position
      transform: [{subtract: default_joint_position}]
      joint_order: policy
      size: 12
    - {source: joint_velocity, transform: [{scale: 0.05}], joint_order: policy, size: 12}
    - {source: previous_action, size: 12}
  history: {frames: 1, order: newest_first}
  expected_size: 45
action:
  space: joint_position_offset
  scale: 0.25
  offset: default_joint_position
  clip: [-100.0, 100.0]
  size: 12
joint_order_policy: [FL_hip_joint, FL_thigh_joint, ...]
```

`joint_order_policy` is YOUR joint ordering. RobustLab builds the permutation to
and from the simulator's ordering — you do not have to match ours.

## Observation sources

Proprioceptive and command fields, available now:

| Source | Unit / frame |
|---|---|
| `joint_position`, `joint_velocity`, `joint_torque` | rad, rad/s, N·m |
| `base_orientation_wxyz` | unit quaternion, world, **w first** |
| `base_euler_rpy` | rad, world |
| `projected_gravity_b` | unit vector, body |
| `base_linear_velocity_b`, `base_angular_velocity_b` | m/s, rad/s, body |
| `base_linear_acceleration_b` | m/s², body |
| `base_height` | m, above local terrain |
| `previous_action` | raw network output, pre-scaling |
| `foot_contact`, `foot_contact_force` | bool per foot, N |
| `foot_position_b`, `foot_velocity_b` | m, m/s, body |
| `velocity_command` | m/s, m/s, rad/s |
| `goal_vector_b`, `goal_distance`, `goal_heading` | m, m, rad |
| `waypoints_b` | m, body |
| `episode_time` | s |

Exteroceptive sources (`height_scan`, `depth_image`, `lidar_rays`, …) are
declared in the schema and validated, but **no simulator adapter produces them
yet**. Requesting one blocks with `adapter_unverified` rather than silently
feeding zeros. See "Current limits".

## Transforms

Applied in the order you declare them. `subtract` then `scale` is not the same
as `scale` then `subtract`.

`scale` (scalar or per-element) · `subtract` (constant, or the name
`default_joint_position`) · `clip` · `select` · `normalize` (mean/std) ·
`quat_to_projected_gravity` · `quat_to_euler` · `to_body_frame` · `one_hot`

`history: {frames: N, order: newest_first|oldest_first, layout: frame_major|term_major}`
stacks N frames. The buffer is zero-filled at reset, never by repeating the
first frame, and cleared between trials.

`order` is **required** once `frames > 1`. Nothing about a stacked vector
reveals which end is the newest frame, so it cannot be inferred from your
artifact and a wrong guess is the same width as a right one.

`layout` says how the frames are arranged:

- `frame_major` (default) — whole frames back to back:
  `[frame_t, frame_t-1, ...]`
- `term_major` — each term's whole history in turn:
  `[N frames of base_ang_vel, N of projected_gravity, ...]`

Isaac Lab and mjlab export `term_major`; their deploy config records it as
`history_layout: isaaclab_term_major`. Both layouts have identical total width,
so the dimension gate cannot separate them — declare the one you trained with.

`default_joint_position` is the **robot's** stance, taken from the robot profile
bound at run time, so you normally do not write it. Declare it under
`references:` only when validating a manifest with no robot bound. If your
declared stance disagrees with the robot you run against, the load is refused —
it is never quietly filled with zeros, which would turn `subtract:` into a no-op
and feed absolute joint angles to a policy trained on offsets.

## What gets checked before you are scored

A wrong manifest is caught before any terrain is scored, and is **never**
recorded as your policy failing.

| Gate | Checks |
|---|---|
| G1 dimension | compiled width == declared width == the width your artifact actually accepts |
| | **Not sufficient alone.** Some exported policies slice their input instead of validating it. The FoReSight exported teacher in `assets/cache/skills/` accepts every width from 530 to 799 and returns 12 actions for each, with *different* values. A wrong width passes G1 silently. G3 and G4 are what catch it. |
| G2 finiteness | no NaN or inf in the compiled observation |
| G3 known pose | at the default pose: `projected_gravity_b` ≈ `[0,0,-1]`, joint term ≈ 0 |
| G4 flat ground | the policy must walk on flat ground before terrain is scored |

### G2/G3 evidence

Preflight measures G2 and G3 by placing the robot level at its default pose on
flat ground and reading one observation after a kinematic reset. It does not
step or settle the simulator. `provenance: measured` means the values came from
that reset; `provenance: declared` means they came from
`observation_gate_evidence` in the examiner configuration. If a backend cannot
perform the reset and there is no declared fallback, provenance is
`unavailable` and the gates do not pass.

Any failure returns `adapter_unverified`, distinct from `policy_fail`. If we
cannot construct your test faithfully, we say so instead of publishing a number.

## Outcomes

- `policy_pass` — ran and succeeded
- `policy_fail` — ran and failed (fell, stalled, timed out)
- `adapter_unverified` — we could not build your observation correctly
- `environment_unsupported` — the terrain physics is not implemented for this backend
- `infrastructure_error` — simulator or GPU fault
- `pruned` — intentionally not run given earlier results

## Current limits

- **Exteroceptive policies are not yet runnable.** Height scans, depth cameras
  and lidar are specified and geometry-validated, but no adapter produces them.
  A parkour or vision policy will block at preflight.
- **Privileged and recurrent inputs** are specified but not wired.
- **MuJoCo is the working backend.** Isaac Lab starts but episodes do not yet
  complete; see `docs/isaac_lab_setup.md`.
- **Multi-input policies are not supported yet.** The compiler builds one
  concatenated vector. A policy whose forward signature takes several tensors —
  e.g. the FoReSight student, `forward(x, scandots_latent)`, whose second input
  comes from a separate depth-encoder artifact — cannot run through this path.
  Single-tensor policies only.
- **Width acceptance is weak evidence.** See the note under G1. Never treat "the
  artifact ran" as confirmation that your manifest is right; rely on G3 and G4.
- **The scaffold drafts a hypothesis, not an answer.** A tensor width cannot
  establish semantics. Always check `needs_review`.
