# Submissions

Everything a researcher brings lives **here**, in its own folder. Nothing inside
`src/` ever needs editing.

```
submissions/
  _template/                      copy this to start
  go2_blind/                      a complete, runnable submission
  <your-policy-name>/
      policy.yaml                 the mapping  (the only file you must write)
      weights/policy.onnx         your trained network
      robot/  (optional)          your URDF/MJCF if not using a built-in robot
      notes.md (optional)         anything you want recorded with the result
```

## What you bring

**1. Your trained network** — TorchScript (`.pt`) or ONNX (`.onnx`).
It must load standalone. A training checkpoint that needs your repo to
reconstruct is not enough; export it first.

**2. `policy.yaml`** — how your network's input maps onto the benchmark's
observations. This is the only thing you write.

**3. Your robot** — optional. Go2 is built in. For anything else, drop a URDF or
MJCF in `robot/` and point at it.

## The mapping, simplest form

List your observations **in the order your network expects them**. Widths are
worked out from your robot, so you type no numbers:

```yaml
observation:
  sources:
    - base_linear_velocity_b
    - base_angular_velocity_b
    - projected_gravity_b
    - velocity_command
    - joint_position
    - joint_velocity
    - previous_action
```

That is 48 wide on a 12-joint robot, 66 on an 18-joint one. Same file.

**One source is a different width?** Override just that entry:

```yaml
    - {source: velocity_command, size: 2, transform: [{select: [0, 2]}]}
```

**Using a camera, lidar or height scan?** Add its geometry so we know what to
sample:

```yaml
    - height_scan
  geometry:
    height_scan: {kind: grid, x_range: [-0.8, 0.8], y_range: [-0.5, 0.5],
                  resolution: 0.1}
```

**Stacking past frames?** Say how many, which end is newest, and how they are
laid out. All three are required because getting any of them wrong produces a
vector of exactly the right width:

```yaml
  history: {frames: 100, order: oldest_first, layout: term_major}
```

`frame_major` (the default) stacks whole frames. `term_major` concatenates each
term's whole history in turn — that is what Isaac Lab and mjlab export, shown in
their deploy config as `history_layout: isaaclab_term_major`.

**Subtracting the default pose?** `default_joint_position` comes from the robot
you run against. You only declare it yourself if you are validating the manifest
without a robot bound — and if what you declare disagrees with the robot, we stop
rather than offset every joint.

**Preprocessing?** Any source takes a transform chain, applied in order:

```yaml
    - {source: joint_velocity, transform: [{scale: 0.05}, {clip: [-10, 10]}]}
```

The full source list is in `docs/observation_sources.md` (76 of them).
The transform list is in `docs/bring_your_own_policy.md`.

## Don't want to write it by hand?

```bash
robustlab policy scaffold submissions/mine/weights/policy.pt \
    --robot-profile runs/go2_profile.json \
    --output submissions/mine/policy.yaml
```

It probes your network, drafts the mapping, and lists what it could not infer
under `needs_review`. Correct those and you are done.

## Then run it

```bash
robustlab preflight submissions/mine/policy.yaml     # checks before spending compute
robustlab launch    submissions/mine/policy.yaml     # runs the benchmark
```

Preflight tells you exactly what is wrong before anything runs. If your declared
width disagrees with what your network actually accepts, it says both numbers.
G2/G3 evidence is measured from one kinematic reset at the robot's level default
pose: `measured` provenance came from that reset, while `declared` provenance
came from examiner configuration. A backend that cannot reset reports
`unavailable` and does not pass unless declared fallback evidence is present.

## What we will not do

If we cannot build your observation faithfully, the run stops with
`adapter_unverified` instead of scoring you. A wrong number is worse than no
number.
