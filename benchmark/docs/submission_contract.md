# Submission contract: who provides what

The platform builds every input your policy needs — including running your own
preprocessing networks — but some facts cannot be inferred from weights and must
come from you. This page draws that line precisely.

## The short version

| Thing | Who |
|---|---|
| Terrain, world geometry, friction, physics | **Platform** |
| The route and its waypoints | **Platform** |
| Robot model, joint ordering, default pose | **Platform** |
| Simulator state: joint angles, IMU, contacts, velocities | **Platform** |
| Depth / height-scan rendering *(not yet implemented)* | **Platform** |
| Your policy weights | **Researcher** |
| Your preprocessing / encoder weights | **Researcher** |
| What each observation slot MEANS | **Researcher** |
| Scales, offsets, clipping, normalisation statistics | **Researcher** |
| Your joint ordering | **Researcher** |
| How many tensors your `forward()` takes, and in what order | **Researcher** (probed, then confirmed) |
| Any slicing of an encoder output | **Researcher** |

The rule behind the split: **the platform provides physical facts; you provide
semantic facts.** We know the robot's knee angle. Only you know that your
network expects it third, scaled by 0.05, with the default pose subtracted.

## What only you can tell us

These are the things a tensor width cannot establish. `robustlab policy scaffold`
guesses them and lists them under `needs_review`; a wrong guess produces a
plausible, wrong benchmark score, so they must be confirmed.

**1. Observation semantics and order.** Which slot is which quantity. A 45-wide
vector could be almost anything.

**2. Transform scales.** RSL-RL policies typically scale angular velocity by
0.25, joint velocity by 0.05, and commands by `[2.0, 2.0, 0.25]`. These live in
your training config, not in your weights.

**3. Your joint ordering.** We remap to the simulator's ordering for you, but we
must know yours.

**4. Action pipeline.** Scale, offset (usually the default joint pose), clipping,
and control frequency.

**5. History depth and stacking order.** Whether you stack N frames, and whether
the newest frame is first or last.

**6. Encoder output slicing.** If your preprocessing network emits more than your
policy consumes, only you know which part to take. The FoReSight student is the
worked case: its depth encoder returns 34 values, the policy takes 32, because
the encoder emits `[latent(32) | yaw(2)]`. Nothing about the weights reveals
that split.

## Multiple inputs

If `forward()` takes several tensors, declare them in call order. We probe your
artifact and refuse the run if the count disagrees, rather than passing the
wrong thing:

```yaml
runtime:
  format: torchscript
  path: policy.pt
  inputs: [x, scandots_latent]     # positional order for forward()
observation:
  tensors:
    - name: x
      terms: [ ... ]
    - name: scandots_latent
      from_encoder: depth_encoder
      transform: [{select: {start: 0, count: 32}}]
      size: 32
```

## Bringing your own preprocessing network

Declare it as an encoder. The platform loads it, feeds it the inputs it needs,
and routes its output into your policy. The weights stay yours and are recorded
as `provided_by: researcher` in the run provenance.

```yaml
encoders:
  - name: depth_encoder
    path: depth_latest.pt
    inputs:
      - {name: depth_image, shape: [58, 87]}
      - {name: proprioception, shape: [53]}
    output: {name: depth_out, size: 34}
```

If the declared output width does not match what the artifact really emits, the
run is refused. A silently reshaped latent is a wrong number, not a small error.

## Worked example: the FoReSight student

Measured from the committed exports, this is the complete chain:

```
platform  ->  depth_image[58,87] + proprioception[53]
                        |
researcher's encoder (depth_latest.pt)
                        |
                      [34]
                        |
researcher's slice  select(start=0, count=32)
                        |
                      [32]
                        |
platform  ->  x[753]  +  scandots_latent[32]
                        |
researcher's policy (policy.pt)
                        |
                     12 actions
```

Everything marked *researcher* is a fact we cannot derive. Everything else we
build for you.

## What we check before scoring you

- **Arity** — your artifact's real input count matches your declaration.
- **Width** — compiled width matches declared width matches what the artifact
  accepts. Weak on its own: some exported policies slice their input and accept
  any width, silently returning different answers.
- **Finiteness** — no NaN or inf reaches your network.
- **Known pose** — at rest, `projected_gravity_b` must be about `[0, 0, -1]` and
  the default-subtracted joint term about zero. This catches a wrong ordering or
  a flipped sign that width checks cannot.
- **Flat ground** — your policy must walk on flat ground before terrain is
  scored. Failure here means we built your observation wrongly, not that your
  policy is bad.

Any of these failing yields `adapter_unverified`, never `policy_fail`. If we
cannot construct your test faithfully, we say so instead of publishing a number.
