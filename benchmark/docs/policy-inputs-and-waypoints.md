# Policy inputs and waypoint handling

The world is identical for every policy. What changes is the adapter between
the simulator and the policy.

```text
canonical world + canonical robot state
                 |
                 v
        policy-specific adapter
                 |
       +---------+----------+----------------+
       |                    |                |
 waypoint headings    velocity command   autonomous policy
       |                    |                |
       +--------- policy action ------------+
                            |
                            v
                 canonical joint command
```

## Inputs

Every policy manifest declares:

- Exact joint order and whether actions mean joint positions or torques.
- Control frequency, history length and recurrent state.
- Proprioception, IMU, contacts and previous actions.
- Camera, depth, lidar or terrain-height inputs when required.
- Normalization and tensor shapes.
- Its navigation interface and the assistance visible to it.

The runner never pads a missing input with zero and calls the result valid. If
an input cannot be recreated exactly, the run is blocked with the missing input
listed in its manifest.

## Navigation modes

| Policy interface | What the benchmark provides | Valid claim |
|---|---|---|
| Waypoint headings | Heading error to current and next benchmark waypoint | Locomotion on a fixed disclosed route |
| Waypoint positions | Relative waypoint positions in the policy's declared frame | Locomotion on a fixed disclosed route |
| Goal pose | Relative goal pose | Goal-conditioned locomotion |
| Velocity command | A versioned route follower converts the same route to velocity/yaw commands | Locomotion only; the route follower is disclosed |
| Autonomous | Sensors and final goal, but no hidden route instruction | Route choice/navigation if the policy declares that capability |
| None | No navigation input | Straight-course locomotion only |

This makes policies with different interfaces runnable on the same physical
test bed without pretending they solve the same task. A velocity-command policy
and a waypoint policy can be compared for obstacle traversal, but not for
unaided route choice unless both are evaluated without route assistance.

## FoReSight policies in this workspace

The FoReSight teacher and depth student use a hybrid interface:

- Forward speed arrives through the native velocity-command term.
- The 753-value state contains heading error to the current waypoint and the
  next waypoint.
- The student additionally uses an `87 x 58` depth image through its native
  depth encoder.
- The benchmark bridge samples the ordered canonical route densely enough that
  consecutive goals are at most about 2 m apart (minimum eight goals). The
  policy still sees only the current and next goal, so its tensor shape is
  unchanged on long courses.

For that reason their declared navigation mode is `waypoint_heading` with a
lookahead of two. Native execution preserves the source observation pipeline.
The portable student export remains blocked from generic execution until the
exact depth preprocessing, history and yaw injection are available.

## Branching worlds

Two scores are kept separate:

1. **Locomotion score:** the benchmark may disclose a route through waypoints
   or a route-follower command. This tests whether the legs can traverse it.
2. **Navigation score:** no external route choice is hidden in an adapter. Only
   a policy that can choose among branches is eligible.

Thus a policy that needs waypoints still works on chains, rings, grids and
mazes for locomotion evaluation. It does not receive an unaided-navigation
claim. A policy that needs no waypoints can run the same routes with a disclosed
route-follower adapter, or run autonomously if it has its own planner.

## Ordered testing

Each policy has an independent resumable examination. Results from one policy
never prune another policy's tree.

1. Flat qualification checks model, action order and basic forward locomotion.
2. RS1--RS11 are screened independently at easy, medium and hard severity.
3. The pass/fail boundary is refined and statistically confirmed per RS.
4. Only terrain severities supported by that policy enter ordered AE pairs.
5. Passing pairs are eligible to become longer chains and graph edges.
6. Longer/topological worlds are increased only after their shorter parent
   passes; stress and locked OOD evaluation come last.

A failure removes only harder descendants of that capability branch. It does
not remove unrelated terrains, and it does not erase an easier condition that
the policy passed.

The two source-native configurations are:

- `configs/examiners/adaptive_go2_native_baseline.yaml`
- `configs/examiners/adaptive_go2_native_foresight.yaml`

They launch the original repository's task and checkpoint loader. Screening
runs are headless and resumable; evidence videos are recorded separately for
selected boundary passes/failures so rendering does not dominate the suite.

## Native-bridge fidelity gate

FoReSight normally combines a generated terrain into one mesh with one physics
material. The bridge can reconstruct per-material USD face subsets, but that
experimental path did not pass behavioral conformance: the baseline policy
failed a flat control that passes with the source global material. It is
therefore disabled for official runs. Mixed-material conditions are explicitly
skipped rather than reported as policy failures or passes.

RS11 also remains unsupported because its viscous drag/deformation is not
supplied by this source task. Effective per-region materials must pass the
backend calibration fixtures before friction/compliance numbers are labelled
comparable.
