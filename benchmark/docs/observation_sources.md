# Observation sources

Every source a policy may declare, with how a robot obtains it and whether a
simulator currently produces it.

**Declarable:** all 76 compile — `tests/test_all_sources_usable.py` builds a real
term for each on every run, so nothing is listed here without working.

**Produced:** 29 are supplied by the MuJoCo backend at runtime. A source that is
declarable but not produced BLOCKS at preflight as `adapter_unverified` — you are
never silently handed zeros.

`multi` marks sources a robot may carry several of; name the instance
(`{source: depth_image, instance: front}`).

Provenance: assembled from the legged-locomotion literature, not a citation
review. Check against the sensor suites of the robots you benchmark.


## `onboard` — 20 sources, 12 produced

Direct sensor reading on the robot.

| Source | Unit | Frame | Multi | Produced |
|---|---|---|---|---|
| `base_angular_velocity_b` | rad/s | body |  | MuJoCo |
| `base_linear_acceleration_b` | m/s^2 | body |  | — |
| `base_orientation_wxyz` | unit quaternion | body to world, wxyz |  | MuJoCo |
| `battery_state` | declared encoding | electrical system |  | — |
| `depth_image` | declared encoding | camera | yes | MuJoCo |
| `event_camera` | events | camera image plane | yes | — |
| `foot_contact` | bool | canonical foot order |  | MuJoCo |
| `foot_contact_force` | N | body, canonical foot order |  | MuJoCo |
| `gnss_position` | declared coordinate unit | declared GNSS frame |  | — |
| `joint_position` | rad | canonical joint order |  | MuJoCo |
| `joint_temperature` | degC | canonical joint order |  | — |
| `joint_torque` | N.m | canonical joint order |  | MuJoCo |
| `joint_velocity` | rad/s | canonical joint order |  | MuJoCo |
| `lidar_rays` | m | body ray pattern | yes | MuJoCo |
| `magnetometer` | unit vector | body |  | — |
| `motor_current` | A | canonical joint order |  | — |
| `point_cloud` | m | declared sensor frame | yes | MuJoCo |
| `range_sensor` | m | declared sensor frame | yes | MuJoCo |
| `rgb_image` | declared encoding | camera | yes | MuJoCo |
| `thermal_image` | declared encoding | thermal camera | yes | — |

## `simulator_only` — 9 sources, 0 produced

Physics ground truth no robot can measure. Marks a result simulation-only.

| Source | Unit | Frame | Multi | Produced |
|---|---|---|---|---|
| `base_velocity_truth` | m/s, rad/s | world |  | — |
| `com_offset_truth` | m | body |  | — |
| `external_wrench_truth` | N, N.m | body |  | — |
| `friction_coefficient` | unitless | contact pair |  | — |
| `motor_strength_scale` | unitless | canonical joint order |  | — |
| `payload_mass_truth` | kg | body |  | — |
| `restitution` | unitless | contact pair |  | — |
| `terrain_class_truth` | declared encoding | world terrain |  | — |
| `terrain_height_truth` | m | world terrain |  | — |

## `command` — 13 sources, 1 produced

Supplied by an operator or high-level planner.

| Source | Unit | Frame | Multi | Produced |
|---|---|---|---|---|
| `body_height_command` | m | body height above local terrain |  | — |
| `footswing_height_command` | m | canonical foot order |  | — |
| `gait_command` | declared encoding | gait selector |  | — |
| `goal_distance` | m | frame-invariant scalar |  | — |
| `goal_heading` | rad | body-relative heading |  | — |
| `goal_vector_b` | m | body |  | — |
| `heading_command` | rad | world heading |  | — |
| `pitch_command` | rad | body pitch |  | — |
| `roll_command` | rad | body roll |  | — |
| `stance_width_command` | m | body lateral |  | — |
| `terrain_style_command` | declared encoding | terrain-style selector |  | — |
| `velocity_command` | m/s, m/s, rad/s | body x, body y, body yaw |  | MuJoCo |
| `waypoints_b` | m | body, N x 2 or N x 3 |  | — |

## `onboard_derived` — 33 sources, 15 produced

Computed on-robot from onboard data or the policy's own memory.

| Source | Unit | Frame | Multi | Produced |
|---|---|---|---|---|
| `actuator_saturation` | bool | canonical joint order |  | — |
| `adaptation_latent` | unitless | policy internal |  | — |
| `base_euler_rpy` | rad | world, roll-pitch-yaw |  | MuJoCo |
| `base_height` | m | above local terrain |  | — |
| `base_linear_velocity_b` | m/s | body |  | MuJoCo |
| `elevation_map` | m | declared map frame |  | MuJoCo |
| `episode_time` | s | episode clock |  | MuJoCo |
| `estimator_output` | declared encoding | declared estimator frame |  | — |
| `foot_position_b` | m | body, canonical foot order |  | MuJoCo |
| `foot_slip` | m/s | body, canonical foot order |  | — |
| `foot_velocity_b` | m/s | body, canonical foot order |  | — |
| `gait_clock` | unitless | gait cycle |  | — |
| `gait_phase` | cycle fraction | gait cycle |  | — |
| `height_scan` | m | body scan pattern |  | MuJoCo |
| `height_scan_per_foot` | m | body scan pattern per foot |  | MuJoCo |
| `imu_bias` | m/s^2, rad/s | body |  | — |
| `joint_acceleration` | rad/s^2 | canonical joint order |  | — |
| `joint_position_error` | rad | canonical joint order |  | — |
| `occupancy_grid` | occupancy probability | declared map frame |  | MuJoCo |
| `odometry_heading` | rad | odometry heading |  | — |
| `odometry_position` | m | odometry |  | — |
| `optical_flow` | pixel displacement | camera image plane | yes | MuJoCo |
| `previous_action` | raw network output | canonical joint order, pre-scaling |  | MuJoCo |
| `projected_gravity_b` | unit vector | body |  | MuJoCo |
| `recurrent_state` | unitless | policy internal |  | — |
| `self_collision` | bool | canonical body link pairs |  | — |
| `semantic_segmentation` | class id | camera image plane | yes | MuJoCo |
| `signed_distance_field` | m | declared map frame |  | MuJoCo |
| `stereo_disparity` | pixel disparity | stereo camera image plane | yes | MuJoCo |
| `terrain_normal_b` | unit vector | body |  | MuJoCo |
| `time_remaining` | s | episode clock |  | — |
| `traversability_map` | unitless | declared map frame |  | — |
| `visual_latent` | unitless | policy internal |  | — |

## `infrastructure` — 1 sources, 1 produced

Needs off-robot systems (motion capture, RTK). Real, but not field-autonomous.

| Source | Unit | Frame | Multi | Produced |
|---|---|---|---|---|
| `base_position_w` | m | world |  | MuJoCo |

## Totals

| Class | Declarable | Produced |
|---|---|---|
| `onboard` | 20 | 12 |
| `simulator_only` | 9 | 0 |
| `command` | 13 | 1 |
| `onboard_derived` | 33 | 15 |
| `infrastructure` | 1 | 1 |
| **total** | **76** | **29** |

## Not produced, and why

- `thermal_image`, `event_camera` — MuJoCo models neither thermal radiation
  nor event pixels. Claiming them would mean inventing data.
- `simulator_only` fields — deliberately withheld from deployable runs.
- The rest await backend work; each blocks honestly rather than returning zeros.
