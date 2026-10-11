"""Every observation source declares HOW a robot obtains it.

The point is not to forbid information, it is to record honestly what kind of
information a result depended on. Almost everything a real legged robot uses is
available in deployment -- including lidar, point clouds, elevation and
traversability maps, learned state estimates and adaptation latents. What is NOT
available is physics ground truth the simulator alone knows.

Classes:
  onboard          direct sensor reading (encoders, IMU, cameras, lidar,
                   contact/force sensors, motor current, temperature)
  onboard_derived  computed on-robot from onboard data or the policy's own
                   memory: projected gravity, forward kinematics, elevation and
                   traversability maps, SLAM odometry, state estimators,
                   adaptation latents, gait clocks, previous actions,
                   recurrent state
  command          supplied by an operator or high-level planner
  infrastructure   needs off-robot systems: motion capture, RTK base station.
                   Real, but not field-autonomous
  simulator_only   physics ground truth no robot can measure: true friction,
                   restitution, true payload/COM, motor strength scale, applied
                   external wrench, ground-truth terrain height or base velocity

DEPLOYABLE is onboard + onboard_derived + command. `infrastructure` is usable
but must be labelled. `simulator_only` marks a result as simulation-only.
"""

import pytest

DEPLOYABLE = {"onboard", "onboard_derived", "command"}

EXPECTED = {
    # --- direct sensors -------------------------------------------------
    "joint_position": "onboard",
    "joint_velocity": "onboard",
    "joint_torque": "onboard",
    "joint_temperature": "onboard",
    "motor_current": "onboard",
    "base_orientation_wxyz": "onboard",
    "base_angular_velocity_b": "onboard",
    "base_linear_acceleration_b": "onboard",
    "foot_contact": "onboard",
    "foot_contact_force": "onboard",
    "depth_image": "onboard",
    "rgb_image": "onboard",
    "lidar_rays": "onboard",
    "point_cloud": "onboard",
    # --- computed on-robot ----------------------------------------------
    "joint_acceleration": "onboard_derived",
    "joint_position_error": "onboard_derived",
    "base_euler_rpy": "onboard_derived",
    "projected_gravity_b": "onboard_derived",
    "base_linear_velocity_b": "onboard_derived",
    "base_height": "onboard_derived",
    "foot_position_b": "onboard_derived",
    "foot_velocity_b": "onboard_derived",
    "previous_action": "onboard_derived",
    "episode_time": "onboard_derived",
    "time_remaining": "onboard_derived",
    "gait_phase": "onboard_derived",
    "gait_clock": "onboard_derived",
    "recurrent_state": "onboard_derived",
    "estimator_output": "onboard_derived",
    "adaptation_latent": "onboard_derived",
    "visual_latent": "onboard_derived",
    "height_scan": "onboard_derived",
    "height_scan_per_foot": "onboard_derived",
    "elevation_map": "onboard_derived",
    "traversability_map": "onboard_derived",
    "terrain_normal_b": "onboard_derived",
    "odometry_position": "onboard_derived",
    "odometry_heading": "onboard_derived",
    # --- operator / planner ----------------------------------------------
    "velocity_command": "command",
    "heading_command": "command",
    "body_height_command": "command",
    "pitch_command": "command",
    "roll_command": "command",
    "gait_command": "command",
    "footswing_height_command": "command",
    "stance_width_command": "command",
    "terrain_style_command": "command",
    "goal_vector_b": "command",
    "goal_distance": "command",
    "goal_heading": "command",
    "waypoints_b": "command",
    # --- off-robot infrastructure ----------------------------------------
    "base_position_w": "infrastructure",
    # --- physics ground truth --------------------------------------------
    "friction_coefficient": "simulator_only",
    "restitution": "simulator_only",
    "terrain_class_truth": "simulator_only",
    "terrain_height_truth": "simulator_only",
    "payload_mass_truth": "simulator_only",
    "com_offset_truth": "simulator_only",
    "motor_strength_scale": "simulator_only",
    "external_wrench_truth": "simulator_only",
    "base_velocity_truth": "simulator_only",
}


def test_every_expected_source_exists():
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    missing = sorted(set(EXPECTED) - set(CANONICAL_FIELDS))
    assert not missing, "observation sources not offered: %s" % missing


def test_every_canonical_field_declares_an_availability_class():
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    missing = [n for n, m in CANONICAL_FIELDS.items() if "availability" not in m]
    assert not missing, "fields without an availability class: %s" % missing


def test_availability_values_are_from_the_known_set():
    from robustness_lab.canonical_state import AVAILABILITY_CLASSES, CANONICAL_FIELDS
    for name, meta in CANONICAL_FIELDS.items():
        assert meta["availability"] in AVAILABILITY_CLASSES, name


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_source_is_classified_as_expected(name, expected):
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    assert CANONICAL_FIELDS[name]["availability"] == expected


def test_perception_products_are_deployable_not_privileged():
    """Elevation maps, scans and lidar run onboard on real robots."""
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    for name in ("height_scan", "elevation_map", "traversability_map",
                 "lidar_rays", "point_cloud", "terrain_normal_b"):
        assert CANONICAL_FIELDS[name]["availability"] in DEPLOYABLE, name


def test_deployable_run_rejects_physics_ground_truth():
    from robustness_lab.observation.gates import check_deployability
    result = check_deployability(
        sources={"joint_position", "friction_coefficient"}, allowed=DEPLOYABLE)
    assert result["status"] == "blocked"
    assert "friction_coefficient" in result["rejected"]
    assert result["outcome_class"] == "adapter_unverified"


def test_deployable_run_accepts_a_full_perception_stack():
    from robustness_lab.observation.gates import check_deployability
    result = check_deployability(
        sources={"joint_position", "projected_gravity_b", "height_scan",
                 "lidar_rays", "elevation_map", "estimator_output",
                 "recurrent_state", "velocity_command"},
        allowed=DEPLOYABLE)
    assert result["status"] == "ready"
    assert result["rejected"] == []


def test_result_records_which_availability_classes_were_used():
    from robustness_lab.observation.gates import check_deployability
    result = check_deployability(
        sources={"joint_position", "height_scan", "velocity_command"},
        allowed=DEPLOYABLE)
    assert set(result["classes_used"]) == {"onboard", "onboard_derived", "command"}
