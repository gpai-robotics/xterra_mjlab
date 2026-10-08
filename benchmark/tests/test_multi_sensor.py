"""Robots carry more than one of a sensor, and more modalities than a depth cam.

ANYmal carries several depth cameras; Go2 has front and rear. Declaring a single
`depth_image` cannot express that, so a term must be able to name WHICH sensor
instance it reads. Separately, the vocabulary was missing several real
modalities.
"""

import pytest

def _cam(resolution):
    return {"kind": "camera", "resolution": resolution, "hfov_deg": 87.0,
            "near": 0.1, "far": 5.0, "position_b": [0.3, 0.0, 0.05],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "encoding": "metres"}


CAM = _cam([2, 2])      # 4 values
CAM2 = _cam([1, 2])     # 2 values

EXPECTED_NEW = {
    "semantic_segmentation": "onboard_derived",
    "optical_flow": "onboard_derived",
    "thermal_image": "onboard",
    "event_camera": "onboard",
    "stereo_disparity": "onboard_derived",
    "occupancy_grid": "onboard_derived",
    "signed_distance_field": "onboard_derived",
    "range_sensor": "onboard",
    "gnss_position": "onboard",
    "magnetometer": "onboard",
    "battery_state": "onboard",
    "foot_slip": "onboard_derived",
    "actuator_saturation": "onboard_derived",
    "self_collision": "onboard_derived",
    "imu_bias": "onboard_derived",
}


def test_new_modalities_exist():
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    missing = sorted(set(EXPECTED_NEW) - set(CANONICAL_FIELDS))
    assert not missing, "missing modalities: %s" % missing


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_NEW.items()))
def test_new_modality_availability(name, expected):
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    assert CANONICAL_FIELDS[name]["availability"] == expected


def test_sensor_sources_are_marked_multi_instance():
    """Anything a robot can carry several of must say so."""
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    for name in ("depth_image", "rgb_image", "lidar_rays", "range_sensor",
                 "thermal_image", "event_camera", "semantic_segmentation"):
        assert CANONICAL_FIELDS[name].get("multi_instance") is True, name


def test_scalar_sources_are_not_multi_instance():
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    for name in ("joint_position", "projected_gravity_b", "velocity_command"):
        assert not CANONICAL_FIELDS[name].get("multi_instance", False), name


def test_two_depth_cameras_compile_as_distinct_terms():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([
        {"source": "depth_image", "instance": "front", "size": 4, "geometry": CAM},
        {"source": "depth_image", "instance": "rear", "size": 4, "geometry": CAM},
    ])
    assert plan.size == 8
    built = plan.build({"depth_image": {"front": [1.0, 2.0, 3.0, 4.0],
                                        "rear": [5.0, 6.0, 7.0, 8.0]}})
    assert built == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_instances_keep_their_declared_order():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([
        {"source": "depth_image", "instance": "rear", "size": 2, "geometry": CAM2},
        {"source": "depth_image", "instance": "front", "size": 2, "geometry": CAM2},
    ])
    built = plan.build({"depth_image": {"front": [9.0, 9.0], "rear": [1.0, 1.0]}})
    assert built == pytest.approx([1.0, 1.0, 9.0, 9.0])


def test_missing_instance_is_reported_by_name():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([
        {"source": "depth_image", "instance": "left", "size": 2, "geometry": CAM2}])
    with pytest.raises(ValueError, match="left"):
        plan.build({"depth_image": {"front": [1.0, 2.0]}})


def test_repeating_a_multi_instance_source_without_instances_is_ambiguous():
    from robustness_lab.observation.compiler import compile_terms
    with pytest.raises(ValueError, match="instance"):
        compile_terms([
            {"source": "depth_image", "size": 2, "geometry": CAM2},
            {"source": "depth_image", "size": 2, "geometry": CAM2},
        ])


def test_single_instance_declaration_still_accepts_a_flat_sequence():
    """One camera, no instance name: the old flat form must keep working."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "depth_image", "size": 4, "geometry": CAM}])
    assert plan.build({"depth_image": [1.0, 2.0, 3.0, 4.0]}) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0])
