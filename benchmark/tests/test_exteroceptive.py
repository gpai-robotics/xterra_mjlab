"""Exteroceptive sensing: height scan, depth, lidar.

Every Tier E field MUST carry a geometry declaration. Two policies asking for
a "height scan" with different grids are asking for different things, and
silently handing them the same array is exactly the class of error this
platform exists to prevent.

Convention: height_scan[i] = base_position_z - terrain_height_at(sample_i),
so on flat ground every sample equals the base height.
"""

import math

import pytest


def test_legged_gym_default_grid_has_187_samples():
    """x in [-0.8, 0.8] and y in [-0.5, 0.5] at 0.1 m -> 17 x 11 = 187."""
    from robustness_lab.observation.sensor_geometry import height_scan_points
    points = height_scan_points(x_range=(-0.8, 0.8), y_range=(-0.5, 0.5), resolution=0.1)
    assert len(points) == 187


def test_flat_ground_height_scan_equals_base_height():
    from robustness_lab.observation.sensor_geometry import sample_height_scan
    scan = sample_height_scan(base_position=[0.0, 0.0, 0.35], base_yaw=0.0,
                              points=[(0.0, 0.0), (0.3, 0.2), (-0.4, -0.1)],
                              terrain_height=lambda x, y: 0.0)
    assert scan == pytest.approx([0.35, 0.35, 0.35], abs=1e-9)


def test_step_terrain_height_scan_reads_the_step():
    from robustness_lab.observation.sensor_geometry import sample_height_scan
    scan = sample_height_scan(base_position=[0.0, 0.0, 0.35], base_yaw=0.0,
                              points=[(-0.2, 0.0), (0.2, 0.0)],
                              terrain_height=lambda x, y: 0.1 if x > 0.0 else 0.0)
    assert scan == pytest.approx([0.35, 0.25], abs=1e-9)


def test_scan_rotates_with_base_yaw():
    """The bug this catches is invisible on flat ground.

    At 90 deg yaw a point declared at body +x must sample world +y. A
    body-vs-world frame error passes every flat-terrain test and silently
    feeds the policy terrain from the wrong direction.
    """
    from robustness_lab.observation.sensor_geometry import sample_height_scan
    scan = sample_height_scan(base_position=[0.0, 0.0, 0.35],
                              base_yaw=math.radians(90.0),
                              points=[(1.0, 0.0)],
                              terrain_height=lambda x, y: 1.0 if y > 0.5 else 0.0)
    assert scan == pytest.approx([-0.65], abs=1e-9)


def test_scan_is_clipped_to_the_declared_range():
    from robustness_lab.observation.sensor_geometry import sample_height_scan
    scan = sample_height_scan(base_position=[0.0, 0.0, 0.35], base_yaw=0.0,
                              points=[(0.0, 0.0)],
                              terrain_height=lambda x, y: -5.0,
                              clip=(-1.0, 1.0))
    assert scan == pytest.approx([1.0], abs=1e-9)


def test_missing_geometry_declaration_is_rejected():
    """A height scan with an undeclared grid is unanswerable, not under-specified."""
    from robustness_lab.observation.compiler import compile_terms
    with pytest.raises(ValueError, match="geometry"):
        compile_terms([{"source": "height_scan", "size": 187}])


def test_declared_geometry_must_match_declared_size():
    from robustness_lab.observation.compiler import compile_terms
    with pytest.raises(ValueError):
        compile_terms([{
            "source": "height_scan", "size": 100,
            "geometry": {"kind": "grid", "x_range": [-0.8, 0.8],
                         "y_range": [-0.5, 0.5], "resolution": 0.1},
        }])


def test_height_scan_term_compiles_with_valid_geometry():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{
        "source": "height_scan", "size": 187,
        "geometry": {"kind": "grid", "x_range": [-0.8, 0.8],
                     "y_range": [-0.5, 0.5], "resolution": 0.1},
    }])
    assert plan.size == 187


def test_backend_without_height_scan_blocks_rather_than_zero_filling():
    """An unsupported exteroceptive field must block, never silently zero-fill."""
    from robustness_lab.observation.gates import check_exteroceptive_support
    result = check_exteroceptive_support(
        required={"height_scan"}, supported=set())
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"
    assert "height_scan" in result["missing"]


def test_depth_camera_requires_full_intrinsics_and_extrinsics():
    from robustness_lab.observation.sensor_geometry import validate_camera_geometry
    with pytest.raises(ValueError):
        validate_camera_geometry({"resolution": [64, 64]})  # no fov/near/far/pose


def test_valid_depth_camera_geometry_is_accepted():
    from robustness_lab.observation.sensor_geometry import validate_camera_geometry
    geometry = validate_camera_geometry({
        "resolution": [64, 64], "hfov_deg": 87.0, "near": 0.1, "far": 5.0,
        "position_b": [0.3, 0.0, 0.05], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "encoding": "metres",
    })
    assert geometry["resolution"] == [64, 64]
