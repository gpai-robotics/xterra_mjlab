"""MuJoCo must actually PRODUCE exteroceptive data, not merely accept it.

Until now 76 sources were declarable but only 16 produced. A source that can be
declared and never supplied is the same defect as a capability advertised
without an implementation.

Approach follows the standard practice for MuJoCo perception: mj_ray casting for
scans, lidar and range sensors; offscreen rendering for depth, RGB and
segmentation; elevation and derived maps built from the ray grid.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ROBOT = FIXTURES / "mini_robot.xml"

GRID = {"kind": "grid", "x_range": [-0.2, 0.2], "y_range": [-0.2, 0.2],
        "resolution": 0.2}                                   # 3 x 3 = 9 points
CAMERA = {"kind": "camera", "resolution": [8, 8], "hfov_deg": 87.0, "near": 0.05,
          "far": 6.0, "position_b": [0.2, 0.0, 0.1],
          "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "encoding": "metres"}
LIDAR = {"kind": "lidar", "ray_count": 16, "angular_pattern": "uniform",
         "max_range": 8.0, "origin_b": [0.0, 0.0, 0.15]}


def _sensors():
    pytest.importorskip("mujoco")
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    import mujoco
    from robustness_lab.backends.mujoco import build_mjcf_scene
    import tempfile, os
    profile = inspect_mjcf(str(ROBOT))
    world = generate_terrain("rs1", profile, 0.2, 11)
    scene = os.path.join(tempfile.mkdtemp(), "scene.xml")
    build_mjcf_scene(str(ROBOT), world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoSensorSuite(model, data, profile), model, data


# ------------------------------------------------------------------ declared

def test_mujoco_now_advertises_exteroceptive_sources():
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    for name in ("height_scan", "height_scan_per_foot", "elevation_map",
                 "terrain_normal_b", "lidar_rays", "point_cloud",
                 "depth_image", "rgb_image", "range_sensor"):
        assert name in declared, name


def test_mujoco_does_not_advertise_what_it_cannot_simulate():
    """No thermal or event-camera physics exists; they must stay unsupported."""
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    for name in ("thermal_image", "event_camera"):
        assert name not in declared, name


# ------------------------------------------------------------------ produced

def test_height_scan_has_the_declared_point_count():
    suite, _, _ = _sensors()
    scan = suite.height_scan(GRID)
    assert len(scan) == 9


def test_height_scan_on_flat_ground_is_uniform_and_positive():
    """Every sample measures the same drop to flat ground."""
    suite, _, _ = _sensors()
    scan = suite.height_scan(GRID)
    assert all(v > 0.0 for v in scan)
    assert max(scan) - min(scan) < 1e-3


def test_height_scan_is_finite_everywhere():
    suite, _, _ = _sensors()
    assert all(abs(v) < 1e6 for v in suite.height_scan(GRID))


def test_lidar_returns_one_range_per_declared_ray():
    suite, _, _ = _sensors()
    rays = suite.lidar_rays(LIDAR)
    assert len(rays) == 16
    assert all(0.0 <= v <= LIDAR["max_range"] for v in rays)


def test_lidar_ignores_the_robots_own_body():
    """Rays that strike the robot must be retried past the self hit, or the
    scan reports a few centimetres in every direction."""
    suite, _, _ = _sensors()
    rays = suite.lidar_rays(LIDAR)
    assert max(rays) > 0.3, "all returns are self-hits: %r" % rays[:4]


def test_depth_image_matches_declared_resolution_and_range():
    suite, _, _ = _sensors()
    depth = suite.depth_image(CAMERA)
    assert len(depth) == 64
    assert all(CAMERA["near"] <= v <= CAMERA["far"] for v in depth)


def test_rgb_image_is_three_channels_per_pixel():
    suite, _, _ = _sensors()
    rgb = suite.rgb_image(CAMERA)
    assert len(rgb) == 64 * 3
    assert all(0.0 <= v <= 1.0 for v in rgb)


def test_point_cloud_is_three_values_per_point():
    suite, _, _ = _sensors()
    cloud = suite.point_cloud({"kind": "point_cloud", "point_count": 16,
                               "ray_count": 16, "angular_pattern": "uniform",
                               "max_range": 8.0, "origin_b": [0.0, 0.0, 0.15]})
    assert len(cloud) == 48


def test_elevation_map_matches_grid_and_terrain_normal_is_unit_length():
    import math
    suite, _, _ = _sensors()
    assert len(suite.elevation_map(GRID)) == 9
    normal = suite.terrain_normal_b(GRID)
    assert len(normal) == 3
    assert math.sqrt(sum(v * v for v in normal)) == pytest.approx(1.0, abs=1e-6)


def test_range_sensor_returns_a_single_finite_reading():
    suite, _, _ = _sensors()
    reading = suite.range_sensor({"kind": "range", "direction_b": [1.0, 0.0, 0.0],
                                  "max_range": 8.0, "origin_b": [0.0, 0.0, 0.15]})
    assert len(reading) == 1
    assert 0.0 <= reading[0] <= 8.0


def test_height_scan_per_foot_returns_one_reading_per_foot():
    suite, _, profile = _sensors()
    values = suite.height_scan_per_foot({"kind": "per_foot", "offsets": [[0.0, 0.0]]})
    assert len(values) >= 1
    assert all(abs(v) < 1e6 for v in values)


# --------------------------------------------------- off-terrain robustness

WIDE = {"kind": "grid", "x_range": [-1.0, 1.0], "y_range": [-0.2, 0.2],
        "resolution": 0.2, "clip": [-1.0, 1.0]}


def test_samples_that_miss_the_terrain_respect_the_declared_clip():
    """A robot near the world edge casts rays into empty space. Those samples
    must be clamped to the declared range, not returned as a metres-long value
    that a policy would read as a cliff."""
    suite, _, _ = _sensors()
    scan = suite.height_scan(WIDE)
    assert all(-1.0 <= v <= 1.0 for v in scan), (
        "unclamped no-hit samples: %r" % [v for v in scan if not -1.0 <= v <= 1.0][:5])


def test_terrain_normal_stays_upright_when_some_samples_miss():
    """No-hit samples must not corrupt the surface normal. Over flat ground
    with part of the grid off the world edge the normal is still up."""
    import math
    suite, _, _ = _sensors()
    normal = suite.terrain_normal_b(WIDE)
    assert math.sqrt(sum(v * v for v in normal)) == pytest.approx(1.0, abs=1e-6)
    assert normal[2] > 0.9, "normal tipped over by off-terrain samples: %r" % normal


def test_flat_ground_grid_fully_on_terrain_is_exactly_uniform():
    """Tight grid, entirely on flat ground: every sample is the base height."""
    tight = {"kind": "grid", "x_range": [-0.1, 0.1], "y_range": [-0.1, 0.1],
             "resolution": 0.1}
    suite, _, _ = _sensors()
    scan = suite.height_scan(tight)
    assert max(scan) - min(scan) < 1e-6
    normal = suite.terrain_normal_b(tight)
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_elevation_map_on_flat_ground_is_the_ground_plane():
    tight = {"kind": "grid", "x_range": [-0.1, 0.1], "y_range": [-0.1, 0.1],
             "resolution": 0.1}
    suite, _, _ = _sensors()
    assert suite.elevation_map(tight) == pytest.approx([0.0] * 9, abs=1e-6)


# ------------------------------------------------- values reach the policy

def test_height_scan_tracks_terrain_as_the_robot_moves():
    """The decisive check that the backend really raycasts.

    On a slope, clearance must FALL as the ground RISES. Constant values would
    mean the scan is a placeholder rather than a measurement.
    """
    pytest.importorskip("mujoco")
    import mujoco, os, tempfile
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.backends.mujoco_sensors import MujocoSensorSuite
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain

    profile = inspect_mjcf(str(ROBOT))
    world = generate_terrain("rs3", profile, 0.4, 7)          # sloped
    scene = os.path.join(tempfile.mkdtemp(), "scene.xml")
    build_mjcf_scene(str(ROBOT), world, scene, 0.005)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    suite = MujocoSensorSuite(model, data, profile)
    geometry = {"kind": "grid", "x_range": [-0.1, 0.1], "y_range": [-0.1, 0.1],
                "resolution": 0.1}

    origin = data.qpos.copy()
    clearances, elevations = [], []
    for x in (0.5, 1.2, 1.9, 2.6):
        data.qpos[:] = origin
        data.qpos[0] = x
        mujoco.mj_forward(model, data)
        clearances.append(suite.height_scan(geometry)[0])
        elevations.append(suite.elevation_map(geometry)[0])

    assert len(set(round(v, 4) for v in clearances)) > 1, \
        "scan never changes with position: %r" % clearances
    assert elevations[-1] > elevations[0], "ground should rise along the slope"
    assert clearances[-1] < clearances[0], "clearance should fall as ground rises"


def test_compiled_observation_carries_real_scan_values():
    """The scan must survive compilation into the vector the policy receives."""
    suite, _, _ = _sensors()
    from robustness_lab.observation.compiler import compile_terms
    geometry = {"kind": "grid", "x_range": [-0.1, 0.1], "y_range": [-0.1, 0.1],
                "resolution": 0.1}
    plan = compile_terms([
        {"source": "projected_gravity_b", "size": 3},
        {"source": "height_scan", "size": 9, "geometry": geometry},
    ])
    built = plan.build({"projected_gravity_b": [0.0, 0.0, -1.0],
                        "height_scan": suite.height_scan(geometry)})
    assert len(built) == 12
    assert not all(abs(v) < 1e-9 for v in built[-9:]), "scan arrived as zeros"
