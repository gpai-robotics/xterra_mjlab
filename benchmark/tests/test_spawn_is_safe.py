"""The robot must spawn ON the ground, with floor under every foot, on every world.

Two defects made spawning unsafe, and they compound on the banked/curved L1
variants, which is where the rear legs were seen dropping at t=0:

1. `build_mjcf_scene` set only the free body's `pos` from the world's
   spawn_pose. The spawn ORIENTATION was discarded. On a 12 degree cambered
   floor the robot therefore spawned level: one side's feet about 3 cm in the
   air, the other side's about 3 cm inside the floor, and the first thing the
   physics does is shove it sideways.

2. There is no ground plane; the approach floor starts at x = 0 and the spawn
   sits 0.6 body-lengths in, so the rear feet land at x = 0.04 with a 0.022 m
   foot radius -- 1.6 cm of margin. Curvature sweeps them to x = 0.02, where the
   foot sphere already overhangs the edge. Any rocking as the policy starts and
   the rear feet slide off the back of the world into nothing.

Neither has anything to do with the policy under test, so neither may be
allowed to cost it the episode.
"""

import math
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
ASSET = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
ALL_RS = ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs9", "rs10", "rs11"]
VARIANTS = ((0.0, 0.0), (-12.0, 65.0), (12.0, -65.0))


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _world(code, camber, curve, difficulty=0.5, seed=5):
    from robustness_lab.worlds import generate_terrain
    from robustness_lab.worlds.modifiers import apply_geometry_modifiers
    world = generate_terrain(code, _robot(), difficulty, seed)
    if camber or curve:
        world = apply_geometry_modifiers(world, camber, curve, 28)
    return world


def _feet_at_spawn(world, robot):
    """World-frame xy of the four feet at the spawn pose."""
    x, y, _ = world.spawn_pose.position
    yaw = world.spawn_pose.rpy[2] if world.spawn_pose.rpy else 0.0
    half_l, half_w = robot.stance_length_m / 2.0, robot.stance_width_m / 2.0
    out = {}
    for name, (fx, fy) in {"FL": (half_l, half_w), "FR": (half_l, -half_w),
                           "RL": (-half_l, half_w), "RR": (-half_l, -half_w)}.items():
        out[name] = (x + math.cos(yaw) * fx - math.sin(yaw) * fy,
                     y + math.sin(yaw) * fx + math.cos(yaw) * fy)
    return out


def _supported(world, x, y, foot_radius=0.022):
    """A foot is a disc, not a point: support anywhere within the foot radius
    counts. Adjacent rotated curve segments leave wedge seams of a few
    millimetres on the outside of a bend, which a 22 mm foot bridges; a real
    absence of floor is wider than a foot."""
    offsets = ((0.0, 0.0), (foot_radius, 0.0), (-foot_radius, 0.0),
               (0.0, foot_radius), (0.0, -foot_radius))
    return any(_point_supported(world, x + dx, y + dy) for dx, dy in offsets)


def _point_supported(world, x, y):
    for item in world.geometry:
        px, py, _ = item.pose.position
        sx, sy, _ = item.size
        yaw = item.pose.rpy[2] if item.pose.rpy else 0.0
        dx, dy = x - px, y - py
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if abs(lx) <= sx / 2.0 + 1e-9 and abs(ly) <= sy / 2.0 + 1e-9:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Orientation reaches the simulator
# ---------------------------------------------------------------------------

def test_the_mjcf_carries_the_spawn_orientation(tmp_path):
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")
    from robustness_lab.backends.mujoco import build_mjcf_scene

    world = _world("rs1", -12.0, 65.0)
    assert abs(world.spawn_pose.rpy[0]) > 0.1, "fixture has no roll to carry"
    out = tmp_path / "scene.xml"
    build_mjcf_scene(str(ASSET), world, str(out), 0.005)

    root = ET.parse(out).getroot()
    free = None
    for body in root.find("worldbody").findall("body"):
        if body.find("freejoint") is not None or any(
                j.get("type") == "free" for j in body.findall("joint")):
            free = body
            break
    assert free is not None
    assert free.get("quat") or free.get("euler"), (
        "the free body has a position but no orientation; the camber roll was dropped")


def test_the_spawn_quaternion_matches_the_world_roll(tmp_path):
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")
    import mujoco
    from robustness_lab.backends.mujoco import build_mjcf_scene
    from robustness_lab.canonical_state import quaternion_to_euler_rpy

    world = _world("rs1", -12.0, 65.0)
    out = tmp_path / "scene.xml"
    build_mjcf_scene(str(ASSET), world, str(out), 0.005)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    quat = data.qpos[3:7]                       # free joint: xyz then wxyz
    roll, _, yaw = quaternion_to_euler_rpy(list(quat))
    assert roll == pytest.approx(world.spawn_pose.rpy[0], abs=1e-3)
    assert yaw == pytest.approx(world.spawn_pose.rpy[2], abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Floor under and behind every foot, on every world
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ALL_RS)
@pytest.mark.parametrize("camber,curve", VARIANTS)
def test_every_foot_has_floor_at_spawn(code, camber, curve):
    robot = _robot()
    world = _world(code, camber, curve)
    for name, (x, y) in _feet_at_spawn(world, robot).items():
        assert _supported(world, x, y), "%s %+.0f/%+.0f: foot %s at (%.3f, %.3f) has no floor" % (
            code, camber, curve, name, x, y)


@pytest.mark.parametrize("code", ALL_RS)
@pytest.mark.parametrize("camber,curve", VARIANTS)
def test_there_is_a_body_length_of_floor_behind_the_rear_feet(code, camber, curve):
    """Rocking backwards at start-up must not step off the end of the world."""
    robot = _robot()
    world = _world(code, camber, curve)
    feet = _feet_at_spawn(world, robot)
    yaw = world.spawn_pose.rpy[2] if world.spawn_pose.rpy else 0.0
    back = robot.body_length_m
    for name in ("RL", "RR"):
        x, y = feet[name]
        bx, by = x - math.cos(yaw) * back, y - math.sin(yaw) * back
        assert _supported(world, bx, by), (
            "%s %+.0f/%+.0f: only %s has floor; nothing %.2f m behind %s" % (
                code, camber, curve, name, back, name))


# ---------------------------------------------------------------------------
# 3. The physics agrees
# ---------------------------------------------------------------------------

def _hold_on(world, robot, seconds=0.4):
    import tempfile
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.io import dump_json
    from robustness_lab.models import EpisodeSpec
    with tempfile.TemporaryDirectory() as tmp:
        pp, wp = Path(tmp) / "r.json", Path(tmp) / "w.json"
        dump_json(str(pp), robot)
        dump_json(str(wp), world)
        episode = EpisodeSpec(
            episode_id="spawn", backend="mujoco", robot_profile=str(pp),
            robot_asset=str(ASSET), robot_format="mjcf", world=str(wp),
            controller={"type": "pd_hold"}, output_dir=str(Path(tmp) / "o"),
            seed=3, timeout_s=seconds)
        return MujocoBackend().run(episode, robot, world)


def test_all_four_feet_touch_the_ground_from_the_first_step_on_a_bank():
    """The spawn criterion, measured in the physics.

    With the roll discarded the robot spawned level on a -12 degree floor: at
    t = 0 only the two right feet touched (`.#.#`), the left legs hung in the
    air for two control steps, and the robot dropped onto its side before the
    policy had acted. With the roll applied every foot is in contact from the
    first sample.

    This deliberately does NOT assert where a passive hold controller settles a
    second later -- on a side-slope a PD stance creeps downhill and leans, and
    correcting that is the policy's job, not the spawn's.
    """
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")
    robot = _robot()
    log = _hold_on(_world("rs1", -12.0, 65.0), robot)
    first = log.samples[0]
    assert all(first.foot_contacts.values()), (
        "at t=0 only these feet touched: %s"
        % [k for k, v in first.foot_contacts.items() if v])
    # And it stays down through the settling transient.
    for sample in log.samples[:10]:
        assert all(sample.foot_contacts.values()), (
            "a foot left the ground at t=%.2f: %s" % (
                float(sample.time_s), sample.foot_contacts))


def test_the_level_spawn_really_was_the_defect():
    """Reproduce the pre-fix behaviour on the same world, so the test above is
    known to discriminate rather than pass vacuously."""
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")
    import copy
    robot = _robot()
    world = copy.deepcopy(_world("rs1", -12.0, 65.0))
    world.spawn_pose.rpy = [0.0, 0.0, 0.0]          # what the writer used to do
    log = _hold_on(world, robot)
    assert not all(log.samples[0].foot_contacts.values()), (
        "a level spawn on a banked floor should leave feet in the air")


def test_the_spawn_is_stable_on_every_plain_terrain():
    """No RS terrain may drop a foot at spawn, curved or not."""
    pytest.importorskip("mujoco")
    if not ASSET.is_file():
        pytest.skip("go2 MJCF not cached")
    robot = _robot()
    for code in ALL_RS:
        log = _hold_on(_world(code, 0.0, 0.0), robot, seconds=0.2)
        assert all(log.samples[0].foot_contacts.values()), (
            "%s: feet off the ground at spawn: %s" % (code, log.samples[0].foot_contacts))


# ---------------------------------------------------------------------------
# 4. Composed worlds build their own approach and need the same apron
# ---------------------------------------------------------------------------

def _composed_worlds():
    from robustness_lab.worlds.compose import generate_ae, generate_aee
    robot = _robot()
    yield "AE rs1->rs5", generate_ae(robot, "rs1", "rs5", 0.5, 5)
    yield "AE rs9->rs10", generate_ae(robot, "rs9", "rs10", 0.6, 13)
    yield "AE reversed", generate_ae(robot, "rs1", "rs5", 0.5, 5, True)
    yield "chain x4", generate_aee(robot, "long", 0.5, 600,
                                   options={"rows": 1, "columns": 4,
                                            "pair_mode": "cycle", "topology": "chain"})
    yield "grid 3x3", generate_aee(robot, "long", 0.5, 601,
                                   options={"rows": 3, "columns": 3,
                                            "pair_mode": "cycle", "topology": "grid"})


def test_composed_worlds_have_floor_under_every_foot_at_spawn():
    robot = _robot()
    for label, world in _composed_worlds():
        for name, (x, y) in _feet_at_spawn(world, robot).items():
            assert _supported(world, x, y), "%s: foot %s has no floor at spawn" % (label, name)


def test_composed_worlds_have_a_body_length_of_floor_behind_the_rear_feet():
    robot = _robot()
    for label, world in _composed_worlds():
        feet = _feet_at_spawn(world, robot)
        yaw = world.spawn_pose.rpy[2] if world.spawn_pose.rpy else 0.0
        for name in ("RL", "RR"):
            x, y = feet[name]
            bx, by = x - math.cos(yaw) * robot.body_length_m, y - math.sin(yaw) * robot.body_length_m
            assert _supported(world, bx, by), (
                "%s: nothing one body-length behind %s" % (label, name))
