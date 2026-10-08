"""The mud model must reach EVERY world that contains marsh, not just RS11 alone.

Marsh geometry is emitted in exactly one place (worlds/catalog.py, the rs11
branch), but it reaches worlds by several routes: RS11 on its own, RS11 inside
an AE challenge pair, and RS11 inside composed graph worlds. A fix applied to
only one of those routes would leave mud behaving differently depending on which
level you were on -- the same defect in a new shape.

Worlds already materialised keep whatever tier is recorded in their JSON, so old
runs stay interpretable rather than being silently re-interpreted under new
physics. That is what the legacy-tier test below pins.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"


def _profile():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _marsh_geometry(world):
    return [item for item in world.geometry if item.semantic == "marsh"]


REQUIRED = (
    "support_stress_pa",
    "depth_exponent",
    "reference_depth_m",
    "yield_stress_pa",
)


def _assert_carries_mud_parameters(world, where):
    from robustness_lab.environment_physics import DEPTH_RESISTIVE_TIER

    patches = _marsh_geometry(world)
    assert patches, "no marsh geometry in %s" % where
    for patch in patches:
        for key in REQUIRED:
            assert key in patch.parameters, (
                "%s: marsh patch %s is missing %s" % (where, patch.name, key))
            assert float(patch.parameters[key]) > 0.0
        assert patch.parameters.get("capability_tier") == DEPTH_RESISTIVE_TIER, where


def test_standalone_rs11_carries_the_mud_parameters():
    from robustness_lab.worlds import generate_terrain
    _assert_carries_mud_parameters(
        generate_terrain("rs11", _profile(), 0.5, 7), "rs11")


def test_every_difficulty_carries_them():
    from robustness_lab.worlds import generate_terrain
    for difficulty in (0.0, 0.25, 0.5, 0.75, 1.0):
        _assert_carries_mud_parameters(
            generate_terrain("rs11", _profile(), difficulty, 3),
            "rs11 at difficulty %.2f" % difficulty)


def test_rs11_inside_an_ae_pair_carries_them():
    """The AE pair route must not get different mud from the standalone route."""
    from robustness_lab.worlds.compose import generate_ae
    _assert_carries_mud_parameters(
        generate_ae(_profile(), "rs11", "rs4", 0.5, 5), "AE pair rs11->rs4")
    _assert_carries_mud_parameters(
        generate_ae(_profile(), "rs4", "rs11", 0.5, 5), "AE pair rs4->rs11")


def test_composed_worlds_carry_them():
    from robustness_lab.worlds.compose import generate_aee
    world = generate_aee(_profile(), "long", 0.5, 600,
                         options={"rows": 1, "columns": 4,
                                  "pair_mode": "cycle", "topology": "chain",
                                  "purple": "rs11", "brown": "rs4"})
    _assert_carries_mud_parameters(world, "composed chain world")


def test_difficulty_still_makes_mud_harder():
    """Softer, deeper, grabbier as difficulty rises -- monotonic, not arbitrary."""
    from robustness_lab.worlds import generate_terrain
    easy = _marsh_geometry(generate_terrain("rs11", _profile(), 0.0, 3))[0]
    hard = _marsh_geometry(generate_terrain("rs11", _profile(), 1.0, 3))[0]
    assert float(hard.parameters["yield_stress_pa"]) > float(
        easy.parameters["yield_stress_pa"]), "harder mud must grab harder"
    assert float(hard.parameters["support_stress_pa"]) < float(
        easy.parameters["support_stress_pa"]), "harder mud must be softer underfoot"


# ---------------------------------------------------------------------------
# The physics plan
# ---------------------------------------------------------------------------

def test_the_plan_requires_the_depth_resistive_effect():
    from robustness_lab.environment_physics import (
        DEPTH_RESISTIVE_EFFECT, world_physics_plan)
    from robustness_lab.worlds import generate_terrain

    plan = world_physics_plan(generate_terrain("rs11", _profile(), 0.5, 7))
    assert DEPTH_RESISTIVE_EFFECT in plan["required_effects"]


def test_a_legacy_world_keeps_its_recorded_tier():
    """Already-materialised worlds must not be re-interpreted under new physics."""
    from robustness_lab.environment_physics import (
        RIGID_VISCOUS_TIER, world_physics_plan)
    from robustness_lab.worlds import generate_terrain

    world = generate_terrain("rs11", _profile(), 0.5, 7)
    world.resolved_parameters["capability_tier"] = RIGID_VISCOUS_TIER
    for patch in _marsh_geometry(world):
        patch.parameters["capability_tier"] = RIGID_VISCOUS_TIER
        patch.parameters.setdefault("viscous_drag", 1.0)
    plan = world_physics_plan(world)
    assert plan["requested_tier"] == RIGID_VISCOUS_TIER
    assert "linear_velocity_drag" in plan["required_effects"]


def test_mujoco_declares_the_new_effect():
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.environment_physics import DEPTH_RESISTIVE_EFFECT
    assert DEPTH_RESISTIVE_EFFECT in MujocoBackend.probe().environment_effects


def test_rs11_is_supported_on_mujoco():
    """The whole point: mud must not become 'Unsupported' again."""
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.environment_physics import (
        assess_environment_support, world_physics_plan)
    from robustness_lab.worlds import generate_terrain

    plan = world_physics_plan(generate_terrain("rs11", _profile(), 0.5, 7))
    report = assess_environment_support(
        plan, MujocoBackend.probe().environment_effects)
    assert report["status"] == "environment_supported", report


# ---------------------------------------------------------------------------
# It has to actually change what the simulator does
# ---------------------------------------------------------------------------

def test_the_robot_sinks_into_mud_but_not_into_flat_ground():
    """The behavioural claim. Rigid patches could never do this."""
    pytest.importorskip("mujoco")
    asset = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    if not asset.is_file():
        pytest.skip("go2 MJCF not cached")

    from robustness_lab.backends.mujoco import MujocoBackend
    robot = _profile()
    state = MujocoBackend().reset_canonical_state(robot, str(asset), "mjcf")
    flat_height = float(state.base_position_w[2])

    from robustness_lab.io import dump_json
    import tempfile
    from robustness_lab.models import EpisodeSpec
    from robustness_lab.worlds import generate_terrain

    world = generate_terrain("rs11", robot, 0.8, 7)
    with tempfile.TemporaryDirectory() as tmp:
        profile_path = Path(tmp) / "robot.json"
        world_path = Path(tmp) / "world.json"
        dump_json(str(profile_path), robot)
        dump_json(str(world_path), world)
        # Start the robot ON the marsh. A stationary controller cannot walk
        # there, and moving the spawn is a property of THIS test -- RS11 itself
        # keeps its approach run (see tests/test_mud_scaling.py).
        marsh_region = [r for r in world.regions if r.kind == "marsh"][0]
        world.spawn_pose.position[0] = float(marsh_region.minimum[0]) + 0.05
        dump_json(str(world_path), world)
        episode = EpisodeSpec(
            episode_id="mud-sink", backend="mujoco",
            robot_profile=str(profile_path), robot_asset=str(asset),
            robot_format="mjcf", world=str(world_path),
            controller={"type": "pd_hold"}, output_dir=str(Path(tmp) / "run"),
            seed=3, timeout_s=4.0,
        )
        log = MujocoBackend().run(episode, robot, world)

    marsh = [region for region in world.regions if region.kind == "marsh"][0]
    over_mud = [
        float(sample.base_position_m[2]) for sample in log.samples
        if marsh.minimum[0] <= float(sample.base_position_m[0]) <= marsh.maximum[0]
    ]
    assert over_mud, "the robot never reached the marsh"
    assert min(over_mud) < flat_height - 0.005, (
        "the robot did not sink into the mud: lowest %.4f m vs flat %.4f m"
        % (min(over_mud), flat_height))
