"""Episode time budgets must come from the route, not from one global constant.

Every episode got `timeout_s: 20` regardless of world size. Measured route
lengths against the route follower's 0.6 m/s:

    rs1            2.78 m ->   4.6 s needed   20 s budget   fine
    AE rs1->rs5    4.48 m ->   7.5 s needed   20 s budget   fine
    chain x4      67.58 m -> 112.6 s needed   20 s budget   IMPOSSIBLE
    grid 3x3      70.22 m -> 117.0 s needed   20 s budget   IMPOSSIBLE

Covering 68 m in 20 s needs 3.4 m/s, well past the 0.8 m/s the reference policy
even declares. So every composed level failed by construction: a flawless policy
scored zero on L3-L7, and the failure was recorded as `timeout` -- indistinguishable
in the report from a policy that genuinely could not walk.

The budget is now route_length / reference_speed * slack, clamped. The floor
means short worlds are generous rather than tight; the ceiling means nothing can
run away. At the measured 167x realtime, a 180 s episode costs about 1.4 s of
wall clock, so the ceiling is nearly free.
"""

import math
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"

MIN_S = 40.0
MAX_S = 180.0
SPEED = 0.6


def _profile():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _route_length(world):
    points = world.routes[0].waypoints
    return sum(math.dist(points[i][:2], points[i + 1][:2])
               for i in range(len(points) - 1))


def _budget(world):
    from robustness_lab.episode_budget import episode_timeout_s
    return episode_timeout_s(world, reference_speed_mps=SPEED, slack=2.0,
                             minimum_s=MIN_S, maximum_s=MAX_S)


def _single(code, difficulty=0.5, seed=5):
    from robustness_lab.worlds import generate_terrain
    return generate_terrain(code, _profile(), difficulty, seed)


def _grid():
    from robustness_lab.worlds.compose import generate_aee
    return generate_aee(_profile(), "long", 0.5, 601,
                        options={"rows": 3, "columns": 3,
                                 "pair_mode": "cycle", "topology": "grid"})


# ---------------------------------------------------------------------------
# Short worlds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["rs1", "rs5", "rs8", "rs11"])
def test_single_rs_terrains_get_the_floor_and_no_more(code):
    """Small worlds get a generous but bounded 40 s."""
    assert _budget(_single(code)) == pytest.approx(MIN_S, abs=1e-9)


def test_an_ae_pair_still_sits_at_the_floor():
    from robustness_lab.worlds.compose import generate_ae
    assert _budget(generate_ae(_profile(), "rs1", "rs5", 0.5, 5)) == pytest.approx(
        MIN_S, abs=1e-9)


def test_the_floor_is_more_generous_than_the_old_flat_budget():
    """A slow or careful policy on RS11 mud now has room it did not have."""
    assert _budget(_single("rs11")) > 20.0


# ---------------------------------------------------------------------------
# Long worlds
# ---------------------------------------------------------------------------

def test_a_composed_world_gets_far_more_than_a_single_terrain():
    assert _budget(_grid()) > 4.0 * _budget(_single("rs1"))


def test_a_composed_world_is_actually_finishable():
    """The regression that matters: the budget must exceed the time needed."""
    world = _grid()
    needed = _route_length(world) / SPEED
    assert _budget(world) > needed, (
        "budget %.1f s cannot cover a %.1f m route needing %.1f s"
        % (_budget(world), _route_length(world), needed))


def test_nothing_exceeds_the_ceiling():
    for world in (_single("rs1"), _grid()):
        assert _budget(world) <= MAX_S + 1e-9


def test_the_ceiling_binds_on_very_long_routes():
    from robustness_lab.worlds.compose import generate_aee
    world = generate_aee(_profile(), "long", 0.5, 602,
                         options={"rows": 4, "columns": 4,
                                  "pair_mode": "cycle", "topology": "grid"})
    assert _budget(world) == pytest.approx(MAX_S, abs=1e-9)


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------

def test_the_budget_scales_with_route_length_between_the_bounds():
    from robustness_lab.episode_budget import episode_timeout_s

    class _Route:
        def __init__(self, points):
            self.waypoints = points

    class _World:
        def __init__(self, points):
            self.routes = [_Route(points)]

    short = _World([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])   # 30 m -> 100 s
    long = _World([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]])    # 45 m -> 150 s
    args = dict(reference_speed_mps=SPEED, slack=2.0, minimum_s=MIN_S, maximum_s=MAX_S)
    assert episode_timeout_s(short, **args) == pytest.approx(100.0, abs=1e-6)
    assert episode_timeout_s(long, **args) == pytest.approx(150.0, abs=1e-6)


def test_invalid_settings_are_rejected():
    from robustness_lab.episode_budget import episode_timeout_s
    world = _single("rs1")
    for bad in (dict(reference_speed_mps=0.0), dict(slack=0.0),
                dict(minimum_s=-1.0), dict(maximum_s=0.0)):
        args = dict(reference_speed_mps=SPEED, slack=2.0,
                    minimum_s=MIN_S, maximum_s=MAX_S)
        args.update(bad)
        with pytest.raises(ValueError):
            episode_timeout_s(world, **args)


def test_a_ceiling_below_the_floor_is_rejected():
    from robustness_lab.episode_budget import episode_timeout_s
    with pytest.raises(ValueError):
        episode_timeout_s(_single("rs1"), reference_speed_mps=SPEED, slack=2.0,
                          minimum_s=100.0, maximum_s=50.0)


# ---------------------------------------------------------------------------
# It has to reach the episode the scheduler actually writes
# ---------------------------------------------------------------------------

def test_the_scheduler_writes_the_scaled_budget_into_the_episode(tmp_path):
    """A budget nothing reads is not a budget."""
    from robustness_lab.discrete_examiner import _prepare_trial
    from robustness_lab.io import load_data
    from robustness_lab.skill import load_skill

    skill_path = ROOT / "submissions/go2_blind/skill.yaml"
    asset = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    if not (skill_path.is_file() and asset.is_file()):
        pytest.skip("go2_blind submission or asset not present")
    skill = load_skill(str(skill_path), check_files=False)

    condition = {
        "id": "L1-rs1", "level": "L1", "kind": "primitive", "terrain": "rs1",
        "world_seed": 5, "attempts": [], "geometry_label": "nominal",
        "geometry_level": 0.5, "static_friction": 0.5, "dynamic_friction": 0.4,
        "stressors": [],
    }
    state = {
        "examiner_id": "budget", "condition_order": ["L1-rs1"],
        "settings": {
            "output_dir": str(tmp_path), "backend": "mujoco", "seed": 1,
            "robot_profile": str(PROFILE), "robot_asset": str(asset),
            "robot_format": "mjcf", "timeout_s": 20.0, "event_thresholds": {},
            "min_timeout_s": MIN_S, "max_timeout_s": MAX_S,
            "timeout_reference_speed_mps": SPEED, "timeout_slack": 2.0,
        },
    }
    prepared = _prepare_trial(state, condition, _profile(), skill)
    episode = load_data(str(prepared["episode_path"]))
    assert episode["timeout_s"] == pytest.approx(MIN_S, abs=1e-6), (
        "scheduler still wrote the flat budget: %r" % episode["timeout_s"])
