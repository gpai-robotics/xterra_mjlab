"""Camber, curvature and latency were built but never evaluated.

`worlds/modifiers.py` produces a genuinely banked/curved world -- its own
docstring says "a spatial sweep: long boxes are segmented so their collision
geometry follows the arc instead of merely rotating as one straight slab", so
this is real geometry, not a render trick. The project ships a 22-world gallery
of it (configs/galleries/go2_curvature_camber.yaml).

But `apply_geometry_modifiers` was called only from gallery.py, from the `world
generate` CLI, and from the separate benchmark.py path. `discrete_examiner`'s
`_build_world` never called it, so no L0-L7 condition was ever cambered or
curved. A 757-trial sweep evaluated none of those 22 worlds.

The same for stressors: the MuJoCo adapter supports four --

    {"push", "push_impulse_ns", "action_latency_s", "perception_latency_s"}

-- and the progression table used exactly one, a push impulse at L6. Control and
perception latency, arguably the more important robustness axes for a deployed
policy, were never exercised.

Latency lives at L6, the stressed level, built from `_passed_pair_library` like
every progression level. Camber and curvature are single-terrain properties and
live at L1 as variants of the terrains that passed (tests/test_l1_camber_variants.py);
the progression-world modifier mechanism tested below remains available.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _settings():
    return {
        "seed": 7,
        "terrain_codes": ["rs1", "rs2", "rs5", "rs6", "rs7"],
        "fixed_pair_static_friction": 0.5,
        "fixed_pair_dynamic_friction": 0.4,
    }


def _templates(level):
    from robustness_lab.discrete_examiner import PROGRESSION
    return [item for item in PROGRESSION if item[0] == level]


# ---------------------------------------------------------------------------
# The templates exist
# ---------------------------------------------------------------------------

def test_camber_and_curvature_are_not_composed_chain_worlds():
    """They were first added here as banked L7 chain worlds. That was the wrong
    shape: the gallery shows banked SINGLE terrains. They now live at L1 as
    variants of the terrains that passed -- see tests/test_l1_camber_variants.py.
    L7 keeps only its locked out-of-distribution world."""
    from robustness_lab.discrete_examiner import progression_modifier
    for template in _templates("L7"):
        assert not progression_modifier(template), template[1]


def test_l6_exercises_control_and_perception_latency():
    kinds = {s.get("type") for t in _templates("L6") for s in t[4]}
    assert "action_latency_s" in kinds, kinds
    assert "perception_latency_s" in kinds, kinds


def test_l6_keeps_its_push_stressor():
    kinds = {s.get("type") for t in _templates("L6") for s in t[4]}
    assert "push_impulse_ns" in kinds


def test_every_declared_stressor_is_one_the_backend_supports():
    from robustness_lab.discrete_examiner import PROGRESSION
    supported = {"push", "push_impulse_ns", "action_latency_s", "perception_latency_s"}
    for item in PROGRESSION:
        for stressor in item[4]:
            assert stressor.get("type") in supported, stressor


def test_the_earlier_levels_are_left_alone():
    """L3 stays plain; this must not silently change what L3 measures."""
    from robustness_lab.discrete_examiner import progression_modifier
    for template in _templates("L3"):
        assert not progression_modifier(template), template[1]
        assert not template[4], "L3 gained a stressor"


# ---------------------------------------------------------------------------
# The world is actually modified, not merely labelled
# ---------------------------------------------------------------------------

def _condition(**values):
    base = {
        "id": "L7-x", "level": "L7", "kind": "progression", "world_seed": 11,
        "topology": "chain", "horizon": "L", "geometry_level": 0.5,
        "geometry_label": "medium", "allowed_pairs": [["rs1", "rs2"]],
        "covered_pairs": [["rs1", "rs2"]], "stressors": [],
    }
    base.update(values)
    return base


def test_a_cambered_condition_builds_a_different_world():
    from robustness_lab.discrete_examiner import _build_world
    robot, settings = _robot(), _settings()
    plain = _build_world(_condition(), robot, settings)
    banked = _build_world(
        _condition(modifier={"camber_deg": -12.0, "curve_angle_deg": 0.0}),
        robot, settings)

    def ys(world):
        return [round(g.pose.position[2], 6) for g in world.geometry]

    assert ys(banked) != ys(plain), "camber changed nothing in the geometry"


def test_a_curved_condition_builds_a_different_world():
    from robustness_lab.discrete_examiner import _build_world
    robot, settings = _robot(), _settings()
    plain = _build_world(_condition(), robot, settings)
    curved = _build_world(
        _condition(modifier={"camber_deg": 0.0, "curve_angle_deg": 65.0}),
        robot, settings)
    assert len(curved.geometry) != len(plain.geometry) or [
        round(g.pose.position[1], 6) for g in curved.geometry
    ] != [round(g.pose.position[1], 6) for g in plain.geometry], (
        "curvature changed nothing in the geometry")


def test_a_condition_without_a_modifier_is_untouched():
    from robustness_lab.discrete_examiner import _build_world
    robot, settings = _robot(), _settings()
    a = _build_world(_condition(), robot, settings)
    b = _build_world(_condition(modifier={}), robot, settings)
    assert [g.pose.position for g in a.geometry] == [g.pose.position for g in b.geometry]


def test_the_modifier_is_recorded_on_the_world():
    """It must be auditable in the run record, not implicit."""
    from robustness_lab.discrete_examiner import _build_world
    world = _build_world(
        _condition(modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0}),
        _robot(), _settings())
    recorded = world.metadata.get("geometry_modifier", {})
    assert float(recorded.get("camber_deg", 0.0)) == pytest.approx(-12.0)
    assert float(recorded.get("curve_angle_deg", 0.0)) == pytest.approx(65.0)
