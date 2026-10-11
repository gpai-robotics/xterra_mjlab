"""Cambered and curved single terrains belong at L1, next to the terrains they vary.

The shipped gallery (configs/galleries/go2_curvature_camber.yaml) is "Universal
Camber and Curvature -- RS1-RS11": each single terrain, banked and curved two
ways. Those are single-terrain worlds, so evaluating them as composed chain
worlds at L7 -- as P51 did -- does not test what the gallery shows.

They are now materialised at L1, after the plain single terrains have all
reached a verdict, and only for the terrains that PASSED. That keeps the same
discipline as every other level -- cost proportional to what the policy earned
-- and means a terrain the policy cannot walk flat is not re-run banked. With 44
of 99 plain conditions passing in a real sweep, that is 88 variant conditions
rather than 198.

They are diagnostic. Pairs and later levels stay built from the plain primitives
(`_materialize_pairs` filters on kind == "primitive"), so a variant result never
changes what L2 is assembled from.
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


def _state():
    """L0 passed; six L1 primitives finished, four passed and two failed."""
    from robustness_lab.discrete_examiner import _condition

    conditions, order = {}, []

    def add(item):
        conditions[item["id"]] = item
        order.append(item["id"])

    q = _condition("L0-flat", "L0", "qualification", 1)
    q["status"] = "passed"
    add(q)
    spec = [("rs1", "passed"), ("rs2", "passed"), ("rs5", "failed"),
            ("rs6", "passed"), ("rs7", "failed"), ("rs10", "passed")]
    for index, (terrain, status) in enumerate(spec):
        item = _condition("L1-%s-g050-f050" % terrain, "L1", "primitive", 10 + index,
                          terrain=terrain, geometry_level=0.5, geometry_label="medium",
                          friction_label="moderate", static_friction=0.5,
                          dynamic_friction=0.4)
        item["status"] = status
        add(item)
    return {
        "conditions": conditions, "condition_order": order,
        "materialized": {"L1": True, "L2": False, "L3": False, "L4": False,
                         "L5": False, "L6": False, "L7": False},
        "level_results": {}, "status": "running", "stage": "L1-single-terrains",
        "settings": {"seed": 7, "terrain_codes": ["rs1", "rs2", "rs5", "rs6", "rs7", "rs10"],
                     "screening_repetitions": 3, "confirmation_repetitions": 10,
                     "confirmation_successes": 7, "qualification_friction_label": "moderate",
                     "fixed_pair_static_friction": 0.5, "fixed_pair_dynamic_friction": 0.4,
                     "geometry_levels": [{"label": "medium", "value": 0.5}],
                     "friction_levels": [{"label": "moderate", "static": 0.5, "dynamic": 0.4}]},
        "trials": [],
    }


def _variants(state):
    return [c for c in state["conditions"].values() if c["kind"] == "variant"]


def _advance(state):
    from robustness_lab.discrete_examiner import _next_condition
    return _next_condition(state, _robot())


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_variants_are_created_once_the_plain_terrains_are_decided():
    state = _state()
    _advance(state)
    assert _variants(state), "no camber/curvature variants were materialised"


def test_variants_live_at_l1():
    state = _state()
    _advance(state)
    assert {c["level"] for c in _variants(state)} == {"L1"}


def test_only_terrains_that_passed_get_variants():
    state = _state()
    _advance(state)
    terrains = {c["terrain"] for c in _variants(state)}
    assert terrains == {"rs1", "rs2", "rs6", "rs10"}, terrains
    assert "rs5" not in terrains and "rs7" not in terrains


def test_two_gallery_variants_per_passed_terrain():
    """Left-curve/left-low and right-curve/right-low, as the gallery ships."""
    state = _state()
    _advance(state)
    assert _variants(state)
    by_terrain = {}
    for c in _variants(state):
        by_terrain.setdefault(c["terrain"], []).append(c["modifier"])
    for terrain, modifiers in by_terrain.items():
        assert len(modifiers) == 2, (terrain, modifiers)
        cambers = sorted(float(m["camber_deg"]) for m in modifiers)
        curves = sorted(float(m["curve_angle_deg"]) for m in modifiers)
        assert cambers == pytest.approx([-12.0, 12.0]), (terrain, cambers)
        assert curves == pytest.approx([-65.0, 65.0]), (terrain, curves)


def test_a_variant_records_which_terrain_it_varies():
    state = _state()
    _advance(state)
    assert _variants(state)
    for c in _variants(state):
        assert c.get("parents"), "%s does not name its plain parent" % c["id"]
        assert state["conditions"][c["parents"][0]]["kind"] == "primitive"


def test_variants_are_materialised_only_once():
    state = _state()
    _advance(state)
    first = len(_variants(state))
    assert first > 0
    _advance(state)
    assert len(_variants(state)) == first


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def test_variants_are_the_next_thing_to_run():
    state = _state()
    nxt = _advance(state)
    assert nxt is not None and nxt["kind"] == "variant", nxt and nxt["id"]


def test_pairs_wait_until_the_variants_are_decided():
    state = _state()
    _advance(state)
    assert state["materialized"]["L2"] is False, "L2 was built while variants were pending"


def test_pairs_are_still_built_from_plain_primitives_only():
    state = _state()
    _advance(state)
    for c in _variants(state):
        c["status"] = "failed"          # even if every variant fails ...
    _advance(state)
    assert state["materialized"]["L2"] is True
    pairs = [c for c in state["conditions"].values() if c["kind"] == "pair"]
    assert pairs, "no pairs were built"
    for pair in pairs:
        for parent in pair["parents"]:
            assert state["conditions"][parent]["kind"] == "primitive"


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

def test_a_variant_builds_a_genuinely_modified_single_terrain():
    from robustness_lab.discrete_examiner import _build_world
    state = _state()
    _advance(state)
    variant = _variants(state)[0]
    plain = state["conditions"][variant["parents"][0]]
    robot = _robot()
    a = _build_world(plain, robot, state["settings"])
    b = _build_world(variant, robot, state["settings"])
    assert [g.pose.position for g in a.geometry] != [g.pose.position for g in b.geometry]
    assert b.metadata.get("geometry_modifier", {}).get("camber_deg") == pytest.approx(
        float(variant["modifier"]["camber_deg"]))
