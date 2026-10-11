"""Composed worlds must not add gap-crossing at both ends of every cell.

`compose.transition()` laid five platforms of `platform * 0.65` on a `platform`
pitch, so 35% of each entry and exit transition was a hole -- five ~3.5 cm gaps
before the challenge and five after it, on a region kind literally named
`rough_friction_rs8`.

Both `generate_aee` and the graph-world builder compose their cells from
`generate_ae`, so every cell carried both transitions: a 12-cell grid meant 24
gap transitions, 120 holes, none of them the thing the level was measuring. A
pair world is supposed to test the ordered pair of challenges, not add a
small-gap course at each end.

This removes the holes and keeps the transitions: same length, same rough
low-friction material, continuous support.

The guard that matters is the last one. RS8 and RS9 ARE gap terrains and their
gaps must survive untouched -- removing those would delete two of the eleven
challenges.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
STEP = 0.005


def _profile():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _supported(world, x, y=0.0):
    """True when some geometry spans (x, y) from above the floor plane."""
    for item in world.geometry:
        px, py, _ = item.pose.position
        sx, sy, _ = item.size
        if abs(x - px) <= sx / 2.0 + 1e-9 and abs(y - py) <= sy / 2.0 + 1e-9:
            return True
    return False


def _unsupported_spans(world, x_min, x_max):
    spans = []
    run = None
    x = x_min
    while x <= x_max:
        if _supported(world, x):
            if run is not None:
                spans.append(run)
                run = None
        else:
            run = [x, x] if run is None else [run[0], x]
        x += STEP
    if run is not None:
        spans.append(run)
    return spans


def _transition_regions(world):
    return [r for r in world.regions if r.name.endswith("-transition")]


def _assert_transitions_are_continuous(world, where):
    regions = _transition_regions(world)
    assert regions, "no transition regions in %s" % where
    for region in regions:
        holes = _unsupported_spans(
            world, float(region.minimum[0]), float(region.maximum[0]))
        assert not holes, (
            "%s: %s still has unsupported spans %s"
            % (where, region.name, [[round(a, 3), round(b, 3)] for a, b in holes]))


# ---------------------------------------------------------------------------
# The transitions themselves
# ---------------------------------------------------------------------------

def test_ae_pair_transitions_have_no_holes():
    from robustness_lab.worlds.compose import generate_ae
    _assert_transitions_are_continuous(
        generate_ae(_profile(), "rs1", "rs5", 0.5, 5), "AE rs1->rs5")


def test_transitions_have_no_holes_at_any_difficulty():
    from robustness_lab.worlds.compose import generate_ae
    for difficulty in (0.0, 0.5, 1.0):
        _assert_transitions_are_continuous(
            generate_ae(_profile(), "rs2", "rs7", difficulty, 11),
            "AE rs2->rs7 at %.1f" % difficulty)


def test_transitions_have_no_holes_for_gap_terrain_pairs():
    """Even when the pair itself is a gap terrain, the transitions stay solid."""
    from robustness_lab.worlds.compose import generate_ae
    _assert_transitions_are_continuous(
        generate_ae(_profile(), "rs9", "rs10", 0.6, 13), "AE rs9->rs10")


def test_composed_worlds_have_no_transition_holes():
    from robustness_lab.worlds.compose import generate_aee
    world = generate_aee(_profile(), "long", 0.5, 600,
                         options={"rows": 1, "columns": 4,
                                  "pair_mode": "cycle", "topology": "chain"})
    _assert_transitions_are_continuous(world, "composed chain world")


def test_grid_worlds_have_no_transition_holes():
    from robustness_lab.worlds.compose import generate_aee
    world = generate_aee(_profile(), "long", 0.5, 601,
                         options={"rows": 3, "columns": 3,
                                  "pair_mode": "cycle", "topology": "grid"})
    _assert_transitions_are_continuous(world, "3x3 grid world")


def test_the_transitions_still_exist_and_keep_their_length():
    """Holes removed, transition preserved -- not deleted."""
    from robustness_lab.worlds.compose import generate_ae
    world = generate_ae(_profile(), "rs1", "rs5", 0.5, 5)
    regions = _transition_regions(world)
    names = sorted(r.name for r in regions)
    assert names == ["entry-transition", "exit-transition"], names
    for region in regions:
        length = float(region.maximum[0]) - float(region.minimum[0])
        assert length > 0.05, "%s collapsed to %.3f m" % (region.name, length)


def test_nothing_still_claims_to_be_a_transition_gap():
    from robustness_lab.worlds.compose import generate_ae
    world = generate_ae(_profile(), "rs1", "rs5", 0.5, 5)
    assert not [g for g in world.geometry if g.semantic == "transition_gap"]


# ---------------------------------------------------------------------------
# The guard: real gap terrains keep their gaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["rs8", "rs9"])
def test_the_real_gap_terrains_still_have_gaps(code):
    """RS8 and RS9 are gap challenges. Removing their gaps would delete them."""
    from robustness_lab.worlds import generate_terrain
    world = generate_terrain(code, _profile(), 0.6, 5)
    challenge = [r for r in world.regions if r.kind not in {"approach", "goal"}]
    assert challenge, "no challenge region in %s" % code
    holes = []
    for region in challenge:
        holes.extend(_unsupported_spans(
            world, float(region.minimum[0]), float(region.maximum[0])))
    assert holes, "%s lost its gaps" % code


def test_a_pair_of_gap_terrains_keeps_the_gaps_inside_its_slots():
    from robustness_lab.worlds.compose import generate_ae
    world = generate_ae(_profile(), "rs9", "rs10", 0.6, 13)
    slots = [r for r in world.regions if r.name.endswith("-slot")]
    assert slots
    holes = []
    for region in slots:
        holes.extend(_unsupported_spans(
            world, float(region.minimum[0]), float(region.maximum[0])))
    assert holes, "the gap terrains lost their gaps inside the pair world"


def test_a_nested_terrains_spawn_floor_never_leaks_into_a_pair_slot():
    """The composer keeps only geometry overlapping the challenge span. A
    nested terrain's own approach/landing floors end exactly AT that boundary
    and slipped through on floating-point equality; once the approach grew an
    apron behind the spawn, the leaked floor reached far enough back to pave
    over the previous slot's gap (RS9's, in rs9->rs10)."""
    from robustness_lab.worlds.compose import generate_ae
    world = generate_ae(_profile(), "rs9", "rs10", 0.6, 13)
    leaked = [g.name for g in world.geometry
              if g.name.endswith("-approach") or g.name.endswith("-landing")]
    assert not leaked, "nested spawn/goal floors were transplanted: %s" % leaked
