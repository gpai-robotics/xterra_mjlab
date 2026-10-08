"""RS1-RS11 difficulty scaling, redesigned to be intuitive (implementation_plan.md).

Each difficulty knob maps to a clear physical quantity, and every terrain scales
a DIFFERENT axis so the eleven tests stay distinct:

  RS1 slippery flat   friction only (materials); geometry fixed
  RS2 rough flat      bump height   = foot_length * difficulty
  RS3 slope           angle         = 45 deg * difficulty
  RS4 rough slope     angle 45*d, roughness = foot_length * difficulty
  RS5 stairs          step COUNT    = 3 + round(10*d)  [6/8/11]; rise fixed
  RS6 waves           FREQUENCY     = 1 + round(4*d) full waves [2/3/4]; amp fixed
  RS7 curved hill     JAGGEDNESS    = 24 - round(20*d) segments [19/14/9]; amp fixed
  RS8 small gaps      gap width     = foot_length * 2*d; five gaps fixed
  RS9 large gap       gap width     = body_length * difficulty
  RS10 high step      rise          = limb_reach * difficulty
  RS11 mud            softer + stickier with difficulty (unchanged model)

Counts use round-half-up (int(x + 0.5)) so difficulty 0.25 gives the plan's
stated 6 steps / 2 waves, not Python's banker's-rounded 5 / 1.

Lengths that used to grow with difficulty are fixed at 5 body lengths, so a
terrain's difficulty changes its named axis and nothing else.
"""

import math
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
EASY, MEDIUM, HARD = 0.25, 0.50, 0.75


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _world(code, difficulty):
    from robustness_lab.worlds import generate_terrain
    return generate_terrain(code, _robot(), difficulty, 7)


def _res(code, difficulty):
    return _world(code, difficulty).resolved_parameters


def _scale():
    r = _robot()
    return {"body": r.body_length_m, "foot": r.foot_size_m[0],
            "stand": r.standing_height_m, "reach": r.limb_reach_m}


# --- RS2 rough flat: bump height = foot_length * difficulty -----------------

@pytest.mark.parametrize("d", [EASY, MEDIUM, HARD])
def test_rs2_bump_height_is_foot_length_times_difficulty(d):
    s = _scale()
    assert _res("rs2", d)["roughness_amplitude_m"] == pytest.approx(s["foot"] * d, rel=1e-6)


def test_rs2_length_is_fixed_at_five_body_lengths():
    s = _scale()
    for d in (EASY, MEDIUM, HARD):
        assert _res("rs2", d)["length_m"] == pytest.approx(s["body"] * 5.0, rel=1e-6)


# --- RS3 / RS4 slope = 45 deg * difficulty ----------------------------------

@pytest.mark.parametrize("code", ["rs3", "rs4"])
@pytest.mark.parametrize("d", [EASY, MEDIUM, HARD])
def test_slope_angle_is_45_degrees_times_difficulty(code, d):
    assert math.degrees(_res(code, d)["slope_rad"]) == pytest.approx(45.0 * d, rel=1e-6)


def test_rs4_roughness_is_foot_length_times_difficulty():
    s = _scale()
    for d in (EASY, MEDIUM, HARD):
        assert _res("rs4", d)["roughness_amplitude_m"] == pytest.approx(s["foot"] * d, rel=1e-6)


# --- RS5 stairs: more steps, fixed rise -------------------------------------

@pytest.mark.parametrize("d,steps", [(EASY, 6), (MEDIUM, 8), (HARD, 11)])
def test_rs5_step_count_grows_with_difficulty(d, steps):
    assert _res("rs5", d)["step_count"] == steps


def test_rs5_step_rise_is_fixed_at_40_percent_standing_height():
    s = _scale()
    for d in (EASY, MEDIUM, HARD):
        assert _res("rs5", d)["step_rise_m"] == pytest.approx(s["stand"] * 0.4, rel=1e-6)


# --- RS6 waves: frequency, fixed amplitude ----------------------------------

@pytest.mark.parametrize("d,waves", [(EASY, 2), (MEDIUM, 3), (HARD, 4)])
def test_rs6_wave_count_grows_with_difficulty(d, waves):
    assert _res("rs6", d)["wave_count"] == waves


def test_rs6_amplitude_and_length_are_fixed():
    s = _scale()
    for d in (EASY, MEDIUM, HARD):
        r = _res("rs6", d)
        assert r["amplitude_m"] == pytest.approx(s["stand"] * 0.3, rel=1e-6)
        assert r["length_m"] == pytest.approx(s["body"] * 5.0, rel=1e-6)


# --- RS7 curved hill: jaggedness, fixed amplitude ---------------------------

@pytest.mark.parametrize("d,segs", [(EASY, 19), (MEDIUM, 14), (HARD, 9)])
def test_rs7_gets_more_jagged_with_difficulty(d, segs):
    assert _res("rs7", d)["segments"] == segs


def test_rs7_amplitude_is_fixed_at_70_percent_standing_height():
    s = _scale()
    for d in (EASY, MEDIUM, HARD):
        assert _res("rs7", d)["amplitude_m"] == pytest.approx(s["stand"] * 0.7, rel=1e-6)


# --- RS8 small gaps: five gaps, width scales --------------------------------

def test_rs8_has_five_gaps_at_every_difficulty():
    for d in (EASY, MEDIUM, HARD):
        assert _res("rs8", d)["gap_count"] == 5


@pytest.mark.parametrize("d", [EASY, MEDIUM, HARD])
def test_rs8_gap_is_two_foot_lengths_times_difficulty(d):
    s = _scale()
    assert _res("rs8", d)["gap_m"] == pytest.approx(s["foot"] * 2.0 * d, rel=1e-6)


# --- RS9 large gap = body_length * difficulty -------------------------------

@pytest.mark.parametrize("d", [EASY, MEDIUM, HARD])
def test_rs9_gap_is_body_length_times_difficulty(d):
    s = _scale()
    assert _res("rs9", d)["gap_m"] == pytest.approx(s["body"] * d, rel=1e-6)


# --- RS10 high step = limb_reach * difficulty -------------------------------

@pytest.mark.parametrize("d", [EASY, MEDIUM, HARD])
def test_rs10_rise_is_limb_reach_times_difficulty(d):
    s = _scale()
    assert _res("rs10", d)["rise_m"] == pytest.approx(s["reach"] * d, rel=1e-6)


# --- Cross-cutting: harder is monotonically harder, one axis at a time -------

def test_every_terrain_still_generates_at_the_three_difficulties():
    for code in ["rs%d" % i for i in range(1, 12)]:
        for d in (EASY, MEDIUM, HARD):
            _world(code, d).validate()


def test_rs11_still_gets_softer_and_stickier():
    easy, hard = _res("rs11", EASY), _res("rs11", HARD)
    assert hard["support_stress_pa"] < easy["support_stress_pa"]
    assert hard["yield_stress_pa"] > easy["yield_stress_pa"]
