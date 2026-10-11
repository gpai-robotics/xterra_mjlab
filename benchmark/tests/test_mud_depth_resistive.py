"""RS11 mud must resist by DEPTH, not by speed, and must grab on the way out.

The shipped model was one line in the MuJoCo adapter:

    xfrc_applied[body, :3] -= marsh_drag * body_mass * cvel[body, 3:6]

a linear drag proportional to VELOCITY, applied to whole bodies whose origin sat
inside an axis-aligned box, over rigid patches that never moved. Three things
follow, and all three are why it does not feel like mud:

  * nothing sinks -- the patches are rigid boxes;
  * the resistance vanishes exactly when it should peak, because a foot in
    stance is nearly stationary. The literature is explicit that at low speed
    "frictional hydrostatic forces dominate, while the inertial hydrodynamic
    response of the material dominates at high speeds" (Ortiz-Jimenez et al.,
    Front. Robot. AI 2023);
  * pulling a foot out is exactly as easy as pushing it in, when suction is
    mud's signature. Jiang et al. (arXiv:2403.02617) measure peak suction of
    6 N at 25% water content, rising to 30 N at 15%.

The replacement is the reduced-order visco-elasto-plastic law of
arXiv:2403.02617: a rate-INDEPENDENT depth term alpha*(z/H)^beta, a damping
term, and a yield-capped suction that only acts while the foot withdraws.

Golden values below are hand-derived from that law and cannot be produced by a
stub. With alpha = 5e4 Pa, A = 1e-3 m^2:

    z = H         -> 5e4 * 1.0   * 1e-3  =  50.0 N
    z = H/4, b=.5 -> 5e4 * 0.5   * 1e-3  =  25.0 N
    one tau of suction, sigma_y = 6 kPa  -> -6.0 * (1 - e^-1) = -3.792723352971346 N
"""

import math
import pytest


ALPHA = 5.0e4          # Pa      support stress at the reference depth
BETA = 0.5             # -       depth exponent (paper range 0.38-0.56)
REF_DEPTH = 0.04       # m       H, reference/normalising depth
YIELD = 6.0e3          # Pa      sigma_y, suction cap (paper: 2-17 kPa)
TAU = 0.05             # s       suction build time constant
AREA = 1.0e-3          # m^2     contact patch


def _params(**overrides):
    from robustness_lab.environment_physics import MudParameters
    values = dict(
        support_stress_pa=ALPHA,
        depth_exponent=BETA,
        reference_depth_m=REF_DEPTH,
        damping_pa_s_per_m=0.0,
        yield_stress_pa=YIELD,
        suction_time_constant_s=TAU,
        max_force_n=1.0e4,
    )
    values.update(overrides)
    return MudParameters(**values)


def _contact(**overrides):
    from robustness_lab.environment_physics import MudFootContact
    return MudFootContact(_params(**overrides))


# ---------------------------------------------------------------------------
# Depth, not speed
# ---------------------------------------------------------------------------

def test_no_penetration_means_no_force():
    assert _contact().step(0.0, 0.0, AREA, 0.005) == 0.0
    assert _contact().step(-0.01, 0.0, AREA, 0.005) == 0.0


def test_support_at_the_reference_depth():
    force = _contact().step(REF_DEPTH, 0.0, AREA, 0.005)
    assert force == pytest.approx(50.0, abs=1e-9)


def test_support_follows_the_depth_power_law():
    """Quarter depth at beta = 0.5 is half the force, not a quarter."""
    force = _contact().step(REF_DEPTH / 4.0, 0.0, AREA, 0.005)
    assert force == pytest.approx(25.0, abs=1e-9)


def test_support_is_rate_independent():
    """THE regression test against the old model.

    Sinking twice as fast must not change the support force at all. The old
    law was proportional to velocity, so it would double.
    """
    slow = _contact().step(REF_DEPTH, -0.1, AREA, 0.005)
    fast = _contact().step(REF_DEPTH, -0.2, AREA, 0.005)
    assert slow == pytest.approx(fast, abs=1e-12)
    assert slow == pytest.approx(50.0, abs=1e-9)


def test_a_stationary_foot_still_carries_load():
    """A foot in stance has near-zero velocity; mud must still hold it up."""
    assert _contact().step(REF_DEPTH, 0.0, AREA, 0.005) > 40.0


def test_damping_opposes_motion_when_declared():
    contact = _contact(damping_pa_s_per_m=1.0e4)
    force = contact.step(REF_DEPTH, -0.1, AREA, 0.005)
    # support 50.0 + damping (-1e4 * -0.1 * 1e-3) = +1.0
    assert force == pytest.approx(51.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Suction: mud grabs on the way out
# ---------------------------------------------------------------------------

def test_suction_builds_over_one_time_constant():
    contact = _contact(support_stress_pa=0.0)
    force = contact.step(REF_DEPTH, +0.1, AREA, TAU)
    assert force == pytest.approx(-3.792723352971346, abs=1e-9)


def test_suction_pulls_downward_and_never_exceeds_the_yield_cap():
    contact = _contact(support_stress_pa=0.0)
    force = 0.0
    for _ in range(400):
        force = contact.step(REF_DEPTH, +0.1, AREA, 0.005)
    assert force < 0.0, "suction must oppose withdrawal"
    assert abs(force) <= YIELD * AREA + 1e-9, "suction exceeded the yield cap"
    assert abs(force) == pytest.approx(YIELD * AREA, abs=1e-3)


def test_insertion_and_withdrawal_are_not_symmetric():
    """The old drag made pulling out exactly as easy as pushing in."""
    inserting = _contact().step(REF_DEPTH, -0.1, AREA, 0.005)
    withdrawing = _contact().step(REF_DEPTH, +0.1, AREA, 0.005)
    assert inserting != pytest.approx(withdrawing, abs=1e-6)
    assert inserting > withdrawing


def test_suction_releases_once_the_foot_clears_the_surface():
    """A foot that is out must be free, or the robot gets trapped."""
    contact = _contact(support_stress_pa=0.0)
    for _ in range(200):
        contact.step(REF_DEPTH, +0.1, AREA, 0.005)
    assert contact.step(0.0, +0.1, AREA, 0.005) == 0.0
    assert contact.step(-0.02, +0.1, AREA, 0.005) == 0.0


def test_reset_clears_suction_state():
    contact = _contact(support_stress_pa=0.0)
    for _ in range(200):
        contact.step(REF_DEPTH, +0.1, AREA, 0.005)
    contact.reset()
    assert contact.step(REF_DEPTH, +0.1, AREA, TAU) == pytest.approx(
        -3.792723352971346, abs=1e-9)


# ---------------------------------------------------------------------------
# Safety: this must never be the thing that breaks a run
# ---------------------------------------------------------------------------

def test_force_is_clamped():
    contact = _contact(max_force_n=25.0)
    assert contact.step(10.0, 0.0, AREA, 0.005) == pytest.approx(25.0, abs=1e-9)


def test_force_is_always_finite():
    contact = _contact()
    for depth in (0.0, 1e-9, 0.005, 0.04, 5.0):
        for velocity in (-5.0, 0.0, 5.0):
            assert math.isfinite(contact.step(depth, velocity, AREA, 0.005))


def test_a_zero_area_contact_produces_no_force():
    assert _contact().step(REF_DEPTH, 0.0, 0.0, 0.005) == 0.0


def test_non_positive_timestep_is_rejected():
    with pytest.raises(ValueError):
        _contact().step(REF_DEPTH, 0.0, AREA, 0.0)


# ---------------------------------------------------------------------------
# It has to hold up an actual robot
# ---------------------------------------------------------------------------

def test_four_feet_support_a_go2_at_half_the_reference_depth():
    """Sinks a couple of centimetres and then holds -- not through, not on top.

    Go2 foot patch about 1.5e-3 m^2 at that depth; robot weight about 147 N.
    """
    contact = _contact()
    per_foot = contact.step(REF_DEPTH / 2.0, 0.0, 1.5e-3, 0.005)
    assert 4.0 * per_foot > 15.0 * 9.81, "the robot would sink straight through"
    assert 4.0 * per_foot < 10.0 * 15.0 * 9.81, "the mud is behaving like concrete"
