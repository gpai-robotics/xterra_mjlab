"""Route CHOICE must be measurable, not just route following.

Every guidance mode hands the policy its path: waypoints give the route, the
steered velocity command follows the route, goal vectors give the endpoint. So
L3-L5 measure sustained traversal over many cells, not decision-making, and
calling them "navigation" oversells them.

A choice test gives the policy a GOAL and the available routes, lets it pick,
and records which one it actually took.
"""

import pytest


def _routes():
    """Two ways round a challenge: through the middle, or a longer detour."""
    return {
        "central": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        "upper-bypass": [[1.0, 0.0, 0.0], [1.0, 0.6, 0.0], [3.0, 0.6, 0.0], [3.0, 0.0, 0.0]],
    }


def test_route_choice_is_a_declared_navigation_scope():
    from robustness_lab.navigation import NAVIGATION_SCOPES
    assert "route_choice" in NAVIGATION_SCOPES
    assert "fixed_route" in NAVIGATION_SCOPES


def test_classify_which_route_a_trajectory_took():
    from robustness_lab.navigation import classify_route
    straight = [[1.0, 0.0], [2.0, 0.02], [3.0, 0.0]]
    assert classify_route(straight, _routes()) == "central"


def test_a_detour_trajectory_is_classified_as_the_bypass():
    from robustness_lab.navigation import classify_route
    detour = [[1.0, 0.0], [1.0, 0.58], [2.0, 0.61], [3.0, 0.55], [3.0, 0.0]]
    assert classify_route(detour, _routes()) == "upper-bypass"


def test_an_unrecognised_path_is_reported_rather_than_forced():
    """Wandering off both routes must not be silently assigned to one."""
    from robustness_lab.navigation import classify_route
    astray = [[1.0, -2.0], [2.0, -2.5], [3.0, -2.0]]
    assert classify_route(astray, _routes()) is None


def test_choice_run_does_not_hand_over_a_route():
    """The whole point: under route_choice the policy gets a goal, not a path."""
    from robustness_lab.navigation import guidance_for_scope
    guidance = guidance_for_scope("route_choice", routes=_routes(),
                                  goal=[3.0, 0.0, 0.0])
    assert guidance["waypoints"] == [], "route_choice must not supply waypoints"
    assert guidance["goal"] == [3.0, 0.0, 0.0]


def test_fixed_route_run_does_hand_over_the_path():
    from robustness_lab.navigation import guidance_for_scope
    guidance = guidance_for_scope("fixed_route", routes=_routes(),
                                  goal=[3.0, 0.0, 0.0])
    assert guidance["waypoints"], "fixed_route should supply the central path"


def test_claim_scope_distinguishes_choice_from_following():
    from robustness_lab.navigation import navigation_claim
    chosen = navigation_claim("route_choice", route_taken="upper-bypass")
    followed = navigation_claim("fixed_route", route_taken="central")
    assert chosen != followed
    assert "chose" in chosen.lower() or "choice" in chosen.lower()
    assert "given" in followed.lower() or "supplied" in followed.lower() or "follow" in followed.lower()
