"""A result must record WHAT THE POLICY COULD PERCEIVE, not just that it ran.

Deployability separates privileged from deployable, but blind, exteroceptive,
vision, recurrent and waypoint-driven policies all received the identical claim:
"deployable evaluation using onboard information and commands".

That is not enough to compare fairly. A blind policy failing at a gap it cannot
see is expected; a policy with a height scan failing the same gap is a real
failure. Schema v1 carried `sensor_track`; the v2 path lost it.

The track is derived from the sources actually declared, so it cannot be
overclaimed: a submitter cannot label themselves blind while reading a height
scan.
"""

import pytest

PROPRIO = {"joint_position", "joint_velocity", "projected_gravity_b",
           "base_angular_velocity_b", "previous_action"}
CMD = {"velocity_command"}


def _track(sources):
    from robustness_lab.observation.gates import derive_sensor_track
    return derive_sensor_track(set(sources))


def test_proprioception_and_command_only_is_blind():
    assert _track(PROPRIO | CMD) == "blind"


def test_a_height_scan_makes_it_exteroceptive():
    assert _track(PROPRIO | CMD | {"height_scan"}) == "exteroceptive"


def test_a_depth_camera_makes_it_exteroceptive():
    assert _track(PROPRIO | CMD | {"depth_image"}) == "exteroceptive"


def test_lidar_and_maps_are_exteroceptive():
    for source in ("lidar_rays", "elevation_map", "point_cloud", "traversability_map"):
        assert _track(PROPRIO | CMD | {source}) == "exteroceptive", source


def test_ground_truth_makes_it_privileged_whatever_else_is_present():
    assert _track(PROPRIO | CMD | {"height_scan", "friction_coefficient"}) == "privileged"


def test_privileged_beats_exteroceptive_beats_blind():
    """Ordering matters: the strongest information present sets the track."""
    assert _track(PROPRIO) == "blind"
    assert _track(PROPRIO | {"height_scan"}) == "exteroceptive"
    assert _track(PROPRIO | {"height_scan", "terrain_height_truth"}) == "privileged"


def test_recurrent_memory_alone_does_not_change_the_track():
    """Internal memory is not perception."""
    assert _track(PROPRIO | CMD | {"recurrent_state"}) == "blind"


def test_waypoints_alone_do_not_make_it_exteroceptive():
    """A route handed to the policy is guidance, not sensing. It changes the
    navigation claim, which is recorded separately."""
    assert _track(PROPRIO | CMD | {"waypoints_b", "goal_heading"}) == "blind"


def test_claim_scope_names_the_track():
    from robustness_lab.observation.gates import claim_scope_for_sources
    blind = claim_scope_for_sources(PROPRIO | CMD)
    extero = claim_scope_for_sources(PROPRIO | CMD | {"height_scan"})
    assert "blind" in blind.lower()
    assert "exteroceptive" in extero.lower() or "perception" in extero.lower()
    assert blind != extero


def test_route_assistance_is_recorded_separately_from_the_track():
    """Being given waypoints weakens a navigation claim even though the policy
    is still blind for perception purposes."""
    from robustness_lab.observation.gates import claim_scope_for_sources
    guided = claim_scope_for_sources(PROPRIO | CMD | {"waypoints_b"})
    unguided = claim_scope_for_sources(PROPRIO | CMD)
    assert guided != unguided
    assert "waypoint" in guided.lower() or "route" in guided.lower() or "guided" in guided.lower()
