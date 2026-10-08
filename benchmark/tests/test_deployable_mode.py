"""A run can be restricted to information a real robot could actually obtain.

Without this, a policy can be scored using simulator ground truth and the result
reads as a deployment claim. The classes are already recorded per source; this
enforces them at run level and records what was used.
"""

import pytest

DEPLOYABLE = {"onboard", "onboard_derived", "command"}


def test_examination_can_declare_a_deployable_only_run():
    from robustness_lab.observation.gates import resolve_allowed_classes
    allowed = resolve_allowed_classes("deployable_only")
    assert allowed == DEPLOYABLE


def test_unrestricted_mode_allows_every_class():
    from robustness_lab.canonical_state import AVAILABILITY_CLASSES
    from robustness_lab.observation.gates import resolve_allowed_classes
    assert resolve_allowed_classes("unrestricted") == set(AVAILABILITY_CLASSES)


def test_unknown_mode_is_rejected_by_name():
    from robustness_lab.observation.gates import resolve_allowed_classes
    with pytest.raises(ValueError, match="wobble"):
        resolve_allowed_classes("wobble")


def test_deployable_run_rejects_ground_truth_and_names_it():
    from robustness_lab.observation.gates import check_deployability, resolve_allowed_classes
    result = check_deployability(
        sources={"joint_position", "friction_coefficient", "base_velocity_truth"},
        allowed=resolve_allowed_classes("deployable_only"))
    assert result["status"] == "blocked"
    assert set(result["rejected"]) == {"friction_coefficient", "base_velocity_truth"}
    assert result["outcome_class"] == "adapter_unverified"


def test_deployable_run_accepts_a_realistic_perception_stack():
    """Lidar, elevation maps, learned estimators and recurrent memory all run
    onboard on real quadrupeds and must not be treated as privileged."""
    from robustness_lab.observation.gates import check_deployability, resolve_allowed_classes
    result = check_deployability(
        sources={"joint_position", "joint_velocity", "base_angular_velocity_b",
                 "projected_gravity_b", "depth_image", "lidar_rays",
                 "height_scan", "elevation_map", "estimator_output",
                 "recurrent_state", "gait_phase", "velocity_command",
                 "previous_action", "foot_contact"},
        allowed=resolve_allowed_classes("deployable_only"))
    assert result["status"] == "ready", result.get("rejected")


def test_infrastructure_sources_are_excluded_from_deployable_runs():
    """Motion capture is real but not field-autonomous."""
    from robustness_lab.observation.gates import check_deployability, resolve_allowed_classes
    result = check_deployability(sources={"base_position_w"},
                                 allowed=resolve_allowed_classes("deployable_only"))
    assert result["status"] == "blocked"
    assert "base_position_w" in result["rejected"]


def test_unrestricted_run_permits_ground_truth_but_still_reports_it():
    """Teacher and distillation runs stay possible; the passport must say so."""
    from robustness_lab.observation.gates import check_deployability, resolve_allowed_classes
    result = check_deployability(
        sources={"joint_position", "friction_coefficient"},
        allowed=resolve_allowed_classes("unrestricted"))
    assert result["status"] == "ready"
    assert "simulator_only" in result["classes_used"]


def test_claim_scope_degrades_when_ground_truth_is_used():
    from robustness_lab.observation.gates import claim_scope_for_classes
    deployable = claim_scope_for_classes({"onboard", "onboard_derived", "command"})
    privileged = claim_scope_for_classes({"onboard", "simulator_only"})
    assert "deployable" in deployable.lower()
    assert deployable != privileged
    assert "simulation" in privileged.lower() or "privileged" in privileged.lower()
