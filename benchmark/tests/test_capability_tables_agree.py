"""Preflight's static contract must agree with each backend's runtime probe.

Two independent lists of the same fact drift apart. When the depth-resistive mud
effect was added to MujocoBackend.probe(), preflight's BACKEND_CONTRACTS table
still listed only linear_velocity_drag, so preflight reported RS11 as
`environment_unsupported` while an actual MuJoCo run supported it.

That is the same defect that opened this project -- a terrain reported
Unsupported because of a declaration, not because of the simulator -- and the
same shape as the doctor/preflight disagreement about runtime availability.

The fix is not to edit one number: it is to assert the two agree, for every
backend, so the next capability added to a probe cannot silently fail to reach
preflight.
"""

import pytest


def _contracts():
    from robustness_lab import preflight
    for value in vars(preflight).values():
        if isinstance(value, dict) and "mujoco" in value and "fake" in value:
            return value
    pytest.fail("could not locate preflight's backend contract table")


def _probe(name):
    from robustness_lab.backends.fake import FakeBackend
    from robustness_lab.backends.mujoco import MujocoBackend
    backends = {"fake": FakeBackend, "mujoco": MujocoBackend}
    if name not in backends:
        pytest.skip("no importable backend class for %r" % name)
    return backends[name].probe()


@pytest.mark.parametrize("name", ["fake", "mujoco"])
def test_declared_environment_effects_match_the_runtime_probe(name):
    declared = set(_contracts()[name]["environment_effects"])
    actual = set(_probe(name).environment_effects)
    assert declared == actual, (
        "preflight declares %s for %r but the backend probe reports %s"
        % (sorted(declared), name, sorted(actual)))


def test_rs11_is_supported_by_preflights_own_table():
    """The end-to-end symptom: mud must not read as Unsupported before a run."""
    from pathlib import Path
    from robustness_lab.environment_physics import (
        assess_environment_support, world_physics_plan)
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    from robustness_lab.worlds import generate_terrain

    profile_path = Path(__file__).resolve().parents[1] / "assets/cache/go2_mjcf_profile.json"
    if not profile_path.is_file():
        pytest.skip("go2 profile not cached")
    profile = RobotProfile.from_dict(load_data(str(profile_path)))

    plan = world_physics_plan(generate_terrain("rs11", profile, 0.5, 7))
    report = assess_environment_support(
        plan, _contracts()["mujoco"]["environment_effects"])
    assert report["status"] == "environment_supported", report
