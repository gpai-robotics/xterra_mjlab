"""The sensor gate must be computed from adapter declarations, not a literal.

examiner._backend_sensor_gate previously hardcoded the set of supported sensor
kinds, so preflight passed policies requiring fields no adapter emits. That is
the same declared-vs-verified defect as the RS11 terrain block.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mujoco_declares_the_derived_proprioceptive_fields():
    from robustness_lab.backends.mujoco import MujocoBackend
    declared = MujocoBackend.probe().observation_capabilities()
    assert "projected_gravity_b" in declared
    assert "previous_action" in declared
    assert "base_angular_velocity_b" in declared


def test_no_hardcoded_capability_literal_remains():
    source = (ROOT / "src" / "robustness_lab" / "examiner.py").read_text(encoding="utf-8")
    assert "backend_supported = {" not in source, (
        "the hardcoded capability literal must be deleted, not shadowed")


def test_fake_backend_does_not_claim_what_it_cannot_produce():
    """fake does not execute the policy, so it must not advertise the full set."""
    from robustness_lab.backends.fake import FakeBackend
    declared = FakeBackend.probe().observation_capabilities()
    assert "depth_image" not in declared
    assert "height_scan" not in declared


def test_every_backend_declares_a_capability_set():
    from robustness_lab.backends.fake import FakeBackend
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.backends.isaaclab import IsaacLabBackend
    from robustness_lab.backends.isaacgym import IsaacGymBackend
    for backend in (FakeBackend, MujocoBackend, IsaacLabBackend, IsaacGymBackend):
        declared = backend.probe().observation_capabilities()
        assert isinstance(declared, (set, frozenset)), backend.__name__


def test_declared_capabilities_are_real_canonical_fields():
    """A backend may not advertise a name the canonical state bus does not define.

    Without this, observation_capabilities() becomes another unverified
    declaration -- the exact defect it exists to remove.
    """
    from robustness_lab.canonical_state import CANONICAL_FIELDS
    from robustness_lab.backends.fake import FakeBackend
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.backends.isaaclab import IsaacLabBackend
    from robustness_lab.backends.isaacgym import IsaacGymBackend
    for backend in (FakeBackend, MujocoBackend, IsaacLabBackend, IsaacGymBackend):
        declared = backend.probe().observation_capabilities()
        unknown = sorted(set(declared) - set(CANONICAL_FIELDS))
        assert not unknown, "%s advertises unknown fields: %s" % (backend.__name__, unknown)


def test_foresight_native_defers_observation_construction_to_the_source_task():
    """The native integration launches the policy's OWN Isaac Lab task.

    That task builds its own observation vector — terrain scan, estimator,
    camera encoder, waypoint heading — so the platform's canonical bus is not
    what supplies them and must not gate on it. The weaker claim is recorded
    in claim_scope instead of being silently granted.
    """
    from robustness_lab.examiner import _backend_sensor_gate
    from robustness_lab.skill import load_skill
    skill = load_skill(str(ROOT / "configs" / "skills" / "go2_baseline_teacher_native.yaml"),
                       check_files=False)
    assert skill.controller.get("type") == "native_rsl_rl"
    gate = _backend_sensor_gate(skill, "foresight_native",
                                str(ROOT / "configs" / "runtimes.yaml"))
    assert gate["missing_sensor_kinds"] == []
    assert "command" in gate["supported_sensor_kinds"]
    assert "waypoints" in gate["supported_sensor_kinds"]
    assert "source-native" in gate["claim_scope"]


def test_foresight_native_blocks_a_controller_the_source_task_does_not_own():
    """Negative control: the exemption is tied to native_rsl_rl, not to the
    backend name. A portable artifact on that backend would need the platform
    to construct observations, which it cannot do here."""
    from robustness_lab.examiner import _backend_sensor_gate
    from robustness_lab.skill import load_skill
    skill = load_skill(str(ROOT / "configs" / "skills" / "go2_baseline_teacher_native.yaml"),
                       check_files=False)
    skill.controller = dict(skill.controller, type="torchscript")
    gate = _backend_sensor_gate(skill, "foresight_native",
                                str(ROOT / "configs" / "runtimes.yaml"))
    assert gate["status"] == "blocked"
    assert "command" in gate["missing_sensor_kinds"]
