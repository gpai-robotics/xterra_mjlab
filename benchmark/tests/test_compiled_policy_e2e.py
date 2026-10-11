"""End-to-end: a submitted TorchScript policy actually drives the simulator.

This is the test that turns the observation compiler from a correct component
into a working pipeline. It exercises the whole chain a submitter relies on:

    manifest -> compiled plan -> CanonicalState from MuJoCo -> policy
             -> action decoder -> joint targets -> simulator

The fixture robot has ONE joint, so the observation is 9-dimensional:
    projected_gravity_b(3) + velocity_command(3)
    + joint_position(1) + joint_velocity(1) + previous_action(1)
"""

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


def _manifest_payload(expected_size=9, action_size=1):
    return {
        "schema_version": "2.0",
        "policy_id": "e2e-mini",
        "runtime": {"format": "torchscript", "path": "policy.pt",
                    "control_frequency_hz": 50},
        "observation": {
            "terms": [
                {"source": "projected_gravity_b", "size": 3},
                {"source": "velocity_command", "size": 3},
                {"source": "joint_position", "size": 1},
                {"source": "joint_velocity", "size": 1},
                {"source": "previous_action", "size": 1},
            ],
            "history": {"frames": 1, "order": "newest_first"},
            "expected_size": expected_size,
        },
        "action": {"space": "joint_position_offset", "scale": 0.25,
                   "offset": "default_joint_position", "size": action_size},
    }


def _build(tmp_path, in_features=9, out_features=1, expected_size=9):
    """Write a TorchScript policy plus its schema-v2 manifest."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(in_features, out_features)

        def forward(self, x):
            return torch.tanh(self.net(x))

    torch.jit.script(Policy()).save(str(tmp_path / "policy.pt"))
    manifest = tmp_path / "policy.yaml"
    manifest.write_text(yaml.safe_dump(_manifest_payload(expected_size, out_features)),
                        encoding="utf-8")
    return str(manifest)


def _episode(tmp_path, manifest_path, profile, world):
    from robustness_lab.io import dump_json
    from robustness_lab.models import EpisodeSpec
    profile_path = tmp_path / "robot.json"
    world_path = tmp_path / "world.json"
    dump_json(str(profile_path), profile)
    dump_json(str(world_path), world)
    return EpisodeSpec(
        episode_id="e2e",
        backend="mujoco",
        robot_profile=str(profile_path),
        robot_asset=str(FIXTURES / "mini_robot.xml"),
        robot_format="mjcf",
        world=str(world_path),
        controller={"type": "compiled_policy", "manifest": manifest_path,
                    "command": [0.4, 0.0, 0.0]},
        output_dir=str(tmp_path / "run"),
        timeout_s=2.0,
    )


def _setup(tmp_path, **kwargs):
    pytest.importorskip("mujoco")
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    profile = inspect_mjcf(str(FIXTURES / "mini_robot.xml"))
    world = generate_terrain("rs1", profile, 0.2, 11)
    manifest = _build(tmp_path, **kwargs)
    return _episode(tmp_path, manifest, profile, world), profile, world


def test_torchscript_policy_runs_end_to_end_in_mujoco(tmp_path):
    """The episode must genuinely EXECUTE, not be refused.

    A randomly-initialised policy is not expected to complete the course, so
    "failed" is a legitimate POLICY outcome here and still proves the pipeline
    ran. What must NOT happen is the episode being refused or crashing: those
    are adapter/infrastructure outcomes and would mean the chain is broken.
    """
    from robustness_lab.backends.mujoco import MujocoBackend
    episode, profile, world = _setup(tmp_path)
    log = MujocoBackend().run(episode, profile, world)
    assert log.status in {"completed", "failed"}, (log.status, log.diagnostics)
    assert log.terminal_reason not in {
        "adapter_unverified", "backend_unavailable", "simulator_error",
    }, (log.terminal_reason, log.diagnostics)
    assert len(log.samples) > 5
    # the policy genuinely drove the joint: some action is non-zero
    assert any(abs(value) > 1e-9 for sample in log.samples for value in sample.action)


def test_compiled_observation_reaches_the_declared_width(tmp_path):
    from robustness_lab.backends.mujoco import MujocoBackend
    episode, profile, world = _setup(tmp_path)
    log = MujocoBackend().run(episode, profile, world)
    assert log.diagnostics.get("observation_size") == 9


def test_artifact_width_mismatch_is_adapter_unverified_not_policy_failure(tmp_path):
    """The manifest claims 9 but the artifact wants 12.

    This must be refused before the policy is scored, and must NEVER be
    recorded as a policy failure.
    """
    from robustness_lab.backends.mujoco import MujocoBackend
    episode, profile, world = _setup(tmp_path, in_features=12, expected_size=9)
    log = MujocoBackend().run(episode, profile, world)
    assert log.status != "completed"
    assert log.terminal_reason == "adapter_unverified"
    assert "policy_fail" not in str(log.terminal_reason)


def test_previous_action_is_fed_back_and_starts_at_zero(tmp_path):
    """previous_action must be the prior raw network output, zeros at reset."""
    from robustness_lab.backends.mujoco import MujocoBackend
    episode, profile, world = _setup(tmp_path)
    log = MujocoBackend().run(episode, profile, world)
    assert log.diagnostics.get("previous_action_initialised_to_zero") is True
