"""A policy that cannot walk on flat ground has not been tested — the setup has.

G4 exists and passes its unit tests but is not wired into the episode path. The
HuggingFace policy hit exactly this: it fell on rs1, rs2 and rs3, moving 0.152 m
BACKWARDS, identically on every terrain — i.e. failing before terrain could
matter — and was recorded as `policy_fail / fall`.

We cannot distinguish "this policy is bad" from "our scene differs from its
training environment". The honest verdict is adapter_unverified, and terrain
must not be scored until flat ground is cleared.
"""

import pytest


def test_flat_ground_gate_exists_and_names_its_outcome():
    from robustness_lab.observation.gates import check_flat_ground
    result = check_flat_ground(distance_m=-0.15, duration_s=0.8,
                               required_distance_m=0.5)
    assert result["status"] == "blocked"
    assert result["outcome_class"] == "adapter_unverified"


def test_walking_forward_passes():
    from robustness_lab.observation.gates import check_flat_ground
    result = check_flat_ground(distance_m=0.9, duration_s=2.0,
                               required_distance_m=0.5)
    assert result["status"] == "ready"


def test_moving_backwards_is_blocked_however_far():
    from robustness_lab.observation.gates import check_flat_ground
    assert check_flat_ground(distance_m=-2.0, duration_s=2.0,
                             required_distance_m=0.5)["status"] == "blocked"


def test_falling_immediately_is_blocked():
    from robustness_lab.observation.gates import check_flat_ground
    result = check_flat_ground(distance_m=0.05, duration_s=0.2,
                               required_distance_m=0.5, fell=True)
    assert result["status"] == "blocked"
    assert "flat" in result["reason"].lower()


def test_reason_states_it_is_the_setup_not_the_policy():
    """The message must not read as a policy verdict."""
    from robustness_lab.observation.gates import check_flat_ground
    reason = check_flat_ground(distance_m=-0.15, duration_s=0.8,
                               required_distance_m=0.5)["reason"].lower()
    assert "adapter" in reason or "setup" in reason or "not scored" in reason


def test_episode_reports_adapter_unverified_when_flat_ground_fails(tmp_path):
    """End to end: a policy that cannot stand must not produce a policy score."""
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    import torch, torch.nn as nn, yaml, os
    from robustness_lab.io import dump_json
    from robustness_lab.models import EpisodeSpec
    from robustness_lab.backends.mujoco import MujocoBackend
    from robustness_lab.robot import inspect_mjcf
    from robustness_lab.worlds import generate_terrain
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    robot = root / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    if not robot.is_file():
        pytest.skip("go2 MJCF not present")
    profile = inspect_mjcf(str(robot))
    joints = profile.controller_joint_order

    # A policy that always commands a large constant offset: it will fall.
    class Faller(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 12) * 3.0

    artifact = tmp_path / "policy.pt"
    torch.jit.script(Faller()).save(str(artifact))
    manifest = tmp_path / "policy.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": "2.0", "policy_id": "faller",
        "runtime": {"format": "torchscript", "path": str(artifact),
                    "control_frequency_hz": 50},
        "canonical_joint_order": joints, "joint_order_policy": joints,
        "references": {"default_joint_position":
                       [profile.default_joint_positions[j] for j in joints]},
        "observation": {"sources": ["projected_gravity_b", "joint_position"],
                        "history": {"frames": 1, "order": "newest_first"}},
        "action": {"space": "joint_position_offset", "scale": 1.0,
                   "offset": "default_joint_position", "size": 12},
    }), encoding="utf-8")

    world = generate_terrain("rs4", profile, 0.5, 7)          # a real challenge
    profile_path = tmp_path / "robot.json"
    world_path = tmp_path / "world.json"
    dump_json(str(profile_path), profile)
    dump_json(str(world_path), world)
    episode = EpisodeSpec(
        episode_id="g4", backend="mujoco", robot_profile=str(profile_path),
        robot_asset=str(robot), robot_format="mjcf", world=str(world_path),
        controller={"type": "compiled_policy", "manifest": str(manifest),
                    "command": [0.6, 0.0, 0.0], "flat_ground_gate": True},
        output_dir=str(tmp_path / "run"), timeout_s=4.0)
    log = MujocoBackend().run(episode, profile, world)
    assert log.terminal_reason == "adapter_unverified", (
        "a policy that cannot stand was scored as a policy failure: %s"
        % log.terminal_reason)
