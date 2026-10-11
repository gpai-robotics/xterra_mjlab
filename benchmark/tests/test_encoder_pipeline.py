"""Encoder sub-artifacts: running the researcher's own preprocessing weights.

Some policies need a tensor the simulator cannot produce directly. The
FoReSight student is the real case, measured from the committed exports:

    depth_encoder : (depth_image[58,87], proprioception[53]) -> 34
    student       : (x[753], scandots_latent[32])            -> 12

The encoder emits 34 but the policy consumes 32 -- Extreme Parkour's encoder
returns [latent(32) | yaw(2)] and the caller slices it. That slice is a
researcher-supplied fact the platform cannot infer, so the manifest must carry
it and the platform must apply it.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "assets" / "cache" / "skills" / "go2_extreme_parkour_student" / "exported_deploy"
STUDENT = DEPLOY / "policy.pt"
ENCODER = DEPLOY / "depth_latest.pt"


def _require(*paths):
    pytest.importorskip("torch")
    for path in paths:
        if not path.is_file():
            pytest.skip("artifact not present: %s" % path)


def _encoder_spec():
    return {
        "name": "depth_encoder",
        "path": str(ENCODER),
        "inputs": [
            {"name": "depth_image", "shape": [58, 87]},
            {"name": "proprioception", "shape": [53]},
        ],
        "output": {"name": "depth_out", "size": 34},
    }


def test_encoder_runs_and_returns_its_declared_width():
    _require(ENCODER)
    from robustness_lab.observation.encoders import compile_encoder
    encoder = compile_encoder(_encoder_spec())
    output = encoder.run({"depth_image": [[0.0] * 87 for _ in range(58)],
                          "proprioception": [0.0] * 53})
    assert len(output) == 34


def test_encoder_output_width_mismatch_is_blocked():
    """Declaring the wrong output width must fail loudly, not silently truncate."""
    _require(ENCODER)
    from robustness_lab.observation.encoders import compile_encoder
    spec = _encoder_spec()
    spec["output"] = {"name": "depth_out", "size": 32}   # real width is 34
    with pytest.raises(ValueError, match="34"):
        compile_encoder(spec).run({"depth_image": [[0.0] * 87 for _ in range(58)],
                                   "proprioception": [0.0] * 53})


def test_missing_encoder_artifact_is_adapter_unverified():
    from robustness_lab.observation.encoders import compile_encoder
    spec = _encoder_spec()
    spec["path"] = str(DEPLOY / "does_not_exist.pt")
    with pytest.raises(ValueError, match="does_not_exist"):
        compile_encoder(spec)


def test_select_transform_slices_encoder_output_to_the_policy_width():
    """34 -> 32: the [latent|yaw] split the researcher must declare."""
    from robustness_lab.observation.transforms import TRANSFORMS
    sliced = TRANSFORMS["select"](list(range(34)), {"start": 0, "count": 32})
    assert len(sliced) == 32
    assert sliced[0] == 0 and sliced[-1] == 31


def test_real_student_chain_encoder_then_policy_yields_12_actions():
    """The whole point: run the researcher's encoder, slice it, feed the policy."""
    _require(STUDENT, ENCODER)
    from robustness_lab.observation.encoders import compile_encoder
    from robustness_lab.observation.transforms import TRANSFORMS
    import torch

    encoder = compile_encoder(_encoder_spec())
    latent34 = encoder.run({"depth_image": [[0.0] * 87 for _ in range(58)],
                            "proprioception": [0.0] * 53})
    latent32 = TRANSFORMS["select"](latent34, {"start": 0, "count": 32})
    assert len(latent32) == 32

    policy = torch.jit.load(str(STUDENT), map_location="cpu")
    policy.eval()
    with torch.no_grad():
        action = policy(torch.zeros(1, 753), torch.tensor([latent32], dtype=torch.float32))
    assert tuple(action.shape) == (1, 12)


def test_encoder_declaration_names_who_supplies_what():
    """The manifest must record that the encoder is researcher-supplied."""
    _require(ENCODER)
    from robustness_lab.observation.encoders import compile_encoder
    encoder = compile_encoder(_encoder_spec())
    assert encoder.provided_by == "researcher"
