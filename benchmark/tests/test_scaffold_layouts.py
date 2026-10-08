"""The scaffold must recognise the standard legged-locomotion layouts.

Tested against four real policies pulled from HuggingFace:

    UWRobotics/Go2BlindFlat          torchscript  48 -> 12
    Kyu3224 anymal_c/flat.onnx       onnx         48 -> 12
    Kyu3224 hound1/rough.onnx        onnx        235 -> 12
    nepyope/unitree_rl_locomotion    torchscript  47 -> 12

Only the 45-wide form was recognised, so three of four got an undecomposed
skeleton. 48 is the COMMON legged_gym layout (45 plus base linear velocity) and
235 is 48 plus a 187-point height scan, so the gap hit the most standard cases.

For N joints the family is:
    9  + 3N          angular velocity, gravity, command, pos, vel, prev action
    12 + 3N          the above plus base linear velocity
    either, plus a height scan of the declared grid size
"""

import pytest

JOINTS = ["j%d" % i for i in range(12)]


def _artifact(tmp_path, width, name="p.pt"):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    path = tmp_path / name
    torch.jit.script(nn.Sequential(nn.Linear(width, 32), nn.ELU(), nn.Linear(32, 12))).save(str(path))
    return str(path)


def _layout(tmp_path, width):
    import yaml
    from robustness_lab.scaffold import scaffold_policy_manifest
    out = str(tmp_path / "m.yaml")
    scaffold_policy_manifest(_artifact(tmp_path, width), joint_order=JOINTS, output=out)
    manifest = yaml.safe_load(open(out, encoding="utf-8"))
    return manifest.get("observation", {}).get("terms") or []


def _sources(terms):
    return [t["source"] for t in terms]


def test_45_wide_layout_without_base_linear_velocity(tmp_path):
    terms = _layout(tmp_path, 45)
    assert sum(t["size"] for t in terms) == 45
    assert "base_linear_velocity_b" not in _sources(terms)


def test_48_wide_layout_includes_base_linear_velocity(tmp_path):
    """The most common legged_gym observation. Previously undecomposed."""
    terms = _layout(tmp_path, 48)
    assert terms, "48-wide layout was not recognised"
    assert sum(t["size"] for t in terms) == 48
    assert "base_linear_velocity_b" in _sources(terms)


def test_235_wide_layout_is_48_plus_a_height_scan(tmp_path):
    """The standard rough-terrain policy."""
    terms = _layout(tmp_path, 235)
    assert terms, "235-wide layout was not recognised"
    assert sum(t["size"] for t in terms) == 235
    scan = [t for t in terms if t["source"] == "height_scan"]
    assert scan, "no height_scan term in a 235-wide layout"
    assert scan[0]["size"] == 187
    assert "geometry" in scan[0], "a height scan must carry its grid"


def test_every_decomposition_sums_to_the_probed_width(tmp_path):
    for width in (45, 48, 235):
        terms = _layout(tmp_path, width)
        assert sum(t["size"] for t in terms) == width, width


def test_unrecognised_width_still_writes_a_skeleton(tmp_path):
    """A width outside the family must not be forced into a false layout."""
    import yaml
    from pathlib import Path
    from robustness_lab.scaffold import scaffold_policy_manifest
    out = str(tmp_path / "m.yaml")
    result = scaffold_policy_manifest(_artifact(tmp_path, 37), joint_order=JOINTS, output=out)
    assert Path(out).exists()
    assert result["input_size"] == 37
    terms = yaml.safe_load(open(out, encoding="utf-8")).get("observation", {}).get("terms") or []
    assert sum(t["size"] for t in terms) in (0, 37), "must not invent a mismatched decomposition"


def test_layout_choice_is_reported_for_review(tmp_path):
    """A chosen layout is still a hypothesis and must be flagged."""
    from robustness_lab.scaffold import scaffold_policy_manifest
    result = scaffold_policy_manifest(_artifact(tmp_path, 48), joint_order=JOINTS,
                                      output=str(tmp_path / "m.yaml"))
    assert result["needs_review"]
    assert any("layout" in item.lower() or "semantics" in item.lower()
               for item in result["needs_review"])
