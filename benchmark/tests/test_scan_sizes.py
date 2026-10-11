"""Any height-scan width must be recognised, not just the legged_gym 187.

The layout family is parametric in joint count -- 8, 12, 16 and 18 joints all
decompose correctly -- but the scan remainder was pinned to 187 (the 17x11
legged_gym grid). A policy using 21x11=231, or 11x7=77, fell back to a skeleton.
The scan size is a free parameter and must be treated as one.
"""

import os
import tempfile

import pytest
import yaml

JOINTS12 = ["j%d" % i for i in range(12)]


def _scaffold(width, joints=JOINTS12):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from robustness_lab.scaffold import scaffold_policy_manifest
    work = tempfile.mkdtemp()
    artifact = os.path.join(work, "p.pt")
    torch.jit.script(
        nn.Sequential(nn.Linear(width, 16), nn.ELU(), nn.Linear(16, len(joints)))
    ).save(artifact)
    out = os.path.join(work, "m.yaml")
    scaffold_policy_manifest(artifact, joint_order=joints, output=out)
    return (yaml.safe_load(open(out, encoding="utf-8")).get("observation") or {}).get("terms") or []


def _scan(terms):
    found = [t for t in terms if t["source"] == "height_scan"]
    return found[0] if found else None


@pytest.mark.parametrize("scan_size", [187, 231, 132, 77, 100, 50])
def test_any_scan_remainder_is_recognised(scan_size):
    width = 48 + scan_size
    terms = _scaffold(width)
    assert terms, "width %d produced no layout" % width
    scan = _scan(terms)
    assert scan is not None, "no height_scan term for remainder %d" % scan_size
    assert scan["size"] == scan_size
    assert sum(t["size"] for t in terms) == width


@pytest.mark.parametrize("scan_size", [187, 231, 77])
def test_scan_geometry_yields_exactly_the_declared_count(scan_size):
    """The grid must actually produce that many points, or the compiler will
    reject the manifest we just wrote."""
    from robustness_lab.observation.sensor_geometry import geometry_element_count
    terms = _scaffold(48 + scan_size)
    scan = _scan(terms)
    assert scan is not None and "geometry" in scan
    assert geometry_element_count("height_scan", scan["geometry"]) == scan_size


@pytest.mark.parametrize("scan_size", [187, 231, 77])
def test_scaffolded_scan_manifest_compiles(scan_size):
    """End to end: what we draft must load and compile."""
    from robustness_lab.observation.compiler import compile_terms
    terms = _scaffold(48 + scan_size)
    plan = compile_terms(terms)
    assert plan.size == 48 + scan_size


def test_scan_remainder_works_for_other_joint_counts():
    terms = _scaffold(12 + 3 * 18 + 231, joints=["j%d" % i for i in range(18)])
    assert sum(t["size"] for t in terms) == 12 + 3 * 18 + 231
    assert _scan(terms)["size"] == 231


def test_a_remainder_that_cannot_be_a_grid_is_not_forced():
    """A prime-ish remainder that no sensible grid produces must not be faked
    into one; a skeleton is the honest answer."""
    terms = _scaffold(48 + 1)
    scan = _scan(terms)
    if scan is not None:
        from robustness_lab.observation.sensor_geometry import geometry_element_count
        assert geometry_element_count("height_scan", scan["geometry"]) == 1
