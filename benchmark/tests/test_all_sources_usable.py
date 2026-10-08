"""Every advertised observation source must actually be usable in a manifest.

A source listed but not compilable is the same defect as a capability declared
but not implemented: it looks available and fails only when a researcher tries
it. This walks the entire vocabulary and compiles a real term for each one.
"""

import pytest

from robustness_lab.canonical_state import CANONICAL_FIELDS
from robustness_lab.observation.compiler import compile_terms

GRID = {"kind": "grid", "x_range": [-0.05, 0.05], "y_range": [-0.05, 0.05],
        "resolution": 0.1}                                    # 2 x 2 = 4 points
CAMERA = {"kind": "camera", "resolution": [2, 2], "hfov_deg": 87.0, "near": 0.1,
          "far": 5.0, "position_b": [0.3, 0.0, 0.05],
          "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "encoding": "metres"}
LIDAR = {"kind": "lidar", "ray_count": 4, "angular_pattern": "uniform",
         "max_range": 10.0, "origin_b": [0.0, 0.0, 0.1]}
POINTS = {"kind": "point_cloud", "point_count": 4}

# (geometry, declared size) combinations tried for each source, in order.
ATTEMPTS = ((None, 4), (GRID, 4), (CAMERA, 4), (CAMERA, 12), (LIDAR, 4), (POINTS, 12))


@pytest.mark.parametrize("source", sorted(CANONICAL_FIELDS))
def test_source_compiles_and_builds(source):
    meta = CANONICAL_FIELDS[source]
    last = None
    for geometry, size in ATTEMPTS:
        term = {"source": source, "size": size}
        if geometry is not None:
            term["geometry"] = geometry
        if meta.get("multi_instance"):
            term["instance"] = "probe"
        try:
            plan = compile_terms([term])
            values = [1.0] * size
            state = {source: {"probe": values} if meta.get("multi_instance") else values}
            assert len(plan.build(state)) == size
            return
        except Exception as error:      # try the next geometry shape
            last = error
    raise AssertionError("source %r is advertised but not usable: %s" % (source, last))


def test_vocabulary_has_not_shrunk():
    """Guards against a source quietly disappearing."""
    assert len(CANONICAL_FIELDS) >= 76
