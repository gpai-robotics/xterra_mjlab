"""A foot walking the route must always have terrain under it — including on curves.

The camber/curve modifier tiles the arc with short rigid boxes, each rotated to
the local heading. Rotating rigid rectangles along an arc fans them apart on the
convex (outer) edge, leaving wedge-shaped holes: on one right-curve rs1 world a
grid scan found ~900 uncovered points, and a replay confirmed the front-left
foot dropped into one at (1.22, 0.038) — no box beneath it — wedged, and the
robot fell. ~9% of all trials showed a foot pinned like this, mostly on curved
variants. That is a terrain-generation defect scored as a policy failure.

This test samples the actual walkable strip — the route centerline, offset
laterally across the track to each foot line — and requires terrain under every
sample. It is checked on the curved variants at several difficulties, where the
holes live, and on the plain terrains as a control.
"""

import math
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
VARIANTS = [(-12.0, 65.0), (12.0, -65.0)]     # left-curve-left-low, right-curve-right-low


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _world(code, difficulty, camber, curve):
    from robustness_lab.worlds import generate_terrain
    from robustness_lab.worlds.modifiers import apply_geometry_modifiers
    w = generate_terrain(code, _robot(), difficulty, 5)
    return apply_geometry_modifiers(w, camber, curve, 28) if (camber or curve) else w


def _covered(world, x, y):
    """True if some box geom's footprint contains (x, y), honouring its yaw."""
    for g in world.geometry:
        if g.shape != "box":
            continue
        px, py, _ = g.pose.position
        sx, sy, _ = g.size
        yaw = (g.pose.rpy or [0.0, 0.0, 0.0])[2]
        dx, dy = x - px, y - py
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if abs(lx) <= sx / 2.0 + 1e-6 and abs(ly) <= sy / 2.0 + 1e-6:
            return True
    return False


def _uncovered_on_walkable_strip(world, robot):
    """Sample the route centerline and offset to each foot line; return holes."""
    route = world.routes[0].waypoints
    half = robot.stance_width_m / 2.0
    lateral_offsets = [-half, -half / 2.0, 0.0, half / 2.0, half]
    holes = []
    for a, b in zip(route, route[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6:
            continue
        hx, hy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg     # local heading
        nx, ny = -hy, hx                                       # left-normal
        steps = max(2, int(seg / 0.05))
        for i in range(steps + 1):
            t = i / steps
            cx, cy = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            for off in lateral_offsets:
                x, y = cx + off * nx, cy + off * ny
                if not _covered(world, x, y):
                    holes.append((round(x, 3), round(y, 3)))
    return holes


@pytest.mark.parametrize("camber,curve", VARIANTS)
@pytest.mark.parametrize("code", ["rs1", "rs2", "rs7"])
@pytest.mark.parametrize("difficulty", [0.25, 0.5, 0.75])
def test_curved_terrain_covers_the_whole_walkable_strip(code, difficulty, camber, curve):
    robot = _robot()
    world = _world(code, difficulty, camber, curve)
    holes = _uncovered_on_walkable_strip(world, robot)
    assert not holes, "%s d%.2f c%+.0f: %d holes under the route, e.g. %s" % (
        code, difficulty, curve, len(holes), holes[:5])


def test_the_specific_reported_hole_is_gone():
    """(1.22, 0.038) on right-curve rs1 had no box; a foot wedged there and fell."""
    world = _world("rs1", 0.5, 12.0, -65.0)
    assert _covered(world, 1.22, 0.038), "the reported hole at (1.22, 0.038) is still open"


def test_plain_terrain_is_still_fully_covered():
    """Control: the un-curved terrains have no holes either."""
    robot = _robot()
    for code in ["rs%d" % i for i in (1, 2, 3, 4, 6, 7)]:
        world = _world(code, 0.5, 0.0, 0.0)
        holes = _uncovered_on_walkable_strip(world, robot)
        assert not holes, "%s (plain): %d holes, e.g. %s" % (code, len(holes), holes[:5])
