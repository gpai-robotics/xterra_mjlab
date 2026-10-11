"""Routes in composed worlds must follow terrain height, as single worlds do.

A single terrain sets waypoint z from the surface the robot will stand on:

    rs4 (gap): [[0.23,0,0.00], [0.94,0,0.00], [2.07,0,0.41], [2.82,0,0.41]]
                                                       ^ the far side is raised

Composed worlds (AEE chain/grid/long) set EVERY waypoint to z = 0.0 while their
terrain rises to 0.41 m. The route is therefore drawn through the obstacles, and
`goal_vector_b` handed to a policy points INTO them: standing on a 0.41 m
platform, the next waypoint reads as 0.41 m below.

Steering is unaffected -- RouteFollower uses planar distance -- but any policy
consuming goal_vector_b, goal_distance or waypoints_b receives a wrong vertical
component.
"""

import pytest


def _profile():
    from robustness_lab.robot import inspect_mjcf
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    if not path.is_file():
        pytest.skip("go2 MJCF not present")
    return inspect_mjcf(str(path))


def _composed(profile):
    from robustness_lab.worlds.compose import generate_aee
    return generate_aee(profile, "long", 0.5, 600,
                        options={"rows": 1, "columns": 4, "pair_mode": "cycle",
                                 "topology": "chain"})


def test_single_terrain_route_follows_height():
    """The behaviour composed worlds must match."""
    from robustness_lab.worlds import generate_terrain
    world = generate_terrain("rs4", _profile(), 0.5, 7)
    central = [r for r in world.routes if r.route_kind == "central"][0]
    assert len({round(w[2], 3) for w in central.waypoints}) > 1, \
        "rs4 should raise its exit waypoints"


def test_composed_world_route_is_not_flat_when_terrain_is_not():
    profile = _profile()
    world = _composed(profile)
    highest = max(g.pose.position[2] + g.size[2] / 2.0 for g in world.geometry)
    if highest < 0.05:
        pytest.skip("this composed world is flat")
    zs = {round(w[2], 3) for r in world.routes for w in r.waypoints}
    assert zs != {0.0}, (
        "every waypoint sits at z=0 while terrain reaches %.2f m" % highest)


def test_no_waypoint_is_buried_inside_terrain():
    """A waypoint below the surface beneath it points a policy into the ground."""
    profile = _profile()
    world = _composed(profile)

    def surface_at(x, y):
        best = 0.0
        for g in world.geometry:
            px, py, pz = g.pose.position
            sx, sy, sz = g.size
            if abs(x - px) <= sx / 2.0 + 1e-6 and abs(y - py) <= sy / 2.0 + 1e-6:
                best = max(best, pz + sz / 2.0)
        return best

    buried = []
    for route in world.routes:
        for point in route.waypoints:
            top = surface_at(point[0], point[1])
            if point[2] < top - 0.02:
                buried.append((route.name, [round(v, 2) for v in point], round(top, 2)))
    assert not buried, "waypoints below the surface: %s" % buried[:4]


def test_goal_pose_is_above_its_local_surface():
    profile = _profile()
    world = _composed(profile)
    goal = world.goal_pose.position
    top = 0.0
    for g in world.geometry:
        px, py, pz = g.pose.position
        sx, sy, sz = g.size
        if abs(goal[0] - px) <= sx / 2.0 + 1e-6 and abs(goal[1] - py) <= sy / 2.0 + 1e-6:
            top = max(top, pz + sz / 2.0)
    assert goal[2] >= top - 1e-6, "goal at %.2f is below its surface %.2f" % (goal[2], top)
