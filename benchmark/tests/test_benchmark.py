import math
from pathlib import Path

import pytest

from robustness_lab.benchmark import HORIZONS_BODY_LENGTHS, _preview_payload, _reverse_world, materialize_benchmark, run_benchmark
from robustness_lab.io import dump_json, load_data, sha256_data
from robustness_lab.models import RobotProfile, WorldSpec
from robustness_lab.worlds import generate_aee, generate_terrain


ROOT = Path(__file__).resolve().parents[1]


def profile():
    return RobotProfile.from_dict(load_data(str(ROOT / "assets/cache/go2_mjcf_profile.json")))


def test_all_graph_topologies_are_connected_and_deterministic():
    robot = profile()
    for topology in ("chain", "ring", "tree", "grid", "maze", "random_connected"):
        options = {"rows": 1 if topology == "chain" else 3, "columns": 4, "pair_mode": "random", "topology": topology}
        left = generate_aee(robot, "grid", 0.5, 77, options=options)
        right = generate_aee(robot, "grid", 0.5, 77, options=options)
        assert sha256_data(left.to_dict()) == sha256_data(right.to_dict())
        assert left.metadata["connected"] is True
        assert left.metadata["topology"] == topology
        assert left.routes[0].waypoints[0] == left.metadata["graph_nodes"][0]["position_m"]


def test_primary_graph_route_covers_every_rs_pair_edge_and_ends_at_goal():
    robot = profile()
    for topology in ("chain", "ring", "tree", "grid", "maze", "random_connected"):
        world = generate_aee(
            robot,
            "grid",
            0.5,
            77,
            options={
                "rows": 1 if topology == "chain" else 3,
                "columns": 4,
                "pair_mode": "random",
                "topology": topology,
            },
        )
        traversed = world.metadata["canonical_route_edges"]
        edge_records = {edge["id"]: edge for edge in world.metadata["graph_edges"]}
        graph_edges = set(edge_records)
        assert set(traversed) == graph_edges, topology
        assert world.metadata["canonical_route_covers_all_edges"] is True
        assert world.routes[0].waypoints[0] == world.spawn_pose.position[:2] + [0.0]
        assert world.routes[0].waypoints[-1][:2] == world.goal_pose.position[:2]

        current = "n-00-00"
        forward_edges = set()
        for edge_id in traversed:
            edge = edge_records[edge_id]
            if current == edge["source"]:
                forward_edges.add(edge_id)
                current = edge["target"]
            else:
                assert current == edge["target"], (topology, current, edge_id)
                current = edge["source"]
        assert forward_edges == graph_edges, "%s omitted a directional RS pair" % topology
        assert set(world.metadata["canonical_route_forward_edges"]) == graph_edges
        assert world.metadata["canonical_route_edge_traversal_count"] == len(traversed)


def test_branching_routes_repeat_edges_when_needed_but_remain_continuous():
    robot = profile()
    tree = generate_aee(
        robot,
        "grid",
        0.5,
        91,
        options={"rows": 3, "columns": 3, "pair_mode": "cycle", "topology": "tree"},
    )
    traversed = tree.metadata["canonical_route_edges"]
    assert len(traversed) > len(set(traversed)), "a tree branch needs a return traversal"
    points = tree.routes[0].waypoints
    assert all(math.dist(a, b) < tree.resolved_parameters["pitch_m"] for a, b in zip(points, points[1:])), (
        "route contains a teleport instead of a connected walk"
    )


def test_reverse_geometry_swaps_spawn_goal_and_preserves_route_length():
    robot = profile()
    world = generate_terrain("rs3", robot, 0.8, 5)
    reverse = _reverse_world(world)
    assert math.isclose(reverse.spawn_pose.position[0], world.dimensions_m[0] - world.spawn_pose.position[0])
    assert reverse.metadata["direction"] == "reverse"
    assert len(reverse.geometry) == len(world.geometry)


def test_preview_hash_excludes_nonvisible_episode_metadata():
    robot = profile()
    world = generate_terrain("rs4", robot, 0.5, 9)
    assert sha256_data(_preview_payload(world, robot)) == sha256_data(_preview_payload(world, robot))
    changed = world.to_dict()
    changed["metadata"]["controller"] = "different"
    assert sha256_data(_preview_payload(WorldSpec.from_dict(changed), robot)) == sha256_data(_preview_payload(world, robot))


def test_smoke_materialization_counts_and_seed_nesting(tmp_path):
    output = tmp_path / "smoke"
    result = materialize_benchmark(
        str(ROOT / "configs/benchmarks/tiered_go2.yaml"),
        "smoke",
        str(ROOT / "assets/cache/go2_mjcf_profile.json"),
        ["fake"],
        {"fake": {"format": "mjcf", "path": str(ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml")}},
        str(ROOT / "configs/controllers/pd_hold.yaml"),
        str(output),
    )
    assert result["schema_version"] == "2.0"
    assert result["condition_count"] == 14
    assert result["episode_count"] == 14
    assert len(result["preview_worlds"]) == result["unique_preview_count"]
    assert set(HORIZONS_BODY_LENGTHS) == {"XS", "S", "M", "L", "XL", "Endurance"}
    assert all(Path(path).exists() for path in result["episodes"])

    manifest_path = output / "benchmark_manifest.json"
    changed = load_data(str(manifest_path))
    changed["profile"] = "tampered"
    dump_json(str(manifest_path), changed)
    with pytest.raises(ValueError, match="changed after materialization"):
        run_benchmark(str(manifest_path), str(ROOT / "configs/runtimes.yaml"), limit=1)


def test_benchmark_geometry_modifier_cases_expand_selected_rs_conditions(tmp_path):
    result = materialize_benchmark(
        str(ROOT / "configs/benchmarks/camber_curvature_go2.yaml"),
        "smoke",
        str(ROOT / "assets/cache/go2_mjcf_profile.json"),
        ["fake"],
        {"fake": {"format": "mjcf", "path": str(ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml")}},
        str(ROOT / "configs/controllers/pd_hold.yaml"),
        str(tmp_path / "modified"),
    )
    assert result["condition_count"] == 36
    modified = [item for item in result["conditions"] if "clockwise" in item["key"]]
    assert len(modified) == 22
    for item in modified:
        world = load_data(item["world"])
        assert world["metadata"]["geometry_modifier"]["collision_geometry_transformed"] is True
