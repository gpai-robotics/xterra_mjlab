from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trimesh")  # the foresight bridge's only extra dependency

from robustness_lab.integrations.foresight_world import (
    bridge_manifest,
    combine_native_result,
    foresight_terrain_requirements,
    material_face_tag,
    world_to_foresight_terrain,
)
from robustness_lab.environment_physics import apply_linear_drag, assess_environment_support, world_physics_plan
from robustness_lab.io import dump_json, load_data
from robustness_lab.models import RobotProfile
from robustness_lab.worlds import generate_aee


ROOT = Path(__file__).resolve().parents[1]


def _single_lane_world():
    for candidate in sorted((ROOT / "runs/tiered_benchmark/smoke/worlds").glob("*.json")):
        if load_data(str(candidate)).get("terrain_type") in {"rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs9", "rs10", "rs11", "ae"}:
            return candidate
    raise AssertionError("single-lane test world is unavailable")


def test_worldspec_maps_to_native_policy_cell():
    world = _single_lane_world()
    meshes, origin, goals, heights, mask = world_to_foresight_terrain(str(world), 8)
    assert meshes
    assert np.allclose(origin, [8.0, 2.0, 0.0])
    assert goals.shape == (8, 2)
    assert np.isclose(goals[0, 0], -7.0)
    assert goals[-1, 0] > goals[0, 0]
    assert heights.shape == (8,)
    assert mask.shape == (201, 51)


def test_bridge_manifest_is_auditable():
    world = _single_lane_world()
    result = bridge_manifest(str(world), num_goals=8)
    assert result["geometry_count"] > 0
    assert result["triangle_count"] == result["geometry_count"] * 12
    assert result["world_sha256"]
    assert result["limitations"]
    assert result["material_fidelity"]["mode"] == "single_global_material"
    assert result["environment_physics"]["support"]["implementation_owner"] == "simulator_adapter"
    assert result["environment_physics"]["support"]["policy_interface_modified"] is False


def test_rs11_has_policy_independent_rigid_viscous_plan():
    candidates = sorted((ROOT / "runs/tiered_benchmark/smoke/worlds").glob("*.json"))
    world_path = next(path for path in candidates if load_data(str(path)).get("terrain_type") == "rs11")
    from robustness_lab.models import WorldSpec

    world = WorldSpec.from_dict(load_data(str(world_path)))
    offset = (-7.0 - world.spawn_pose.position[0], -world.spawn_pose.position[1], 0.0)
    plan = world_physics_plan(world, offset)
    support = assess_environment_support(plan, {"linear_velocity_drag"})
    assert plan["requested_tier"] == "rigid_viscous"
    assert plan["required_effects"] == ["linear_velocity_drag"]
    assert plan["policy_interface_modified"] is False
    assert support["status"] == "environment_supported"
    effect = plan["effects"][0]
    inside = [(effect["minimum"][i] + effect["maximum"][i]) / 2.0 for i in range(3)]
    damped, regions = apply_linear_drag(inside, [1.0, -0.5, 0.25], plan["effects"], 0.02)
    assert regions == [effect["region"]]
    assert 0.0 < damped[0] < 1.0
    assert -0.5 < damped[1] < 0.0
    outside, regions = apply_linear_drag([effect["maximum"][0] + 1.0, 0.0, 0.0], [1.0, 0.0, 0.0], plan["effects"], 0.02)
    assert outside == [1.0, 0.0, 0.0]
    assert regions == []


def test_environment_preflight_names_missing_backend_effects():
    plan = {"requested_tier": "rigid_viscous", "required_effects": ["linear_velocity_drag"]}
    support = assess_environment_support(plan, set())
    assert support["status"] == "environment_unsupported"
    assert support["capability_claim_eligible"] is False
    assert support["missing_effects"] == ["linear_velocity_drag"]
    assert "simulator adapter" in support["reason"]


def test_native_cell_requirements_preserve_small_world_defaults():
    world = _single_lane_world()
    requirements = foresight_terrain_requirements(str(world))
    assert requirements["size"][0] >= 16.0
    assert requirements["size"][1] >= 4.0
    assert requirements["num_goals"] >= 8
    meshes, _, goals, _, _ = world_to_foresight_terrain(
        str(world), requirements["num_goals"], size=tuple(requirements["size"])
    )
    assert meshes
    assert goals.shape[0] == requirements["num_goals"]


def test_graph_composition_can_be_restricted_to_passed_pairs():
    robot = RobotProfile.from_dict(load_data(str(ROOT / "assets/cache/go2_mjcf_profile.json")))
    allowed = [["rs9", "rs10"], ["rs10", "rs9"]]
    world = generate_aee(
        robot,
        "grid",
        0.5,
        42,
        options={"rows": 2, "columns": 2, "topology": "grid", "pair_mode": "cycle", "allowed_pairs": allowed},
    )
    assert world.metadata["graph_edges"]
    assert all(edge["pair"] in allowed for edge in world.metadata["graph_edges"])


def test_material_face_tags_survive_native_mesh_concatenation():
    world_path = _single_lane_world()
    world = load_data(str(world_path))
    meshes, _, _, _, _ = world_to_foresight_terrain(str(world_path), 8)
    observed = {tuple(color) for mesh in meshes for color in np.asarray(mesh.visual.face_colors)}
    from robustness_lab.models import WorldSpec

    spec = WorldSpec.from_dict(world)
    expected = {
        tuple(material_face_tag(item["material"], spec.materials[item["material"]]))
        for item in world["geometry"]
    }
    assert expected.issubset(observed)


def test_native_result_keeps_world_and_policy_evidence(tmp_path):
    world = _single_lane_world()
    bridge_path = tmp_path / "bridge.json"
    summary_path = tmp_path / "summary.json"
    dump_json(str(bridge_path), bridge_manifest(str(world), num_goals=8))
    dump_json(str(summary_path), {"checkpoint": "/policy.pt", "task": "native", "seed": 7, "steps": 500, "success": True, "max_goal_index": 7, "num_goals": 8, "episode_terminations": 2, "aggregate_path_distance_m": 4.2, "trajectory": "/trajectory.json"})
    result = combine_native_result(str(bridge_path), str(summary_path))
    assert result["success"] is True
    assert result["world_sha256"]
    assert result["episode_terminations"] == 2
    assert result["aggregate_path_distance_m"] == 4.2
    assert result["trajectory"] == "/trajectory.json"
    assert "not a robust success-rate" in result["claim_scope"]
