from pathlib import Path

from robustness_lab.io import sha256_data
from robustness_lab.robot import inspect_urdf
from robustness_lab.worlds import TERRAIN_CODES, apply_geometry_modifiers, generate_ae, generate_ae_pairs, generate_aee, generate_terrain


FIXTURES = Path(__file__).parent / "fixtures"


def profile():
    return inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))


def test_every_rs_generator_is_valid_and_deterministic():
    robot = profile()
    for code in TERRAIN_CODES:
        left = generate_terrain(code, robot, difficulty=0.6, seed=123)
        right = generate_terrain(code, robot, difficulty=0.6, seed=123)
        left.validate()
        assert sha256_data(left.to_dict()) == sha256_data(right.to_dict())
        assert left.robot_profile_sha256 == robot.source_sha256
        assert any(region.mandatory for region in left.regions)


def test_every_rs_supports_physical_camber_and_route_curvature():
    robot = profile()
    for code in TERRAIN_CODES:
        source = generate_terrain(code, robot, difficulty=0.6, seed=123)
        left = apply_geometry_modifiers(source, camber_deg=-12.0, curve_angle_deg=65.0, curve_segments=20)
        repeated = apply_geometry_modifiers(source, camber_deg=-12.0, curve_angle_deg=65.0, curve_segments=20)
        left.validate()
        assert sha256_data(left.to_dict()) == sha256_data(repeated.to_dict())
        assert left.metadata["geometry_modifier"]["collision_geometry_transformed"]
        assert len(left.geometry) >= len(source.geometry)
        assert left.goal_pose.position[1] > source.goal_pose.position[1]
        assert left.goal_pose.rpy[0] < 0.0
        assert left.materials == source.materials


def test_signed_modifier_directions_mirror_curve_and_camber():
    source = generate_terrain("rs2", profile(), difficulty=0.5, seed=7)
    left = apply_geometry_modifiers(source, camber_deg=-10.0, curve_angle_deg=55.0)
    right = apply_geometry_modifiers(source, camber_deg=10.0, curve_angle_deg=-55.0)
    assert left.goal_pose.position[1] > 0.0
    assert right.goal_pose.position[1] < 0.0
    assert left.goal_pose.rpy[0] < 0.0 < right.goal_pose.rpy[0]
    assert left.metadata["geometry_modifier"]["curve_direction"] == "left / anticlockwise"
    assert right.metadata["geometry_modifier"]["curve_direction"] == "right / clockwise"


def test_ae_has_all_ordered_pairs_and_routes():
    pairs = generate_ae_pairs()
    assert len(pairs) == 100
    assert len(set(pairs)) == 100
    assert ("rs1", "rs1") in pairs
    assert ("rs11", "rs10") in pairs
    world = generate_ae(profile(), "rs3", "rs10", 0.5, 9)
    assert {route.route_kind for route in world.routes} == {"central", "bypass"}
    assert world.normalized_parameters["ordered_pair_index"] == 28
    world.validate()


def test_ae_slot_color_does_not_replace_physical_material():
    world = generate_ae(profile(), "rs11", "rs4", 0.5, 9)
    marsh = next(item for item in world.geometry if item.semantic == "marsh")
    rough_slope = next(item for item in world.geometry if item.semantic == "slope")
    assert marsh.material == "compliant"
    assert marsh.parameters["display_material"] == "purple"
    assert rough_slope.material == "rough_low"
    assert rough_slope.parameters["display_material"] == "challenge"


def test_ae_accepts_morphology_relative_course_and_slot_overrides():
    robot = profile()
    default = generate_ae(robot, "rs11", "rs4", 0.5, 9)
    enlarged = generate_ae(
        robot,
        "rs11",
        "rs4",
        0.5,
        9,
        overrides={
            "approach_m": 4.0 * robot.body_length_m,
            "landing_m": 3.0 * robot.body_length_m,
            "purple_length_m": 8.0 * robot.body_length_m,
            "brown_run_m": 6.0 * robot.body_length_m,
            "brown_roughness_amplitude_m": 0.05,
        },
    )
    assert enlarged.dimensions_m[0] > default.dimensions_m[0]
    assert enlarged.normalized_parameters["course_body_lengths"] > default.normalized_parameters["course_body_lengths"]
    assert enlarged.resolved_parameters["slots"]["purple"]["resolved"]["length_m"] == 4.0
    assert enlarged.resolved_parameters["slots"]["brown"]["resolved"]["roughness_amplitude_m"] == 0.05


def test_exhaustive_aee_contains_one_cell_per_pair():
    world = generate_aee(profile(), "exhaustive", 0.2, 3)
    assert len(world.metadata["cells"]) == 100
    assert len({tuple(cell["pair"]) for cell in world.metadata["cells"]}) == 100
    assert world.metadata["exhaustive_pair_coverage"]
    world.validate()


def test_long_aee_has_explicit_reachable_connectors():
    world = generate_aee(profile(), "long", 0.4, 12, count=5)
    assert len(world.metadata["graph_edges"]) == 4
    assert len([region for region in world.regions if region.kind == "connector"]) == 4
    assert world.routes[0].name == "aee-central"
    assert world.routes[0].waypoints[0][0] < world.routes[0].waypoints[-1][0]
    world.validate()


def test_grid_aee_has_four_direction_ports_loops_and_repeatable_edges():
    robot = profile()
    options = {
        "rows": 3,
        "columns": 3,
        "pair_mode": "repeat",
        "purple": "rs11",
        "brown": "rs4",
        "cell_overrides": {
            "approach_m": 0.8,
            "transition_length_m": 0.4,
            "landing_m": 0.8,
            "purple_length_m": 1.2,
            "brown_run_m": 1.0,
        },
    }
    world = generate_aee(robot, "grid", 0.4, 12, options=options)
    repeated = generate_aee(robot, "grid", 0.4, 12, options=options)
    assert sha256_data(world.to_dict()) == sha256_data(repeated.to_dict())
    assert world.metadata["connected"]
    assert world.metadata["bidirectional_edges"]
    assert len(world.metadata["graph_nodes"]) == 9
    assert len(world.metadata["graph_edges"]) == 12
    center = next(node for node in world.metadata["graph_nodes"] if node["id"] == "n-01-01")
    assert set(center["ports"]) == {"north", "east", "south", "west"}
    assert center["degree"] == 4
    assert {tuple(edge["pair"]) for edge in world.metadata["graph_edges"]} == {("rs11", "rs4")}
    assert {edge["orientation"] for edge in world.metadata["graph_edges"]} == {"east", "north"}
    assert world.routes[0].name == "canonical-grid-all-edges"
    assert {route.name for route in world.routes[1:3]} == {
        "grid-east-then-north",
        "grid-north-then-east",
    }
    world.validate()
