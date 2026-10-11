from pathlib import Path

from robustness_lab.io import dump_json, load_data
from robustness_lab.robot import inspect_urdf
from robustness_lab.sampling import latin_hypercube, sobol
from robustness_lab.suites import materialize_suite


FIXTURES = Path(__file__).parent / "fixtures"


def test_stressor_samples_are_bounded_and_reproducible():
    bounds = {"friction": (0.2, 1.0), "latency": (0.0, 0.1)}
    assert sobol(bounds, 8, 9) == sobol(bounds, 8, 9)
    rows = latin_hypercube(bounds, 8, 9)
    assert len(rows) == 8
    assert all(0.2 <= row["friction"] <= 1.0 and 0.0 <= row["latency"] <= 0.1 for row in rows)


def test_suite_materialization_freezes_worlds_and_episode_seeds(tmp_path):
    robot_path = FIXTURES / "mini_quadruped.urdf"
    profile = inspect_urdf(str(robot_path))
    profile_path = tmp_path / "profile.json"
    dump_json(str(profile_path), profile)
    suite = tmp_path / "suite.yaml"
    suite.write_text("suite_id: small\nrepetitions: 2\nterrain_codes: [rs1, ae]\ndifficulties: [0.5]\nseed_base: 50\nstressors:\n  sampling: sobol\n  dimensions: [friction_scale, action_latency_s]\n  samples: 2\n", encoding="utf-8")
    controller = tmp_path / "controller.yaml"
    controller.write_text("type: pd_hold\n", encoding="utf-8")
    manifest = materialize_suite(str(suite), str(profile_path), "fake", str(robot_path), "urdf", str(controller), str(tmp_path / "materialized"))
    assert manifest["condition_count"] == 2
    assert manifest["episode_count"] == 4
    assert len(set(manifest["episodes"])) == 4
    episode = load_data(manifest["episodes"][0])
    assert {item["type"] for item in episode["stressors"]} == {"friction_scale", "action_latency_s"}
