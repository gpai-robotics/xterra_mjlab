from pathlib import Path

from robustness_lab.backends.fake import FakeBackend
from robustness_lab.events import detect_events
from robustness_lab.metrics import compute_episode_metrics, metric_map
from robustness_lab.models import EpisodeSpec
from robustness_lab.robot import inspect_urdf
from robustness_lab.worlds import generate_terrain


FIXTURES = Path(__file__).parent / "fixtures"


def test_fake_episode_produces_goal_events_and_metrics(tmp_path):
    robot = inspect_urdf(str(FIXTURES / "mini_quadruped.urdf"))
    world = generate_terrain("rs2", robot, 0.4, 5)
    episode = EpisodeSpec("fake", "fake", "unused", str(FIXTURES / "mini_quadruped.urdf"), "urdf", "unused", {"type": "pd_hold", "command_velocity_mps": 1.0}, str(tmp_path), timeout_s=20.0)
    log = FakeBackend().run(episode, robot, world)
    detect_events(log, world, robot)
    values = metric_map(compute_episode_metrics(log, world, robot))
    assert log.status == "completed"
    assert any(event.event_type == "goal" for event in log.events)
    assert any(event.event_type == "route_complete" for event in log.events)
    assert values["success"] == 1.0
    assert values["completion_time"] > 0
    assert 0 <= values["slip_ratio"] <= 1
    assert values["inference_latency_p99"] >= values["inference_latency_p50"]
