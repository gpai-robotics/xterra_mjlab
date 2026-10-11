"""A run you cannot watch is hard to trust.

`MujocoBackend.run` raised outright:

    if episode.record_video:
        raise BackendUnavailable(
            "MuJoCo v1 adapter does not yet expose a reproducible video renderer")

so `record_video_all_trials` turned every trial into an infrastructure error
rather than a recording. The video plumbing downstream (discrete_examiner's
`media.video` record with its sha256) was only ever fed by the native Isaac
path.

The renderer itself already exists and works headless -- `rendering.py` drives
`mujoco.Renderer`, and the demo figures are produced through it under
MUJOCO_GL=egl. OpenCV is present, so encoding needs no new dependency.

The criterion that matters most is the LAST one: recording must not change what
happened. A recorder that steps the model to grab frames, or that reseeds, would
produce a video of a different episode than the one being scored.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
ASSET = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"


def _setup(tmp_path, record):
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    if not (PROFILE.is_file() and ASSET.is_file()):
        pytest.skip("go2 assets not cached")
    from robustness_lab.io import dump_json, load_data
    from robustness_lab.models import EpisodeSpec, RobotProfile
    from robustness_lab.worlds import generate_terrain

    robot = RobotProfile.from_dict(load_data(str(PROFILE)))
    world = generate_terrain("rs1", robot, 0.2, 5)
    profile_path = tmp_path / "robot.json"
    world_path = tmp_path / "world.json"
    dump_json(str(profile_path), robot)
    dump_json(str(world_path), world)
    episode = EpisodeSpec(
        episode_id="video-%s" % record, backend="mujoco",
        robot_profile=str(profile_path), robot_asset=str(ASSET),
        robot_format="mjcf", world=str(world_path),
        controller={"type": "pd_hold"},
        output_dir=str(tmp_path / ("run-%s" % record)),
        seed=11, timeout_s=2.0, record_video=record,
    )
    return episode, robot, world


def _run(tmp_path, record):
    from robustness_lab.backends.mujoco import MujocoBackend
    episode, robot, world = _setup(tmp_path, record)
    return episode, MujocoBackend().run(episode, robot, world)


def _video_path(episode, log):
    declared = log.diagnostics.get("video")
    if declared:
        return Path(declared)
    candidates = sorted(Path(episode.output_dir).rglob("*.mp4"))
    return candidates[0] if candidates else None


def test_recording_is_no_longer_refused(tmp_path):
    episode, log = _run(tmp_path, True)
    assert log.status in {"completed", "failed"}, log.terminal_reason
    assert log.terminal_reason != "adapter_unverified"


def test_a_video_file_is_written_and_is_not_empty(tmp_path):
    episode, log = _run(tmp_path, True)
    path = _video_path(episode, log)
    assert path is not None and path.is_file(), "no video was written"
    assert path.stat().st_size > 1024, "video file is a stub"


def test_the_video_holds_more_than_one_frame(tmp_path):
    """A single still is not a recording."""
    import cv2
    episode, log = _run(tmp_path, True)
    capture = cv2.VideoCapture(str(_video_path(episode, log)))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    assert frames > 1, "video contains %d frame(s)" % frames


def test_the_episode_log_points_at_the_video_with_a_hash(tmp_path):
    episode, log = _run(tmp_path, True)
    assert log.diagnostics.get("video"), "the log must say where the video is"
    digest = log.diagnostics.get("video_sha256")
    assert isinstance(digest, str) and len(digest) == 64, digest
    overlays = log.diagnostics.get("video_overlays")
    assert overlays and overlays["coordinate_frame"] == "world"
    assert overlays["simulation_coupling"] == "renderer_only"
    assert overlays["scope"] == "3d_episode_video_only"
    assert overlays["command_velocity_body_to_world"] == "orange"
    assert overlays["measured_velocity_world"] == "green"


def test_nothing_is_written_when_recording_was_not_asked_for(tmp_path):
    episode, log = _run(tmp_path, False)
    assert not log.diagnostics.get("video")
    assert not list(Path(episode.output_dir).rglob("*.mp4"))


def test_recording_does_not_change_what_happened(tmp_path):
    """The video must be OF the scored episode, not of a second one.

    Same seed, same world, recording on and off: identical outcome and identical
    final base position. A recorder that steps the model to grab frames, or that
    touches the RNG, fails here.
    """
    _, plain = _run(tmp_path / "plain", False)
    _, taped = _run(tmp_path / "taped", True)

    assert plain.terminal_reason == taped.terminal_reason
    assert len(plain.samples) == len(taped.samples)
    assert list(taped.samples[-1].base_position_m) == pytest.approx(
        list(plain.samples[-1].base_position_m), abs=1e-9)
