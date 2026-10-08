"""`record_video_all_trials` must reach the simulator path, not just the native one.

`_prepare_trial` hardcoded

    "record_video": False,

in the episode it writes for the simulator-adapter backends, while the setting
was threaded only into the native batch payload (`"record_video": bool(
state["settings"].get("record_video_all_trials", False))`). So a MuJoCo
examination with `record_video_all_trials: true` silently recorded nothing --
the option was accepted, validated, and ignored.

An option that is read from your config and then dropped is worse than one that
does not exist: the run looks configured.
"""

import pytest
from pathlib import Path


def _episode_payload(tmp_path, record_all):
    """Write one trial's episode spec and read back what it declared."""
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "assets/cache/go2_mjcf_profile.json"
    asset = root / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
    if not (profile_path.is_file() and asset.is_file()):
        pytest.skip("go2 assets not cached")

    from robustness_lab.discrete_examiner import _prepare_trial
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    from robustness_lab.skill import load_skill

    robot = RobotProfile.from_dict(load_data(str(profile_path)))
    skill_path = root / "submissions/go2_blind/skill.yaml"
    if not skill_path.is_file():
        pytest.skip("go2_blind submission not present")
    skill = load_skill(str(skill_path), check_files=False)

    condition = {
        "id": "L0-flat", "level": "L0", "kind": "qualification",
        "world_seed": 5, "attempts": [], "geometry_label": "flat",
        "geometry_level": 0.0, "static_friction": 0.5, "dynamic_friction": 0.4,
        "stressors": [],
    }
    state = {
        "examiner_id": "video-option", "condition_order": ["L0-flat"],
        "settings": {
            "output_dir": str(tmp_path), "backend": "mujoco", "seed": 1,
            "robot_profile": str(profile_path), "robot_asset": str(asset),
            "robot_format": "mjcf",
            "timeout_s": 2.0, "event_thresholds": {},
            "record_video_all_trials": record_all,
        },
    }
    prepared = _prepare_trial(state, condition, robot, skill)
    return load_data(str(prepared["episode_path"]))


def test_the_option_switches_recording_on(tmp_path):
    assert _episode_payload(tmp_path, True)["record_video"] is True


def test_recording_stays_off_by_default(tmp_path):
    assert _episode_payload(tmp_path, False)["record_video"] is False
