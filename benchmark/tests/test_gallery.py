import os
from pathlib import Path

import pytest

from robustness_lab.gallery import generate_variation_gallery
from robustness_lab.io import dump_json
from robustness_lab.robot import inspect_mjcf


FIXTURES = Path(__file__).parent / "fixtures"


def test_gallery_generates_clickable_render_catalogue(tmp_path):
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    pytest.importorskip("mujoco")
    pytest.importorskip("PIL")
    robot_asset = FIXTURES / "mini_robot.xml"
    profile = inspect_mjcf(str(robot_asset))
    profile_path = tmp_path / "profile.json"
    config_path = tmp_path / "gallery.json"
    dump_json(str(profile_path), profile)
    dump_json(
        str(config_path),
        {
            "title": "Test gallery",
            "gl_backend": "osmesa",
            "variations": [
                {"name": "rough", "terrain": "rs2", "difficulty": 0.4, "seed": 1},
                {"name": "mud-slope", "terrain": "ae", "purple": "rs11", "brown": "rs4", "difficulty": 0.4, "seed": 2},
            ],
        },
    )
    result = generate_variation_gallery(str(config_path), str(profile_path), str(robot_asset), str(tmp_path / "gallery"))
    assert result["variation_count"] == 2
    assert result["rendered_count"] == 2
    assert result["failed_count"] == 0
    assert Path(result["index"]).is_file()
    assert Path(result["contact_sheet"]).is_file()
    assert len(list((tmp_path / "gallery" / "images").glob("*.png"))) == 2
