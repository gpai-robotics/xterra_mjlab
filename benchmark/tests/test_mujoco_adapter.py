from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from robustness_lab.backends.mujoco import build_mjcf_scene
from robustness_lab.robot import inspect_mjcf
from robustness_lab.worlds import TERRAIN_CODES, generate_ae, generate_terrain


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("terrain", TERRAIN_CODES)
def test_generated_mjcf_compiles(tmp_path, terrain):
    mujoco = pytest.importorskip("mujoco")
    asset = FIXTURES / "mini_robot.xml"
    profile = inspect_mjcf(str(asset))
    world = generate_terrain(terrain, profile, 0.3, 5)
    scene = tmp_path / "scene.xml"
    build_mjcf_scene(str(asset), world, str(scene), 0.005)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.ngeom > 2


def test_generated_mjcf_preserves_world_material_colors(tmp_path):
    asset = FIXTURES / "mini_robot.xml"
    profile = inspect_mjcf(str(asset))
    world = generate_terrain("rs3", profile, 0.3, 5)
    scene = tmp_path / "scene.xml"
    build_mjcf_scene(str(asset), world, str(scene), 0.005)
    root = ET.parse(str(scene)).getroot()
    slope = root.find(".//geom[@name='terrain_slope']")
    assert slope is not None
    assert slope.get("rgba") == "1 0.94902 0 1"


def test_ae_mjcf_separates_slot_color_from_contact_physics(tmp_path):
    asset = FIXTURES / "mini_robot.xml"
    profile = inspect_mjcf(str(asset))
    world = generate_ae(profile, "rs11", "rs4", 0.5, 5)
    scene = tmp_path / "scene.xml"
    build_mjcf_scene(str(asset), world, str(scene), 0.005)
    root = ET.parse(str(scene)).getroot()
    marsh = next(item for item in root.findall(".//geom") if (item.get("name") or "").startswith("terrain_purple-marsh-patch"))
    assert marsh.get("rgba") == "0.470588 0.215686 0.666667 1"
    assert marsh.get("friction").startswith("0.5 ")
    assert marsh.get("solref") is not None
