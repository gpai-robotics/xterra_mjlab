"""An exported mesh must say which faces are mud and which are ice.

`export_obj` wrote one `o <name>` object per box and nothing else. A receiver
loading it in another engine got one uniform surface: a slope at mu = 0.25 and
the same slope at mu = 0.9 were byte-identical files, and a marsh patch was
indistinguishable from asphalt. The shape travelled; the physics did not.

OBJ cannot hold friction, but it can tag faces with a material name (`usemtl`)
and every engine can assign physics per material group. So the export now
writes `mtllib` + `usemtl` per object, an `.mtl` beside it carrying the colours
for viewers, and a `.materials.json` sidecar carrying the actual numbers --
friction, restitution, compliance, damping, roughness -- plus, for marsh
patches, the depth-resistive mud parameters. `.obj` + `.materials.json` is a
complete, repo-independent description.
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _export(tmp_path, code="rs11", camber=0.0, curve=0.0):
    from robustness_lab.worlds import generate_terrain
    from robustness_lab.worlds.export import export_obj
    from robustness_lab.worlds.modifiers import apply_geometry_modifiers
    world = generate_terrain(code, _robot(), 0.5, 5)
    if camber or curve:
        world = apply_geometry_modifiers(world, camber, curve, 28)
    out = tmp_path / ("%s.obj" % code)
    export_obj(world, str(out))
    return world, out


def test_every_object_is_tagged_with_its_material(tmp_path):
    world, out = _export(tmp_path)
    text = out.read_text()
    objects = [line for line in text.splitlines() if line.startswith("o ")]
    tags = [line for line in text.splitlines() if line.startswith("usemtl ")]
    assert objects, "no objects exported"
    assert len(tags) == len(objects), (
        "%d objects but %d usemtl tags" % (len(objects), len(tags)))


def test_material_tags_match_the_world(tmp_path):
    world, out = _export(tmp_path)
    lines = out.read_text().splitlines()
    tagged = {}
    current = None
    for line in lines:
        if line.startswith("o "):
            current = line[2:]
        elif line.startswith("usemtl ") and current:
            tagged[current] = line[7:]
    for item in world.geometry:
        if item.shape == "box":
            assert tagged.get(item.name.replace(" ", "_")) == item.material, item.name


def test_mud_faces_are_distinguishable_from_the_rest(tmp_path):
    """The whole point: a receiver can find the marsh."""
    _, out = _export(tmp_path, "rs11")
    text = out.read_text()
    assert "usemtl compliant" in text
    assert "usemtl nominal" in text


def test_an_mtl_and_a_materials_sidecar_are_written(tmp_path):
    _, out = _export(tmp_path)
    assert out.with_suffix(".mtl").is_file()
    assert Path(str(out).replace(".obj", ".materials.json")).is_file()
    assert "mtllib" in out.read_text()


def test_the_sidecar_carries_the_physics_numbers(tmp_path):
    world, out = _export(tmp_path, "rs11")
    sidecar = json.loads(Path(str(out).replace(".obj", ".materials.json")).read_text())
    assert set(sidecar["materials"]) >= {g.material for g in world.geometry}
    nominal = sidecar["materials"]["nominal"]
    for key in ("static_friction", "dynamic_friction", "restitution",
                "compliance", "damping", "roughness"):
        assert key in nominal, key
    assert nominal["static_friction"] == pytest.approx(0.9)


def test_the_sidecar_carries_the_mud_model_for_marsh_patches(tmp_path):
    world, out = _export(tmp_path, "rs11")
    sidecar = json.loads(Path(str(out).replace(".obj", ".materials.json")).read_text())
    marsh = [o for o in sidecar["objects"] if o["semantic"] == "marsh"]
    assert marsh, "no marsh objects recorded"
    for key in ("support_stress_pa", "yield_stress_pa", "reference_depth_m",
                "depth_exponent", "capability_tier"):
        assert key in marsh[0]["parameters"], key


def test_every_object_in_the_sidecar_is_in_the_obj(tmp_path):
    _, out = _export(tmp_path, "rs5", -12.0, 65.0)
    sidecar = json.loads(Path(str(out).replace(".obj", ".materials.json")).read_text())
    names = {line[2:] for line in out.read_text().splitlines() if line.startswith("o ")}
    assert {o["name"] for o in sidecar["objects"]} == names


def test_a_curved_world_exports_every_segment(tmp_path):
    world, out = _export(tmp_path, "rs5", -12.0, 65.0)
    boxes = sum(1 for g in world.geometry if g.shape == "box")
    objects = sum(1 for line in out.read_text().splitlines() if line.startswith("o "))
    assert objects == boxes


# ---------------------------------------------------------------------------
# MuJoCo loads one shape per OBJ file
# ---------------------------------------------------------------------------
#
# Verified directly: a 35-box world exported as one multi-object OBJ loaded in
# MuJoCo with all 280 vertices but only the FIRST object's 12 faces. The same
# file with the `o` lines removed loaded all 420. So a receiver using MuJoCo
# would see a single box and believe that was the terrain.
#
# The export therefore also writes one single-shape OBJ per material, which
# MuJoCo loads in full -- and grouping by material is exactly how physics gets
# assigned after loading anyway.

def _per_material_files(out):
    return sorted(p for p in out.parent.glob(out.stem + ".*.obj"))


def test_a_single_shape_obj_is_written_per_material(tmp_path):
    world, out = _export(tmp_path, "rs11")
    files = _per_material_files(out)
    materials = {g.material for g in world.geometry if g.shape == "box"}
    assert {p.name.split(".")[-2] for p in files} == materials, files


def test_per_material_files_have_no_object_splits(tmp_path):
    _, out = _export(tmp_path, "rs11")
    for path in _per_material_files(out):
        assert not any(l.startswith("o ") for l in path.read_text().splitlines()), path.name


def test_mujoco_loads_every_face_of_each_per_material_file(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    world, out = _export(tmp_path, "rs5", -12.0, 65.0)
    total = 0
    for path in _per_material_files(out):
        material = path.name.split(".")[-2]
        boxes = sum(1 for g in world.geometry if g.shape == "box" and g.material == material)
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="t" file="%s"/></asset>'
            '<worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>' % path)
        assert model.nmeshface == boxes * 12, (
            "%s: MuJoCo loaded %d triangles for %d boxes" % (path.name, model.nmeshface, boxes))
        total += model.nmeshface
    all_boxes = sum(1 for g in world.geometry if g.shape == "box")
    assert total == all_boxes * 12, "per-material files do not cover the whole world"


def test_the_sidecar_names_the_per_material_files(tmp_path):
    _, out = _export(tmp_path, "rs11")
    sidecar = json.loads(Path(str(out).replace(".obj", ".materials.json")).read_text())
    for material, entry in sidecar["materials"].items():
        if entry.get("mesh"):
            assert (out.parent / entry["mesh"]).is_file(), entry["mesh"]
    assert any(e.get("mesh") for e in sidecar["materials"].values())
