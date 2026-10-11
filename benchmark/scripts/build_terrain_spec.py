#!/usr/bin/env python3
"""Build a self-contained, engine-independent terrain specification.

Everything here is generated from the current code, so re-running it after any
terrain change produces an updated set. Nothing is copied from the stale demo
bundle.

Coverage matches the benchmark's own L1 matrix, plus the banked/curved
variants:

    RS1-RS11
    x {plain, left-curve-left-low, right-curve-right-low}
    x difficulty {0.25, 0.50, 0.75}
    x friction   {high, moderate, low}        (the same levels the examiner uses)
    = 297 worlds

For each world, files sharing one name:

    <name>.json             WorldSpec -- boxes + materials + spawn/goal/routes.
                            The portable source; any engine with an adapter.
    <name>.obj              mesh, every face tagged with its material (usemtl)
    <name>.mtl              colours for viewers
    <name>.materials.json   the physics numbers per material, per-object
                            semantics, mud parameters for marsh patches
    <name>.xml              MuJoCo scene (terrain + a Go2 for scale)
    <name>.png              image

Usage:
    MUJOCO_GL=egl PYTHONPATH=src python3 scripts/build_terrain_spec.py [--output DIR] [--no-render]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustness_lab.io import dump_json, load_data  # noqa: E402
from robustness_lab.models import RobotProfile  # noqa: E402
from robustness_lab.worlds import apply_challenge_friction, generate_terrain  # noqa: E402
from robustness_lab.worlds.export import export_obj  # noqa: E402
from robustness_lab.worlds.modifiers import apply_geometry_modifiers  # noqa: E402

TERRAINS = ["rs%d" % index for index in range(1, 12)]
TERRAIN_NAMES = {
    "rs1": "Slippery flat ground", "rs2": "Rough slippery ground", "rs3": "Slippery slope",
    "rs4": "Rough slippery slope", "rs5": "Stairs", "rs6": "Repeated waves", "rs7": "Curved hill",
    "rs8": "Small repeated gaps", "rs9": "Large gap", "rs10": "High step", "rs11": "Mud / marsh",
}
# The two variants the gallery ships, and their seeds.
VARIANTS = (
    ("plain", 0.0, 0.0, 3100),
    ("left-curve-left-low", -12.0, 65.0, 3101),
    ("right-curve-right-low", 12.0, -65.0, 3102),
)
DIFFICULTIES = (("d025", 0.25), ("d050", 0.50), ("d075", 0.75))
# Identical to discrete_examiner.DEFAULT_FRICTION_LEVELS.
FRICTIONS = (("fhigh", 0.75, 0.60), ("fmoderate", 0.50, 0.40), ("flow", 0.25, 0.20))


def build(output: Path, robot_profile: str, robot_asset: str, render: bool) -> dict:
    robot = RobotProfile.from_dict(load_data(robot_profile))
    worlds_dir = output / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=True)
    index = []
    started = time.perf_counter()

    if render:
        from robustness_lab.backends.mujoco import build_mjcf_scene
        from robustness_lab.rendering import render_mjcf_scene

    for terrain in TERRAINS:
        for variant, camber, curve, seed in VARIANTS:
            for d_label, difficulty in DIFFICULTIES:
                for f_label, static, dynamic in FRICTIONS:
                    name = "%s-%s-%s-%s" % (terrain, variant, d_label, f_label)
                    world = generate_terrain(terrain, robot, difficulty, seed)
                    if camber or curve:
                        world = apply_geometry_modifiers(world, camber, curve, 28)
                    world = apply_challenge_friction(world, static, dynamic)
                    world.name = name

                    base = worlds_dir / name
                    dump_json(str(base) + ".json", world)
                    export_obj(world, str(base) + ".obj")
                    if render:
                        build_mjcf_scene(robot_asset, world, str(base) + ".xml", 0.005)
                        render_mjcf_scene(str(base) + ".xml", world, str(base) + ".png",
                                          gl_backend="egl")
                    index.append({
                        "name": name, "terrain": terrain, "terrain_name": TERRAIN_NAMES[terrain],
                        "variant": variant, "camber_deg": camber, "curve_angle_deg": curve,
                        "difficulty": difficulty, "friction": f_label[1:],
                        "static_friction": static, "dynamic_friction": dynamic,
                        "seed": seed, "length_m": round(float(world.dimensions_m[0]), 3),
                        "boxes": sum(1 for g in world.geometry if g.shape == "box"),
                    })
        print("  %s done (%d worlds so far, %.0fs)" % (terrain, len(index), time.perf_counter() - started), flush=True)

    dump_json(str(output / "index.json"), {"schema_version": "1.0", "worlds": index,
                                            "count": len(index), "robot_profile": robot_profile})
    (output / "README.md").write_text(_readme(len(index)), encoding="utf-8")
    return {"count": len(index), "output": str(output)}


def _readme(count: int) -> str:
    return f"""# RobustLab terrain specification

{count} single-terrain worlds, generated from the current code, covering the
benchmark's L1 matrix plus its banked/curved variants:

    RS1-RS11  x  {{plain, left-curve-left-low, right-curve-right-low}}
              x  difficulty {{0.25, 0.50, 0.75}}
              x  friction   {{high 0.75/0.60, moderate 0.50/0.40, low 0.25/0.20}}

Each world is a set of files sharing one name under `worlds/`:

| file | what it is | use it for |
|---|---|---|
| `.json` | WorldSpec: boxes with pose/size, named materials with friction, compliance, damping; spawn, goal, routes, regions | any engine with an adapter (MuJoCo, Isaac Lab, Isaac Gym have one) -- this is the source of truth |
| `.obj` | triangle mesh, one object per box, **every face tagged with its material** (`usemtl`) | Blender, Unity, Isaac Sim, trimesh -- anything that reads multi-object OBJ |
| `.<material>.obj` | one single-shape mesh per material (e.g. `.nominal.obj`, `.compliant.obj`) | **MuJoCo**, which loads one shape per file -- and any engine where you want to assign physics per file |
| `.mtl` | colours per material | viewers |
| `.materials.json` | the physics numbers per material; each object's semantic tag; the mud model parameters for marsh patches | apply the right physics to the right faces after loading the `.obj` |
| `.xml` | MuJoCo scene with a Go2 for scale | drop straight into MuJoCo |
| `.png` | image | looking |

`index.json` lists every world with its terrain, variant, difficulty, friction, seed and size.

**MuJoCo note.** MuJoCo's OBJ loader takes exactly one shape per file: given the
combined multi-object `.obj` it silently loads only the first box. Use the
per-material files instead -- each is one shape and loads in full:

    <asset>
      <mesh name="ground" file="rs5-plain-d050-fmoderate.nominal.obj"/>
      <mesh name="steps"  file="rs5-plain-d050-fmoderate.challenge.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="ground" friction="0.9 0.005 0.0001"/>
      <geom type="mesh" mesh="steps"  friction="0.75 0.005 0.0001"/>
    </worldbody>

The friction values are in `.materials.json`. RS11's `compliant` mesh is the mud
volume and should not be a plain collider -- see the mud note below.

## The terrains

| code | terrain | what difficulty scales |
|---|---|---|
| RS1 | Slippery flat ground | friction only |
| RS2 | Rough slippery ground | roughness amplitude |
| RS3 | Slippery slope | slope angle |
| RS4 | Rough slippery slope | slope angle + roughness |
| RS5 | Stairs | step height / count |
| RS6 | Repeated waves | wave amplitude |
| RS7 | Curved hill | hill height |
| RS8 | Small repeated gaps | gap width |
| RS9 | Large gap | gap width |
| RS10 | High step | step height |
| RS11 | Mud / marsh | mud depth, softness (support stress), suction (yield stress) |

Every dimension is scaled to the robot's morphology (body length, foot size,
stance), so the same code produces proportionally different worlds for a
different robot.

## Banked and curved variants

`left-curve-left-low`: the route bends anticlockwise by 65 degrees and the
floor is banked 12 degrees with the left (+Y) side low. `right-curve-right-low`
is the mirror. These are real geometry: long boxes are segmented so collision
follows the arc, not a visual rotation. The spawn pose carries the matching
roll and yaw.

## Physics

Materials (per world; `rough_low` roughness and `compliant` softness scale with
difficulty):

| material | static / dynamic friction | notes |
|---|---|---|
| nominal | 0.90 / 0.80 | firm ground |
| low_friction | set by the friction level | slippery |
| rough_low | set by the friction level | slippery + rough |
| challenge / purple | 0.75 / 0.65 | challenge pieces |
| compliant | 0.60 / 0.50 | the mud volume (non-colliding; force model below) |

Mud (RS11) is not a rigid surface. The marsh volume does not collide; a rigid
backstop sits beneath it, and a per-foot force model provides support by
penetration depth (rate-independent, `support_stress_pa * (z/H)^beta`),
damping, and suction on withdrawal capped at `yield_stress_pa`. The
parameters are on each marsh object in `.materials.json`. Reference:
arXiv:2403.02617. Not modelled: rutting, thixotropy, shear-strength traction.

## Reproducing or extending

Any world here is a pure function of (terrain, robot profile, difficulty,
seed, camber, curve, friction). To regenerate the whole set after a code
change:

    MUJOCO_GL=egl PYTHONPATH=src python3 scripts/build_terrain_spec.py --output <dir>

To make one world for another robot, pass that robot's profile.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT / "terrain_spec"))
    parser.add_argument("--robot-profile", default=str(ROOT / "assets/cache/go2_mjcf_profile.json"))
    parser.add_argument("--robot-asset", default=str(ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"))
    parser.add_argument("--no-render", action="store_true", help="skip .xml and .png (json/obj only)")
    args = parser.parse_args()
    result = build(Path(args.output), args.robot_profile, args.robot_asset, render=not args.no_render)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
