"""Reproducible visual galleries for morphology-scaled world variations."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List

from .backends.mujoco import build_mjcf_scene
from .io import atomic_write_text, dump_json, load_data
from .models import RobotProfile
from .rendering import render_mjcf_scene
from .worlds import apply_geometry_modifiers, generate_ae, generate_aee, generate_terrain


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError("variation name must contain a letter or number")
    return result


def _make_world(profile: RobotProfile, item: Dict[str, Any]):
    terrain = str(item["terrain"]).lower()
    difficulty = float(item.get("difficulty", 0.5))
    seed = int(item.get("seed", 0))
    overrides = dict(item.get("set", {}))
    if terrain == "ae":
        world = generate_ae(
            profile,
            str(item.get("purple", "rs1")),
            str(item.get("brown", "rs2")),
            difficulty,
            seed,
            bool(item.get("reversed", False)),
            overrides,
        )
    elif terrain == "aee":
        for key in ("rows", "columns", "pair_mode", "purple", "brown"):
            if key in item:
                overrides.setdefault(key, item[key])
        world = generate_aee(profile, str(item.get("layout", "long")), difficulty, seed, int(item.get("count", 6)), overrides)
    else:
        world = generate_terrain(terrain, profile, difficulty, seed, overrides)
    return apply_geometry_modifiers(world, float(item.get("camber_deg", 0.0)), float(item.get("curve_angle_deg", 0.0)), int(item.get("curve_segments", 24)))


def _gallery_html(title: str, records: List[Dict[str, Any]]) -> str:
    cards = []
    for record in records:
        if record["status"] != "rendered":
            cards.append(
                '<article class="card error"><h2>%s</h2><p>%s</p></article>'
                % (html.escape(record["title"]), html.escape(record.get("error", "unknown error")))
            )
            continue
        dimensions = record["dimensions_m"]
        parameters = html.escape(str(record["parameters"]))
        cards.append(
            """<article class="card">
<a href="{image}"><img loading="lazy" src="{image}" alt="{title}"></a>
<div class="body"><h2>{title}</h2><p class="type">{terrain}</p>
<p>{length:.2f} × {width:.2f} m · difficulty {difficulty:.2f} · seed {seed}</p><p>{modifier}</p>
<details><summary>Resolved parameters</summary><pre>{parameters}</pre></details>
<p class="links"><a href="{world}">World JSON</a> · <a href="{scene}">MuJoCo XML</a> · <a href="{image}">PNG</a></p></div>
</article>""".format(
                image=html.escape(record["image"]),
                title=html.escape(record["title"]),
                terrain=html.escape(record["terrain"]),
                length=dimensions[0],
                width=dimensions[1],
                difficulty=record["difficulty"],
                seed=record["seed"],
                parameters=parameters,
                modifier=html.escape(record.get("modifier_summary", "Straight and level geometry")),
                world=html.escape(record["world"]),
                scene=html.escape(record["scene"]),
            )
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b1017; color: #e8eef7; }}
header {{ padding: 28px clamp(18px,4vw,60px); background: #111a25; border-bottom: 1px solid #263548; }}
h1 {{ margin: 0 0 8px; }} header p {{ margin: 0; color: #aebed1; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 18px; padding: 24px; }}
.card {{ overflow: hidden; border: 1px solid #29394d; border-radius: 12px; background: #121c28; box-shadow: 0 8px 24px #0005; }}
.card img {{ display: block; width: 100%; aspect-ratio: 16/9; object-fit: cover; background: #000; }}
.body {{ padding: 16px; }} h2 {{ margin: 0; font-size: 1.08rem; }}
.type {{ color: #83c5ff; text-transform: uppercase; letter-spacing: .08em; font-size: .78rem; }}
p {{ color: #b9c7d7; }} a {{ color: #76c7ff; }} details {{ color: #cdd8e5; }}
pre {{ white-space: pre-wrap; font-size: .75rem; background: #091019; padding: 10px; border-radius: 8px; max-height: 220px; overflow: auto; }}
.error {{ border-color: #a33; padding: 18px; }}
</style></head><body><header><h1>{title}</h1><p>{count} deterministic morphology-scaled MuJoCo world variations. Click any render for the full PNG.</p></header>
<main>{cards}</main></body></html>
""".format(title=html.escape(title), count=len(records), cards="\n".join(cards))


def _contact_sheet(output: Path, records: List[Dict[str, Any]]) -> str:
    from PIL import Image, ImageDraw

    rendered = [record for record in records if record["status"] == "rendered"]
    columns = 4
    cell_width, image_height, caption_height = 320, 180, 46
    rows = max(1, (len(rendered) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + caption_height)), "#0b1017")
    draw = ImageDraw.Draw(sheet)
    resampling = getattr(Image, "Resampling", Image)
    for index, record in enumerate(rendered):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * (image_height + caption_height)
        source = Image.open(str(output / record["image"])).convert("RGB")
        source.thumbnail((cell_width, image_height), resampling.LANCZOS)
        offset_x = x + (cell_width - source.width) // 2
        offset_y = y + (image_height - source.height) // 2
        sheet.paste(source, (offset_x, offset_y))
        title = record["title"].replace("\u2014", "-").replace("\u00b0", " deg").replace("\u00d7", "x")
        if len(title) > 43:
            title = title[:40] + "..."
        draw.text((x + 8, y + image_height + 7), title, fill="#e8eef7")
        draw.text(
            (x + 8, y + image_height + 24),
            "%.2f x %.2f m | d=%.2f" % (record["dimensions_m"][0], record["dimensions_m"][1], record["difficulty"]),
            fill="#91a8c1",
        )
    destination = output / "contact-sheet.png"
    sheet.save(str(destination))
    return str(destination.resolve())


def generate_variation_gallery(config_path: str, robot_profile_path: str, robot_asset: str, output_dir: str) -> Dict[str, Any]:
    config = load_data(config_path)
    profile = RobotProfile.from_dict(load_data(robot_profile_path))
    output = Path(output_dir).resolve()
    worlds_dir, scenes_dir, images_dir = output / "worlds", output / "scenes", output / "images"
    for directory in (worlds_dir, scenes_dir, images_dir):
        directory.mkdir(parents=True, exist_ok=True)
    variation_items = list(config.get("variations", []))
    matrix = config.get("modifier_matrix", {})
    for terrain in matrix.get("terrains", []):
        for variant in matrix.get("variants", []):
            item = {**matrix.get("defaults", {}), **variant, "terrain": terrain}
            suffix = str(item.pop("suffix", "variation"))
            item["name"] = "%s-%s" % (terrain, suffix)
            item["title"] = str(item.get("title", "{terrain} — " + suffix)).format(terrain=str(terrain).upper())
            variation_items.append(item)
    records: List[Dict[str, Any]] = []
    seen = set()
    for index, raw_item in enumerate(variation_items):
        item = dict(raw_item)
        title = str(item.get("title", item.get("name", "Variation %d" % (index + 1))))
        slug = _slug(str(item.get("name", title)))
        if slug in seen:
            raise ValueError("duplicate gallery variation name: %s" % slug)
        seen.add(slug)
        record: Dict[str, Any] = {"name": slug, "title": title, "status": "error"}
        try:
            world = _make_world(profile, item)
            world_path = worlds_dir / (slug + ".json")
            scene_path = scenes_dir / (slug + ".xml")
            image_path = images_dir / (slug + ".png")
            dump_json(str(world_path), world)
            build_mjcf_scene(robot_asset, world, str(scene_path), float(config.get("dt_s", 0.002)))
            render = render_mjcf_scene(
                str(scene_path),
                world,
                str(image_path),
                int(config.get("width_px", 640)),
                int(config.get("height_px", 360)),
                float(item.get("azimuth_deg", config.get("azimuth_deg", 138.0))),
                float(item.get("elevation_deg", config.get("elevation_deg", -30.0))),
                float(item.get("distance_scale", config.get("distance_scale", 0.78))),
                str(config.get("gl_backend", "osmesa")),
            )
            record.update(
                {
                    "status": "rendered",
                    "terrain": world.terrain_type,
                    "difficulty": float(item.get("difficulty", 0.5)),
                    "seed": int(item.get("seed", 0)),
                    "dimensions_m": world.dimensions_m,
                    "parameters": world.resolved_parameters,
                    "modifier_summary": (
                        "Sideways tilt %.1f° · route curvature %+.1f° (%s)"
                        % (
                            float(world.metadata.get("geometry_modifier", {}).get("camber_deg", 0.0)),
                            float(world.metadata.get("geometry_modifier", {}).get("curve_angle_deg", 0.0)),
                            world.metadata.get("geometry_modifier", {}).get("curve_direction", "straight"),
                        )
                    ),
                    "clips": world.clips,
                    "world": "worlds/%s.json" % slug,
                    "scene": "scenes/%s.xml" % slug,
                    "image": "images/%s.png" % slug,
                    "render": render,
                }
            )
        except Exception as error:  # keep the catalogue auditable when one entry fails
            record["error"] = "%s: %s" % (type(error).__name__, error)
        records.append(record)
    result = {
        "title": str(config.get("title", "World variation gallery")),
        "config": str(Path(config_path).resolve()),
        "robot_profile": str(Path(robot_profile_path).resolve()),
        "robot_asset": str(Path(robot_asset).resolve()),
        "variation_count": len(records),
        "rendered_count": sum(record["status"] == "rendered" for record in records),
        "failed_count": sum(record["status"] != "rendered" for record in records),
        "variations": records,
    }
    result["contact_sheet"] = _contact_sheet(output, records)
    dump_json(str(output / "manifest.json"), result)
    atomic_write_text(str(output / "index.html"), _gallery_html(result["title"], records))
    result["index"] = str((output / "index.html").resolve())
    result["manifest"] = str((output / "manifest.json").resolve())
    return result
