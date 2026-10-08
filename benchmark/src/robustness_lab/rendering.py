"""Headless MuJoCo rendering for simulator-world inspection artifacts."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .models import WorldSpec


def make_mjcf_camera(
    mujoco: Any,
    world: WorldSpec,
    azimuth_deg: float = 138.0,
    elevation_deg: float = -30.0,
    distance_scale: float = 0.78,
    focus_bounds: Optional[Sequence[float]] = None,
) -> Any:
    """Build the overview camera shared by still and episode renderers."""
    x_values = []
    y_values = []
    z_values = []
    for geometry in world.geometry:
        if geometry.shape != "box":
            continue
        for axis, values in enumerate((x_values, y_values, z_values)):
            center = geometry.pose.position[axis]
            half = geometry.size[axis] / 2.0
            values.extend((center - half, center + half))
    minimum_x, maximum_x = min(x_values, default=0.0), max(
        x_values, default=world.dimensions_m[0]
    )
    minimum_y, maximum_y = min(y_values, default=-0.5), max(
        y_values, default=0.5
    )
    if focus_bounds is not None:
        if len(focus_bounds) != 4:
            raise ValueError("focus_bounds must be [min_x, max_x, min_y, max_y]")
        minimum_x, maximum_x, minimum_y, maximum_y = [
            float(value) for value in focus_bounds
        ]
    minimum_z, maximum_z = min(z_values, default=0.0), max(
        z_values, default=0.5
    )
    span_x = max(maximum_x - minimum_x, 0.5)
    span_y = max(maximum_y - minimum_y, 0.5)
    span_z = max(maximum_z - minimum_z, 0.25)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [
        (minimum_x + maximum_x) / 2.0,
        (minimum_y + maximum_y) / 2.0,
        max(0.1, (minimum_z + maximum_z) / 2.0),
    ]
    camera.distance = max(
        span_x * distance_scale, span_y * 1.7, span_z * 3.0, 2.0
    )
    camera.azimuth = azimuth_deg
    camera.elevation = elevation_deg
    return camera


def render_mjcf_scene(
    scene_path: str,
    world: WorldSpec,
    output_path: str,
    width: int = 640,
    height: int = 360,
    azimuth_deg: float = 138.0,
    elevation_deg: float = -30.0,
    distance_scale: float = 0.78,
    gl_backend: str = "osmesa",
    focus_bounds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Render a deterministic overview PNG without requiring DISPLAY.

    MuJoCo's default offscreen framebuffer is 640x480, so v1 deliberately
    caps previews at that size. Interactive inspection remains available via
    ``python -m mujoco.viewer``.
    """

    if width <= 0 or height <= 0 or width > 640 or height > 480:
        raise ValueError("preview dimensions must be positive and no larger than 640x480")
    os.environ.setdefault("MUJOCO_GL", gl_backend)
    if gl_backend == "osmesa":
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    try:
        import mujoco
        from PIL import Image
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("headless rendering requires robustness-lab[mujoco]: %s" % error)

    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    camera = make_mjcf_camera(
        mujoco,
        world,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        distance_scale=distance_scale,
        focus_bounds=focus_bounds,
    )

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(str(destination))
    finally:
        renderer.close()

    return {
        "output": str(Path(output_path).resolve()),
        "width_px": width,
        "height_px": height,
        "gl_backend": gl_backend,
        "camera": {
            "lookat_m": [float(value) for value in camera.lookat],
            "distance_m": float(camera.distance),
            "azimuth_deg": float(camera.azimuth),
            "elevation_deg": float(camera.elevation),
        },
        "model": {"bodies": int(model.nbody), "geometries": int(model.ngeom), "meshes": int(model.nmesh)},
    }


def render_topdown(world: WorldSpec, output_path: str, width: int = 900, height: int = 700) -> Dict[str, Any]:
    """Render a semantic plan view showing the route, graph nodes, spawn and goal."""
    try:
        from PIL import Image, ImageDraw
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("top-down rendering requires Pillow: %s" % error)
    if width < 200 or height < 200:
        raise ValueError("top-down image must be at least 200x200")
    boxes = [item for item in world.geometry if item.shape == "box"]
    x0 = min((item.pose.position[0] - item.size[0] / 2 for item in boxes), default=0.0)
    x1 = max((item.pose.position[0] + item.size[0] / 2 for item in boxes), default=world.dimensions_m[0])
    y0 = min((item.pose.position[1] - item.size[1] / 2 for item in boxes), default=-world.dimensions_m[1] / 2)
    y1 = max((item.pose.position[1] + item.size[1] / 2 for item in boxes), default=world.dimensions_m[1] / 2)
    margin = 48
    scale = min((width - 2 * margin) / max(x1 - x0, 0.1), (height - 2 * margin) / max(y1 - y0, 0.1))

    def pixel(point: Sequence[float]) -> tuple:
        return (margin + (point[0] - x0) * scale, height - margin - (point[1] - y0) * scale)

    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    for item in boxes:
        material_name = str(item.parameters.get("display_material", item.material))
        material = world.materials.get(material_name) or world.materials.get(item.material)
        color = material.color if material else "#718096"
        roll, pitch, yaw = item.pose.rpy
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rotation = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        polygon = []
        for local_x, local_y in ((-item.size[0] / 2, -item.size[1] / 2), (item.size[0] / 2, -item.size[1] / 2), (item.size[0] / 2, item.size[1] / 2), (-item.size[0] / 2, item.size[1] / 2)):
            local = [local_x, local_y, item.size[2] / 2]
            point = [item.pose.position[axis] + sum(rotation[axis][column] * local[column] for column in range(3)) for axis in range(3)]
            polygon.append(pixel(point))
        draw.polygon(polygon, fill=color, outline="#334e68")
    route = next((item for item in world.routes if item.route_kind in {"graph-route", "central"}), world.routes[0] if world.routes else None)
    if route:
        draw.line([pixel(point) for point in route.waypoints], fill="#00a896", width=4, joint="curve")
    for label, pose, color in (("S", world.spawn_pose, "#18a558"), ("G", world.goal_pose, "#d64545")):
        x, y = pixel(pose.position)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=2)
        draw.text((x + 12, y - 9), label, fill="#102a43")
    for node in world.metadata.get("graph_nodes", []):
        x, y = pixel(node["position_m"])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ffd166", outline="#102a43")
    draw.text((12, 12), "%s | top-down | %.1f m x %.1f m" % (world.name, x1 - x0, y1 - y0), fill="#102a43")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(destination))
    return {"output": str(destination.resolve()), "width_px": width, "height_px": height}
