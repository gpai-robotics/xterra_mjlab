"""Generate plain-language annotated PNG diagrams for RS1--RS11."""

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .io import atomic_write_text, dump_json


WIDTH, HEIGHT = 1800, 1000
INK = "#102a43"
MUTED = "#627d98"
BG = "#eef3f7"
PANEL = "#ffffff"
NOMINAL = "#236b8e"
SLIPPERY = "#f5df00"
ROUGH = "#e7a900"
CHALLENGE = "#873e13"
MUD = "#687b51"
ACCENT = "#00a896"
RED = "#d64545"


TERRAINS: Dict[str, Dict[str, Any]] = {
    "rs1": {"title": "RS1 — Slippery flat ground", "subtitle": "Tests traction loss on a level surface", "difficulty": "Higher difficulty reduces surface friction; the slippery section length is fixed at five body lengths.", "labels": [("Slippery-section length", "How long traction remains reduced"), ("Surface friction", "Lower friction causes more foot slip"), ("Travel direction", "Forward or reverse traversal"), ("Approach / recovery length", "Space before and after the challenge")]},
    "rs2": {"title": "RS2 — Rough and slippery ground", "subtitle": "Combines uncertain footholds with reduced traction", "difficulty": "Higher difficulty raises bump height (up to one foot length); section length and slipperiness are fixed.", "labels": [("Rough-section length", "Total distance over uneven ground"), ("Bump height", "Vertical size of surface irregularities"), ("Bump arrangement", "Seed controls reproducible positions and heights"), ("Surface friction", "Traction available on the base and bumps")]},
    "rs3": {"title": "RS3 — Slippery straight slope", "subtitle": "Tests climbing or descending with limited traction", "difficulty": "Higher difficulty steepens the slope in proportion to difficulty (up to 45°); length is fixed.", "labels": [("Slope angle", "Steepness of the incline or decline"), ("Horizontal slope length", "Ground distance covered by the ramp"), ("Vertical rise", "Resulting height change"), ("Surface friction", "Available climbing/braking traction"), ("Direction", "Uphill in forward mode; downhill when reversed")]},
    "rs4": {"title": "RS4 — Rough slippery slope", "subtitle": "Adds irregular footholds to a low-friction incline", "difficulty": "Higher difficulty steepens the slope (up to 45°) and raises roughness with it; length is fixed.", "labels": [("Slope angle", "Steepness of the incline or decline"), ("Slope length", "Horizontal distance across the ramp"), ("Roughness height", "Size of bumps on the slope"), ("Roughness arrangement", "Seed controls repeatable bump placement"), ("Surface friction", "Traction on the ramp and bumps"), ("Direction", "Uphill or downhill traversal")]},
    "rs5": {"title": "RS5 — Slippery stairs", "subtitle": "Tests repeated step-up or step-down contacts", "difficulty": "Higher difficulty adds more steps — a longer flight to climb — while step height stays fixed at 40% of standing height.", "labels": [("Number of steps", "How many consecutive level changes occur"), ("Step height", "Vertical rise of each stair"), ("Step depth", "Available foothold length on each tread"), ("Surface friction", "Traction available on stair surfaces"), ("Direction", "Upstairs forward; downstairs when reversed")]},
    "rs6": {"title": "RS6 — Repeated up/down waveform", "subtitle": "Tests continuous body-height and pitch adjustment", "difficulty": "Higher difficulty packs more full waves into a fixed length; wave height stays fixed.", "labels": [("Course length", "Total distance occupied by the waves"), ("Wave height", "Maximum vertical excursion"), ("Surface smoothness", "More segments create a smoother approximation"), ("Number of waves", "Repeated elevation cycles along the route"), ("Direction", "Forward or reversed waveform traversal")]},
    "rs7": {"title": "RS7 — Smooth curved hill", "subtitle": "Tests gradual ascent, cresting and descent", "difficulty": "Higher difficulty makes the fixed-height hill blockier — fewer, sharper segments — not taller.", "labels": [("Hill length", "Horizontal distance from start to end"), ("Hill height", "Maximum elevation at the crest"), ("Surface smoothness", "Number of segments used for the curve"), ("Direction", "Normal or reversed traversal"), ("Surface friction", "Traction over the curved surface")]},
    "rs8": {"title": "RS8 — Repeated small gaps", "subtitle": "Tests precise repeated foothold placement", "difficulty": "Higher difficulty adds gaps and makes each gap wider.", "labels": [("Number of gaps", "How many interruptions must be crossed"), ("Gap width", "Unsupported distance between platforms"), ("Platform length", "Available foothold space between gaps"), ("Gap depth", "Failure/contact region below the surface"), ("Track width", "Side-to-side room available to the robot")]},
    "rs9": {"title": "RS9 — One large gap", "subtitle": "Tests a committed long step or jump", "difficulty": "Higher difficulty increases the unsupported crossing distance.", "labels": [("Large-gap width", "Distance from takeoff edge to landing edge"), ("Takeoff length", "Space available to prepare for crossing"), ("Landing length", "Safe recovery area after crossing"), ("Gap depth", "Failure region beneath the gap"), ("Direction", "Crossing may be attempted from either side")]},
    "rs10": {"title": "RS10 — High step / climb", "subtitle": "Tests climbing onto an elevated platform", "difficulty": "Higher difficulty raises the step in proportion to the robot's limb reach.", "labels": [("Obstacle height", "Vertical climb required"), ("Top-platform length", "Space available after climbing"), ("Approach length", "Run-up or preparation distance"), ("Recovery length", "Stable area after the obstacle"), ("Direction", "Step-up forward; drop-down when reversed")]},
    "rs11": {"title": "RS11 — Depth-resistive mud", "subtitle": "Tests sinking support, vertical damping and suction during withdrawal", "difficulty": "Higher difficulty lowers support stress and raises yield-capped suction.", "labels": [("Marsh length", "Total distance through the mud region"), ("Patch size", "Horizontal extent of each mud volume"), ("Patch arrangement", "Seed controls repeatable placement and size"), ("Reference depth", "Depth used to normalise the support power law"), ("Support law", "Alpha and beta set rate-independent depth resistance"), ("Damping / suction", "Vertical damping plus a time-building yield cap"), ("Fidelity level", "Depth-resistive mud v1, with rigid-viscous retained for old worlds")]},
}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/%s" % name, size)


def _arrow(draw, start: Tuple[float, float], end: Tuple[float, float], color: str = ACCENT, width: int = 5, both: bool = False) -> None:
    draw.line([start, end], fill=color, width=width)
    for source, target in ((start, end), (end, start)) if both else ((start, end),):
        angle = math.atan2(target[1] - source[1], target[0] - source[0])
        size = 14
        points = [target, (target[0] - size * math.cos(angle - 0.55), target[1] - size * math.sin(angle - 0.55)), (target[0] - size * math.cos(angle + 0.55), target[1] - size * math.sin(angle + 0.55))]
        draw.polygon(points, fill=color)


def _dimension(draw, start, end, label: str, offset: Tuple[int, int] = (0, -40), color: str = ACCENT) -> None:
    _arrow(draw, start, end, color, 4, True)
    midpoint = ((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1])
    box = draw.textbbox(midpoint, label, font=_font(24, True), anchor="mm")
    draw.rounded_rectangle((box[0] - 8, box[1] - 4, box[2] + 8, box[3] + 4), radius=8, fill="#e4f7f3", outline=color, width=2)
    draw.text(midpoint, label, font=_font(24, True), fill=INK, anchor="mm")


def _callout(draw, anchor: Tuple[int, int], box: Tuple[int, int], title: str, description: str, side: str = "left") -> None:
    wrapped = textwrap.wrap(description, width=31)
    width, height = 360, 56 + 27 * len(wrapped)
    x, y = box
    end = (x + width if side == "right" else x, y + height / 2)
    elbow = ((anchor[0] + end[0]) / 2, anchor[1])
    draw.line([anchor, elbow, end], fill=ACCENT, width=3)
    draw.ellipse((anchor[0] - 6, anchor[1] - 6, anchor[0] + 6, anchor[1] + 6), fill=ACCENT)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12, fill=PANEL, outline="#9fb3c8", width=2)
    draw.text((x + 16, y + 11), title, font=_font(22, True), fill=INK)
    draw.multiline_text((x + 16, y + 40), "\n".join(wrapped), font=_font(19), fill=MUTED, spacing=5)


def _dog(draw, x: int, ground_y: int, scale: float = 1.0) -> None:
    body = (x, ground_y - 115 * scale, x + 145 * scale, ground_y - 60 * scale)
    draw.rounded_rectangle(body, radius=int(20 * scale), fill="#283747", outline="#ffffff", width=3)
    draw.ellipse((x + 125 * scale, ground_y - 125 * scale, x + 180 * scale, ground_y - 75 * scale), fill="#34495e", outline="#ffffff", width=3)
    for hip, foot in ((x + 25 * scale, x + 5 * scale), (x + 55 * scale, x + 70 * scale), (x + 110 * scale, x + 90 * scale), (x + 135 * scale, x + 155 * scale)):
        draw.line((hip, ground_y - 65 * scale, foot, ground_y), fill="#283747", width=max(5, int(10 * scale)))
        draw.ellipse((foot - 11 * scale, ground_y - 5 * scale, foot + 11 * scale, ground_y + 5 * scale), fill="#18232e")


def _base(draw, code: str, title: str, subtitle: str) -> None:
    draw.text((70, 45), title, font=_font(42, True), fill=INK)
    draw.text((70, 102), subtitle, font=_font(25), fill=MUTED)
    draw.rounded_rectangle((1470, 45, 1725, 115), radius=18, fill=INK)
    draw.text((1597, 80), code.upper(), font=_font(34, True), fill="white", anchor="mm")
    draw.rounded_rectangle((55, 155, 1745, 835), radius=22, fill=PANEL, outline="#bcccdc", width=2)


def _terrain_scene(draw, code: str) -> Dict[str, Tuple[int, int]]:
    y = 650
    anchors: Dict[str, Tuple[int, int]] = {}
    draw.rectangle((95, y, 355, y + 65), fill=NOMINAL, outline=INK, width=2)
    draw.rectangle((1390, y, 1695, y + 65), fill=NOMINAL, outline=INK, width=2)
    _dog(draw, 145, y, 0.85)
    _arrow(draw, (210, 760), (340, 760), RED, 5)
    anchors["direction"] = (280, 760)
    if code == "rs1":
        draw.rectangle((355, y, 1390, y + 65), fill=SLIPPERY, outline=INK, width=2)
        _dimension(draw, (370, 590), (1375, 590), "Slippery-section length")
        anchors.update({"length": (875, 650), "friction": (950, 680), "approach": (250, 650)})
    elif code == "rs2":
        draw.rectangle((355, y, 1390, y + 65), fill=SLIPPERY, outline=INK, width=2)
        for index, height in enumerate((35, 70, 45, 90, 55, 75, 40, 82, 52)):
            x = 390 + index * 105
            draw.rectangle((x, y - height, x + 55, y), fill=ROUGH, outline=INK, width=2)
        _dimension(draw, (370, 550), (1375, 550), "Rough-section length")
        _dimension(draw, (1220, y), (1220, y - 82), "Bump height", (105, 0))
        anchors.update({"length": (875, 650), "height": (1220, 580), "arrangement": (800, 590), "friction": (1020, 680)})
    elif code in {"rs3", "rs4"}:
        top = 385
        polygon = [(355, y), (1285, top), (1390, top), (1390, top + 65), (355, y + 65)]
        draw.polygon(polygon, fill=SLIPPERY, outline=INK)
        if code == "rs4":
            for index, height in enumerate((22, 38, 18, 48, 30, 42, 20)):
                x = 520 + index * 105
                surface_y = y - (x - 355) * (y - top) / 930
                draw.ellipse((x - 23, surface_y - height, x + 23, surface_y + 4), fill=ROUGH, outline=INK, width=2)
        _dimension(draw, (370, 745), (1280, 745), "Horizontal slope length", (0, 35))
        _dimension(draw, (1335, y), (1335, top), "Vertical rise", (100, 0))
        draw.arc((315, y - 145, 600, y + 140), 195, 225, fill=ACCENT, width=5)
        draw.text((445, 600), "Slope angle", font=_font(23, True), fill=INK)
        anchors.update({"angle": (480, 610), "length": (820, 745), "rise": (1335, 520), "friction": (900, 520), "roughness": (940, 450), "arrangement": (720, 510)})
    elif code == "rs5":
        step_w, step_h = 165, 58
        for index in range(6):
            x = 355 + index * step_w
            height = (index + 1) * step_h
            draw.rectangle((x, y - height, x + step_w, y + 65), fill=SLIPPERY, outline=INK, width=2)
        top_y = y - 6 * step_h
        draw.rectangle((1345, top_y, 1695, top_y + 65), fill=NOMINAL, outline=INK, width=2)
        _dimension(draw, (355, 790), (1345, 790), "Number of steps", (0, 34))
        _dimension(draw, (1190, y - 5 * step_h), (1190, y - 6 * step_h), "Step height", (110, 0))
        _dimension(draw, (1180, y - 6 * step_h - 42), (1345, y - 6 * step_h - 42), "Step depth")
        anchors.update({"count": (850, 790), "rise": (1190, 330), "tread": (1260, 260), "friction": (850, 460)})
    elif code in {"rs6", "rs7"}:
        points = []
        for index in range(101):
            x = 355 + index * 10.35
            t = index / 100.0
            height = 150 * (0.5 * (1 - math.cos(4 * math.pi * t)) if code == "rs6" else math.sin(math.pi * t))
            points.append((x, y - height))
        polygon = [(355, y + 65), *points, (1390, y + 65)]
        draw.polygon(polygon, fill=SLIPPERY, outline=INK)
        _dimension(draw, (370, 790), (1375, 790), "Course / hill length", (0, 34))
        peak = max(points, key=lambda value: y - value[1])
        _dimension(draw, (peak[0], y), peak, "Wave height" if code == "rs6" else "Hill height", (115, 0))
        anchors.update({"length": (875, 790), "height": peak, "segments": (1050, points[67][1]), "waves": (875, 530), "friction": (600, 620)})
    elif code == "rs8":
        cursor = 355
        for index in range(6):
            draw.rectangle((cursor, y, cursor + 120, y + 65), fill=ROUGH, outline=INK, width=2)
            cursor += 120
            if index < 5:
                draw.rectangle((cursor, y, cursor + 80, 805), fill="#17212b")
                cursor += 80
        _dimension(draw, (475, 565), (555, 565), "Gap width")
        _dimension(draw, (755, 565), (875, 565), "Platform length")
        _dimension(draw, (515, y), (515, 805), "Gap depth", (105, 0))
        anchors.update({"count": (955, 720), "gap": (515, 650), "platform": (815, 650), "depth": (515, 780), "width": (1100, 650)})
    elif code == "rs9":
        draw.rectangle((355, y, 780, y + 65), fill=CHALLENGE, outline=INK, width=2)
        draw.rectangle((780, y, 1180, 820), fill="#17212b")
        draw.rectangle((1180, y, 1550, y + 65), fill=CHALLENGE, outline=INK, width=2)
        _dimension(draw, (370, 570), (765, 570), "Takeoff length")
        _dimension(draw, (795, 570), (1165, 570), "Large-gap width")
        _dimension(draw, (1195, 570), (1535, 570), "Landing length")
        _dimension(draw, (980, y), (980, 820), "Gap depth", (105, 0))
        anchors.update({"takeoff": (560, 650), "gap": (980, 650), "landing": (1360, 650), "depth": (980, 795)})
    elif code == "rs10":
        rise_y = 370
        draw.rectangle((355, y, 760, y + 65), fill=NOMINAL, outline=INK, width=2)
        draw.rectangle((760, rise_y, 1390, y + 65), fill=CHALLENGE, outline=INK, width=2)
        draw.rectangle((1390, rise_y, 1695, rise_y + 65), fill=NOMINAL, outline=INK, width=2)
        _dimension(draw, (705, y), (705, rise_y), "Obstacle height", (-115, 0))
        _dimension(draw, (775, 310), (1375, 310), "Top-platform length")
        _dimension(draw, (370, 790), (745, 790), "Approach length", (0, 34))
        anchors.update({"rise": (760, 500), "platform": (1080, 370), "approach": (550, 650), "recovery": (1535, 370)})
    else:
        draw.rectangle((355, y, 1390, y + 65), fill="#9eb28a", outline=INK, width=2)
        cursor = 370
        patch_index = 0
        while cursor < 1370:
            patch_w = (125, 175, 110, 205, 145, 120)[patch_index % 6]
            sink = (22, 55, 35, 70, 45, 60)[patch_index % 6]
            draw.rounded_rectangle((cursor, y + sink, min(cursor + patch_w, 1380), y + sink + 45), radius=8, fill=MUD, outline=INK, width=2)
            cursor += patch_w + 16
            patch_index += 1
        _dimension(draw, (370, 555), (1375, 555), "Marsh length")
        _dimension(draw, (386, 770), (511, 770), "Patch size", (0, 35))
        _dimension(draw, (448, y), (448, y + 55), "Sinking depth", (105, 0))
        for x in (700, 800, 900):
            _arrow(draw, (x + 55, 620), (x, 620), RED, 4)
        draw.text((810, 575), "Motion resistance", font=_font(23, True), fill=INK, anchor="mm")
        anchors.update({"length": (875, 650), "patch": (448, 720), "arrangement": (1120, 720), "sink": (448, 700), "drag": (800, 620), "softness": (1050, 690), "fidelity": (1280, 650)})
    return anchors


def _draw_parameter_diagram(code: str, output: Path) -> None:
    from PIL import Image, ImageDraw

    specification = TERRAINS[code]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _base(draw, code, specification["title"], specification["subtitle"])
    anchors = _terrain_scene(draw, code)
    # Positions are intentionally consistent so the series is easy to scan.
    positions = [(75, 180), (75, 300), (75, 420), (1365, 180), (1365, 300), (1365, 420), (1365, 540)]
    preferred = {
        "rs1": ["length", "friction", "direction", "approach"],
        "rs2": ["length", "height", "arrangement", "friction"],
        "rs3": ["angle", "length", "rise", "friction", "direction"],
        "rs4": ["angle", "length", "roughness", "arrangement", "friction", "direction"],
        "rs5": ["count", "rise", "tread", "friction", "direction"],
        "rs6": ["length", "height", "segments", "waves", "direction"],
        "rs7": ["length", "height", "segments", "direction", "friction"],
        "rs8": ["count", "gap", "platform", "depth", "width"],
        "rs9": ["gap", "takeoff", "landing", "depth", "direction"],
        "rs10": ["rise", "platform", "approach", "recovery", "direction"],
        "rs11": ["length", "patch", "arrangement", "sink", "drag", "softness", "fidelity"],
    }[code]
    for index, ((title, description), anchor_key) in enumerate(zip(specification["labels"], preferred)):
        x, y = positions[index]
        _callout(draw, anchors.get(anchor_key, anchors.get("direction", (850, 650))), (x, y), title, description, "right" if x < WIDTH / 2 else "left")
    draw.rounded_rectangle((55, 860, 1745, 950), radius=18, fill="#102a43")
    draw.text((85, 880), "DIFFICULTY EFFECT", font=_font(21, True), fill="#77e0ca")
    draw.text((315, 878), specification["difficulty"], font=_font(23), fill="white")
    draw.text((85, 918), "Morphology-scaled dimensions plus universal sideways camber and clockwise/anticlockwise route curvature can be layered onto this RS.", font=_font(19), fill="#d9e2ec")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _draw_universal_modifiers(output: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "Universal geometry modifiers — available to every RS", font=_font(40, True), fill=INK)
    draw.text((70, 102), "These transform collision geometry, routes, regions, spawn and goal poses before simulator export.", font=_font(24), fill=MUTED)
    draw.rounded_rectangle((55, 155, 865, 810), radius=22, fill=PANEL, outline="#bcccdc", width=2)
    draw.rounded_rectangle((935, 155, 1745, 810), radius=22, fill=PANEL, outline="#bcccdc", width=2)
    draw.text((90, 185), "SIDEWAYS TILT / CAMBER", font=_font(29, True), fill=INK)
    draw.text((970, 185), "LEFT / RIGHT ROUTE CURVATURE", font=_font(29, True), fill=INK)

    # Front cross-section: the actual collision slab is rolled, not merely painted.
    center_x, center_y = 455, 525
    angle = math.radians(-14)
    half_w, half_h = 290, 30
    corners = []
    for x, y in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        corners.append((center_x + x * math.cos(angle) - y * math.sin(angle), center_y + x * math.sin(angle) + y * math.cos(angle)))
    draw.polygon(corners, fill=SLIPPERY, outline=INK)
    draw.line((center_x, 385, center_x, 690), fill="#9fb3c8", width=3)
    draw.arc((325, 395, 585, 655), 263, 282, fill=ACCENT, width=6)
    draw.text((455, 690), "Signed sideways angle", font=_font(24, True), fill=INK, anchor="mm")
    draw.text((150, 355), "Left side low", font=_font(23, True), fill=RED)
    draw.text((650, 355), "Right side high", font=_font(23, True), fill=ACCENT)
    _arrow(draw, (455, 750), (275, 750), RED, 5)
    _arrow(draw, (455, 750), (635, 750), ACCENT, 5)
    draw.text((455, 785), "Change the sign to reverse the tilt", font=_font(21), fill=MUTED, anchor="mm")

    # Top view: centreline and subdivided collision slabs follow an arc.
    points = []
    center = (1110, 720)
    radius = 390
    for index in range(41):
        theta = math.radians(5 + 75 * index / 40)
        points.append((center[0] + radius * math.cos(theta), center[1] - radius * math.sin(theta)))
    draw.line(points, fill=ACCENT, width=8, joint="curve")
    for index in range(0, 41, 5):
        x, y = points[index]
        theta = math.radians(5 + 75 * index / 40)
        tangent = theta + math.pi / 2
        dx, dy = 55 * math.cos(tangent), -55 * math.sin(tangent)
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=SLIPPERY, width=24)
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=INK, width=2)
    _arrow(draw, points[12], points[17], RED, 5)
    draw.text((1340, 300), "Anticlockwise / left turn", font=_font(24, True), fill=INK, anchor="mm")
    draw.text((1340, 340), "Negative angle mirrors it into a clockwise / right turn", font=_font(20), fill=MUTED, anchor="mm")
    draw.text((1340, 610), "Curve radius", font=_font(24, True), fill=INK, anchor="mm")
    _arrow(draw, center, points[20], ACCENT, 4, True)
    draw.text((1340, 755), "More segments = smoother collision arc", font=_font(22, True), fill=INK, anchor="mm")

    draw.rounded_rectangle((55, 850, 1745, 950), radius=18, fill=INK)
    draw.text((85, 875), "COMPOSITION", font=_font(21, True), fill="#77e0ca")
    draw.text((260, 873), "Base RS → terrain parameters → route curvature → sideways camber → friction/material → AE/AEE placement", font=_font(23), fill="white")
    draw.text((85, 916), "Because collision boxes are subdivided and rotated, gravity, contact, slip and force calculations use the modified surface in every backend.", font=_font(19), fill="#d9e2ec")
    image.save(output)


def generate_parameter_diagrams(output_dir: str) -> Dict[str, Any]:
    from PIL import Image, ImageDraw

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    universal = root / "universal-camber-curvature.png"
    _draw_universal_modifiers(universal)
    records = []
    for code in TERRAINS:
        output = root / (code + "-parameters.png")
        _draw_parameter_diagram(code, output)
        records.append({"code": code, "title": TERRAINS[code]["title"], "png": str(output), "parameters": [title for title, _ in TERRAINS[code]["labels"]]})
    thumb_w, thumb_h, columns = 600, 340, 3
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (thumb_w * columns, thumb_h * rows), BG)
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        image = Image.open(record["png"]).convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 36))
        x, y = (index % columns) * thumb_w, (index // columns) * thumb_h
        sheet.paste(image, (x + 6, y + 32))
        draw.text((x + 10, y + 5), record["title"], font=_font(21, True), fill=INK)
    contact_sheet = root / "contact-sheet.png"
    sheet.save(contact_sheet)
    cards = "".join("<article><h2>%s</h2><a href='%s'><img src='%s' loading='lazy'></a><p><b>Labels:</b> %s</p></article>" % (html.escape(item["title"]), Path(item["png"]).name, Path(item["png"]).name, html.escape(", ".join(item["parameters"]))) for item in records)
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>RS parameter diagrams</title><style>body{font-family:system-ui;background:#eef3f7;color:#102a43;margin:0}.wrap{max-width:1500px;margin:auto;padding:28px}.hero{background:#102a43;color:white;padding:24px;border-radius:14px}.feature{background:white;padding:14px;border-radius:12px;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px;margin-top:20px}article{background:white;padding:14px;border-radius:12px}img{width:100%%;border-radius:8px}p{color:#486581}.hero p{color:#d9e2ec}</style></head><body><div class='wrap'><div class='hero'><h1>RS1–RS11 changeable-parameter diagrams</h1><p>Click any diagram for the full 1800×1000 labelled image. These are simulator-neutral morphology-scaled schematics.</p><a style='color:#77e0ca' href='contact-sheet.png'>Open combined RS contact sheet</a></div><div class='feature'><h2>Universal camber and curvature</h2><a href='universal-camber-curvature.png'><img src='universal-camber-curvature.png'></a></div><div class='grid'>%s</div></div></body></html>""" % cards
    index = root / "index.html"
    atomic_write_text(str(index), page)
    manifest = {"schema_version": "1.1", "diagram_count": len(records), "records": records, "universal_modifier_diagram": str(universal), "contact_sheet": str(contact_sheet), "index": str(index)}
    dump_json(str(root / "manifest.json"), manifest)
    return manifest
