"""Deterministic RS1--RS11 morphology-scaled terrain generators."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..environment_physics import DEPTH_RESISTIVE_TIER
from ..models import GeometrySpec, MaterialSpec, Pose, RegionSpec, RobotProfile, RouteSpec, WorldSpec


TERRAIN_CODES = tuple("rs%d" % index for index in range(1, 12))


def apply_challenge_friction(world: WorldSpec, static_friction: float, dynamic_friction: float) -> WorldSpec:
    """Set challenge traction independently from morphology difficulty."""
    static_friction = float(static_friction)
    dynamic_friction = float(dynamic_friction)
    if static_friction <= 0.0 or dynamic_friction <= 0.0:
        raise ValueError("friction coefficients must be positive")
    if dynamic_friction > static_friction:
        raise ValueError("dynamic friction cannot exceed static friction")
    challenge_materials = {
        geometry.material
        for geometry in world.geometry
        if geometry.material != "nominal" and geometry.semantic not in {"approach", "landing", "junction", "connector"}
    }
    for name in challenge_materials:
        material = world.materials.get(name)
        if material is not None:
            material.static_friction = static_friction
            material.dynamic_friction = dynamic_friction
    world.resolved_parameters["explicit_challenge_friction"] = {
        "static": static_friction,
        "dynamic": dynamic_friction,
        "materials": sorted(challenge_materials),
    }
    world.metadata["friction_policy"] = "explicit_challenge_only"
    world.validate()
    return world


def _clip(name: str, value: float, lower: float, upper: float, clips: List[Dict[str, Any]]) -> float:
    resolved = min(max(value, lower), upper)
    if resolved != value:
        clips.append({"parameter": name, "requested": value, "resolved": resolved, "minimum": lower, "maximum": upper})
    return resolved


def default_materials(difficulty: float) -> Dict[str, MaterialSpec]:
    low = max(0.02, 0.22 - 0.18 * difficulty)
    return {
        "nominal": MaterialSpec("nominal", 0.9, 0.8, color="#236b8e"),
        "low_friction": MaterialSpec("low_friction", low, low * 0.8, color="#fff200"),
        "rough_low": MaterialSpec("rough_low", low, low * 0.8, roughness=0.5 + 0.5 * difficulty, color="#ffe600"),
        "challenge": MaterialSpec("challenge", 0.75, 0.65, color="#873e13"),
        "purple": MaterialSpec("purple", 0.75, 0.65, color="#7837aa"),
        "compliant": MaterialSpec("compliant", 0.6, 0.5, compliance=0.01 + 0.04 * difficulty, damping=2.0 + 8.0 * difficulty, color="#4b5d3a"),
    }


@dataclass
class _Scale:
    body_length: float
    body_width: float
    body_height: float
    standing_height: float
    foot_length: float
    foot_width: float
    reach: float
    track_width: float
    approach: float
    landing: float


def _scale(profile: RobotProfile, clips: List[Dict[str, Any]]) -> _Scale:
    body_length = _clip("body_length_m", profile.body_length_m, 0.15, 2.0, clips)
    body_width = _clip("body_width_m", profile.body_width_m, 0.08, 1.2, clips)
    return _Scale(
        body_length=body_length,
        body_width=body_width,
        body_height=_clip("body_height_m", profile.body_size_m[2], 0.08, 1.0, clips),
        standing_height=_clip("standing_height_m", profile.standing_height_m, 0.15, 1.5, clips),
        foot_length=_clip("foot_length_m", profile.foot_size_m[0], 0.02, 0.4, clips),
        foot_width=_clip("foot_width_m", profile.foot_size_m[1], 0.015, 0.3, clips),
        reach=_clip("limb_reach_m", profile.limb_reach_m, 0.15, 1.5, clips),
        track_width=_clip("track_width_m", max(4.0 * body_width, profile.stance_width_m + 2.0 * body_width), 1.0, 4.0, clips),
        approach=_clip("approach_m", 2.5 * body_length, 0.8, 3.0, clips),
        landing=_clip("landing_m", 2.0 * body_length, 0.7, 2.5, clips),
    )


def _box(name: str, center: List[float], size: List[float], material: str, semantic: str, rpy: List[float] = None, parameters: Dict[str, Any] = None) -> GeometrySpec:
    return GeometrySpec(
        name=name,
        shape="box",
        size=size,
        pose=Pose(center, rpy or [0.0, 0.0, 0.0]),
        material=material,
        semantic=semantic,
        parameters=parameters or {},
    )


def _floor(name: str, start: float, length: float, width: float, top_z: float = 0.0, material: str = "nominal", semantic: str = "terrain") -> GeometrySpec:
    thickness = 0.1
    return _box(name, [start + length / 2.0, 0.0, top_z - thickness / 2.0], [length, width, thickness], material, semantic)


def _region(name: str, kind: str, start: float, end: float, width: float, mandatory: bool = False, z_min: float = -0.5, z_max: float = 3.0, metadata: Dict[str, Any] = None) -> RegionSpec:
    return RegionSpec(name, kind, [start, -width / 2.0, z_min], [end, width / 2.0, z_max], mandatory, metadata or {})


def _finish_world(code: str, profile: RobotProfile, seed: int, difficulty: float, scale: _Scale, challenge_start: float, challenge_end: float, final_z: float, geometry: List[GeometrySpec], regions: List[RegionSpec], normalized: Dict[str, Any], resolved: Dict[str, Any], clips: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> WorldSpec:
    total_length = challenge_end + scale.landing
    # The approach floor extends one body length BEHIND the origin. There is no
    # ground plane, and the spawn sits 0.6 body-lengths in, so without this the
    # rear feet land about 1.6 cm from the end of the world -- and curvature
    # sweeps them onto the edge itself. A robot that rocks back as its policy
    # starts must not step off the back of the world; that is not what any
    # terrain here is testing. Routes, regions and the goal are unchanged.
    apron = scale.body_length
    geometry.insert(0, _floor("approach", -apron, challenge_start + apron, scale.track_width))
    geometry.append(_floor("landing", challenge_end, scale.landing, scale.track_width, final_z))
    regions.insert(0, _region("approach", "approach", 0.0, challenge_start, scale.track_width))
    regions.append(_region("goal", "goal", challenge_end, total_length, scale.track_width, True))
    world = WorldSpec(
        name="%s-d%.3f-s%d" % (code.upper(), difficulty, seed),
        terrain_type=code,
        seed=seed,
        robot_profile_sha256=profile.source_sha256,
        dimensions_m=[total_length, scale.track_width, max(2.0, final_z + scale.standing_height * 2.0)],
        spawn_pose=Pose([scale.body_length * 0.6, 0.0, scale.standing_height]),
        goal_pose=Pose([total_length - scale.body_length * 0.5, 0.0, final_z + scale.standing_height]),
        materials=default_materials(difficulty),
        geometry=geometry,
        regions=regions,
        routes=[RouteSpec("central", [[scale.body_length * 0.6, 0.0, 0.0], [challenge_start, 0.0, 0.0], [challenge_end, 0.0, final_z], [total_length, 0.0, final_z]], [region.name for region in regions], "central")],
        checkpoints=[[challenge_start, 0.0, 0.0], [challenge_end, 0.0, final_z]],
        normalized_parameters=normalized,
        resolved_parameters=resolved,
        clips=clips,
        metadata={"difficulty": difficulty, "source": "Robustness xTerra Plaksha.pdf", **(metadata or {})},
    )
    world.validate()
    return world


def generate_terrain(code: str, profile: RobotProfile, difficulty: float = 0.5, seed: int = 0, overrides: Dict[str, Any] = None) -> WorldSpec:
    """Generate one RS terrain in SI units.

    ``difficulty`` is bounded to [0, 1]. Overrides use resolved parameter names
    (for example ``gap_m`` or ``slope_rad``) and are recorded verbatim.
    """
    code = code.lower()
    if code not in TERRAIN_CODES:
        raise ValueError("terrain code must be one of %s" % (TERRAIN_CODES,))
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    profile.validate()
    overrides = overrides or {}
    clips: List[Dict[str, Any]] = []
    scale = _scale(profile, clips)
    rng = random.Random(seed)
    start = scale.approach
    geometry: List[GeometrySpec] = []
    regions: List[RegionSpec] = []
    normalized: Dict[str, Any] = {"difficulty": difficulty}
    resolved: Dict[str, Any] = dict(overrides)
    final_z = 0.0

    if code == "rs1":
        length = float(overrides.get("length_m", scale.body_length * 5.0))  # fixed; RS1 difficulty is friction
        geometry.append(_floor("low-friction-zone", start, length, scale.track_width, material="low_friction", semantic="low_friction"))
        regions.append(_region("challenge", "low_friction", start, start + length, scale.track_width, True))
        normalized["length_body_lengths"] = length / scale.body_length
        resolved.update({"length_m": length, "friction": default_materials(difficulty)["low_friction"].dynamic_friction})
        end = start + length
    elif code == "rs2":
        length = float(overrides.get("length_m", scale.body_length * 5.0))
        amplitude = _clip("roughness_amplitude_m", float(overrides.get("roughness_amplitude_m", scale.foot_length * difficulty)), 0.001, scale.foot_length * 2.0, clips)
        spacing = max(scale.foot_length * 1.5, 0.04)
        geometry.append(_floor("rough-low-base", start, length, scale.track_width, material="rough_low", semantic="rough_low_friction"))
        count = max(1, int(length / spacing))
        for index in range(count):
            height = amplitude * (0.35 + 0.65 * rng.random())
            geometry.append(_box("roughness-%03d" % index, [start + (index + 0.5) * length / count, rng.uniform(-0.35, 0.35) * scale.track_width, height / 2.0], [spacing * 0.55, scale.track_width * 0.25, height], "rough_low", "roughness"))
        regions.append(_region("challenge", "rough_low_friction", start, start + length, scale.track_width, True))
        normalized.update({"length_body_lengths": length / scale.body_length, "amplitude_foot_lengths": amplitude / scale.foot_length})
        resolved.update({"length_m": length, "roughness_amplitude_m": amplitude})
        end = start + length
    elif code in {"rs3", "rs4"}:
        angle = _clip("slope_rad", float(overrides.get("slope_rad", math.radians(45.0 * difficulty))), math.radians(0.0), math.radians(45.0), clips)
        run = float(overrides.get("run_m", scale.body_length * 5.0))
        ramp_length = run / max(math.cos(angle), 1e-6)
        rise = run * math.tan(angle)
        thickness = 0.12
        material = "low_friction" if code == "rs3" else "rough_low"
        geometry.append(_box("slope", [start + run / 2.0, 0.0, rise / 2.0 - thickness / 2.0], [ramp_length, scale.track_width, thickness], material, "slope", [0.0, -angle, 0.0]))
        if code == "rs4":
            roughness_amplitude = _clip(
                "roughness_amplitude_m",
                float(overrides.get("roughness_amplitude_m", scale.foot_length * difficulty)),
                0.001,
                scale.foot_length * 2.0,
                clips,
            )
            count = max(3, int(run / max(scale.foot_length * 2.0, 0.08)))
            for index in range(count):
                x = start + (index + 0.5) * run / count
                height = roughness_amplitude * (0.35 + 0.65 * rng.random())
                z = (x - start) * math.tan(angle) + height / 2.0
                geometry.append(_box("slope-roughness-%03d" % index, [x, rng.uniform(-0.3, 0.3) * scale.track_width, z], [run / count * 0.5, scale.track_width * 0.3, height], material, "roughness", [0.0, -angle, 0.0]))
        regions.append(_region("challenge", "rough_slope" if code == "rs4" else "low_friction_slope", start, start + run, scale.track_width, True, z_max=rise + 2.0))
        normalized.update({"run_body_lengths": run / scale.body_length, "slope_deg": math.degrees(angle)})
        resolved.update({"run_m": run, "rise_m": rise, "slope_rad": angle})
        if code == "rs4":
            resolved["roughness_amplitude_m"] = roughness_amplitude
        end, final_z = start + run, rise
    elif code == "rs5":
        steps = int(overrides.get("step_count", 3 + int(10.0 * difficulty + 0.5)))  # round-half-up: 6/8/11
        rise = _clip("step_rise_m", float(overrides.get("step_rise_m", scale.standing_height * 0.4)), 0.01, scale.reach * 0.6, clips)
        tread = float(overrides.get("step_tread_m", max(scale.foot_length * 2.5, scale.body_length * 0.35)))
        for index in range(steps):
            height = (index + 1) * rise
            geometry.append(_box("stair-%02d" % index, [start + (index + 0.5) * tread, 0.0, height / 2.0 - 0.05], [tread, scale.track_width, height + 0.1], "low_friction", "stair"))
        end, final_z = start + steps * tread, steps * rise
        regions.append(_region("challenge", "stairs", start, end, scale.track_width, True, z_max=final_z + 2.0))
        normalized.update({"rise_standing_heights": rise / scale.standing_height, "tread_foot_lengths": tread / scale.foot_length})
        resolved.update({"step_count": steps, "step_rise_m": rise, "step_tread_m": tread, "total_rise_m": final_z})
    elif code in {"rs6", "rs7"}:
        length = float(overrides.get("length_m", scale.body_length * 5.0))
        if code == "rs6":
            # Difficulty packs more full waves into a fixed length; amplitude fixed.
            wave_count = int(overrides.get("wave_count", 1 + int(4.0 * difficulty + 0.5)))  # 2/3/4
            count = int(overrides.get("segments", 30))
            amplitude = float(overrides.get("amplitude_m", scale.standing_height * 0.3))
        else:
            # Difficulty makes the single hill blockier -- fewer segments, sharper
            # foot contacts -- at a fixed height.
            count = int(overrides.get("segments", max(4, 24 - int(20.0 * difficulty + 0.5))))  # 19/14/9
            amplitude = float(overrides.get("amplitude_m", scale.standing_height * 0.7))
            wave_count = None
        dx = length / count
        heights = []
        for index in range(count + 1):
            t = index / count
            if code == "rs6":
                heights.append(amplitude * 0.5 * (1.0 - math.cos(2.0 * math.pi * wave_count * t)))
            else:
                heights.append(amplitude * math.sin(math.pi * t))
        for index in range(count):
            dz = heights[index + 1] - heights[index]
            angle = math.atan2(dz, dx)
            segment_length = math.sqrt(dx * dx + dz * dz)
            geometry.append(_box("curve-%02d" % index, [start + (index + 0.5) * dx, 0.0, (heights[index] + heights[index + 1]) / 2.0 - 0.05], [segment_length, scale.track_width, 0.1], "low_friction", "wave" if code == "rs6" else "curved_slope", [0.0, -angle, 0.0]))
        end, final_z = start + length, heights[-1]
        regions.append(_region("challenge", "wave" if code == "rs6" else "curved_slope", start, end, scale.track_width, True, z_max=amplitude + 2.0))
        normalized.update({"length_body_lengths": length / scale.body_length, "amplitude_standing_heights": amplitude / scale.standing_height})
        resolved.update({"length_m": length, "amplitude_m": amplitude, "segments": count})
        if code == "rs6":
            resolved["wave_count"] = wave_count
            normalized["wave_count"] = wave_count
    elif code == "rs8":
        count = int(overrides.get("gap_count", 5))
        gap = _clip("gap_m", float(overrides.get("gap_m", scale.foot_length * 2.0 * difficulty)), 0.005, scale.body_length * 0.5, clips)
        platform = float(overrides.get("platform_m", max(scale.foot_length * 2.0, scale.body_length * 0.3)))
        cursor = start
        for index in range(count + 1):
            geometry.append(_floor("gap-platform-%02d" % index, cursor, platform, scale.track_width, material="rough_low", semantic="gap_platform"))
            cursor += platform
            if index < count:
                regions.append(_region("gap-%02d" % index, "gap", cursor, cursor + gap, scale.track_width, True, z_min=-2.0, metadata={"gap_m": gap}))
                cursor += gap
        end = cursor
        regions.append(_region("challenge", "small_gaps", start, end, scale.track_width, True))
        normalized.update({"gap_foot_lengths": gap / scale.foot_length, "platform_body_lengths": platform / scale.body_length})
        resolved.update({"gap_count": count, "gap_m": gap, "platform_m": platform})
    elif code == "rs9":
        gap = _clip("gap_m", float(overrides.get("gap_m", scale.body_length * difficulty)), 0.02, scale.body_length * 1.5, clips)
        takeoff = float(overrides.get("takeoff_m", max(scale.body_length, 0.5)))
        geometry.append(_floor("takeoff-platform", start, takeoff, scale.track_width, material="challenge", semantic="takeoff"))
        geometry.append(_floor("landing-platform", start + takeoff + gap, scale.landing, scale.track_width, material="challenge", semantic="landing"))
        regions.extend([_region("challenge", "large_gap", start + takeoff, start + takeoff + gap, scale.track_width, True, z_min=-3.0), _region("takeoff", "takeoff", start, start + takeoff, scale.track_width)])
        end = start + takeoff + gap + scale.landing
        normalized["gap_body_lengths"] = gap / scale.body_length
        resolved.update({"gap_m": gap, "takeoff_m": takeoff})
    elif code == "rs10":
        rise = _clip("rise_m", float(overrides.get("rise_m", scale.reach * difficulty)), 0.02, scale.reach * 1.1, clips)
        platform = float(overrides.get("platform_m", scale.body_length * 2.0))
        geometry.append(_box("high-rise", [start + platform / 2.0, 0.0, rise / 2.0 - 0.05], [platform, scale.track_width, rise + 0.1], "challenge", "high_rise"))
        end, final_z = start + platform, rise
        regions.append(_region("challenge", "high_rise", start, end, scale.track_width, True, z_max=rise + 2.0))
        normalized["rise_standing_heights"] = rise / scale.standing_height
        resolved.update({"rise_m": rise, "platform_m": platform})
    else:  # rs11
        length = float(overrides.get("length_m", scale.body_length * (3.0 + 2.0 * difficulty)))
        patch = float(overrides.get("patch_m", max(scale.foot_length * 2.0, 0.08)))
        mud_parameters = {
            # Chen et al. report alpha=0.01--0.17 MPa, beta=0.38--0.56,
            # and sigma_y=2--17 kPa. Harder levels are softer but grabbier.
            "support_stress_pa": float(overrides.get("support_stress_pa", 7.0e4 - 4.5e4 * difficulty)),
            "depth_exponent": float(overrides.get("depth_exponent", 0.38 + 0.18 * difficulty)),
            "reference_depth_m": float(overrides.get(
                "reference_depth_m",
                # H normalises penetration. In arXiv:2403.02617 it is the
                # intruder width, so it belongs to the foot, not to a
                # constant: a larger robot must find proportionally
                # deeper mud, not implausibly firm ground.
                min(scale.foot_length, scale.foot_width),
            )),
            "damping_pa_s_per_m": float(overrides.get("damping_pa_s_per_m", 1.2e4 + 8.0e3 * difficulty)),
            "yield_stress_pa": float(overrides.get("yield_stress_pa", 2.0e3 + 1.5e4 * difficulty)),
            "suction_time_constant_s": float(overrides.get("suction_time_constant_s", 0.05)),
            # Per-foot safety clamp: twice this robot's weight.
            "max_force_n": float(overrides.get("max_force_n", max(50.0, 2.0 * profile.total_mass_kg * 9.81))),
            "capability_tier": str(overrides.get("capability_tier", DEPTH_RESISTIVE_TIER)),
            "viscous_drag": float(overrides.get("viscous_drag", 0.5 + 3.0 * difficulty)),
        }
        cursor = start
        index = 0
        while cursor < start + length:
            width = min(patch * rng.uniform(1.0, 2.0), start + length - cursor)
            sink = min(scale.foot_length, scale.foot_width) * (0.3 + 1.5 * difficulty)
            geometry.append(_box("marsh-patch-%03d" % index, [cursor + width / 2.0, rng.uniform(-0.15, 0.15) * scale.track_width, -sink / 2.0 - 0.01], [width, scale.track_width * rng.uniform(0.7, 1.0), max(0.04, sink)], "compliant", "marsh", parameters=dict(mud_parameters)))
            cursor += width
            index += 1
        end = start + length
        regions.append(_region("challenge", "marsh", start, end, scale.track_width, True))
        normalized["length_body_lengths"] = length / scale.body_length
        resolved.update({"length_m": length, "patch_m": patch, **mud_parameters})

    world = _finish_world(
        code,
        profile,
        seed,
        difficulty,
        scale,
        start,
        end,
        final_z,
        geometry,
        regions,
        normalized,
        resolved,
        clips,
        {"overrides": overrides},
    )
    world.validate()
    return world
