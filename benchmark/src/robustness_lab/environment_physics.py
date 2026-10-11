"""Policy-independent environment-physics planning and reference effects.

Policies consume observations and produce actions.  They never decide how a
terrain material is simulated.  This module turns a canonical ``WorldSpec``
into an auditable backend plan and provides deterministic reference
implementations for the RS11 rigid-viscous and depth-resistive fidelity tiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import WorldSpec


RIGID_VISCOUS_TIER = "rigid_viscous"
DEPTH_RESISTIVE_TIER = "depth_resistive_mud_v1"
DEPTH_RESISTIVE_EFFECT = "depth_resistive_mud"

_MUD_PARAMETER_NAMES = (
    "support_stress_pa",
    "depth_exponent",
    "reference_depth_m",
    "damping_pa_s_per_m",
    "yield_stress_pa",
    "suction_time_constant_s",
    "max_force_n",
)


@dataclass(frozen=True)
class MudParameters:
    support_stress_pa: float
    depth_exponent: float
    reference_depth_m: float
    damping_pa_s_per_m: float
    yield_stress_pa: float
    suction_time_constant_s: float
    max_force_n: float

    def __post_init__(self) -> None:
        values = {
            name: float(getattr(self, name)) for name in _MUD_PARAMETER_NAMES
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("mud parameters must be finite")
        if values["support_stress_pa"] < 0.0:
            raise ValueError("support_stress_pa must be non-negative")
        if values["depth_exponent"] <= 0.0:
            raise ValueError("depth_exponent must be positive")
        if values["reference_depth_m"] <= 0.0:
            raise ValueError("reference_depth_m must be positive")
        if values["damping_pa_s_per_m"] < 0.0:
            raise ValueError("damping_pa_s_per_m must be non-negative")
        if values["yield_stress_pa"] < 0.0:
            raise ValueError("yield_stress_pa must be non-negative")
        if values["suction_time_constant_s"] <= 0.0:
            raise ValueError("suction_time_constant_s must be positive")
        if values["max_force_n"] <= 0.0:
            raise ValueError("max_force_n must be positive")


class MudFootContact:
    """Per-foot state for one mud contact."""

    def __init__(self, parameters: MudParameters) -> None:
        self.parameters = parameters
        self.reset()

    def reset(self) -> None:
        self._suction_state = 0.0

    def step(
        self,
        depth_m: float,
        vertical_velocity_mps: float,
        contact_area_m2: float,
        dt_s: float,
    ) -> float:
        """Return signed vertical force in newtons, positive upward."""
        dt_s = float(dt_s)
        if dt_s <= 0.0 or not math.isfinite(dt_s):
            raise ValueError("dt_s must be positive and finite")
        depth_m = float(depth_m)
        contact_area_m2 = float(contact_area_m2)
        vertical_velocity_mps = float(vertical_velocity_mps)
        if depth_m <= 0.0 or contact_area_m2 <= 0.0:
            self.reset()
            return 0.0

        parameters = self.parameters
        try:
            support = (
                parameters.support_stress_pa
                * (depth_m / parameters.reference_depth_m)
                ** parameters.depth_exponent
                * contact_area_m2
            )
            damping = (
                -parameters.damping_pa_s_per_m
                * vertical_velocity_mps
                * contact_area_m2
            )
        except OverflowError:
            support = math.inf
            damping = 0.0

        suction = 0.0
        if vertical_velocity_mps > 0.0:
            self._suction_state = 1.0 - (1.0 - self._suction_state) * math.exp(
                -dt_s / parameters.suction_time_constant_s
            )
            self._suction_state = min(1.0, max(0.0, self._suction_state))
            suction = (
                -parameters.yield_stress_pa
                * contact_area_m2
                * self._suction_state
            )
        else:
            self.reset()

        total = support + damping + suction
        limit = parameters.max_force_n
        if math.isnan(total):
            return 0.0
        if not math.isfinite(total):
            return math.copysign(limit, total)
        return min(limit, max(-limit, total))


def _box_bounds(item: Any, offset: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return conservative world-axis bounds for a possibly yawed box."""
    half_x = 0.5 * float(item.size[0])
    half_y = 0.5 * float(item.size[1])
    yaw = float(item.pose.rpy[2])
    extent_x = abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y
    extent_y = abs(math.sin(yaw)) * half_x + abs(math.cos(yaw)) * half_y
    center = [float(item.pose.position[i]) + float(offset[i]) for i in range(3)]
    half_z = 0.5 * float(item.size[2])
    return (
        [center[0] - extent_x, center[1] - extent_y, center[2] - half_z],
        [center[0] + extent_x, center[1] + extent_y, center[2] + half_z],
    )


def world_physics_plan(
    world: WorldSpec,
    coordinate_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """Describe required environment effects without referring to a policy.

    Effect bounds are translated into the backend's local terrain frame.
    Materialised worlds retain their recorded RS11 fidelity tier.
    """
    offset = [float(value) for value in coordinate_offset]
    if len(offset) != 3:
        raise ValueError("coordinate_offset must contain three values")
    marsh_regions = [region for region in world.regions if region.kind == "marsh"]
    marsh_geometry = [item for item in world.geometry if item.semantic == "marsh"]
    has_marsh = bool(marsh_regions or marsh_geometry)
    recorded_geometry_tiers = {
        str(item.parameters["capability_tier"])
        for item in marsh_geometry
        if item.parameters.get("capability_tier")
    }
    requested_tier = str(
        world.resolved_parameters.get(
            "capability_tier",
            next(iter(recorded_geometry_tiers))
            if len(recorded_geometry_tiers) == 1
            else RIGID_VISCOUS_TIER if has_marsh else "rigid_contact",
        )
    )
    drag_values = [
        float(item.parameters.get("viscous_drag", 0.0))
        for item in world.geometry
        if item.semantic == "marsh" or item.material == "compliant"
    ]
    drag = max(drag_values, default=0.0)
    effects: List[Dict[str, Any]] = []
    if has_marsh and requested_tier == DEPTH_RESISTIVE_TIER:
        if not marsh_geometry:
            raise ValueError("depth-resistive mud requires marsh geometry")
        for item in marsh_geometry:
            missing = [
                name for name in _MUD_PARAMETER_NAMES if name not in item.parameters
            ]
            if missing:
                raise ValueError(
                    "depth-resistive marsh %s is missing parameters: %s"
                    % (item.name, ", ".join(missing))
                )
            minimum, maximum = _box_bounds(item, offset)
            effects.append(
                {
                    "type": DEPTH_RESISTIVE_EFFECT,
                    "region": item.name,
                    "minimum": minimum,
                    "maximum": maximum,
                    **{
                        name: float(item.parameters[name])
                        for name in _MUD_PARAMETER_NAMES
                    },
                    "applies_to": "robot_feet",
                }
            )
    elif has_marsh and requested_tier == RIGID_VISCOUS_TIER:
        if drag <= 0.0:
            raise ValueError("RS11 marsh regions require a positive viscous_drag")
        bounds = [
            (
                region.name,
                [float(region.minimum[i]) + offset[i] for i in range(3)],
                [float(region.maximum[i]) + offset[i] for i in range(3)],
            )
            for region in marsh_regions
        ]
        if not bounds:
            bounds = [
                (item.name, *_box_bounds(item, offset)) for item in marsh_geometry
            ]
        for name, minimum, maximum in bounds:
            effects.append(
                {
                    "type": "linear_velocity_drag",
                    "region": name,
                    "minimum": minimum,
                    "maximum": maximum,
                    "coefficient_per_s": drag,
                    "applies_to": "robot_feet",
                }
            )
    elif has_marsh:
        raise ValueError("unsupported recorded marsh capability tier: %s" % requested_tier)
    return {
        "schema_version": "1.0",
        "implementation_owner": "simulator_adapter",
        "policy_interface_modified": False,
        "terrain_type": world.terrain_type,
        "requested_tier": requested_tier,
        "required_effects": sorted({item["type"] for item in effects}),
        "effects": effects,
    }


def assess_environment_support(plan: Dict[str, Any], supported_effects: Iterable[str]) -> Dict[str, Any]:
    """Return an explicit preflight result for one backend implementation."""
    supported = set(str(value) for value in supported_effects)
    required = set(str(value) for value in plan.get("required_effects", []))
    missing = sorted(required - supported)
    return {
        "status": "environment_supported" if not missing else "environment_unsupported",
        "capability_claim_eligible": not missing,
        "requested_tier": plan.get("requested_tier", "rigid_contact"),
        "supported_effects": sorted(supported),
        "missing_effects": missing,
        "implementation_owner": "simulator_adapter",
        "policy_interface_modified": False,
        "reason": "" if not missing else "simulator adapter is missing environment effects: %s" % ", ".join(missing),
    }


def apply_linear_drag(
    position: Sequence[float],
    linear_velocity: Sequence[float],
    effects: Sequence[Dict[str, Any]],
    dt_s: float,
) -> Tuple[List[float], List[str]]:
    """Apply time-step-invariant linear drag for all containing regions.

    ``v(t+dt) = v(t) exp(-k dt)`` is the exact solution of linear drag per
    unit mass.  Using the exponential keeps the effect stable when simulator
    time steps differ.  Region membership uses horizontal coordinates because
    the lowered rigid patches already represent the configured sink depth.
    """
    if dt_s <= 0.0 or not math.isfinite(dt_s):
        raise ValueError("dt_s must be positive and finite")
    if len(position) < 2 or len(linear_velocity) < 3:
        raise ValueError("position requires x/y and linear_velocity requires x/y/z")
    velocity = [float(value) for value in linear_velocity[:3]]
    applied: List[str] = []
    for effect in effects:
        if effect.get("type") != "linear_velocity_drag":
            continue
        minimum = effect["minimum"]
        maximum = effect["maximum"]
        inside = all(float(minimum[axis]) <= float(position[axis]) <= float(maximum[axis]) for axis in (0, 1))
        if not inside:
            continue
        coefficient = max(0.0, float(effect.get("coefficient_per_s", 0.0)))
        decay = math.exp(-coefficient * dt_s)
        velocity = [value * decay for value in velocity]
        applied.append(str(effect.get("region", "unnamed")))
    return velocity, applied
