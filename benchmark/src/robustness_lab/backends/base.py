"""Backend contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List

from ..models import EpisodeLog, EpisodeSpec, RobotProfile, WorldSpec
from ..environment_physics import assess_environment_support, world_physics_plan


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class BackendCapabilities:
    name: str
    available: bool
    version: str = ""
    geometry: List[str] = field(default_factory=lambda: ["box"])
    deformable: bool = False
    camera: bool = False
    contact_forces: bool = False
    environment_effects: List[str] = field(default_factory=list)
    observations: FrozenSet[str] = field(default_factory=frozenset)
    reason: str = ""

    def assess_world(self, world: WorldSpec) -> Dict[str, Any]:
        return assess_environment_support(world_physics_plan(world), self.environment_effects)

    def observation_capabilities(self) -> FrozenSet[str]:
        """Canonical observation fields this adapter can genuinely produce."""

        return frozenset(self.observations)


class Backend:
    name = "base"

    @classmethod
    def probe(cls) -> BackendCapabilities:
        return BackendCapabilities(cls.name, True)

    def run(self, episode: EpisodeSpec, robot: RobotProfile, world: WorldSpec) -> EpisodeLog:
        raise NotImplementedError

    def reset_canonical_state(
        self,
        robot: RobotProfile,
        robot_asset: str,
        robot_format: str,
        velocity_command=(0.0, 0.0, 0.0),
    ):
        """Return a CanonicalState at the default pose on flat ground, or None.

        None means this backend cannot produce gate evidence. Returning
        approximate or invented values instead is forbidden.
        """

        return None

    @classmethod
    def require_world_support(cls, world: WorldSpec) -> Dict[str, Any]:
        report = cls.probe().assess_world(world)
        if report["status"] != "environment_supported":
            raise BackendUnavailable(report["reason"])
        return report
