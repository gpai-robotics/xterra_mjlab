"""Terrain catalog and composition API."""

from .catalog import TERRAIN_CODES, apply_challenge_friction, generate_terrain
from .compose import AE_CHALLENGES, DISCRETE_CHALLENGES, generate_ae, generate_ae_pairs, generate_aee
from .export import export_obj, preview_svg
from .modifiers import apply_geometry_modifiers

__all__ = [
    "AE_CHALLENGES",
    "DISCRETE_CHALLENGES",
    "TERRAIN_CODES",
    "apply_challenge_friction",
    "apply_geometry_modifiers",
    "export_obj",
    "generate_ae",
    "generate_ae_pairs",
    "generate_aee",
    "generate_terrain",
    "preview_svg",
]
