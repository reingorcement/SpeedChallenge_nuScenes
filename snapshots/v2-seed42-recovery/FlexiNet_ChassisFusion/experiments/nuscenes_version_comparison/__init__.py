"""Causal nuScenes comparison between retained legacy models and V2."""

from .bootstrap import SOURCE_PROJECT_ROOT

ALGORITHMS = (
    "base_velocity",
    "legacy_kd",
    "legacy_vkd",
    "v2_geometry",
    "v2_visual",
    "v2_kinematic_pretrain",
    "v2_visual_kinematic",
)

__all__ = ["ALGORITHMS", "SOURCE_PROJECT_ROOT"]
