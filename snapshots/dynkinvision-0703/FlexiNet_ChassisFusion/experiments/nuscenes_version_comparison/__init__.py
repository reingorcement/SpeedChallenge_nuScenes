"""Causal nuScenes comparison with a self-contained legacy default."""

from .bootstrap import SOURCE_PROJECT_ROOT

LEGACY_ALGORITHMS = (
    "base_velocity",
    "legacy_kd",
    "legacy_vkd",
)
V2_ALGORITHMS = (
    "v2_geometry",
    "v2_visual",
    "v2_kinematic_pretrain",
    "v2_visual_kinematic",
)
ALGORITHMS = LEGACY_ALGORITHMS + V2_ALGORITHMS

__all__ = ["ALGORITHMS", "LEGACY_ALGORITHMS", "SOURCE_PROJECT_ROOT", "V2_ALGORITHMS"]
