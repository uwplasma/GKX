"""Linear operator cache facade."""

from __future__ import annotations

from spectraxgk.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
    _numpy_dtype_for_jax,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.cache_model import LinearCache

__all__ = [
    "LinearCache",
    "_build_end_damping_profile_array",
    "_build_gyroaverage_cache_arrays",
    "_build_low_rank_moment_cache_arrays",
    "_numpy_dtype_for_jax",
    "build_linear_cache",
    "collision_damping",
    "hypercollision_damping",
]
