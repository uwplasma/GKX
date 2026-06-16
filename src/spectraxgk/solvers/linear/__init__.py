"""Linear solve policies and optional parallel RHS dispatch."""

from __future__ import annotations

from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.linear.parallel import (
    linear_rhs_electrostatic_slices_velocity_sharded,
    linear_rhs_parallel_cached,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_streaming_velocity_sharded,
)

__all__ = [
    "KrylovConfig",
    "dominant_eigenpair",
    "linear_rhs_electrostatic_slices_velocity_sharded",
    "linear_rhs_parallel_cached",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_streaming_velocity_sharded",
]
