"""Linear solve policies and optional parallel RHS dispatch."""

from __future__ import annotations

from gkx.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from gkx.solvers.linear.implicit import (
    _build_implicit_operator,
    _integrate_linear_implicit_cached,
)
from gkx.solvers.linear.integrators import (
    _integrate_linear_cached,
    _integrate_linear_cached_donate,
    _integrate_linear_cached_impl,
    integrate_linear,
    integrate_linear_diagnostics,
)
from gkx.solvers.linear.parallel import (
    linear_rhs_electrostatic_slices_velocity_sharded,
    linear_rhs_parallel_cached,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_streaming_velocity_sharded,
)

__all__ = [
    "KrylovConfig",
    "_build_implicit_operator",
    "_integrate_linear_cached",
    "_integrate_linear_cached_donate",
    "_integrate_linear_cached_impl",
    "_integrate_linear_implicit_cached",
    "dominant_eigenpair",
    "integrate_linear",
    "integrate_linear_diagnostics",
    "linear_rhs_electrostatic_slices_velocity_sharded",
    "linear_rhs_parallel_cached",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_streaming_velocity_sharded",
]
