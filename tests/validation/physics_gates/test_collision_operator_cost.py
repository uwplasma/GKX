"""Performance guard: cost of the collision operators in the linear RHS.

The finite-Larmor Coulomb operator interpolates its tables at every grid point,
so the compiler sees a distinct moment matrix per ``(ky, kx, z)``. That storage
grows as ``n^2`` in the moment count while the state grows as ``n``, which is
what eventually bounds the reachable resolution:

===================  ==========  ==========================================
moments ``n``        grid        per-point matrix storage
===================  ==========  ==========================================
8   ``(4, 2)``       32x64x32    0.07 GB
18  ``(6, 3)``       32x64x32    0.34 GB
128 ``(16, 8)``      32x64x32    17 GB, past a 16 GB card
512 ``(32, 16)``     32x64x32    275 GB
===================  ==========  ==========================================

Published convergence studies ask for ``(16, 8)`` for linear Cyclone-base-case
ITG and ``(32, 16)`` converged, so this is the mechanism that has to change
before those resolutions are reachable. It is comfortable at the resolutions
GKX ships today, which is why this file measures and bounds the cost rather
than asserting a target that the current implementation cannot meet.

The bounds are ratios against the built-in diagonal operator rather than
absolute byte counts, so they track the structural cost and tolerate compiler
and library changes.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers.time.runners import _resolve_config_collision_operator

# Small enough to compile quickly, large enough that the per-point matrices
# dominate the difference between the two operators.
GRID = GridConfig(Nx=8, Ny=16, Nz=32, Lx=62.8, Ly=62.8)


def compiled_rhs_cost(collision_operator: str, hermite: int, laguerre: int):
    """Return (flops, bytes, temp_bytes) for one compiled linear RHS."""

    config = CycloneBaseCase(grid=GRID)
    grid = build_spectral_grid(config.grid)
    geometry = SAlphaGeometry.from_config(config.geometry)
    parameters = LinearParams(nu=0.05)
    state = jnp.zeros(
        (hermite, laguerre, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex128,
    )
    cache = build_linear_cache(grid, geometry, parameters, hermite, laguerre)
    time_config = dataclasses.replace(
        config.time, collision_operator=collision_operator
    )
    operator = _resolve_config_collision_operator(time_config, parameters, state)

    def rhs(value):
        return linear_rhs_cached(
            value, cache, parameters, use_jit=False, collision_operator=operator
        )[0]

    compiled = jax.jit(rhs).lower(state).compile()
    analysis = compiled.cost_analysis()
    return (
        float(analysis["flops"]),
        float(analysis["bytes accessed"]),
        float(compiled.memory_analysis().temp_size_in_bytes),
    )


@pytest.mark.parametrize(
    ("hermite", "laguerre", "temp_ratio_bound"),
    [(4, 2, 12.0), (6, 3, 24.0)],
)
def test_finite_larmor_overhead_stays_within_its_measured_envelope(
    hermite: int, laguerre: int, temp_ratio_bound: float
) -> None:
    """The finite-Larmor operator's extra temporary storage must not blow up.

    Measured on this grid: 5.9x the diagonal operator at 8 moments and 12.1x at
    18. The bounds are set at roughly twice those so ordinary compiler drift
    passes while a structural regression, such as losing a fusion or
    materializing the tables for every species pair, fails.
    """

    _, _, diagonal_temp = compiled_rhs_cost("lenard_bernstein", hermite, laguerre)
    _, _, coulomb_temp = compiled_rhs_cost("coulomb_finite_kperp", hermite, laguerre)

    assert diagonal_temp > 0.0
    ratio = coulomb_temp / diagonal_temp
    assert ratio < temp_ratio_bound, (
        f"({hermite},{laguerre}): finite-Larmor temporaries are {ratio:.1f}x the "
        f"diagonal operator, above the {temp_ratio_bound}x envelope"
    )


def test_finite_larmor_cost_grows_no_faster_than_the_moment_count_squared() -> None:
    """Cost must track the n^2 matrix, not something steeper.

    Going from 8 to 18 moments raises n^2 by 5.06x. A growth rate materially
    above that would mean the implementation had acquired an extra factor, for
    instance interpolating per species pair or per Runge-Kutta stage.
    """

    _, _, small = compiled_rhs_cost("coulomb_finite_kperp", 4, 2)
    _, _, large = compiled_rhs_cost("coulomb_finite_kperp", 6, 3)
    moment_ratio = (18 / 8) ** 2
    growth = large / small
    assert growth < 1.5 * moment_ratio, (
        f"temporaries grew {growth:.2f}x for an n^2 ratio of {moment_ratio:.2f}x"
    )


def test_diagonal_operator_stays_cheap() -> None:
    """The built-in operator must not acquire per-point matrix storage.

    It is the fallback for every path that cannot carry a moment operator, so a
    regression here would be felt everywhere.
    """

    _, _, small = compiled_rhs_cost("lenard_bernstein", 4, 2)
    _, _, large = compiled_rhs_cost("lenard_bernstein", 6, 3)
    # Diagonal damping stores O(n) per point, so cost should track the state
    # size (2.25x from 8 to 18 moments), not n^2.
    assert large / small < 4.0, f"diagonal operator grew {large / small:.2f}x"
