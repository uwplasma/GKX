"""Linear gyrokinetic RHS assembly entry points."""

from __future__ import annotations

import jax.numpy as jnp

from gkx.operators.collision import CollisionOperator
from gkx.geometry import FluxTubeGeometryLike
from gkx.core_grid import SpectralGrid, _gyrokinetic_moment_shape
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    dt: jnp.ndarray | float | None = None,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS and electrostatic potential from grid/geometry inputs."""

    Nl, Nm = _gyrokinetic_moment_shape(G, name="G")
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        dt=dt,
        collision_operator=collision_operator,
    )


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
    force_electrostatic_fields: bool = False,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry/cache arrays."""

    from gkx.terms.assembly import assemble_linear_rhs_cached

    term_cfg = linear_terms_to_term_config(terms)
    dG, fields = assemble_linear_rhs_cached(
        G,
        cache,
        params,
        terms=term_cfg,
        use_jit=use_jit,
        use_custom_vjp=use_custom_vjp,
        dt=dt,
        electrostatic_fields=force_electrostatic_fields,
        collision_operator=collision_operator,
    )
    return dG, fields.phi


__all__ = ["linear_rhs", "linear_rhs_cached"]
