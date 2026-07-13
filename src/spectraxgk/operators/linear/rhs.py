"""Linear gyrokinetic RHS assembly entry points."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.core.extension_points import CollisionOperator
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.terms.linear_dissipation import (
    custom_collision_contribution,
    terms_without_builtin_collisions,
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

    if G.ndim == 5:
        Nl, Nm = G.shape[0], G.shape[1]
    elif G.ndim == 6:
        Nl, Nm = G.shape[1], G.shape[2]
    else:
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
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

    from spectraxgk.terms.assembly import (
        assemble_rhs_cached,
        assemble_rhs_cached_electrostatic_jit,
        assemble_rhs_cached_jit,
    )

    term_cfg = linear_terms_to_term_config(terms)
    assembled_terms = terms_without_builtin_collisions(term_cfg, collision_operator)

    if use_jit:
        rhs_fn = (
            assemble_rhs_cached_electrostatic_jit
            if force_electrostatic_fields
            else assemble_rhs_cached_jit
        )
        dG, fields = rhs_fn(G, cache, params, assembled_terms, dt)
    else:
        dG, fields = assemble_rhs_cached(
            G,
            cache,
            params,
            terms=assembled_terms,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    collision_rhs = custom_collision_contribution(
        G,
        fields,
        cache,
        params,
        term_cfg,
        collision_operator,
        force_electrostatic_fields=force_electrostatic_fields,
    )
    if collision_rhs is not None:
        dG = dG + collision_rhs
    return dG, fields.phi


__all__ = ["linear_rhs", "linear_rhs_cached"]
