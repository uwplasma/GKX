"""Cached full-RHS assembly for gyrokinetic linear terms."""

from __future__ import annotations

from typing import Tuple

import functools
import jax
import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.moments import build_H
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly_helpers import (
    _normalized_rhs_state,
    _rhs_term_contributions,
    _scalar_params,
    _solve_cached_fields,
    _solved_rhs_fields,
    _species_arrays,
    _sum_rhs_terms,
    _term_weights,
)
from spectraxgk.terms.config import FieldState, TermConfig


def assemble_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
    external_phi: jnp.ndarray | float | None = None,
    force_electrostatic_fields: bool = False,
) -> Tuple[jnp.ndarray, FieldState]:
    """Assemble the cached full RHS from named physics stages.

    The production path shares the same state normalization, species/scalar
    expansion, field/Hamiltonian construction, term contribution, and fixed-order
    summation helpers used by the per-term diagnostic path. This keeps the fast
    RHS and debug RHS numerically aligned while preserving the public return
    convention for single-species states.
    """

    term_cfg = terms or TermConfig()
    state = _normalized_rhs_state(G, cache)
    species = _species_arrays(params, state.G.shape[0], state.real_dtype)
    scalars = _scalar_params(params, state.real_dtype, dt)
    weights = _term_weights(params, term_cfg, state.real_dtype)
    rhs_fields = _solved_rhs_fields(
        state.G,
        cache,
        params,
        term_cfg,
        species,
        weights,
        use_custom_vjp=use_custom_vjp,
        external_phi=external_phi,
        force_electrostatic_fields=force_electrostatic_fields,
        build_H_fn=build_H,
    )
    contrib = _rhs_term_contributions(
        state, cache, species, scalars, weights, rhs_fields
    )
    dG = _sum_rhs_terms(contrib)
    if state.squeeze_species:
        dG = dG[0]
    return dG.astype(state.out_dtype), rhs_fields.fields


def assemble_rhs_cached_jit(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig,
    dt: jnp.ndarray | float | None = None,
    external_phi: jnp.ndarray | float | None = None,
) -> Tuple[jnp.ndarray, FieldState]:
    """Jitted wrapper for cached RHS assembly."""

    return assemble_rhs_cached(
        G, cache, params, terms=terms, dt=dt, external_phi=external_phi
    )


@functools.partial(jax.jit)
def assemble_rhs_cached_electrostatic_jit(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig,
    dt: jnp.ndarray | float | None = None,
    external_phi: jnp.ndarray | float | None = None,
) -> Tuple[jnp.ndarray, FieldState]:
    """Jitted cached RHS assembly for statically electrostatic field terms."""

    return assemble_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        dt=dt,
        external_phi=external_phi,
        force_electrostatic_fields=True,
    )


def compute_fields_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
    external_phi: jnp.ndarray | float | None = None,
) -> FieldState:
    """Compute fields for a cached state without assembling the RHS."""

    term_cfg = terms or TermConfig()
    state = _normalized_rhs_state(G, cache)
    species = _species_arrays(params, state.G.shape[0], state.real_dtype)
    weights = _term_weights(params, term_cfg, state.real_dtype)
    fields = _solve_cached_fields(
        state.G,
        cache,
        params,
        species=species,
        weights=weights,
        use_custom_vjp=use_custom_vjp,
        external_phi=external_phi,
    )
    if state.squeeze_species:
        return FieldState(phi=fields.phi, apar=fields.apar, bpar=fields.bpar)
    return fields


__all__ = [
    "assemble_rhs_cached",
    "assemble_rhs_cached_electrostatic_jit",
    "assemble_rhs_cached_jit",
    "compute_fields_cached",
]
