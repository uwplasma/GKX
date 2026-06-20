"""Per-term RHS decomposition for diagnostics and parity audits."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.moments import build_H
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly_helpers import (
    _normalized_rhs_state,
    _rhs_term_contributions,
    _scalar_params,
    _solved_rhs_fields,
    _species_arrays,
    _squeeze_species_rhs_outputs,
    _sum_rhs_terms,
    _term_weights,
)
from spectraxgk.terms.config import FieldState, TermConfig


def assemble_rhs_terms_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, FieldState, dict[str, jnp.ndarray]]:
    """Assemble per-term RHS contributions for diagnostics and audits.

    Returns ``(total_rhs, fields, term_contributions)``. The production RHS and
    this diagnostic path share the same state normalization, species expansion,
    field solve, Hamiltonian construction, and fixed-order summation helpers so
    a term audit compares named physics pieces without duplicating numerical
    policy.
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
        build_H_fn=build_H,
    )
    contrib = _rhs_term_contributions(
        state, cache, species, scalars, weights, rhs_fields
    )
    total = _sum_rhs_terms(contrib)
    if state.squeeze_species:
        total, contrib = _squeeze_species_rhs_outputs(total, contrib)
    return total.astype(state.out_dtype), rhs_fields.fields, contrib


__all__ = ["assemble_rhs_terms_cached"]
