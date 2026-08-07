"""Compiled multi-candidate extraction for full-operator RK4 propagators."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.solvers.linear.krylov_algorithms import (
    _advance_rk4,
    _apply_operator,
    _arnoldi,
)
from gkx.terms.config import TermConfig


@partial(
    jax.jit,
    static_argnames=("krylov_dim", "propagator_steps", "candidates"),
)
def dominant_eigenpairs_propagator_cached(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    dt: float,
    propagator_steps: int,
    candidates: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract several leading-growth candidates from one filtered subspace.

    Near a branch crossing, roundoff in ``|exp(T lambda)|`` can reverse two
    almost equal growth rates even when both eigenvectors are already resolved.
    Returning the leading projected candidates and ranking their continuous
    Rayleigh values avoids another propagator pass and makes the cold path
    branch-cluster aware.
    """

    if not 1 <= candidates <= krylov_dim:
        raise ValueError("candidates must lie in [1, krylov_dim]")
    if propagator_steps < 1:
        raise ValueError("propagator_steps must be positive")
    dt_val = jnp.asarray(dt, dtype=jnp.real(v0).dtype)

    def apply_prop(x, cache, params, term_cfg):
        return jax.lax.fori_loop(
            0,
            propagator_steps,
            lambda _index, state: _advance_rk4(
                state,
                cache,
                params,
                term_cfg,
                dt_val,
            ),
            x,
        )

    basis, projected = _arnoldi(
        v0,
        apply_prop,
        cache,
        params,
        term_cfg,
        krylov_dim,
    )
    eigenvalues, coefficients = jnp.linalg.eig(
        projected[:krylov_dim, :krylov_dim],
    )
    candidate_indices = jnp.argsort(jnp.abs(eigenvalues))[-candidates:][::-1]
    lifted = jnp.tensordot(
        coefficients[:, candidate_indices].T,
        basis[:krylov_dim],
        axes=1,
    )
    flattened = lifted.reshape((candidates, -1))
    norms = jnp.linalg.norm(flattened, axis=1)
    safe_norms = jnp.where(norms > 0.0, norms, 1.0)
    vectors = (flattened / safe_norms[:, None]).reshape((candidates, *v0.shape))

    def certify(vector):
        image = _apply_operator(vector, cache, params, term_cfg)
        denominator = jnp.vdot(vector, vector)
        safe_denominator = jnp.where(
            denominator != 0.0,
            denominator,
            1.0 + 0.0j,
        )
        value = jnp.vdot(vector, image) / safe_denominator
        residual = jnp.linalg.norm(image - value * vector)
        residual = residual / jnp.maximum(
            jnp.abs(value) * jnp.linalg.norm(vector),
            jnp.finfo(jnp.real(vector).dtype).tiny,
        )
        return value, residual

    values, residuals = jax.vmap(certify)(vectors)
    return values, vectors, residuals


__all__ = ["dominant_eigenpairs_propagator_cached"]
