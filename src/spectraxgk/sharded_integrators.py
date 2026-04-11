"""Sharded linear integrator for strong-scaling experiments."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit

from spectraxgk.linear import LinearCache, LinearParams, LinearTerms, linear_rhs_cached


def integrate_linear_sharded(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    terms: LinearTerms | None = None,
    state_sharding: Any | None = None,
) -> jnp.ndarray:
    """Integrate the linear system with a pjit-sharded RK2 loop.

    This is intentionally minimal: it returns the final state only and avoids
    saving time histories to focus on strong scaling of the RHS.
    """

    if terms is None:
        terms = LinearTerms()
    if steps < 1:
        raise ValueError("steps must be >= 1")

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    dt_val = jnp.asarray(dt, dtype=jnp.real(jnp.empty((), dtype=state_dtype)).dtype)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def step(G, _):
        G = _maybe_shard(G)
        dG, _ = linear_rhs_cached(G, cache, params, terms=terms, dt=dt_val)
        G_half = G + 0.5 * dt_val * dG
        dG_half, _ = linear_rhs_cached(G_half, cache, params, terms=terms, dt=dt_val)
        G_next = G + dt_val * dG_half
        return _maybe_shard(G_next), None

    def run(G_init):
        G_final, _ = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final

    run_pjit = pjit(
        run,
        in_shardings=state_sharding,
        out_shardings=state_sharding,
    )

    if state_sharding is not None:
        G0 = jax.device_put(G0, state_sharding)
        G0 = _maybe_shard(G0)
    return run_pjit(G0)
