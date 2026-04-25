"""Sharded fixed-step integrators for multi-device scaling experiments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit

from spectraxgk.linear import LinearCache, LinearParams, LinearTerms, linear_rhs_cached
from spectraxgk.nonlinear import _make_hermitian_projector, nonlinear_rhs_cached
from spectraxgk.terms.config import FieldState, TermConfig


_EXPLICIT_METHODS = {"euler", "rk2", "rk3", "rk3_gx", "rk3_classic", "rk4", "sspx3"}


def _dt_array(dt: float, state_dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(dt, dtype=jnp.real(jnp.empty((), dtype=state_dtype)).dtype)


def _validate_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")


def _validate_explicit_method(method: str) -> str:
    method_key = str(method).strip().lower()
    if method_key not in _EXPLICIT_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_gx', 'rk3_classic', 'rk4', 'sspx3'}"
        )
    return method_key


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
    _validate_steps(steps)

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    dt_val = _dt_array(dt, state_dtype)

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


def integrate_nonlinear_sharded(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str = "rk2",
    terms: TermConfig | None = None,
    state_sharding: Any | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
    return_fields: bool = True,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system with an explicit pjit-sharded scan.

    The state array can be partitioned along a ``resolve_state_sharding`` axis
    such as ``ky`` or ``kx``. This is a whole-state sharding primitive for
    production multi-device experiments; it preserves the serial numerical
    update and should be paired with a numerical-identity gate before using its
    timings in performance claims.
    """

    _validate_steps(steps)
    method_key = _validate_explicit_method(method)
    term_cfg = terms or TermConfig()

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    dt_val = _dt_array(dt, state_dtype)

    projector = _make_hermitian_projector(np.asarray(cache.ky), int(np.asarray(cache.kx).size)) if gx_real_fft else None

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def _project(state: jnp.ndarray) -> jnp.ndarray:
        if projector is None:
            return state
        return jnp.asarray(projector(state), dtype=state_dtype)

    def _rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        dG, fields = nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )
        return jnp.asarray(dG, dtype=state_dtype), fields

    def _stage(state: jnp.ndarray, increment: jnp.ndarray, scale: float) -> jnp.ndarray:
        return _maybe_shard(_project(state + jnp.asarray(scale, dtype=dt_val.dtype) * dt_val * increment))

    def step(G: jnp.ndarray, _):
        G = _maybe_shard(_project(G))
        k1, _ = _rhs(G)
        if method_key == "euler":
            G_next = G + dt_val * k1
        elif method_key == "rk2":
            k2, _ = _rhs(_stage(G, k1, 0.5))
            G_next = G + dt_val * k2
        elif method_key == "rk3_classic":
            G1 = _stage(G, k1, 1.0)
            k2, _ = _rhs(G1)
            G2 = _maybe_shard(_project(0.75 * G + 0.25 * (G1 + dt_val * k2)))
            k3, _ = _rhs(G2)
            G_next = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)
        elif method_key in {"rk3", "rk3_gx"}:
            k2, _ = _rhs(_stage(G, k1, 1.0 / 3.0))
            k3, _ = _rhs(_stage(G, k2, 2.0 / 3.0))
            G3 = _stage(G, k3, 0.75)
            G_next = G3 + 0.25 * dt_val * k1
        elif method_key == "rk4":
            k2, _ = _rhs(_stage(G, k1, 0.5))
            k3, _ = _rhs(_stage(G, k2, 0.5))
            k4, _ = _rhs(_stage(G, k3, 1.0))
            G_next = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            G1 = _stage(G, k1, 1.0)
            k2, _ = _rhs(G1)
            G2 = _maybe_shard(_project(0.75 * G + 0.25 * (G1 + dt_val * k2)))
            k3, _ = _rhs(G2)
            G_next = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)

        G_next = _maybe_shard(_project(jnp.asarray(G_next, dtype=state_dtype)))
        _dG_next, fields_next = _rhs(G_next)
        return G_next, fields_next

    def run_with_fields(G_init):
        G_final, fields_t = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final, fields_t

    def run_final_only(G_init):
        G_final, _fields_t = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final

    if state_sharding is not None:
        G0 = jax.device_put(G0, state_sharding)
        G0 = _maybe_shard(G0)

    if return_fields:
        return pjit(run_with_fields, in_shardings=state_sharding, out_shardings=None)(G0)
    return pjit(run_final_only, in_shardings=state_sharding, out_shardings=state_sharding)(G0)
