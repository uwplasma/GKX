"""Sharded fixed-step integrators for multi-device scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.nonlinear import _make_hermitian_projector, nonlinear_rhs_cached
from spectraxgk.terms.config import FieldState, TermConfig


_EXPLICIT_METHODS = {"euler", "rk2", "rk3", "rk3_heun", "rk3_classic", "rk4", "sspx3"}
pjit = jax.jit


@dataclass(frozen=True)
class _NonlinearShardSetup:
    G0: jnp.ndarray
    dt_val: jnp.ndarray
    state_dtype: jnp.dtype
    method_key: str
    terms: TermConfig
    projector: Callable[[jnp.ndarray], jnp.ndarray] | None


def _dt_array(dt: float, state_dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(dt, dtype=jnp.real(jnp.empty((), dtype=state_dtype)).dtype)


def _validate_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")


def _validate_explicit_method(method: str) -> str:
    method_key = str(method).strip().lower()
    if method_key not in _EXPLICIT_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_heun', 'rk3_classic', 'rk4', 'sspx3'}"
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


def _rk3_classic_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    project_shard: Callable[[jnp.ndarray], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    G1 = stage(G, k1, 1.0)
    k2, _ = rhs(G1)
    G2 = project_shard(0.75 * G + 0.25 * (G1 + dt_val * k2))
    k3, _ = rhs(G2)
    return (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)


def _rk3_heun_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    k2, _ = rhs(stage(G, k1, 1.0 / 3.0))
    k3, _ = rhs(stage(G, k2, 2.0 / 3.0))
    return stage(G, k3, 0.75) + 0.25 * dt_val * k1


def _rk4_update(
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    k2, _ = rhs(stage(G, k1, 0.5))
    k3, _ = rhs(stage(G, k2, 0.5))
    k4, _ = rhs(stage(G, k3, 1.0))
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _nonlinear_explicit_update(
    method_key: str,
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    stage: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    project_shard: Callable[[jnp.ndarray], jnp.ndarray],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    if method_key == "euler":
        return G + dt_val * k1
    if method_key == "rk2":
        k2, _ = rhs(stage(G, k1, 0.5))
        return G + dt_val * k2
    if method_key == "rk3_classic":
        return _rk3_classic_update(
            G,
            k1,
            rhs=rhs,
            stage=stage,
            project_shard=project_shard,
            dt_val=dt_val,
        )
    if method_key in {"rk3", "rk3_heun"}:
        return _rk3_heun_update(G, k1, rhs=rhs, stage=stage, dt_val=dt_val)
    if method_key == "rk4":
        return _rk4_update(G, k1, rhs=rhs, stage=stage, dt_val=dt_val)
    return _rk3_classic_update(
        G,
        k1,
        rhs=rhs,
        stage=stage,
        project_shard=project_shard,
        dt_val=dt_val,
    )


def _prepare_nonlinear_sharded_setup(
    G0: jnp.ndarray,
    cache: LinearCache,
    *,
    dt: float,
    method: str,
    terms: TermConfig | None,
    compressed_real_fft: bool,
) -> _NonlinearShardSetup:
    """Normalize nonlinear sharded integration inputs and projection policy."""

    method_key = _validate_explicit_method(method)
    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0_arr = jnp.asarray(G0, dtype=state_dtype)
    projector = (
        _make_hermitian_projector(np.asarray(cache.ky), int(np.asarray(cache.kx).size))
        if compressed_real_fft
        else None
    )
    return _NonlinearShardSetup(
        G0=G0_arr,
        dt_val=_dt_array(dt, state_dtype),
        state_dtype=state_dtype,
        method_key=method_key,
        terms=terms or TermConfig(),
        projector=projector,
    )


def _nonlinear_sharding_projector(
    *,
    state_sharding: Any | None,
    projector: Callable[[jnp.ndarray], jnp.ndarray] | None,
    state_dtype: jnp.dtype,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Return sharding and Hermitian-projection helpers for the scan loop."""

    def maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def project_shard(state: jnp.ndarray) -> jnp.ndarray:
        if projector is not None:
            state = jnp.asarray(projector(state), dtype=state_dtype)
        return maybe_shard(state)

    return maybe_shard, project_shard


def _nonlinear_sharded_rhs(
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig,
    state_dtype: jnp.dtype,
    compressed_real_fft: bool,
    laguerre_mode: str,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]]:
    """Build the nonlinear RHS closure used by sharded explicit scans."""

    def rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        dG, fields = nonlinear_rhs_cached(
            state,
            cache,
            params,
            terms,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
        return jnp.asarray(dG, dtype=state_dtype), fields

    return rhs


def _nonlinear_sharded_step_fn(
    *,
    setup: _NonlinearShardSetup,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]],
    project_shard: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray, None], tuple[jnp.ndarray, FieldState]]:
    """Build one explicit sharded nonlinear scan step."""

    def stage(state: jnp.ndarray, increment: jnp.ndarray, scale: float) -> jnp.ndarray:
        return project_shard(
            state
            + jnp.asarray(scale, dtype=setup.dt_val.dtype) * setup.dt_val * increment
        )

    def step(G: jnp.ndarray, _) -> tuple[jnp.ndarray, FieldState]:
        G = project_shard(G)
        k1, _ = rhs(G)
        G_next = _nonlinear_explicit_update(
            setup.method_key,
            G,
            k1,
            rhs=rhs,
            stage=stage,
            project_shard=project_shard,
            dt_val=setup.dt_val,
        )
        G_next = project_shard(jnp.asarray(G_next, dtype=setup.state_dtype))
        _dG_next, fields_next = rhs(G_next)
        return G_next, fields_next

    return step


def _maybe_put_sharded_state(
    G0: jnp.ndarray,
    *,
    state_sharding: Any | None,
    maybe_shard: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    if state_sharding is None:
        return G0
    return maybe_shard(jax.device_put(G0, state_sharding))


def _run_nonlinear_sharded_scan(
    G0: jnp.ndarray,
    *,
    step: Callable[[jnp.ndarray, None], tuple[jnp.ndarray, FieldState]],
    steps: int,
    state_sharding: Any | None,
    return_fields: bool,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Dispatch final-only or field-history nonlinear sharded scans."""

    def run_with_fields(G_init):
        G_final, fields_t = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final, fields_t

    def run_final_only(G_init):
        G_final, _fields_t = jax.lax.scan(step, G_init, xs=None, length=steps)
        return G_final

    if return_fields:
        return pjit(run_with_fields, in_shardings=state_sharding, out_shardings=None)(G0)
    return pjit(run_final_only, in_shardings=state_sharding, out_shardings=state_sharding)(G0)


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
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    return_fields: bool = True,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system with an explicit pjit-sharded scan.

    The state array can be partitioned along a ``resolve_state_sharding`` axis
    such as ``ky`` or ``kx``. This is a diagnostic whole-state sharding
    primitive for identity gates and profiler localization. It is not a
    production nonlinear domain decomposition or speedup claim until the exact
    workload has communication-complete identity, conservation, transport, and
    profiler gates. Domain-sharding identity reports are metadata gates only;
    they do not authorize routing through this whole-state integrator.
    """

    _validate_steps(steps)
    setup = _prepare_nonlinear_sharded_setup(
        G0,
        cache,
        dt=dt,
        method=method,
        terms=terms,
        compressed_real_fft=compressed_real_fft,
    )
    maybe_shard, project_shard = _nonlinear_sharding_projector(
        state_sharding=state_sharding,
        projector=setup.projector,
        state_dtype=setup.state_dtype,
    )
    rhs = _nonlinear_sharded_rhs(
        cache=cache,
        params=params,
        terms=setup.terms,
        state_dtype=setup.state_dtype,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )
    step = _nonlinear_sharded_step_fn(
        setup=setup,
        rhs=rhs,
        project_shard=project_shard,
    )
    G_init = _maybe_put_sharded_state(
        setup.G0,
        state_sharding=state_sharding,
        maybe_shard=maybe_shard,
    )
    return _run_nonlinear_sharded_scan(
        G_init,
        step=step,
        steps=steps,
        state_sharding=state_sharding,
        return_fields=return_fields,
    )


__all__ = ["integrate_linear_sharded", "integrate_nonlinear_sharded"]
