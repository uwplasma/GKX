"""Explicit linear step kernels and growth-rate diagnostic helpers."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import FieldState, TermConfig

LinearStageRhsFn = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = [
    "_SSPX3_ADT",
    "_SSPX3_W1",
    "_SSPX3_W2",
    "_SSPX3_W3",
    "_apply_completed_step_state_mask",
    "_completed_step_state_mask",
    "_diagnostic_midplane_index",
    "_growth_rate_mode_mask",
    "_instantaneous_growth_rate_step",
    "_linear_explicit_step",
    "_linear_term_config",
    "_rk3_heun_step",
    "_rk4_step",
]


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _completed_step_state_mask(cache: LinearCache) -> jnp.ndarray:
    """Return the completed-step state-space mask applied after each completed step."""

    mask = _growth_rate_mode_mask(cache.ky, cache.kx, cache.dealias_mask)
    ky_zero = jnp.isclose(jnp.asarray(cache.ky), 0.0)
    kx_zero = jnp.isclose(jnp.asarray(cache.kx), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _apply_completed_step_state_mask(
    state: jnp.ndarray, cache: LinearCache
) -> jnp.ndarray:
    """Apply the completed-step mask to a spectral state array."""

    mask = _completed_step_state_mask(cache).astype(state.real.dtype)[..., None]
    shape = (1,) * (state.ndim - mask.ndim) + mask.shape
    return state * jnp.reshape(mask, shape)


def _growth_rate_mode_mask(
    ky: jnp.ndarray,
    kx: jnp.ndarray,
    dealias_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return the diagnostic mask used by explicit-time growth-rate extraction.

    Single selected nonzonal ``ky`` slices should remain diagnosable even when
    the originating full nonlinear mesh would mark that representative row as
    dealiased away.
    """

    mask = jnp.asarray(dealias_mask, dtype=bool)
    ky_arr = jnp.asarray(ky, dtype=float).reshape(-1)
    if mask.ndim == 2 and int(mask.shape[0]) == 1:
        promote = (~jnp.any(mask)) & jnp.any(jnp.abs(ky_arr) > 0.0)
        mask = jnp.where(promote, jnp.ones_like(mask, dtype=bool), mask)
    ky_zero = jnp.isclose(ky_arr, 0.0)
    kx_zero = jnp.isclose(jnp.asarray(kx, dtype=float).reshape(-1), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _diagnostic_midplane_index(nz: int) -> int:
    if nz <= 1:
        return 0
    idx = nz // 2 + 1
    return min(idx, nz - 1)


def _instantaneous_growth_rate_step(
    phi_now: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float | jax.Array,
    *,
    z_index: int,
    mask: jnp.ndarray,
    mode_method: str = "z_index",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Instantaneous growth rates from phi ratios at the midplane."""

    if mode_method == "z_index":
        phi_now_z = phi_now[..., z_index]
        phi_prev_z = phi_prev[..., z_index]
    elif mode_method == "max":
        now_idx = jnp.argmax(jnp.abs(phi_now), axis=-1, keepdims=True)
        prev_idx = jnp.argmax(jnp.abs(phi_prev), axis=-1, keepdims=True)
        phi_now_z = jnp.take_along_axis(phi_now, now_idx, axis=-1)[..., 0]
        phi_prev_z = jnp.take_along_axis(phi_prev, prev_idx, axis=-1)[..., 0]
    else:
        raise ValueError("mode_method must be 'z_index' or 'max'")
    # Keep the diagnostic conservative: require non-zero real and imaginary
    # parts of phi at the current step only.
    valid = (jnp.abs(jnp.real(phi_now_z)) > 0.0) & (jnp.abs(jnp.imag(phi_now_z)) > 0.0)
    ratio = jnp.where(phi_prev_z != 0.0, phi_now_z / phi_prev_z, 0.0 + 0.0j)
    log_amp = jnp.log(jnp.abs(ratio))
    phase = jnp.angle(ratio)
    gamma = jnp.where(mask & valid, log_amp / dt, 0.0)
    omega = jnp.where(mask & valid, -phase / dt, 0.0)
    return gamma, omega


def _linear_term_config(terms: LinearTerms | None) -> TermConfig:
    return linear_terms_to_term_config(terms)


def _rk4_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
) -> tuple[jnp.ndarray, FieldState]:
    """Single Explicit RK4 step for linear dynamics."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk4")


def _rk3_heun_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
) -> tuple[jnp.ndarray, FieldState]:
    """Single Explicit RK3/Heun step for linear dynamics."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk3")


def _linear_stage_rhs(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt_val: jnp.ndarray,
    assemble_rhs_cached_fn=assemble_rhs_cached,
) -> LinearStageRhsFn:
    """Return the linear RHS closure used by explicit staged methods."""

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached_fn(
            state, cache, params, terms=term_cfg, dt=dt_val
        )
        return dG

    return rhs


def _linear_rk3_classic_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    k1: jnp.ndarray,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    G1 = G + dt_val * k1
    k2 = rhs(G1)
    G2 = 0.75 * G + 0.25 * (G1 + dt_val * k2)
    k3 = rhs(G2)
    return (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)


def _linear_rk3_heun_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    k1: jnp.ndarray,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    G1 = G + (dt_val / 3.0) * k1
    k2 = rhs(G1)
    G2 = G + (2.0 * dt_val / 3.0) * k2
    k3 = rhs(G2)
    G3 = G + 0.75 * dt_val * k3
    return G3 + 0.25 * dt_val * k1


def _linear_rk4_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    k1: jnp.ndarray,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    k2 = rhs(G + 0.5 * dt_val * k1)
    k3 = rhs(G + 0.5 * dt_val * k2)
    k4 = rhs(G + dt_val * k3)
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _linear_sspx3_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        return G_state + (_SSPX3_ADT * dt_val) * rhs(G_state)

    G1 = euler_step(G)
    G2_euler = euler_step(G1)
    G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
    G3 = euler_step(G2)
    return (
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
        + _SSPX3_W3 * G1
        + (_SSPX3_W2 - 1.0) * G2
        + G3
    )


def _linear_k10_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        return G_state + (dt_val / 6.0) * rhs(G_state)

    G_q1 = G
    G_q2 = G
    for _ in range(5):
        G_q1 = euler_step(G_q1)
    G_q2 = 0.04 * G_q2 + 0.36 * G_q1
    G_q1 = 15.0 * G_q2 - 5.0 * G_q1
    for _ in range(4):
        G_q1 = euler_step(G_q1)
    dG_final = rhs(G_q1)
    return G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final


def _linear_explicit_stage_update(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    method_key: str,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    """Advance one explicit method before post-step masking and field solve."""

    # These schemes build their own first stage; evaluating ``rhs(G)`` here would
    # add an unused full gyrokinetic RHS evaluation to every completed step.
    if method_key == "sspx3":
        return _linear_sspx3_state(G, dt_val, rhs=rhs)
    if method_key == "k10":
        return _linear_k10_state(G, dt_val, rhs=rhs)

    k1 = rhs(G)
    if method_key == "euler":
        return G + dt_val * k1
    if method_key == "rk2":
        return G + dt_val * rhs(G + 0.5 * dt_val * k1)
    if method_key == "rk3_classic":
        return _linear_rk3_classic_state(G, dt_val, k1=k1, rhs=rhs)
    if method_key in {"rk3", "rk3_heun"}:
        return _linear_rk3_heun_state(G, dt_val, k1=k1, rhs=rhs)
    if method_key == "rk4":
        return _linear_rk4_state(G, dt_val, k1=k1, rhs=rhs)
    raise ValueError(
        "explicit linear method must be one of {'euler', 'rk2', 'rk3', "
        "'rk3_classic', 'rk3_heun', 'rk4', 'k10', 'sspx3'}"
    )


def _linear_explicit_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
    *,
    method: str,
    assemble_rhs_cached_fn=assemble_rhs_cached,
) -> tuple[jnp.ndarray, FieldState]:
    """Single explicit linear step matching explicit staged schemes."""

    dt_val = jnp.asarray(dt)
    method_key = method.strip().lower()
    rhs = _linear_stage_rhs(cache, params, term_cfg, dt_val, assemble_rhs_cached_fn)
    G_next = _linear_explicit_stage_update(
        G,
        dt_val,
        method_key=method_key,
        rhs=rhs,
    )

    # Mask inactive modes only after the full explicit step, before the next field solve.
    G_next = _apply_completed_step_state_mask(jnp.asarray(G_next), cache)

    # fields at the end of step
    _, fields = assemble_rhs_cached_fn(
        G_next, cache, params, terms=term_cfg, dt=dt_val
    )
    return G_next, fields
