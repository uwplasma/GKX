"""Diagnostic sampling integration for linear fixed-step solves."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.cache import (
    LinearCache,
    build_linear_cache,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import LinearParams, LinearTerms, _x64_enabled
from spectraxgk.operators.linear.rhs import linear_rhs_cached

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _validate_sampling(steps: int, sample_stride: int) -> None:
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")


def _resolve_cache(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    cache: LinearCache | None,
) -> LinearCache:
    if cache is not None:
        return cache
    if G0.ndim == 5:
        Nl, Nm = G0.shape[0], G0.shape[1]
    elif G0.ndim == 6:
        Nl, Nm = G0.shape[1], G0.shape[2]
    else:
        raise ValueError(
            "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    return build_linear_cache(grid, geom, params, Nl, Nm)


def _initial_state(G0: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    return G, real_dtype


def _linear_damping(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    real_dtype: Any,
) -> jnp.ndarray:
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G.ndim == 5))
        + hyper_damp
    )
    return damping.astype(real_dtype)


def _rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        use_jit=False,
        dt=dt_val,
    )


def _sspx3_step(
    G_in: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        dG_state, _phi_state = _rhs(G_state, cache, params, terms, dt_val)
        return G_state + (_SSPX3_ADT * dt_val) * dG_state

    G1 = euler_step(G_in)
    G2_euler = euler_step(G1)
    G2 = (1.0 - _SSPX3_W1) * G_in + (_SSPX3_W1 - 1.0) * G1 + G2_euler
    G3 = euler_step(G2)
    return (
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_in
        + _SSPX3_W3 * G1
        + (_SSPX3_W2 - 1.0) * G2
        + G3
    )


def _advance_linear_state(
    G_in: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    method: str,
    dt_val: jnp.ndarray,
    damping: jnp.ndarray,
) -> jnp.ndarray:
    dG, _phi = _rhs(G_in, cache, params, terms, dt_val)
    if method == "imex":
        dG_explicit = dG + damping * G_in
        return (G_in + dt_val * dG_explicit) / (1.0 + dt_val * damping)
    if method == "imex2":
        dG_explicit = dG + damping * G_in
        G_half = (G_in + 0.5 * dt_val * dG_explicit) / (
            1.0 + 0.5 * dt_val * damping
        )
        dG_half, _phi = _rhs(G_half, cache, params, terms, dt_val)
        dG_half_exp = dG_half + damping * G_half
        return (G_in + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
    if method == "euler":
        return G_in + dt_val * dG
    if method == "rk2":
        k2, _ = _rhs(G_in + 0.5 * dt_val * dG, cache, params, terms, dt_val)
        return G_in + dt_val * k2
    if method == "sspx3":
        return _sspx3_step(G_in, cache, params, terms, dt_val)
    if method == "rk4":
        k1 = dG
        k2, _ = _rhs(G_in + 0.5 * dt_val * k1, cache, params, terms, dt_val)
        k3, _ = _rhs(G_in + 0.5 * dt_val * k2, cache, params, terms, dt_val)
        k4, _ = _rhs(G_in + dt_val * k3, cache, params, terms, dt_val)
        return G_in + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"Unsupported method '{method}'")


def _density_from_state(
    G: jnp.ndarray,
    cache: LinearCache,
    species_index: int | None,
) -> jnp.ndarray:
    Jl = cache.Jl
    if G.ndim == 5:
        Jl_s = Jl[0] if Jl.ndim == 5 else Jl
        return jnp.sum(Jl_s * G[:, 0, ...], axis=0)
    if Jl.ndim == 5:
        if species_index is None:
            return jnp.sum(jnp.sum(Jl * G[:, :, 0, ...], axis=1), axis=0)
        Jl_s = Jl[int(species_index)]
        return jnp.sum(Jl_s * G[int(species_index), :, 0, ...], axis=0)
    if species_index is None:
        return jnp.sum(jnp.sum(Jl[None, ...] * G[:, :, 0, ...], axis=1), axis=0)
    return jnp.sum(Jl * G[int(species_index), :, 0, ...], axis=0)


def _hl_energy_from_state(G: jnp.ndarray) -> jnp.ndarray:
    if G.ndim == 5:
        return jnp.sum(jnp.abs(G) ** 2, axis=(2, 3, 4))
    return jnp.sum(jnp.abs(G) ** 2, axis=(0, 3, 4, 5))


def _maybe_emit_progress(
    G: jnp.ndarray,
    idx: jnp.ndarray,
    steps: int,
    dt_val: jnp.ndarray,
    phi: jnp.ndarray,
    density: jnp.ndarray,
    *,
    show_progress: bool,
    step_multiplier: int = 1,
) -> jnp.ndarray:
    if not show_progress:
        return G
    from spectraxgk.utils.callbacks import print_callback, should_emit_progress

    completed_step = jnp.minimum((idx + 1) * step_multiplier, steps) - 1
    sim_time = jnp.minimum((idx + 1) * step_multiplier, steps) * dt_val
    sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
    phi_max = jnp.max(jnp.abs(phi))
    density_max = jnp.max(jnp.abs(density))
    return jax.lax.cond(
        should_emit_progress(completed_step, steps),
        lambda state: print_callback(
            state,
            completed_step,
            steps,
            0.0,
            0.0,
            phi_max,
            density_max,
            sim_time,
            sim_total,
            metric_labels=("|phi|_max", "|n|_max"),
        ),
        lambda state: state,
        G,
    )


def _diagnostic_sample(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    species_index: int | None,
    *,
    record_hl_energy: bool,
) -> tuple[jnp.ndarray, ...]:
    _dG, phi = _rhs(G, cache, params, terms, dt_val)
    density = _density_from_state(G, cache, species_index)
    if record_hl_energy:
        return phi, density, _hl_energy_from_state(G)
    return phi, density


def _every_step_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    method: str,
    damping: jnp.ndarray,
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    def step(G_in: jnp.ndarray, idx: jnp.ndarray):
        G_out = _advance_linear_state(
            G_in,
            cache,
            params,
            terms,
            method=method,
            dt_val=dt_val,
            damping=damping,
        )
        outputs = _diagnostic_sample(
            G_out,
            cache,
            params,
            terms,
            dt_val,
            species_index,
            record_hl_energy=record_hl_energy,
        )
        G_out = _maybe_emit_progress(
            G_out,
            idx,
            steps,
            dt_val,
            outputs[0],
            outputs[1],
            show_progress=show_progress,
        )
        return G_out, outputs

    return jax.lax.scan(step, G0, jnp.arange(steps))


def _strided_sample_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    sample_stride: int,
    method: str,
    damping: jnp.ndarray,
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    def sample_step(G_in: jnp.ndarray, idx: jnp.ndarray):
        def inner_step(_i: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
            return _advance_linear_state(
                g,
                cache,
                params,
                terms,
                method=method,
                dt_val=dt_val,
                damping=damping,
            )

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
        outputs = _diagnostic_sample(
            G_out,
            cache,
            params,
            terms,
            dt_val,
            species_index,
            record_hl_energy=record_hl_energy,
        )
        G_out = _maybe_emit_progress(
            G_out,
            idx,
            steps,
            dt_val,
            outputs[0],
            outputs[1],
            show_progress=show_progress,
            step_multiplier=sample_stride,
        )
        return G_out, outputs

    num_samples = steps // sample_stride
    return jax.lax.scan(sample_step, G0, jnp.arange(num_samples))


def integrate_linear_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    species_index: int | None = 0,
    record_hl_energy: bool = False,
    show_progress: bool = False,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    terms_use = terms or LinearTerms()
    _validate_sampling(steps, sample_stride)
    cache_use = _resolve_cache(G0, grid, geom, params, cache)
    G, real_dtype = _initial_state(G0)
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    damping = _linear_damping(G, cache_use, params, real_dtype)
    if sample_stride <= 1:
        G_out, outputs = _every_step_scan(
            G,
            cache_use,
            params,
            terms_use,
            dt_val=dt_val,
            steps=steps,
            method=method,
            damping=damping,
            species_index=species_index,
            record_hl_energy=record_hl_energy,
            show_progress=show_progress,
        )
    else:
        G_out, outputs = _strided_sample_scan(
            G,
            cache_use,
            params,
            terms_use,
            dt_val=dt_val,
            steps=steps,
            sample_stride=sample_stride,
            method=method,
            damping=damping,
            species_index=species_index,
            record_hl_energy=record_hl_energy,
            show_progress=show_progress,
        )
    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t


__all__ = ["integrate_linear_diagnostics"]
