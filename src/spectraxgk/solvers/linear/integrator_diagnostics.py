"""Diagnostic sampling integration for linear fixed-step solves."""

from __future__ import annotations

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

    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    def advance(G_in: jnp.ndarray) -> jnp.ndarray:
        dG, _phi = linear_rhs_cached(
            G_in, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        if method == "imex":
            dG_explicit = dG + damping * G_in
            return (G_in + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G_in
            G_half = (G_in + 0.5 * dt_val * dG_explicit) / (
                1.0 + 0.5 * dt_val * damping
            )
            dG_half, _phi = linear_rhs_cached(
                G_half, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            dG_half_exp = dG_half + damping * G_half
            return (G_in + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G_in + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            return G_in + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _phi_state = linear_rhs_cached(
                    G_state,
                    cache,
                    params,
                    terms=terms,
                    use_jit=False,
                    dt=dt_val,
                )
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G_in)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G_in + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_in
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        if method == "rk4":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k3, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k2,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k4, _ = linear_rhs_cached(
                G_in + dt_val * k3, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            return G_in + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"Unsupported method '{method}'")

    def density_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        Jl = cache.Jl
        if G_in.ndim == 5:
            if Jl.ndim == 5:
                Jl_s = Jl[0]
            else:
                Jl_s = Jl
            return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
        if Jl.ndim == 5:
            if species_index is None:
                return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
            Jl_s = Jl[int(species_index)]
            return jnp.sum(Jl_s * G_in[int(species_index), :, 0, ...], axis=0)
        if species_index is None:
            return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
        return jnp.sum(Jl * G_in[int(species_index), :, 0, ...], axis=0)

    def hl_energy_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        if G_in.ndim == 5:
            return jnp.sum(jnp.abs(G_in) ** 2, axis=(2, 3, 4))
        return jnp.sum(jnp.abs(G_in) ** 2, axis=(0, 3, 4, 5))

    def step(G_in, idx):
        G_out = advance(G_in)
        _dG, phi = linear_rhs_cached(
            G_out, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        density = density_from_G(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi))
            density_max = jnp.max(jnp.abs(density))
            G_out = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
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
                G_out,
            )
        if record_hl_energy:
            hl_energy = hl_energy_from_G(G_out)
            return G_out, (phi, density, hl_energy)
        return G_out, (phi, density)

    if sample_stride <= 1:
        indices = jnp.arange(steps)
        G_out, outputs = jax.lax.scan(step, G0, indices)
    else:

        def sample_step(G_in, idx):
            def inner_step(_i, g):
                return advance(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG, phi_out = linear_rhs_cached(
                G_out_local, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            density_out = density_from_G(G_out_local)
            if show_progress:
                from spectraxgk.utils.callbacks import (
                    print_callback,
                    should_emit_progress,
                )

                completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
                sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
                sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
                phi_max = jnp.max(jnp.abs(phi_out))
                density_max = jnp.max(jnp.abs(density_out))
                G_out_local = jax.lax.cond(
                    should_emit_progress(completed_idx, steps),
                    lambda state: print_callback(
                        state,
                        completed_idx,
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
                    G_out_local,
                )
            if record_hl_energy:
                hl_out = hl_energy_from_G(G_out_local)
                return G_out_local, (phi_out, density_out, hl_out)
            return G_out_local, (phi_out, density_out)

        num_samples = steps // sample_stride
        sample_indices = jnp.arange(num_samples)
        G_out, outputs = jax.lax.scan(sample_step, G0, sample_indices)

    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t


__all__ = ["integrate_linear_diagnostics"]
