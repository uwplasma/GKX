"""Streaming Diffrax linear growth/frequency estimators."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.cache import LinearCache, build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms, linear_terms_to_term_config
from spectraxgk.solvers.time.diffrax_core import (
    _adjoint,
    _assemble_rhs,
    _base_complex_dtype,
    _density_from_G_cached,
    _is_imex_solver,
    _is_implicit_solver,
    _pack_complex_state,
    _progress_meter,
    _require_diffrax,
    _solver_from_name,
    _stepsize_controller,
    _unpack_complex_state,
)
from spectraxgk.terms.assembly import compute_fields_cached

def integrate_linear_diffrax_streaming(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "Dopri8",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    adaptive: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    max_steps: int = 4096,
    show_progress: bool = False,
    progress_bar: bool = False,
    checkpoint: bool = False,
    jit: bool | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    fit_signal: str = "density",
    mode_ky_indices: Sequence[int] | np.ndarray | jnp.ndarray | None = None,
    mode_kx_index: int = 0,
    mode_z_index: int = 0,
    mode_method: str = "z_index",
    amp_floor: float = 1.0e-30,
    density_species_index: int | None = None,
    return_state: bool = True,
    state_sharding: Any | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray, jnp.ndarray]:
    """Integrate the linear system and stream a growth-rate fit without storing time series."""

    dfx, eqx = _require_diffrax()
    state_dtype = jnp.result_type(G0, _base_complex_dtype())
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    if terms is None:
        terms = LinearTerms()
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    term_cfg = linear_terms_to_term_config(terms)

    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))

    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    ky_idx = jnp.arange(grid.ky.size, dtype=jnp.int32)
    if mode_ky_indices is not None:
        ky_idx = jnp.asarray(mode_ky_indices, dtype=jnp.int32)
        if ky_idx.ndim == 0:
            ky_idx = ky_idx[None]

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be one of {'z_index', 'max'} for streaming fits")

    def _extract_mode(field: jnp.ndarray) -> jnp.ndarray:
        data = field[ky_idx, mode_kx_index, :]
        if mode_method == "z_index":
            return data[:, mode_z_index]
        idx = jnp.argmax(jnp.abs(data), axis=-1)
        return jnp.take_along_axis(data, idx[:, None], axis=-1)[:, 0]

    def _density_mode_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        if mode_method != "z_index":
            field = _density_from_G_cached(G_in, cache, density_species_index)
            return _extract_mode(field)
        Jl = cache.Jl
        if G_in.ndim == 5:
            Jl_s = Jl[0] if Jl.ndim == 5 else Jl
            Gm0 = G_in[:, 0, ...]
            Gm0 = jnp.take(Gm0, ky_idx, axis=1)
            Gm0 = Gm0[..., mode_kx_index, mode_z_index]
            Jl_sel = jnp.take(Jl_s, ky_idx, axis=1)
            Jl_sel = Jl_sel[..., mode_kx_index, mode_z_index]
            return jnp.sum(Jl_sel * Gm0, axis=0)
        if Jl.ndim == 5:
            if density_species_index is None:
                Gm0 = G_in[:, :, 0, ...]
                Gm0 = jnp.take(Gm0, ky_idx, axis=2)
                Gm0 = Gm0[..., mode_kx_index, mode_z_index]
                Jl_sel = jnp.take(Jl, ky_idx, axis=2)
                Jl_sel = Jl_sel[..., mode_kx_index, mode_z_index]
                return jnp.sum(Jl_sel * Gm0, axis=1).sum(axis=0)
            species_idx = int(density_species_index)
            Gm0 = G_in[species_idx, :, 0, ...]
            Gm0 = jnp.take(Gm0, ky_idx, axis=1)
            Gm0 = Gm0[..., mode_kx_index, mode_z_index]
            Jl_sel = jnp.take(Jl[species_idx], ky_idx, axis=1)
            Jl_sel = Jl_sel[..., mode_kx_index, mode_z_index]
            return jnp.sum(Jl_sel * Gm0, axis=0)
        if density_species_index is None:
            Gm0 = G_in[:, :, 0, ...]
            Gm0 = jnp.take(Gm0, ky_idx, axis=2)
            Gm0 = Gm0[..., mode_kx_index, mode_z_index]
            Jl_sel = jnp.take(Jl, ky_idx, axis=1)
            Jl_sel = Jl_sel[..., mode_kx_index, mode_z_index]
            return jnp.sum(Jl_sel * Gm0, axis=1).sum(axis=0)
        species_idx = int(density_species_index)
        Gm0 = G_in[species_idx, :, 0, ...]
        Gm0 = jnp.take(Gm0, ky_idx, axis=1)
        Gm0 = Gm0[..., mode_kx_index, mode_z_index]
        Jl_sel = jnp.take(Jl, ky_idx, axis=1)
        Jl_sel = Jl_sel[..., mode_kx_index, mode_z_index]
        return jnp.sum(Jl_sel * Gm0, axis=0)

    amp_floor_val = jnp.asarray(amp_floor, dtype=real_dtype)
    tmin_val = jnp.asarray(0.0 if tmin is None else tmin, dtype=real_dtype)
    tmax_val = jnp.asarray(dt * steps if tmax is None else tmax, dtype=real_dtype)

    def rhs(t, state, args):
        cache_, params_, term_cfg_ = args
        G_packed, acc_re, acc_im, wsum = state
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        dG, fields = _assemble_rhs(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        dG = jnp.asarray(dG, dtype=G.dtype)

        if fit_signal == "phi":
            s = _extract_mode(fields.phi)
            dphi = compute_fields_cached(
                dG, cache_, params_, terms=term_cfg_, use_custom_vjp=use_custom_vjp
            ).phi
            s_dot = _extract_mode(dphi)
        elif fit_signal == "density":
            s = _density_mode_from_G(G)
            s_dot = _density_mode_from_G(dG)
        else:
            raise ValueError("fit_signal must be 'phi' or 'density'")

        abs_s = jnp.abs(s)
        safe_s = jnp.where(abs_s > amp_floor_val, s, jnp.ones_like(s))
        log_deriv = jnp.where(abs_s > amp_floor_val, s_dot / safe_s, jnp.zeros_like(s))
        window = (t >= tmin_val) & (t <= tmax_val)
        window = jnp.asarray(window, dtype=abs_s.dtype)
        weight = jnp.asarray(window * (abs_s > amp_floor_val), dtype=real_dtype)
        acc_re_dot = weight * jnp.asarray(jnp.real(log_deriv), dtype=real_dtype)
        acc_im_dot = weight * jnp.asarray(jnp.imag(log_deriv), dtype=real_dtype)
        wsum_dot = weight
        dG_packed = jnp.asarray(_pack_complex_state(dG), dtype=G_packed.dtype)
        dG_packed = _maybe_shard(dG_packed)
        return (dG_packed, acc_re_dot, acc_im_dot, wsum_dot)

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs)
    if _is_imex_solver(method):
        zero_term = dfx.ODETerm(lambda t, y, args: (jnp.zeros_like(y[0]),) + tuple(jnp.zeros_like(x) for x in y[1:]))
        terms_obj = dfx.MultiTerm(zero_term, explicit_term)
    else:
        terms_obj = explicit_term

    dt_val = jnp.asarray(dt, dtype=real_dtype)
    adaptive_eff = adaptive or _is_imex_solver(method) or _is_implicit_solver(method)

    acc0 = jnp.zeros((ky_idx.shape[0],), dtype=real_dtype)

    def solve(G0_packed_in):
        G0_packed_in = _maybe_shard(G0_packed_in)
        max_steps_eff = max(int(max_steps), int(steps))
        return dfx.diffeqsolve(
            terms_obj,
            solver,
            t0=jnp.asarray(0.0, dtype=real_dtype),
            t1=dt_val * steps,
            dt0=dt_val,
            y0=(G0_packed_in, acc0, acc0, acc0),
            args=(cache, params, term_cfg),
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=_stepsize_controller(adaptive_eff, rtol, atol),
            adjoint=_adjoint(checkpoint),
            max_steps=max_steps_eff,
            throw=state_sharding is None,
            progress_meter=_progress_meter(show_progress or progress_bar),
        )

    if jit is None:
        jit = not (show_progress or progress_bar)
    if jit:
        solve_jit = eqx.filter_jit(solve, donate="all")
        sol = solve_jit(G0_packed)
    else:
        sol = solve(G0_packed)

    (G_last_packed, acc_re, acc_im, wsum) = sol.ys
    if isinstance(G_last_packed, jnp.ndarray) and G_last_packed.ndim > G0_packed.ndim:
        G_last_packed = G_last_packed[0]
    if isinstance(acc_re, jnp.ndarray) and acc_re.ndim > 1:
        acc_re = acc_re[0]
    if isinstance(acc_im, jnp.ndarray) and acc_im.ndim > 1:
        acc_im = acc_im[0]
    if isinstance(wsum, jnp.ndarray) and wsum.ndim > 1:
        wsum = wsum[0]
    wsum_safe = jnp.where(wsum > 0.0, wsum, jnp.nan)
    gamma = acc_re / wsum_safe
    omega = -acc_im / wsum_safe
    if return_state:
        G_last = _unpack_complex_state(G_last_packed)
    else:
        G_last = None
    return G_last, gamma, omega

__all__ = ["integrate_linear_diffrax_streaming"]
