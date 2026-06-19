"""Streaming Diffrax linear growth/frequency estimators."""

from __future__ import annotations

from typing import Any, Callable, Sequence

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


def _cache_for_streaming_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    cache: LinearCache | None,
) -> LinearCache:
    """Return a linear cache compatible with a single- or multi-species state."""

    if cache is not None:
        return cache
    if G0.ndim == 5:
        Nl, Nm = G0.shape[0], G0.shape[1]
    elif G0.ndim == 6:
        Nl, Nm = G0.shape[1], G0.shape[2]
    else:
        raise ValueError(
            "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or "
            "(Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    return build_linear_cache(grid, geom, params, Nl, Nm)


def _maybe_apply_sharding(state: jnp.ndarray, state_sharding: Any | None) -> jnp.ndarray:
    """Apply the optional state sharding constraint used by streaming solves."""

    if state_sharding is None:
        return state
    return jax.lax.with_sharding_constraint(state, state_sharding)


def _streaming_mode_indices(
    grid: SpectralGrid,
    mode_ky_indices: Sequence[int] | np.ndarray | jnp.ndarray | None,
) -> jnp.ndarray:
    """Return the ky indices monitored by the streaming fit."""

    if mode_ky_indices is None:
        return jnp.arange(grid.ky.size, dtype=jnp.int32)
    ky_idx = jnp.asarray(mode_ky_indices, dtype=jnp.int32)
    if ky_idx.ndim == 0:
        return ky_idx[None]
    return ky_idx


def _streaming_mode_extractor(
    ky_idx: jnp.ndarray,
    *,
    mode_kx_index: int,
    mode_z_index: int,
    mode_method: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the field-mode extractor used by streaming growth fits."""

    if mode_method not in {"z_index", "max"}:
        raise ValueError(
            "mode_method must be one of {'z_index', 'max'} for streaming fits"
        )

    def _extract_mode(field: jnp.ndarray) -> jnp.ndarray:
        data = field[ky_idx, mode_kx_index, :]
        if mode_method == "z_index":
            return data[:, mode_z_index]
        idx = jnp.argmax(jnp.abs(data), axis=-1)
        return jnp.take_along_axis(data, idx[:, None], axis=-1)[:, 0]

    return _extract_mode


def _density_mode_from_streaming_state(
    G_in: jnp.ndarray,
    cache: LinearCache,
    *,
    ky_idx: jnp.ndarray,
    mode_kx_index: int,
    mode_z_index: int,
    mode_method: str,
    density_species_index: int | None,
    extract_mode: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Extract the monitored density mode without materializing a full time trace."""

    if mode_method != "z_index":
        field = _density_from_G_cached(G_in, cache, density_species_index)
        return extract_mode(field)
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


def _streaming_signal_and_derivative(
    *,
    G: jnp.ndarray,
    dG: jnp.ndarray,
    fields: Any,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    use_custom_vjp: bool,
    fit_signal: str,
    extract_mode: Callable[[jnp.ndarray], jnp.ndarray],
    ky_idx: jnp.ndarray,
    mode_kx_index: int,
    mode_z_index: int,
    mode_method: str,
    density_species_index: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the fitted observable and its time derivative."""

    if fit_signal == "phi":
        signal = extract_mode(fields.phi)
        dphi = compute_fields_cached(
            dG,
            cache,
            params,
            terms=term_cfg,
            use_custom_vjp=use_custom_vjp,
        ).phi
        return signal, extract_mode(dphi)
    if fit_signal == "density":
        signal = _density_mode_from_streaming_state(
            G,
            cache,
            ky_idx=ky_idx,
            mode_kx_index=mode_kx_index,
            mode_z_index=mode_z_index,
            mode_method=mode_method,
            density_species_index=density_species_index,
            extract_mode=extract_mode,
        )
        signal_dot = _density_mode_from_streaming_state(
            dG,
            cache,
            ky_idx=ky_idx,
            mode_kx_index=mode_kx_index,
            mode_z_index=mode_z_index,
            mode_method=mode_method,
            density_species_index=density_species_index,
            extract_mode=extract_mode,
        )
        return signal, signal_dot
    raise ValueError("fit_signal must be 'phi' or 'density'")


def _streaming_fit_accumulator_derivative(
    *,
    t: Any,
    signal: jnp.ndarray,
    signal_dot: jnp.ndarray,
    amp_floor_val: jnp.ndarray,
    tmin_val: jnp.ndarray,
    tmax_val: jnp.ndarray,
    real_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the instantaneous contribution to the streaming fit averages."""

    abs_signal = jnp.abs(signal)
    safe_signal = jnp.where(abs_signal > amp_floor_val, signal, jnp.ones_like(signal))
    log_deriv = jnp.where(
        abs_signal > amp_floor_val,
        signal_dot / safe_signal,
        jnp.zeros_like(signal),
    )
    window = (t >= tmin_val) & (t <= tmax_val)
    window = jnp.asarray(window, dtype=abs_signal.dtype)
    weight = jnp.asarray(window * (abs_signal > amp_floor_val), dtype=real_dtype)
    acc_re_dot = weight * jnp.asarray(jnp.real(log_deriv), dtype=real_dtype)
    acc_im_dot = weight * jnp.asarray(jnp.imag(log_deriv), dtype=real_dtype)
    return acc_re_dot, acc_im_dot, weight


def _build_streaming_rhs(
    *,
    use_custom_vjp: bool,
    fit_signal: str,
    extract_mode: Callable[[jnp.ndarray], jnp.ndarray],
    ky_idx: jnp.ndarray,
    mode_kx_index: int,
    mode_z_index: int,
    mode_method: str,
    density_species_index: int | None,
    amp_floor_val: jnp.ndarray,
    tmin_val: jnp.ndarray,
    tmax_val: jnp.ndarray,
    real_dtype: jnp.dtype,
    state_sharding: Any | None,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Build the Diffrax RHS that streams fit statistics with the state."""

    def rhs(t: Any, state: tuple[Any, ...], args: tuple[Any, ...]):
        cache_, params_, term_cfg_ = args
        G_packed, _acc_re, _acc_im, _wsum = state
        G_packed = _maybe_apply_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        dG, fields = _assemble_rhs(
            G,
            cache_,
            params_,
            term_cfg_,
            use_custom_vjp=use_custom_vjp,
        )
        dG = jnp.asarray(dG, dtype=G.dtype)
        signal, signal_dot = _streaming_signal_and_derivative(
            G=G,
            dG=dG,
            fields=fields,
            cache=cache_,
            params=params_,
            term_cfg=term_cfg_,
            use_custom_vjp=use_custom_vjp,
            fit_signal=fit_signal,
            extract_mode=extract_mode,
            ky_idx=ky_idx,
            mode_kx_index=mode_kx_index,
            mode_z_index=mode_z_index,
            mode_method=mode_method,
            density_species_index=density_species_index,
        )
        acc_re_dot, acc_im_dot, wsum_dot = _streaming_fit_accumulator_derivative(
            t=t,
            signal=signal,
            signal_dot=signal_dot,
            amp_floor_val=amp_floor_val,
            tmin_val=tmin_val,
            tmax_val=tmax_val,
            real_dtype=real_dtype,
        )
        dG_packed = jnp.asarray(_pack_complex_state(dG), dtype=G_packed.dtype)
        dG_packed = _maybe_apply_sharding(dG_packed, state_sharding)
        return (dG_packed, acc_re_dot, acc_im_dot, wsum_dot)

    return rhs


def _zero_streaming_term_rhs(_t: Any, y: tuple[Any, ...], _args: tuple[Any, ...]):
    """Return the zero implicit term used to route IMEX streaming through Diffrax."""

    return (jnp.zeros_like(y[0]),) + tuple(jnp.zeros_like(x) for x in y[1:])


def _strip_save_axis(value: Any, *, reference_ndim: int) -> Any:
    """Drop Diffrax's saved-time axis when a t1-only solve adds one."""

    if isinstance(value, jnp.ndarray) and value.ndim > reference_ndim:
        return value[0]
    return value


def _finalize_streaming_solution(
    sol: Any,
    *,
    packed_state_ndim: int,
    return_state: bool,
) -> tuple[jnp.ndarray | None, jnp.ndarray, jnp.ndarray]:
    """Convert Diffrax streaming accumulators into growth and frequency arrays."""

    G_last_packed, acc_re, acc_im, wsum = sol.ys
    G_last_packed = _strip_save_axis(G_last_packed, reference_ndim=packed_state_ndim)
    acc_re = _strip_save_axis(acc_re, reference_ndim=1)
    acc_im = _strip_save_axis(acc_im, reference_ndim=1)
    wsum = _strip_save_axis(wsum, reference_ndim=1)
    wsum_safe = jnp.where(wsum > 0.0, wsum, jnp.nan)
    gamma = acc_re / wsum_safe
    omega = -acc_im / wsum_safe
    if return_state:
        return _unpack_complex_state(G_last_packed), gamma, omega
    return None, gamma, omega


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
    cache = _cache_for_streaming_state(G0, grid, geom, params, cache)
    term_cfg = linear_terms_to_term_config(terms)
    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))

    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)

    ky_idx = _streaming_mode_indices(grid, mode_ky_indices)
    extract_mode = _streaming_mode_extractor(
        ky_idx,
        mode_kx_index=mode_kx_index,
        mode_z_index=mode_z_index,
        mode_method=mode_method,
    )
    amp_floor_val = jnp.asarray(amp_floor, dtype=real_dtype)
    tmin_val = jnp.asarray(0.0 if tmin is None else tmin, dtype=real_dtype)
    tmax_val = jnp.asarray(dt * steps if tmax is None else tmax, dtype=real_dtype)
    rhs = _build_streaming_rhs(
        use_custom_vjp=use_custom_vjp,
        fit_signal=fit_signal,
        extract_mode=extract_mode,
        ky_idx=ky_idx,
        mode_kx_index=mode_kx_index,
        mode_z_index=mode_z_index,
        mode_method=mode_method,
        density_species_index=density_species_index,
        amp_floor_val=amp_floor_val,
        tmin_val=tmin_val,
        tmax_val=tmax_val,
        real_dtype=real_dtype,
        state_sharding=state_sharding,
    )

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs)
    if _is_imex_solver(method):
        zero_term = dfx.ODETerm(_zero_streaming_term_rhs)
        terms_obj = dfx.MultiTerm(zero_term, explicit_term)
    else:
        terms_obj = explicit_term

    dt_val = jnp.asarray(dt, dtype=real_dtype)
    adaptive_eff = adaptive or _is_imex_solver(method) or _is_implicit_solver(method)

    acc0 = jnp.zeros((ky_idx.shape[0],), dtype=real_dtype)

    def solve(G0_packed_in):
        G0_packed_in = _maybe_apply_sharding(G0_packed_in, state_sharding)
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

    return _finalize_streaming_solution(
        sol,
        packed_state_ndim=G0_packed.ndim,
        return_state=return_state,
    )

__all__ = ["integrate_linear_diffrax_streaming"]
