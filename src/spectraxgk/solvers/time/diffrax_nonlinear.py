"""Diffrax nonlinear time integration paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.cache import LinearCache, build_linear_cache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.solvers.time.diffrax_core import (
    _adjoint,
    _assemble_rhs,
    _base_complex_dtype,
    _is_imex_solver,
    _is_implicit_solver,
    _pack_complex_state,
    _progress_meter,
    _require_diffrax,
    _save_with_phi,
    _solver_from_name,
    _stepsize_controller,
    _unpack_complex_state,
)
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution


@dataclass(frozen=True)
class _NonlinearDiffraxSetup:
    dfx: Any
    eqx: Any
    G0_packed: jnp.ndarray
    cache: LinearCache
    term_cfg: TermConfig
    state_dtype: Any
    rhs_linear: Any
    rhs_nonlinear: Any
    dt_val: jnp.ndarray
    ts: jnp.ndarray
    real_dtype: Any
    adaptive_eff: bool
    use_custom_vjp: bool


def _infer_nonlinear_velocity_shape(G0: jnp.ndarray) -> tuple[int, int]:
    if G0.ndim == 5:
        return int(G0.shape[0]), int(G0.shape[1])
    if G0.ndim == 6:
        return int(G0.shape[1]), int(G0.shape[2])
    raise ValueError(
        "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
    )


def _prepare_nonlinear_state_and_cache(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    cache: LinearCache | None,
) -> tuple[jnp.ndarray, Any, LinearCache]:
    state_dtype = jnp.result_type(G0, _base_complex_dtype())
    G0 = jnp.asarray(G0, dtype=state_dtype)
    if cache is None:
        Nl, Nm = _infer_nonlinear_velocity_shape(G0)
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return G0, state_dtype, cache


def _apply_state_sharding(state: jnp.ndarray, state_sharding: Any | None) -> jnp.ndarray:
    if state_sharding is None:
        return state
    return jax.lax.with_sharding_constraint(state, state_sharding)


def _prepare_packed_nonlinear_state(
    G0: jnp.ndarray,
    state_sharding: Any | None,
) -> jnp.ndarray:
    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)
    return _apply_state_sharding(G0_packed, state_sharding)


def _pack_nonlinear_rhs(
    dG: jnp.ndarray,
    *,
    G: jnp.ndarray,
    G_packed: jnp.ndarray,
    state_sharding: Any | None,
) -> jnp.ndarray:
    dG = jnp.asarray(dG, dtype=G.dtype)
    packed = jnp.asarray(_pack_complex_state(dG), dtype=G_packed.dtype)
    return _apply_state_sharding(packed, state_sharding)


def _nonlinear_em_rhs_array(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    use_custom_vjp: bool,
    compressed_real_fft: bool,
    laguerre_mode: str,
) -> jnp.ndarray:
    fields = compute_fields_cached(
        G,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=use_custom_vjp,
    )
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    weight = jnp.asarray(term_cfg.nonlinear, dtype=real_dtype)
    return nonlinear_em_contribution(
        G,
        phi=fields.phi,
        apar=fields.apar,
        bpar=fields.bpar,
        Jl=cache.Jl,
        JlB=cache.JlB,
        tz=jnp.asarray(params.tz, dtype=real_dtype),
        vth=jnp.asarray(params.vth, dtype=real_dtype),
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        kx_grid=cache.kx_grid,
        ky_grid=cache.ky_grid,
        dealias_mask=cache.dealias_mask,
        kxfac=cache.kxfac,
        weight=weight,
        apar_weight=float(term_cfg.apar),
        bpar_weight=float(term_cfg.bpar),
        laguerre_to_grid=cache.laguerre_to_grid,
        laguerre_to_spectral=cache.laguerre_to_spectral,
        laguerre_roots=cache.laguerre_roots,
        laguerre_j0=cache.laguerre_j0,
        laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
        b=cache.b,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )


def _make_nonlinear_linear_rhs(
    *,
    use_custom_vjp: bool,
    state_sharding: Any | None,
):
    def rhs_linear(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _apply_state_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        dG, _fields = _assemble_rhs(
            G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp
        )
        return _pack_nonlinear_rhs(
            dG,
            G=G,
            G_packed=G_packed,
            state_sharding=state_sharding,
        )

    return rhs_linear


def _make_nonlinear_explicit_rhs(
    *,
    use_custom_vjp: bool,
    state_sharding: Any | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
):
    def rhs_nonlinear(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        if term_cfg_.nonlinear == 0.0:
            return jnp.zeros_like(G_packed)
        G_packed = _apply_state_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        dG = _nonlinear_em_rhs_array(
            G,
            cache_,
            params_,
            term_cfg_,
            use_custom_vjp=use_custom_vjp,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
        return _pack_nonlinear_rhs(
            dG,
            G=G,
            G_packed=G_packed,
            state_sharding=state_sharding,
        )

    return rhs_nonlinear


def _make_full_nonlinear_rhs(rhs_linear: Any, rhs_nonlinear: Any):
    def rhs_full(t, G_packed, args):
        return rhs_linear(t, G_packed, args) + rhs_nonlinear(t, G_packed, args)

    return rhs_full


def _make_nonlinear_save_fn(
    *,
    use_custom_vjp: bool,
    state_sharding: Any | None,
    state_dtype: Any,
):
    def save_fn(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _apply_state_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        G_out, phi = _save_with_phi(
            G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp
        )
        G_out = jnp.asarray(G_out, dtype=state_dtype)
        phi = jnp.asarray(phi, dtype=state_dtype)
        packed = jnp.asarray(_pack_complex_state(G_out), dtype=G_packed.dtype)
        return _apply_state_sharding(packed, state_sharding), phi

    return save_fn


def _nonlinear_diffrax_terms_obj(
    dfx: Any,
    *,
    method: str,
    rhs_linear: Any,
    rhs_nonlinear: Any,
    rhs_full: Any,
) -> Any:
    explicit_term = dfx.ODETerm(rhs_nonlinear if _is_imex_solver(method) else rhs_full)
    implicit_term = dfx.ODETerm(rhs_linear)
    if _is_imex_solver(method):
        return dfx.MultiTerm(explicit_term, implicit_term)
    return explicit_term


def _nonlinear_save_times(
    *,
    dt: float,
    steps: int,
    state_dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, Any]:
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    ts = dt_val * (jnp.arange(steps, dtype=real_dtype) + 1)
    return dt_val, ts, real_dtype


def _run_nonlinear_diffrax_solve(
    *,
    dfx: Any,
    eqx: Any,
    terms_obj: Any,
    solver: Any,
    save_fn: Any,
    G0_packed: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt_val: jnp.ndarray,
    steps: int,
    ts: jnp.ndarray,
    real_dtype: Any,
    adaptive_eff: bool,
    rtol: float,
    atol: float,
    max_steps: int,
    checkpoint: bool,
    show_progress: bool,
    progress_bar: bool,
    jit: bool | None,
    state_sharding: Any | None,
) -> Any:
    def solve(G0_packed_in):
        G0_packed_in = _apply_state_sharding(G0_packed_in, state_sharding)
        max_steps_eff = max(int(max_steps), int(steps))
        return dfx.diffeqsolve(
            terms_obj,
            solver,
            t0=jnp.asarray(0.0, dtype=real_dtype),
            t1=dt_val * steps,
            dt0=dt_val,
            y0=G0_packed_in,
            args=(cache, params, term_cfg),
            saveat=dfx.SaveAt(ts=ts, fn=save_fn),
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
        return solve_jit(G0_packed)
    return solve(G0_packed)


def _nonlinear_diffrax_output(sol: Any) -> tuple[jnp.ndarray, FieldState]:
    G_t_packed, phi_t = sol.ys
    G_last = _unpack_complex_state(G_t_packed[-1])
    return G_last, FieldState(phi=phi_t)


def _prepare_nonlinear_diffrax_setup(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str,
    cache: LinearCache | None,
    terms: TermConfig | None,
    adaptive: bool,
    state_sharding: Any | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
) -> _NonlinearDiffraxSetup:
    dfx, eqx = _require_diffrax()
    G0, state_dtype, cache = _prepare_nonlinear_state_and_cache(
        G0, grid, geom, params, cache
    )
    term_cfg = terms or TermConfig()
    method_is_special = _is_imex_solver(method) or _is_implicit_solver(method)
    use_custom_vjp = not method_is_special
    rhs_linear = _make_nonlinear_linear_rhs(
        use_custom_vjp=use_custom_vjp,
        state_sharding=state_sharding,
    )
    rhs_nonlinear = _make_nonlinear_explicit_rhs(
        use_custom_vjp=use_custom_vjp,
        state_sharding=state_sharding,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )
    dt_val, ts, real_dtype = _nonlinear_save_times(
        dt=dt,
        steps=steps,
        state_dtype=state_dtype,
    )
    return _NonlinearDiffraxSetup(
        dfx=dfx,
        eqx=eqx,
        G0_packed=_prepare_packed_nonlinear_state(G0, state_sharding),
        cache=cache,
        term_cfg=term_cfg,
        state_dtype=state_dtype,
        rhs_linear=rhs_linear,
        rhs_nonlinear=rhs_nonlinear,
        dt_val=dt_val,
        ts=ts,
        real_dtype=real_dtype,
        adaptive_eff=adaptive or method_is_special,
        use_custom_vjp=use_custom_vjp,
    )


def integrate_nonlinear_diffrax(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "KenCarp4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    adaptive: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    max_steps: int = 4096,
    show_progress: bool = False,
    progress_bar: bool = False,
    checkpoint: bool = False,
    jit: bool | None = None,
    state_sharding: Any | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system with diffrax."""

    setup = _prepare_nonlinear_diffrax_setup(
        G0,
        grid,
        geom,
        params,
        dt=dt,
        steps=steps,
        method=method,
        cache=cache,
        terms=terms,
        adaptive=adaptive,
        state_sharding=state_sharding,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )
    sol = _run_nonlinear_diffrax_solve(
        dfx=setup.dfx,
        eqx=setup.eqx,
        terms_obj=_nonlinear_diffrax_terms_obj(
            setup.dfx,
            method=method,
            rhs_linear=setup.rhs_linear,
            rhs_nonlinear=setup.rhs_nonlinear,
            rhs_full=_make_full_nonlinear_rhs(setup.rhs_linear, setup.rhs_nonlinear),
        ),
        solver=_solver_from_name(method),
        save_fn=_make_nonlinear_save_fn(
            use_custom_vjp=setup.use_custom_vjp,
            state_sharding=state_sharding,
            state_dtype=setup.state_dtype,
        ),
        G0_packed=setup.G0_packed,
        cache=setup.cache,
        params=params,
        term_cfg=setup.term_cfg,
        dt_val=setup.dt_val,
        steps=steps,
        ts=setup.ts,
        real_dtype=setup.real_dtype,
        adaptive_eff=setup.adaptive_eff,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        checkpoint=checkpoint,
        show_progress=show_progress,
        progress_bar=progress_bar,
        jit=jit,
        state_sharding=state_sharding,
    )
    return _nonlinear_diffrax_output(sol)


__all__ = ["integrate_nonlinear_diffrax"]
