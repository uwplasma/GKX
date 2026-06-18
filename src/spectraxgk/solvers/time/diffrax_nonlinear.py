"""Diffrax nonlinear time integration paths."""

from __future__ import annotations

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

    dfx, eqx = _require_diffrax()
    state_dtype = jnp.result_type(G0, _base_complex_dtype())
    G0 = jnp.asarray(G0, dtype=state_dtype)
    term_cfg = terms or TermConfig()
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))

    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def rhs_linear(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        dG, _fields = _assemble_rhs(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        dG = jnp.asarray(dG, dtype=G.dtype)
        return _maybe_shard(jnp.asarray(_pack_complex_state(dG), dtype=G_packed.dtype))

    def rhs_nonlinear(t, G_packed, args):
        _cache, _params, term_cfg_ = args
        if term_cfg_.nonlinear == 0.0:
            return jnp.zeros_like(G_packed)
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        fields = compute_fields_cached(G, _cache, _params, terms=term_cfg_, use_custom_vjp=use_custom_vjp)
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        weight = jnp.asarray(term_cfg_.nonlinear, dtype=real_dtype)
        dG = nonlinear_em_contribution(
            G,
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=_cache.Jl,
            JlB=_cache.JlB,
            tz=_params.tz,
            vth=_params.vth,
            sqrt_m=_cache.sqrt_m,
            sqrt_m_p1=_cache.sqrt_m_p1,
            kx_grid=_cache.kx_grid,
            ky_grid=_cache.ky_grid,
            dealias_mask=_cache.dealias_mask,
            kxfac=_cache.kxfac,
            weight=weight,
            apar_weight=float(term_cfg_.apar),
            bpar_weight=float(term_cfg_.bpar),
            laguerre_to_grid=_cache.laguerre_to_grid,
            laguerre_to_spectral=_cache.laguerre_to_spectral,
            laguerre_roots=_cache.laguerre_roots,
            laguerre_j0=_cache.laguerre_j0,
            laguerre_j1_over_alpha=_cache.laguerre_j1_over_alpha,
            b=_cache.b,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
        dG = jnp.asarray(dG, dtype=G.dtype)
        return _maybe_shard(jnp.asarray(_pack_complex_state(dG), dtype=G_packed.dtype))

    def rhs_full(t, G_packed, args):
        return rhs_linear(t, G_packed, args) + rhs_nonlinear(t, G_packed, args)

    def save_fn(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        G_out, phi = _save_with_phi(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        G_out = jnp.asarray(G_out, dtype=state_dtype)
        phi = jnp.asarray(phi, dtype=state_dtype)
        return _maybe_shard(jnp.asarray(_pack_complex_state(G_out), dtype=G_packed.dtype)), phi

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs_nonlinear if _is_imex_solver(method) else rhs_full)
    implicit_term = dfx.ODETerm(rhs_linear)
    if _is_imex_solver(method):
        terms_obj = dfx.MultiTerm(explicit_term, implicit_term)
    else:
        terms_obj = explicit_term

    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    ts = dt_val * (jnp.arange(steps, dtype=real_dtype) + 1)

    adaptive_eff = adaptive or _is_imex_solver(method) or _is_implicit_solver(method)

    def solve(G0_packed_in):
        G0_packed_in = _maybe_shard(G0_packed_in)
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
        sol = solve_jit(G0_packed)
    else:
        sol = solve(G0_packed)
    G_t_packed, phi_t = sol.ys
    G_last = _unpack_complex_state(G_t_packed[-1])
    return G_last, FieldState(phi=phi_t)

__all__ = ["integrate_nonlinear_diffrax"]
