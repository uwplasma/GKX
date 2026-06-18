"""Diffrax linear time integration with saved fields and mode traces."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.diagnostics.modes import ModeSelection, ModeSelectionBatch
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

def integrate_linear_diffrax(
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
    sample_stride: int = 1,
    return_state: bool = True,
    save_mode: ModeSelection | ModeSelectionBatch | None = None,
    mode_method: str = "z_index",
    save_field: str = "phi",
    density_species_index: int | None = None,
    state_sharding: Any | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Integrate the linear system with diffrax."""

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

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)
        G0_packed = _maybe_shard(G0_packed)

    def rhs(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        dG, _fields = _assemble_rhs(G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp)
        return _maybe_shard(_pack_complex_state(dG))

    def _extract_mode(field: jnp.ndarray) -> jnp.ndarray:
        if save_mode is None:
            raise ValueError("save_mode must be provided when extracting modes")
        if isinstance(save_mode, ModeSelectionBatch):
            ky_idx = jnp.asarray(save_mode.ky_indices, dtype=jnp.int32)
            data = field[ky_idx, save_mode.kx_index, :]
            if mode_method == "z_index":
                return data[:, save_mode.z_index]
            if mode_method == "max":
                idx = jnp.argmax(jnp.abs(data), axis=-1)
                return jnp.take_along_axis(data, idx[:, None], axis=-1)[:, 0]
            raise ValueError(
                "mode_method must be one of {'z_index', 'max'} when save_mode is set"
            )
        data = field[save_mode.ky_index, save_mode.kx_index, :]
        if mode_method == "z_index":
            return data[save_mode.z_index]
        if mode_method == "max":
            idx = jnp.argmax(jnp.abs(data))
            return data[idx]
        raise ValueError("mode_method must be one of {'z_index', 'max'} when save_mode is set")

    def _density_from_G_local(G_in: jnp.ndarray, cache_: LinearCache) -> jnp.ndarray:
        return _density_from_G_cached(G_in, cache_, density_species_index)

    def save_fn(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _maybe_shard(G_packed)
        G = _unpack_complex_state(G_packed)
        if save_field == "phi":
            fields = compute_fields_cached(
                G, cache_, params_, terms=term_cfg_, use_custom_vjp=use_custom_vjp
            )
            field = fields.phi
        elif save_field == "density":
            field = _density_from_G_local(G, cache_)
        elif save_field == "phi+density":
            if save_mode is not None:
                raise ValueError("save_mode cannot be used when save_field='phi+density'")
            fields = compute_fields_cached(
                G, cache_, params_, terms=term_cfg_, use_custom_vjp=use_custom_vjp
            )
            phi_field = fields.phi
            density_field = _density_from_G_local(G, cache_)
            if return_state:
                return _maybe_shard(_pack_complex_state(G)), (phi_field, density_field)
            return (phi_field, density_field)
        else:
            raise ValueError("save_field must be 'phi', 'density', or 'phi+density'")

        if save_mode is not None:
            mode_val = _extract_mode(field)
            if return_state:
                return _maybe_shard(_pack_complex_state(G)), mode_val
            return mode_val
        if return_state:
            return _maybe_shard(_pack_complex_state(G)), field
        return field

    solver = _solver_from_name(method)
    explicit_term = dfx.ODETerm(rhs)
    if _is_imex_solver(method):
        zero_term = dfx.ODETerm(lambda t, y, args: jnp.zeros_like(y))
        terms_obj = dfx.MultiTerm(zero_term, explicit_term)
    else:
        terms_obj = explicit_term

    dt_val = jnp.asarray(dt, dtype=real_dtype)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    num_samples = steps // sample_stride
    ts = dt_val * sample_stride * (jnp.arange(num_samples, dtype=real_dtype) + 1)

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
    if return_state:
        G_t_packed, saved = sol.ys
        G_last = _unpack_complex_state(G_t_packed[-1])
    else:
        saved = sol.ys
        G_last = None
    return G_last, saved

__all__ = ["integrate_linear_diffrax"]
