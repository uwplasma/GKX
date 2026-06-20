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


def _infer_linear_velocity_shape(G0: jnp.ndarray) -> tuple[int, int]:
    if G0.ndim == 5:
        return int(G0.shape[0]), int(G0.shape[1])
    if G0.ndim == 6:
        return int(G0.shape[1]), int(G0.shape[2])
    raise ValueError(
        "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
    )


def _prepare_linear_state_and_cache(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    cache: LinearCache | None,
) -> tuple[jnp.ndarray, Any, LinearCache]:
    state_dtype = jnp.result_type(G0, _base_complex_dtype())
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    if cache is None:
        Nl, Nm = _infer_linear_velocity_shape(G0)
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return G0, real_dtype, cache


def _apply_state_sharding(state: jnp.ndarray, state_sharding: Any | None) -> jnp.ndarray:
    if state_sharding is None:
        return state
    return jax.lax.with_sharding_constraint(state, state_sharding)


def _prepare_packed_linear_state(
    G0: jnp.ndarray,
    state_sharding: Any | None,
) -> jnp.ndarray:
    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)
    return _apply_state_sharding(G0_packed, state_sharding)


def _extract_linear_saved_mode(
    field: jnp.ndarray,
    *,
    save_mode: ModeSelection | ModeSelectionBatch | None,
    mode_method: str,
) -> jnp.ndarray:
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


def _linear_density_field(
    G: jnp.ndarray,
    cache: LinearCache,
    density_species_index: int | None,
) -> jnp.ndarray:
    return _density_from_G_cached(G, cache, density_species_index)


def _linear_saved_field(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    *,
    save_field: str,
    density_species_index: int | None,
    use_custom_vjp: bool,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    if save_field == "phi":
        fields = compute_fields_cached(
            G, cache, params, terms=term_cfg, use_custom_vjp=use_custom_vjp
        )
        return fields.phi
    if save_field == "density":
        return _linear_density_field(G, cache, density_species_index)
    if save_field == "phi+density":
        fields = compute_fields_cached(
            G, cache, params, terms=term_cfg, use_custom_vjp=use_custom_vjp
        )
        return fields.phi, _linear_density_field(G, cache, density_species_index)
    raise ValueError("save_field must be 'phi', 'density', or 'phi+density'")


def _linear_save_payload(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    *,
    save_field: str,
    return_state: bool,
    save_mode: ModeSelection | ModeSelectionBatch | None,
    mode_method: str,
    density_species_index: int | None,
    use_custom_vjp: bool,
    state_sharding: Any | None,
) -> Any:
    if save_field == "phi+density" and save_mode is not None:
        raise ValueError("save_mode cannot be used when save_field='phi+density'")
    field = _linear_saved_field(
        G,
        cache,
        params,
        term_cfg,
        save_field=save_field,
        density_species_index=density_species_index,
        use_custom_vjp=use_custom_vjp,
    )
    saved_value: Any
    if save_mode is not None:
        if isinstance(field, tuple):
            raise ValueError("save_mode cannot be used when save_field='phi+density'")
        saved_value = _extract_linear_saved_mode(
            field,
            save_mode=save_mode,
            mode_method=mode_method,
        )
    else:
        saved_value = field
    if return_state:
        return _apply_state_sharding(_pack_complex_state(G), state_sharding), saved_value
    return saved_value


def _make_linear_diffrax_rhs(
    *,
    use_custom_vjp: bool,
    state_sharding: Any | None,
):
    def rhs(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _apply_state_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        dG, _fields = _assemble_rhs(
            G, cache_, params_, term_cfg_, use_custom_vjp=use_custom_vjp
        )
        return _apply_state_sharding(_pack_complex_state(dG), state_sharding)

    return rhs


def _make_linear_diffrax_save_fn(
    *,
    save_field: str,
    return_state: bool,
    save_mode: ModeSelection | ModeSelectionBatch | None,
    mode_method: str,
    density_species_index: int | None,
    use_custom_vjp: bool,
    state_sharding: Any | None,
):
    def save_fn(t, G_packed, args):
        cache_, params_, term_cfg_ = args
        G_packed = _apply_state_sharding(G_packed, state_sharding)
        G = _unpack_complex_state(G_packed)
        return _linear_save_payload(
            G,
            cache_,
            params_,
            term_cfg_,
            save_field=save_field,
            return_state=return_state,
            save_mode=save_mode,
            mode_method=mode_method,
            density_species_index=density_species_index,
            use_custom_vjp=use_custom_vjp,
            state_sharding=state_sharding,
        )

    return save_fn


def _linear_diffrax_terms_obj(dfx: Any, method: str, rhs: Any) -> Any:
    explicit_term = dfx.ODETerm(rhs)
    if _is_imex_solver(method):
        zero_term = dfx.ODETerm(lambda t, y, args: jnp.zeros_like(y))
        return dfx.MultiTerm(zero_term, explicit_term)
    return explicit_term


def _linear_save_times(
    *,
    dt: float,
    steps: int,
    sample_stride: int,
    real_dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    num_samples = steps // sample_stride
    ts = dt_val * sample_stride * (jnp.arange(num_samples, dtype=real_dtype) + 1)
    return dt_val, ts


def _run_linear_diffrax_solve(
    *,
    dfx: Any,
    eqx: Any,
    terms_obj: Any,
    solver: Any,
    save_fn: Any,
    G0_packed: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    dt_val: jnp.ndarray,
    steps: int,
    ts: jnp.ndarray,
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
            t0=jnp.asarray(0.0, dtype=dt_val.dtype),
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


def _linear_diffrax_output(sol: Any, *, return_state: bool) -> tuple[jnp.ndarray | None, Any]:
    if return_state:
        G_t_packed, saved = sol.ys
        return _unpack_complex_state(G_t_packed[-1]), saved
    return None, sol.ys


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
    G0, real_dtype, cache = _prepare_linear_state_and_cache(
        G0, grid, geom, params, cache
    )
    term_cfg = linear_terms_to_term_config(terms or LinearTerms())
    use_custom_vjp = not (_is_imex_solver(method) or _is_implicit_solver(method))
    G0_packed = _prepare_packed_linear_state(G0, state_sharding)
    rhs = _make_linear_diffrax_rhs(
        use_custom_vjp=use_custom_vjp,
        state_sharding=state_sharding,
    )
    save_fn = _make_linear_diffrax_save_fn(
        save_field=save_field,
        return_state=return_state,
        save_mode=save_mode,
        mode_method=mode_method,
        density_species_index=density_species_index,
        use_custom_vjp=use_custom_vjp,
        state_sharding=state_sharding,
    )
    dt_val, ts = _linear_save_times(
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        real_dtype=real_dtype,
    )
    method_is_special = _is_imex_solver(method) or _is_implicit_solver(method)
    sol = _run_linear_diffrax_solve(
        dfx=dfx,
        eqx=eqx,
        terms_obj=_linear_diffrax_terms_obj(dfx, method, rhs),
        solver=_solver_from_name(method),
        save_fn=save_fn,
        G0_packed=G0_packed,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        dt_val=dt_val,
        steps=steps,
        ts=ts,
        adaptive_eff=adaptive or method_is_special,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        checkpoint=checkpoint,
        show_progress=show_progress,
        progress_bar=progress_bar,
        jit=jit,
        state_sharding=state_sharding,
    )
    return _linear_diffrax_output(sol, return_state=return_state)


__all__ = ["integrate_linear_diffrax"]
