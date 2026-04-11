"""Diffrax-based time integrators for gyrokinetic systems."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

import numpy as np

from spectraxgk.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    LinearTerms,
    build_linear_cache,
    linear_terms_to_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution

if TYPE_CHECKING:  # pragma: no cover
    import diffrax as dfx
    import equinox as eqx
else:  # pragma: no cover - optional dependency loading
    try:
        import diffrax as dfx  # type: ignore[no-redef]
        import equinox as eqx  # type: ignore[no-redef]
    except Exception:
        dfx = None  # type: ignore[assignment]
        eqx = None  # type: ignore[assignment]


def _require_diffrax() -> tuple[Any, Any]:
    if dfx is None or eqx is None:
        raise ImportError("diffrax and equinox must be installed to use diffrax integrators")
    return dfx, eqx


def _solver_from_name(name: str):
    dfx, _ = _require_diffrax()
    key = name.strip().lower()
    aliases = {
        "rk4": "tsit5",
        "rk2": "heun",
        "euler": "euler",
        "heun": "heun",
        "tsit5": "tsit5",
        "dopri5": "dopri5",
        "dopri8": "dopri8",
        "implicit": "kvaerno5",
        "imex": "kencarp4",
        "semi-implicit": "kencarp4",
    }
    key = aliases.get(key, key)
    mapping = {
        "euler": dfx.Euler,
        "heun": dfx.Heun,
        "tsit5": dfx.Tsit5,
        "dopri5": dfx.Dopri5,
        "dopri8": dfx.Dopri8,
        "impliciteuler": dfx.ImplicitEuler,
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno4": dfx.Kvaerno4,
        "kvaerno5": dfx.Kvaerno5,
        "kencarp3": dfx.KenCarp3,
        "kencarp4": dfx.KenCarp4,
        "kencarp5": dfx.KenCarp5,
    }
    if key not in mapping:
        raise ValueError(f"Unknown diffrax solver '{name}'")
    return mapping[key]()


def _is_imex_solver(name: str) -> bool:
    key = name.strip().lower()
    return key in {"kencarp3", "kencarp4", "kencarp5", "imex", "semi-implicit"}


def _is_implicit_solver(name: str) -> bool:
    key = name.strip().lower()
    return key in {"impliciteuler", "kvaerno3", "kvaerno4", "kvaerno5", "implicit"}


def _progress_meter(enabled: bool):
    dfx, _ = _require_diffrax()
    return dfx.TqdmProgressMeter() if enabled else dfx.NoProgressMeter()


def _stepsize_controller(adaptive: bool, rtol: float, atol: float):
    dfx, _ = _require_diffrax()
    if adaptive:
        return dfx.PIDController(rtol=rtol, atol=atol)
    return dfx.ConstantStepSize()


def _adjoint(checkpoint: bool):
    dfx, _ = _require_diffrax()
    if checkpoint:
        return dfx.RecursiveCheckpointAdjoint()
    return dfx.DirectAdjoint()


def _base_complex_dtype() -> jnp.dtype:
    return jnp.complex128 if bool(getattr(jax.config, "x64_enabled", False)) else jnp.complex64


def _pack_complex_state(G: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.real(G), jnp.imag(G)], axis=-1)


def _unpack_complex_state(G_packed: jnp.ndarray) -> jnp.ndarray:
    return G_packed[..., 0] + 1j * G_packed[..., 1]


def _assemble_rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    use_custom_vjp: bool,
) -> tuple[jnp.ndarray, FieldState]:
    if use_custom_vjp:
        return assemble_rhs_cached_jit(G, cache, params, term_cfg)
    return assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)


def _save_with_phi(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    use_custom_vjp: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fields = compute_fields_cached(G, cache, params, terms=term_cfg, use_custom_vjp=use_custom_vjp)
    return G, fields.phi


def _density_from_G_cached(
    G_in: jnp.ndarray,
    cache: LinearCache,
    density_species_index: int | None,
) -> jnp.ndarray:
    Jl = cache.Jl
    if G_in.ndim == 5:
        if Jl.ndim == 5:
            Jl_s = Jl[0]
        else:
            Jl_s = Jl
        return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
    if Jl.ndim == 5:
        if density_species_index is None:
            return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
        Jl_s = Jl[int(density_species_index)]
        return jnp.sum(Jl_s * G_in[int(density_species_index), :, 0, ...], axis=0)
    if density_species_index is None:
        return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
    return jnp.sum(Jl * G_in[int(density_species_index), :, 0, ...], axis=0)


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

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    G0_packed = _pack_complex_state(G0)
    if state_sharding is not None:
        G0_packed = jax.device_put(G0_packed, state_sharding)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)

    def _maybe_shard(state: jnp.ndarray) -> jnp.ndarray:
        if state_sharding is None:
            return state
        return jax.lax.with_sharding_constraint(state, state_sharding)
        G0_packed = _maybe_shard(G0_packed)
        G0_packed = _maybe_shard(G0_packed)
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
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system with diffrax (placeholder nonlinear term)."""

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
            gx_real_fft=gx_real_fft,
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
