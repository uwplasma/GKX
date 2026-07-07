"""Solver-path policies for the Cyclone single-mode linear benchmark."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
)
from spectraxgk.validation.benchmarks.defaults import CYCLONE_KRYLOV_DEFAULT
from spectraxgk.diagnostics.growth_rates import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.defaults import _midplane_index
from spectraxgk.solvers.time.explicit import ExplicitTimeConfig, integrate_linear_explicit
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import linear_terms_to_term_config
from spectraxgk.solvers.linear.krylov import dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


@dataclass(frozen=True)
class _CycloneTimeFitOptions:
    """Private growth-window policy for Cyclone saved-time fits."""

    fit_key: str
    mode_method: str
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_fraction: float
    end_fraction: float
    max_amp_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str


@dataclass(frozen=True)
class _CycloneKrylovSeed:
    """Frequency seed extracted before the Cyclone Krylov solve."""

    gamma: float = 0.0
    omega: float = 0.0
    seed_ok: bool = False
    omega_ok: bool = False


@dataclass(frozen=True)
class _CycloneTimeTrace:
    """Saved field history and sampling stride for a Cyclone time run."""

    phi_t: Any
    density_t: Any | None
    stride: int


@dataclass(frozen=True)
class _CycloneTimePathControls:
    """Resolved runtime and fit controls for one Cyclone time path."""

    time_cfg: Any | None
    fit_options: _CycloneTimeFitOptions


@dataclass(frozen=True)
class _CycloneTimePathRequest:
    """Public Cyclone time-path inputs packed for private solver routing."""

    grid: Any
    cache: Any
    params: Any
    geom: Any
    terms: Any
    cfg: Any
    time_cfg: Any
    sel: Any
    dt: float
    steps: int
    method: str
    sample_stride: int | None
    fit_key: str
    need_density: bool
    reference_aligned: bool
    use_jit: bool
    diagnostic_norm: str
    show_progress: bool
    status: Callable[[str], None]
    fresh_G0: Callable[[], jnp.ndarray]
    mode_method: str
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_fraction: float
    end_fraction: float
    max_amp_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str


def _cyclone_time_path_request_from_locals(values: dict[str, Any]) -> _CycloneTimePathRequest:
    return _CycloneTimePathRequest(
        **{field.name: values[field.name] for field in fields(_CycloneTimePathRequest)}
    )


_PATCHABLE_NAMES = (
    "ModeSelection",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "instantaneous_growth_rate_from_phi",
    "CYCLONE_KRYLOV_DEFAULT",
    "_normalize_growth_rate",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_build_initial_condition",
    "_midplane_index",
    "ExplicitTimeConfig",
    "integrate_linear_explicit",
    "integrate_linear",
    "integrate_linear_diagnostics",
    "build_linear_cache",
    "linear_terms_to_term_config",
    "dominant_eigenpair",
    "integrate_linear_from_config",
    "compute_fields_cached",
)


def sync_path_hooks(source: dict[str, Any]) -> None:
    """Mirror the Cyclone linear owner module's patchable hooks into this module."""

    for name in _PATCHABLE_NAMES:
        if name in source:
            globals()[name] = source[name]


def _fit_cyclone_explicit_seed(
    *,
    state: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
) -> _CycloneKrylovSeed:
    """Estimate a Cyclone growth/frequency seed with a short explicit march."""

    t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
    time_cfg = ExplicitTimeConfig(
        dt=float(kcfg.power_dt),
        t_max=t_seed,
        sample_stride=1,
        fixed_dt=True,
    )
    t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
        state,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    gamma_seed, omega_seed, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
        phi_t,
        t_short,
        selection,
        navg_fraction=0.5,
        mode_method="z_index",
    )
    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
    seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    return _CycloneKrylovSeed(
        gamma=float(gamma_seed),
        omega=float(omega_seed),
        seed_ok=bool(seed_ok),
        omega_ok=bool(omega_ok),
    )


def _estimate_cyclone_primary_seed(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneKrylovSeed:
    """Try the full-resolution explicit seed and preserve silent fallback."""

    try:
        return _fit_cyclone_explicit_seed(
            state=fresh_G0(),
            grid=grid,
            cache=cache,
            params=params,
            geom=geom,
            terms=terms,
            kcfg=kcfg,
            selection=selection,
            show_progress=show_progress,
        )
    except Exception:
        return _CycloneKrylovSeed()


def _estimate_cyclone_reduced_seed(
    *,
    grid: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    kcfg: Any,
    show_progress: bool,
) -> _CycloneKrylovSeed:
    """Try the reduced Hermite-Laguerre explicit seed and preserve fallback."""

    try:
        Nl_seed = min(Nl, 16)
        Nm_seed = min(Nm, 12)
        cache_seed = build_linear_cache(grid, geom, params, Nl_seed, Nm_seed)
        G0_seed = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl_seed,
            Nm=Nm_seed,
            init_cfg=init_cfg,
        )
        selection = ModeSelection(
            ky_index=0,
            kx_index=0,
            z_index=_midplane_index(grid),
        )
        return _fit_cyclone_explicit_seed(
            state=jnp.asarray(np.asarray(G0_seed)),
            grid=grid,
            cache=cache_seed,
            params=params,
            geom=geom,
            terms=terms,
            kcfg=kcfg,
            selection=selection,
            show_progress=show_progress,
        )
    except Exception:
        return _CycloneKrylovSeed()


def _estimate_cyclone_krylov_seed(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneKrylovSeed:
    """Run the Cyclone primary then reduced seed ladder when no shift is given."""

    if kcfg.shift is not None:
        return _CycloneKrylovSeed()
    status("estimating frequency seed with short explicit time march")
    seed = _estimate_cyclone_primary_seed(
        grid=grid,
        cache=cache,
        params=params,
        geom=geom,
        terms=terms,
        kcfg=kcfg,
        selection=selection,
        show_progress=show_progress,
        fresh_G0=fresh_G0,
    )
    if seed.seed_ok:
        return seed
    status("primary seed failed; retrying reduced Hermite-Laguerre seed")
    return _estimate_cyclone_reduced_seed(
        grid=grid,
        params=params,
        geom=geom,
        terms=terms,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
        kcfg=kcfg,
        show_progress=show_progress,
    )


def _cyclone_krylov_shift(seed: _CycloneKrylovSeed) -> complex | None:
    """Convert a valid frequency seed into the shifted-eigenvalue target."""

    if not seed.omega_ok:
        return None
    return complex(float(seed.gamma) if seed.seed_ok else 0.0, float(-seed.omega))


def _solve_cyclone_dominant_pair(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    shift: complex | None,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[Any, Any]:
    """Call the Cyclone dominant-eigenpair solver with one option policy."""

    status("running dominant eigenpair solve")
    return dominant_eigenpair(
        fresh_G0(),
        cache,
        params,
        terms=terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=shift if shift is not None else kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
        status_callback=status,
    )


def _apply_cyclone_seed_branch_guard(
    *,
    gamma: float,
    omega: float,
    seed: _CycloneKrylovSeed,
) -> tuple[float, float]:
    """Prefer a strong explicit seed when Krylov lands on an inconsistent branch."""

    if not seed.seed_ok:
        return gamma, omega
    seed_strong = (seed.gamma > 0.0) and (abs(seed.omega) > 1.0e-6)
    if not seed_strong:
        return gamma, omega
    omega_tol = 0.15 * max(abs(seed.omega), 1.0e-6)
    gamma_tol = 0.15 * max(abs(seed.gamma), 1.0e-6)
    use_seed = (
        not np.isfinite(gamma)
        or not np.isfinite(omega)
        or (seed.gamma > 0.0 and gamma < 0.0)
        or abs(omega - seed.omega) > omega_tol
        or abs(gamma - seed.gamma) > gamma_tol
    )
    if not use_seed:
        return gamma, omega
    return float(seed.gamma), float(seed.omega)


def _pack_cyclone_krylov_result(
    *,
    eig: Any,
    vec: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    seed: _CycloneKrylovSeed,
    diagnostic_norm: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute fields, guard branch selection, normalize, and pack Krylov output."""

    term_cfg = linear_terms_to_term_config(terms)
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    phi_t_out = np.asarray(phi)[None, ...]
    t_out = np.array([0.0])
    gamma_out = float(np.real(eig))
    omega_out = float(-np.imag(eig))
    gamma_out, omega_out = _apply_cyclone_seed_branch_guard(
        gamma=gamma_out,
        omega=omega_out,
        seed=seed,
    )
    if kcfg.omega_sign != 0:
        omega_out = float(np.sign(kcfg.omega_sign)) * abs(omega_out)
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    return gamma_out, omega_out, phi_t_out, t_out


def run_cyclone_krylov_path(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    krylov_cfg: Any,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the Cyclone Krylov branch with the explicit seed policy."""

    status("starting Krylov solve")
    kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    seed = _estimate_cyclone_krylov_seed(
        grid=grid,
        cache=cache,
        params=params,
        geom=geom,
        terms=terms,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
        kcfg=kcfg,
        selection=selection,
        show_progress=show_progress,
        status=status,
        fresh_G0=fresh_G0,
    )
    eig, vec = _solve_cyclone_dominant_pair(
        grid=grid,
        cache=cache,
        params=params,
        terms=terms,
        kcfg=kcfg,
        shift=_cyclone_krylov_shift(seed),
        status=status,
        fresh_G0=fresh_G0,
    )
    gamma_out, omega_out, phi_t_out, t_out = _pack_cyclone_krylov_result(
        eig=eig,
        vec=vec,
        cache=cache,
        params=params,
        terms=terms,
        kcfg=kcfg,
        seed=seed,
        diagnostic_norm=diagnostic_norm,
    )
    status(f"Krylov solve complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return gamma_out, omega_out, phi_t_out, t_out


def _resolve_cyclone_time_config(
    *,
    cfg: Any,
    time_cfg: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
) -> Any | None:
    """Resolve a runtime time config using the existing Cyclone branch policy."""

    method_key = method.lower()
    time_cfg_use = None
    if time_cfg is not None:
        time_cfg_use = replace(time_cfg, dt=float(dt), t_max=float(dt) * int(steps))
    elif cfg.time.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        time_cfg_use = replace(cfg.time, dt=float(dt), t_max=float(dt) * int(steps))
    if time_cfg_use is not None and sample_stride is not None:
        time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    return time_cfg_use


def _run_cyclone_reference_aligned_time(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    time_cfg_use: Any,
    dt: float,
    steps: int,
    sample_stride: int | None,
    diagnostic_norm: str,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the reference-aligned explicit Cyclone time path."""

    t_max_val = float(dt) * int(steps) if time_cfg_use is None else float(time_cfg_use.t_max)
    stride = (
        int(sample_stride)
        if sample_stride is not None
        else (1 if time_cfg_use is None else int(time_cfg_use.sample_stride))
    )
    explicit_time_cfg = ExplicitTimeConfig(
        dt=float(dt),
        t_max=t_max_val,
        sample_stride=stride,
        fixed_dt=True,
    )
    t, phi_ref, _g_t, _o_t = integrate_linear_explicit(
        fresh_G0(),
        grid,
        cache,
        params,
        geom,
        explicit_time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    sel_local = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    gamma, omega, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
        phi_ref,
        t,
        sel_local,
        navg_fraction=0.5,
        mode_method="z_index",
    )
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return gamma, omega, np.asarray(phi_ref), np.asarray(t)


def _integrate_cyclone_configured_time(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_use: Any,
    need_density: bool,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Integrate Cyclone with an explicit or synthesized runtime config."""

    if need_density:
        _, saved = integrate_linear_from_config(
            fresh_G0(),
            grid,
            geom,
            params,
            time_cfg_use,
            terms=terms,
            save_field="phi+density",
            density_species_index=0,
            show_progress=show_progress,
        )
        phi_t, density_t = saved
        return _CycloneTimeTrace(phi_t, density_t, int(time_cfg_use.sample_stride))
    _, phi_t = integrate_linear_from_config(
        fresh_G0(),
        grid,
        geom,
        params,
        time_cfg_use,
        terms=terms,
        show_progress=show_progress,
    )
    return _CycloneTimeTrace(phi_t, None, int(time_cfg_use.sample_stride))


def _integrate_cyclone_unconfigured_time(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Integrate Cyclone with the fixed-step path selected by diagnostics needs."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if need_density or not use_jit:
        diag = integrate_linear_diagnostics(
            fresh_G0(),
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            terms=terms,
            sample_stride=stride,
            species_index=0,
            record_hl_energy=False,
            show_progress=show_progress,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return _CycloneTimeTrace(phi_t, density_t, stride)
    _, phi_out_time = integrate_linear(
        fresh_G0(),
        grid,
        geom,
        params,
        dt=dt,
        steps=steps,
        method=method,
        terms=terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return _CycloneTimeTrace(phi_out_time, None, stride)


def _integrate_cyclone_time_trace(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_use: Any | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Route Cyclone time integration to the configured or fixed-step backend."""

    if time_cfg_use is not None:
        status(
            f"running runtime-configured integrator over {int(steps)} steps with sample_stride={int(time_cfg_use.sample_stride)}"
        )
        if need_density:
            status("saving phi and density diagnostics for automatic fit selection")
        return _integrate_cyclone_configured_time(
            grid=grid,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg_use=time_cfg_use,
            need_density=need_density,
            show_progress=show_progress,
            fresh_G0=fresh_G0,
        )

    stride = 1 if sample_stride is None else int(sample_stride)
    status(
        f"running {'explicit diagnostics' if need_density or not use_jit else 'cached linear'} integrator over {int(steps)} steps with sample_stride={stride}"
    )
    return _integrate_cyclone_unconfigured_time(
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        need_density=need_density,
        use_jit=use_jit,
        show_progress=show_progress,
        fresh_G0=fresh_G0,
    )


def _cyclone_auto_fit_kwargs(options: _CycloneTimeFitOptions) -> dict[str, Any]:
    """Pack automatic-window options shared by Cyclone time-fit branches."""

    return {
        "tmin": options.tmin,
        "tmax": options.tmax,
        "window_fraction": options.window_fraction,
        "min_points": options.min_points,
        "start_fraction": options.start_fraction,
        "growth_weight": options.growth_weight,
        "require_positive": options.require_positive,
        "min_amp_fraction": options.min_amp_fraction,
        "max_amp_fraction": options.max_amp_fraction,
        "window_method": options.window_method,
        "max_fraction": options.max_fraction,
        "end_fraction": options.end_fraction,
        "num_windows": 8,
        "phase_weight": options.phase_weight,
        "length_weight": options.length_weight,
        "min_r2": options.min_r2,
        "late_penalty": options.late_penalty,
        "min_slope": options.min_slope,
        "min_slope_frac": options.min_slope_frac,
        "slope_var_weight": options.slope_var_weight,
    }


def _build_cyclone_time_fit_options(
    *,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    max_amp_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
) -> _CycloneTimeFitOptions:
    """Collect user/runtime fit knobs into the immutable fit policy object."""

    return _CycloneTimeFitOptions(
        fit_key=fit_key,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        max_amp_fraction=max_amp_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
    )


def _fit_cyclone_time_trace(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt: float,
    stride: int,
    sel: Any,
    params: Any,
    diagnostic_norm: str,
    options: _CycloneTimeFitOptions,
    status: Callable[[str], None],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit Cyclone growth/frequency from the saved time trace."""

    phi_t_np = np.asarray(phi_t)
    t_arr = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    status(
        f"integration complete; fitting growth rate from {phi_t_np.shape[0]} saved samples"
    )
    auto_fit_kwargs = _cyclone_auto_fit_kwargs(options)
    if options.fit_key == "auto":
        _signal, name, gamma_out, omega_out = _select_fit_signal_auto(
            t_arr,
            phi_t_np,
            density_np,
            sel,
            mode_method=options.mode_method,
            **auto_fit_kwargs,
        )
        status(f"automatic fit selected signal '{name}'")
        if not np.isfinite(gamma_out) or not np.isfinite(omega_out):
            gamma_out, omega_out = 0.0, 0.0
    else:
        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=options.fit_key,
            mode_method=options.mode_method,
        )
        if options.auto_window and options.tmin is None and options.tmax is None:
            gamma_out, omega_out, _tmin, _tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                **auto_fit_kwargs,
            )
        else:
            gamma_out, omega_out = fit_growth_rate(
                t_arr, signal, tmin=options.tmin, tmax=options.tmax
            )
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    return float(gamma_out), float(omega_out), phi_t_np, t_arr


def _prepare_cyclone_time_path_controls(
    *,
    cfg: Any,
    time_cfg: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    max_amp_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
) -> _CycloneTimePathControls:
    """Resolve time-config overrides and fit-window policy for Cyclone."""

    return _CycloneTimePathControls(
        time_cfg=_resolve_cyclone_time_config(
            cfg=cfg,
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
        ),
        fit_options=_build_cyclone_time_fit_options(
            fit_key=fit_key,
            mode_method=mode_method,
            tmin=tmin,
            tmax=tmax,
            auto_window=auto_window,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            max_fraction=max_fraction,
            end_fraction=end_fraction,
            max_amp_fraction=max_amp_fraction,
            phase_weight=phase_weight,
            length_weight=length_weight,
            min_r2=min_r2,
            late_penalty=late_penalty,
            min_slope=min_slope,
            min_slope_frac=min_slope_frac,
            slope_var_weight=slope_var_weight,
            window_method=window_method,
        ),
    )


def _run_cyclone_saved_time_path(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    sel: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
    controls: _CycloneTimePathControls,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run non-reference Cyclone time integration and fit the saved trace."""

    trace = _integrate_cyclone_time_trace(
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        time_cfg_use=controls.time_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        need_density=need_density,
        use_jit=use_jit,
        show_progress=show_progress,
        status=status,
        fresh_G0=fresh_G0,
    )
    gamma_out, omega_out, phi_t_np, t_arr = _fit_cyclone_time_trace(
        phi_t=trace.phi_t,
        density_t=trace.density_t,
        dt=dt,
        stride=trace.stride,
        sel=sel,
        params=params,
        diagnostic_norm=diagnostic_norm,
        options=controls.fit_options,
        status=status,
    )
    status(f"time integration fit complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return gamma_out, omega_out, phi_t_np, t_arr


def _cyclone_time_path_controls_from_request(
    request: _CycloneTimePathRequest,
) -> _CycloneTimePathControls:
    return _prepare_cyclone_time_path_controls(
        cfg=request.cfg,
        time_cfg=request.time_cfg,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        fit_key=request.fit_key,
        mode_method=request.mode_method,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        max_fraction=request.max_fraction,
        end_fraction=request.end_fraction,
        max_amp_fraction=request.max_amp_fraction,
        phase_weight=request.phase_weight,
        length_weight=request.length_weight,
        min_r2=request.min_r2,
        late_penalty=request.late_penalty,
        min_slope=request.min_slope,
        min_slope_frac=request.min_slope_frac,
        slope_var_weight=request.slope_var_weight,
        window_method=request.window_method,
    )


def _run_cyclone_time_path_request(
    request: _CycloneTimePathRequest,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    request.status(f"starting time integration path with fit_signal={request.fit_key}")
    controls = _cyclone_time_path_controls_from_request(request)
    if request.reference_aligned:
        request.status("running reference-aligned explicit integrator")
        return _run_cyclone_reference_aligned_time(
            grid=request.grid,
            cache=request.cache,
            params=request.params,
            geom=request.geom,
            terms=request.terms,
            time_cfg_use=controls.time_cfg,
            dt=request.dt,
            steps=request.steps,
            sample_stride=request.sample_stride,
            diagnostic_norm=request.diagnostic_norm,
            show_progress=request.show_progress,
            fresh_G0=request.fresh_G0,
        )

    return _run_cyclone_saved_time_path(
        grid=request.grid,
        geom=request.geom,
        params=request.params,
        terms=request.terms,
        sel=request.sel,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        need_density=request.need_density,
        use_jit=request.use_jit,
        diagnostic_norm=request.diagnostic_norm,
        show_progress=request.show_progress,
        status=request.status,
        fresh_G0=request.fresh_G0,
        controls=controls,
    )


def run_cyclone_time_path(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any,
    sel: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    reference_aligned: bool,
    use_jit: bool,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    max_amp_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the Cyclone time-integration branch and fit late-time growth."""

    return _run_cyclone_time_path_request(_cyclone_time_path_request_from_locals(locals()))
