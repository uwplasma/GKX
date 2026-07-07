"""ETG single-ky linear benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
)
from spectraxgk.validation.benchmarks.scan import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.defaults import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _electron_only_params,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from spectraxgk.config import ETGBaseCase, TimeConfig
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached

_ETG_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


@dataclass(frozen=True)
class _ETGLinearSetup:
    """Solver-ready single-ky ETG state shared by Krylov and time paths."""

    cfg: ETGBaseCase
    grid: Any
    geom: Any
    params: Any
    terms: LinearTerms
    selection: ModeSelection
    electron_index: int
    initial_state: Any


@dataclass(frozen=True)
class _ETGTimePathOptions:
    """Private fit and streaming policy for ETG saved-time integrations."""

    fit_key: str
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
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
    diagnostic_norm: str


@dataclass(frozen=True)
class _ETGLinearRequest:
    """Raw public ETG single-ky inputs before solver policies are resolved."""

    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: ETGBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    diagnostic_norm: str
    show_progress: bool


def _etg_linear_request_from_locals(values: dict[str, Any]) -> _ETGLinearRequest:
    """Build an ETG request from ``run_etg_linear`` locals."""

    names = {field.name for field in fields(_ETGLinearRequest)}
    return _ETGLinearRequest(**{name: values[name] for name in names})


def _default_etg_params(cfg: ETGBaseCase, geom: Any, Nm: int) -> LinearParams:
    """Build ETG benchmark species parameters using the tracked normalization."""

    if getattr(cfg.model, "adiabatic_ions", False):
        return _electron_only_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    return _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=Nm,
    )


def _default_etg_terms() -> LinearTerms:
    """Return the electrostatic ETG benchmark term contract."""

    return LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)


def _build_etg_linear_setup(
    *,
    cfg: ETGBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    ky_target: float,
    Nl: int,
    Nm: int,
) -> _ETGLinearSetup:
    """Create the selected-grid initial state for one ETG benchmark point."""

    cfg_use = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = params if params is not None else _default_etg_params(cfg_use, geom, Nm)
    terms_use = terms if terms is not None else _default_etg_terms()

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    charge = np.atleast_1d(np.asarray(params_use.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
    G0 = np.zeros(
        (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg_use.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)
    return _ETGLinearSetup(
        cfg=cfg_use,
        grid=grid,
        geom=geom,
        params=params_use,
        terms=terms_use,
        selection=sel,
        electron_index=electron_index,
        initial_state=jnp.asarray(G0),
    )


def _etg_linear_result(
    setup: _ETGLinearSetup,
    *,
    t: np.ndarray,
    phi_t_np: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack a single-ky ETG run result with the selected physical ky."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(setup.grid.ky[setup.selection.ky_index]),
        selection=setup.selection,
    )


def _valid_etg_growth(
    gamma_val: float, omega_val: float, *, require_positive: bool
) -> bool:
    """Return whether a Krylov ETG result is acceptable for auto solver mode."""

    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _run_etg_krylov_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
    diagnostic_norm: str,
) -> LinearRunResult:
    """Solve one ETG point with the Krylov eigenpath."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    cache = build_linear_cache(setup.grid, setup.geom, setup.params, Nl, Nm)
    krylov_kwargs = {
        "terms": setup.terms,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
    }
    eig, vec = dominant_eigenpair(
        setup.initial_state, cache, setup.params, **krylov_kwargs
    )
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, diagnostic_norm
    )
    return _etg_linear_result(
        setup,
        t=np.array([0.0]),
        phi_t_np=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_etg_time_config(
    cfg: ETGBaseCase,
    time_cfg: TimeConfig | None,
    *,
    streaming_fit: bool,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> tuple[TimeConfig | None, float, int]:
    """Resolve explicit ETG time configuration without changing fit semantics."""

    time_cfg_use = time_cfg
    if time_cfg_use is None and streaming_fit and cfg.time.use_diffrax:
        max_steps = max(int(cfg.time.diffrax_max_steps), int(steps))
        time_cfg_use = replace(
            cfg.time,
            dt=dt,
            t_max=dt * steps,
            diffrax_max_steps=max_steps,
        )
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    if time_cfg_use is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
        if time_cfg is not None:
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
    return time_cfg_use, dt, steps


def _etg_auto_fit_options(
    *,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> dict[str, Any]:
    """Pack the shared automatic-window policy for ETG trace fits."""

    return {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }


def _fit_etg_reference_growth(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    t: np.ndarray,
    reference_navg_fraction: float,
    mode_method: str,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit ETG ``phi`` with the legacy instantaneous-growth reference window."""

    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        phi_t_np,
        t,
        setup.selection,
        navg_fraction=reference_navg_fraction,
        mode_method=mode_method,
    )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_auto_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Select the most stable ETG signal and fit its automatic growth window."""

    gamma, omega = _select_fit_signal_auto(
        t,
        phi_t_np,
        density_np,
        setup.selection,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=0.9,
        window_method="loglinear",
        max_fraction=0.8,
        end_fraction=0.9,
        num_windows=8,
        phase_weight=0.2,
        length_weight=0.05,
        min_r2=0.0,
        late_penalty=0.1,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )[2:]
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_selected_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
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
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit a caller-selected ETG signal with manual-window fallback."""

    signal = _select_fit_signal(
        phi_t_np,
        density_np,
        setup.selection,
        fit_signal=fit_key,
        mode_method=mode_method,
    )
    auto_fit_kwargs = _etg_auto_fit_options(
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )
    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t, tmin, tmax):
        use_auto = True
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_time_trace(
    setup: _ETGLinearSetup,
    *,
    phi_t: Any,
    density_t: Any,
    dt: float,
    stride: int,
    options: _ETGTimePathOptions,
) -> LinearRunResult:
    """Fit ETG growth and frequency from a saved time trace."""

    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    if options.reference_growth_window and options.fit_key == "phi":
        gamma, omega = _fit_etg_reference_growth(
            setup,
            phi_t_np=phi_t_np,
            t=t,
            reference_navg_fraction=options.reference_navg_fraction,
            mode_method=options.mode_method,
            diagnostic_norm=options.diagnostic_norm,
        )
    elif options.fit_key == "auto":
        gamma, omega = _fit_etg_auto_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    else:
        gamma, omega = _fit_etg_selected_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            fit_key=options.fit_key,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            auto_window=options.auto_window,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    return _etg_linear_result(setup, t=t, phi_t_np=phi_t_np, gamma=gamma, omega=omega)


def _run_etg_streaming_density_fit(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run the memory-light Diffrax density streaming fit path."""

    t_total = float(dt * steps)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        options.tmin,
        options.tmax,
        options.start_fraction,
        options.window_fraction,
        1.0,
    )
    G_last, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=time_cfg.diffrax_solver,
        cache=cache,
        terms=setup.terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        show_progress=show_progress,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="density",
        mode_ky_indices=np.array([0], dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(setup.grid),
        mode_method=options.mode_method,
        amp_floor=options.streaming_amp_floor,
        density_species_index=setup.electron_index,
        return_state=True,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, options.diagnostic_norm
    )
    if G_last is not None and G_last.ndim == 7:
        G_last = G_last[0]
    if G_last is None:
        raise ValueError("Expected final state from streaming fit; got None.")
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi_last = compute_fields_cached(
        G_last, cache, setup.params, terms=term_cfg
    ).phi
    return _etg_linear_result(
        setup,
        t=np.array([tmax_i]),
        phi_t_np=np.asarray(jnp.asarray(phi_last)[None, ...]),
        gamma=gamma,
        omega=omega,
    )


def _integrate_etg_configured_history(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    fit_key: str,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history using an explicit or synthesized TimeConfig."""

    if fit_key in {"density", "auto"}:
        if time_cfg.use_diffrax:
            _, saved = integrate_linear_diffrax(
                setup.initial_state,
                setup.grid,
                setup.geom,
                setup.params,
                dt=dt,
                steps=steps,
                method=time_cfg.diffrax_solver,
                cache=cache,
                terms=setup.terms,
                adaptive=time_cfg.diffrax_adaptive,
                rtol=time_cfg.diffrax_rtol,
                atol=time_cfg.diffrax_atol,
                max_steps=time_cfg.diffrax_max_steps,
                show_progress=show_progress,
                progress_bar=time_cfg.progress_bar,
                checkpoint=time_cfg.checkpoint,
                sample_stride=time_cfg.sample_stride,
                return_state=time_cfg.save_state,
                save_field="phi+density",
                density_species_index=setup.electron_index,
            )
            phi_t, density_t = saved
            return phi_t, density_t, time_cfg.sample_stride
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        show_progress=show_progress,
    )
    return phi_t, None, time_cfg.sample_stride


def _integrate_etg_unconfigured_history(
    setup: _ETGLinearSetup,
    *,
    dt: float,
    steps: int,
    method: str,
    fit_key: str,
    sample_stride: int | None,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history without a TimeConfig object."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key in {"density", "auto"}:
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, stride

    _, phi_t = integrate_linear(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, stride


def _run_etg_time_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run ETG saved-time or streaming time paths and fit the trace."""

    time_cfg_use, dt, steps = _resolve_etg_time_config(
        setup.cfg,
        time_cfg,
        streaming_fit=options.streaming_fit,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
    )
    if time_cfg_use is not None:
        cache = build_linear_cache(
            setup.grid,
            setup.geom,
            setup.params,
            Nl,
            Nm,
        )
        if (
            options.fit_key in {"density", "auto"}
            and options.streaming_fit
            and time_cfg_use.use_diffrax
        ):
            return _run_etg_streaming_density_fit(
                setup,
                time_cfg=time_cfg_use,
                cache=cache,
                dt=dt,
                steps=steps,
                options=options,
                show_progress=show_progress,
            )
        phi_t, density_t, stride = _integrate_etg_configured_history(
            setup,
            time_cfg=time_cfg_use,
            cache=cache,
            dt=dt,
            steps=steps,
            fit_key=options.fit_key,
            show_progress=show_progress,
        )
    else:
        phi_t, density_t, stride = _integrate_etg_unconfigured_history(
            setup,
            dt=dt,
            steps=steps,
            method=method,
            fit_key=options.fit_key,
            sample_stride=sample_stride,
            show_progress=show_progress,
        )

    return _fit_etg_time_trace(
        setup,
        phi_t=phi_t,
        density_t=density_t,
        dt=dt,
        stride=stride,
        options=options,
    )


def _run_etg_linear_request(request: _ETGLinearRequest) -> LinearRunResult:
    """Resolve ETG solver policies and execute one single-ky linear point."""

    setup = _build_etg_linear_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
    )
    solver_key = request.solver.strip().lower()
    fit_key = request.fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    streaming_fit = request.streaming_fit
    if fit_key == "auto" and streaming_fit:
        streaming_fit = False
    time_options = _ETGTimePathOptions(
        fit_key=fit_key,
        streaming_fit=streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        reference_growth_window=request.reference_growth_window,
        reference_navg_fraction=request.reference_navg_fraction,
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
        diagnostic_norm=request.diagnostic_norm,
    )
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "krylov"

    if solver_key == "krylov":
        krylov_result = _run_etg_krylov_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            krylov_cfg=request.krylov_cfg,
            diagnostic_norm=request.diagnostic_norm,
        )
        if auto_solver and not _valid_etg_growth(
            krylov_result.gamma,
            krylov_result.omega,
            require_positive=request.require_positive,
        ):
            solver_key = "time"
        else:
            return krylov_result

    if solver_key != "krylov":
        return _run_etg_time_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            time_cfg=request.time_cfg,
            dt=request.dt,
            steps=request.steps,
            method=request.method,
            sample_stride=request.sample_stride,
            options=time_options,
            show_progress=request.show_progress,
        )

    raise ValueError(f"Unsupported ETG linear solver '{request.solver}'.")


def run_etg_linear(
    ky_target: float = 3.0,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: ETGBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    mode_method: str = "project",
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    streaming_fit: bool = False,
    streaming_amp_floor: float = 1.0e-30,
    reference_growth_window: bool = False,
    reference_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    return _run_etg_linear_request(_etg_linear_request_from_locals(locals()))
