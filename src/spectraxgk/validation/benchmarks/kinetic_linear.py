"""Kinetic-electron single-ky benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KINETIC_KRYLOV_DEFAULT,
    KINETIC_KRYLOV_REFERENCE_ALIGNED,
    KINETIC_OMEGA_D_SCALE,
    KINETIC_OMEGA_STAR_SCALE,
    KINETIC_RHO_STAR,
)
from spectraxgk.diagnostics.growth_rates import (
    _normalize_growth_rate,
    _select_fit_signal,
)
from spectraxgk.validation.benchmarks.initialization import (
    _build_initial_condition,
    _kinetic_reference_init_cfg,
)
from spectraxgk.validation.benchmarks.reference import LinearRunResult
from spectraxgk.validation.benchmarks.scan import scan_window_valid
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _apply_reference_hypercollisions,
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.config import KineticElectronBaseCase, TimeConfig
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


@dataclass(frozen=True)
class _KineticLinearSetup:
    cfg: KineticElectronBaseCase
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticLinearState:
    grid: Any
    selection: ModeSelection
    state: jnp.ndarray


@dataclass(frozen=True)
class _KineticHistory:
    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _KineticFitOptions:
    fit_signal: str
    mode_method: str
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


@dataclass(frozen=True)
class _KineticTimePathOptions:
    time_cfg: TimeConfig | None
    dt: float
    steps: int
    method: str
    sample_stride: int | None
    density_species_index: int
    show_progress: bool
    n_laguerre: int
    n_hermite: int
    fit: _KineticFitOptions


def _resolve_kinetic_linear_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticLinearSetup:
    """Resolve kinetic benchmark setup shared by Krylov and time paths."""

    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg_use = _kinetic_reference_init_cfg(
        cfg_use.init, reference_aligned=reference_aligned_use
    )
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KINETIC_OMEGA_D_SCALE,
            omega_star_scale=KINETIC_OMEGA_STAR_SCALE,
            rho_star=KINETIC_RHO_STAR,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        if reference_aligned_use:
            params_use = _apply_reference_hypercollisions(params_use, nhermite=Nm)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    return _KineticLinearSetup(
        cfg=cfg_use,
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg_use,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _validate_kinetic_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    """Validate the kinetic two-species index contract."""

    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _build_kinetic_linear_state(
    setup: _KineticLinearSetup,
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    init_species_index: int,
    density_species_index: int,
) -> _KineticLinearState:
    """Select the ky grid and build the kinetic initial perturbation."""

    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    nspecies = 2
    _validate_kinetic_species_indices(
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        nspecies=nspecies,
    )
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return _KineticLinearState(grid=grid, selection=sel, state=jnp.asarray(G0))


def _prepare_kinetic_linear_setup_and_state(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    ky_target: float,
    n_laguerre: int,
    n_hermite: int,
    init_species_index: int,
    density_species_index: int,
) -> tuple[_KineticLinearSetup, _KineticLinearState]:
    """Resolve the kinetic benchmark setup and selected-ky initial state."""

    setup = _resolve_kinetic_linear_setup(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        Nm=n_hermite,
    )
    state = _build_kinetic_linear_state(
        setup,
        ky_target=ky_target,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    return setup, state


def _kinetic_krylov_config(
    setup: _KineticLinearSetup,
    krylov_cfg: KrylovConfig | None,
) -> KrylovConfig:
    """Return the kinetic Krylov policy, including reference-aligned defaults."""

    if krylov_cfg is not None:
        return krylov_cfg
    if setup.reference_aligned:
        return KINETIC_KRYLOV_REFERENCE_ALIGNED
    return KINETIC_KRYLOV_DEFAULT


def _run_kinetic_krylov_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
) -> LinearRunResult:
    """Solve one kinetic benchmark point with the Krylov eigenpath."""

    cfg_use = _kinetic_krylov_config(setup, krylov_cfg)
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    eig, vec = dominant_eigenpair(
        state.state,
        cache,
        setup.params,
        terms=setup.terms,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )
    return _pack_kinetic_result(
        state,
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_time_config(
    time_cfg: TimeConfig | None,
    *,
    sample_stride: int | None,
) -> TimeConfig | None:
    """Apply user sample-stride override without changing time-config semantics."""

    if time_cfg is None or sample_stride is None:
        return time_cfg
    return replace(time_cfg, sample_stride=sample_stride)


def _integrate_configured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig,
    method_key: str,
    fit_signal: str,
    density_species_index: int,
    Nl: int,
    Nm: int,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history with an explicit runtime TimeConfig."""

    dt = float(time_cfg.dt)
    steps = int(round(time_cfg.t_max / time_cfg.dt))
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    if time_cfg.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        save_field = "density" if fit_signal == "density" else "phi"
        _, phi_t = integrate_linear_from_config(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=cache,
            terms=setup.terms,
            save_field=save_field,
            density_species_index=density_species_index
            if fit_signal == "density"
            else None,
        )
        density_t = phi_t if fit_signal == "density" else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        density_species_index=density_species_index if fit_signal == "density" else None,
    )
    return phi_t, None, dt, time_cfg.sample_stride


def _integrate_unconfigured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history without a runtime TimeConfig."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, float(dt), stride

    _, phi_t = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, float(dt), stride


def _integrate_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    Nl: int,
    Nm: int,
) -> _KineticHistory:
    """Integrate a kinetic time history and preserve saved-observable semantics."""

    time_cfg_use = _resolve_time_config(time_cfg, sample_stride=sample_stride)
    if time_cfg_use is not None:
        phi_t, density_t, dt_eff, stride = _integrate_configured_kinetic_history(
            setup,
            state,
            time_cfg=time_cfg_use,
            method_key=method.lower(),
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            Nl=Nl,
            Nm=Nm,
        )
    else:
        phi_t, density_t, dt_eff, stride = _integrate_unconfigured_kinetic_history(
            setup,
            state,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            show_progress=show_progress,
        )
    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt_eff * stride
    density_np = None if density_t is None else np.asarray(density_t)
    return _KineticHistory(t=t, phi_t=phi_t_np, density_t=density_np)


def _fit_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    history: _KineticHistory,
    *,
    options: _KineticFitOptions,
) -> tuple[float, float]:
    """Fit growth/frequency from a saved kinetic time history."""

    signal = _select_fit_signal(
        history.phi_t,
        history.density_t,
        state.selection,
        fit_signal=options.fit_signal,
        mode_method=options.mode_method,
    )
    use_auto = options.auto_window and options.tmin is None and options.tmax is None
    if not use_auto and not scan_window_valid(history.t, options.tmin, options.tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": options.window_fraction,
        "min_points": options.min_points,
        "start_fraction": options.start_fraction,
        "growth_weight": options.growth_weight,
        "require_positive": options.require_positive,
        "min_amp_fraction": options.min_amp_fraction,
    }
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            history.t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(
                history.t, signal, tmin=options.tmin, tmax=options.tmax
            )
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                history.t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _run_kinetic_time_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    options: _KineticTimePathOptions,
) -> LinearRunResult:
    """Run and fit the saved-time kinetic benchmark path."""

    history = _integrate_kinetic_history(
        setup,
        state,
        time_cfg=options.time_cfg,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        sample_stride=options.sample_stride,
        fit_signal=options.fit.fit_signal,
        density_species_index=options.density_species_index,
        show_progress=options.show_progress,
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
    )
    gamma, omega = _fit_kinetic_history(
        setup,
        state,
        history,
        options=options.fit,
    )
    return _pack_kinetic_result(
        state, t=history.t, phi_t=history.phi_t, gamma=gamma, omega=omega
    )


def _pack_kinetic_result(
    state: _KineticLinearState,
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack the public kinetic linear benchmark result."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(state.grid.ky[state.selection.ky_index]),
        selection=state.selection,
    )


def _kinetic_time_path_options(
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
    n_laguerre: int,
    n_hermite: int,
    fit_signal: str,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> _KineticTimePathOptions:
    """Pack public time-path keyword controls into the internal request object."""

    return _KineticTimePathOptions(
        time_cfg,
        dt,
        steps,
        method,
        sample_stride,
        density_species_index,
        show_progress,
        n_laguerre,
        n_hermite,
        _KineticFitOptions(
            fit_signal,
            mode_method,
            auto_window,
            tmin,
            tmax,
            window_fraction,
            min_points,
            start_fraction,
            growth_weight,
            require_positive,
            min_amp_fraction,
        ),
    )


def run_kinetic_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "krylov",
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
    fit_signal: str = "density",
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a kinetic-electron ITG/TEM benchmark and extract growth rate."""

    setup, state = _prepare_kinetic_linear_setup_and_state(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        ky_target=ky_target,
        n_laguerre=Nl,
        n_hermite=Nm,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    if solver.lower() == "krylov":
        return _run_kinetic_krylov_path(
            setup,
            state,
            Nl=Nl,
            Nm=Nm,
            krylov_cfg=krylov_cfg,
        )
    return _run_kinetic_time_path(
        setup,
        state,
        options=_kinetic_time_path_options(
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            density_species_index=density_species_index,
            show_progress=show_progress,
            n_laguerre=Nl,
            n_hermite=Nm,
            fit_signal=fit_signal,
            mode_method=mode_method,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        ),
    )


__all__ = ["run_kinetic_linear"]
