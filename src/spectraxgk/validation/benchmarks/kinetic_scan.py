"""Kinetic-electron ky-scan benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
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
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
)
from spectraxgk.validation.benchmarks.initialization import (
    _build_initial_condition,
    _kinetic_reference_init_cfg,
)
from spectraxgk.validation.benchmarks.reference import LinearScanResult
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _apply_reference_hypercollisions,
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.config import KineticElectronBaseCase, TimeConfig
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config


@dataclass(frozen=True)
class _KineticScanSetup:
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _KineticScanRunOptions:
    ky_values: np.ndarray
    time_cfg: TimeConfig | None
    solver_key: str
    krylov_cfg: KrylovConfig | None
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    use_batch: bool
    show_progress: bool


@dataclass(frozen=True)
class _KineticScanFitOptions:
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    fit_policy: ScanFitWindowPolicy


@dataclass
class _KineticScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_KineticScanOutput":
        return cls(gammas=[], omegas=[], ky=[])


@dataclass(frozen=True)
class _KineticScanControls:
    setup: _KineticScanSetup
    run_options: _KineticScanRunOptions
    fit_options: _KineticScanFitOptions


def _resolve_kinetic_scan_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticScanSetup:
    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg = _kinetic_reference_init_cfg(
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
    return _KineticScanSetup(
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _validate_kinetic_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _iter_kinetic_scan_batches(options: _KineticScanRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)


def _prepare_kinetic_scan_batch(
    setup: _KineticScanSetup,
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> _KineticScanBatch:
    if use_batch:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices)
        sel_indices = np.arange(len(ky_indices), dtype=int)
        selection: ModeSelection | ModeSelectionBatch = ModeSelectionBatch(
            sel_indices, 0, _midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky_slice[0]))
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices[0])
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    nspecies = 2
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    state = jnp.asarray(G0)
    cache = build_linear_cache(grid, setup.geom, setup.params, Nl, Nm)
    return _KineticScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )


def _kinetic_scan_time_config(
    time_cfg: TimeConfig | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> TimeConfig | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _run_kinetic_scan_krylov(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    krylov_cfg: KrylovConfig | None,
) -> tuple[float, float]:
    cfg_use = krylov_cfg or (
        KINETIC_KRYLOV_REFERENCE_ALIGNED
        if setup.reference_aligned
        else KINETIC_KRYLOV_DEFAULT
    )
    eig, _vec = dominant_eigenpair(
        batch.state,
        batch.cache,
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
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _append_kinetic_streaming_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    streaming_amp_floor: float,
    density_species_index: int,
    output: _KineticScanOutput,
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        setup.geom,
        setup.params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=setup.terms,
        adaptive=time_cfg.diffrax_adaptive,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=np.arange(batch.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            setup.params,
            setup.diagnostic_norm,
        )
        output.gammas.append(gamma_i)
        output.omegas.append(omega_i)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _integrate_kinetic_scan_history(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig | None,
    method: str,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
        _, phi_t = integrate_linear_from_config(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=batch.cache,
            terms=setup.terms,
            save_mode=batch.selection if (mode_only and fit_key == "phi") else None,
            mode_method=save_mode_method,
            save_field="density" if fit_key == "density" else "phi",
            density_species_index=density_species_index
            if fit_key == "density"
            else None,
        )
        return np.asarray(phi_t), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key == "density":
        _diag = integrate_linear_diagnostics(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = _diag[1]
        density_t = _diag[2] if len(_diag) > 2 else None
    else:
        _, phi_t = integrate_linear(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        density_t = None
    return (
        np.asarray(phi_t),
        None if density_t is None else np.asarray(density_t),
        stride,
    )


def _append_kinetic_sampled_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    fit_policy: ScanFitWindowPolicy,
    density_species_index: int,
    stride: int,
    output: _KineticScanOutput,
) -> None:
    density_np = phi_t if fit_key == "density" and density_t is None else density_t
    for local_idx in range(batch.valid_count):
        if mode_only and fit_key == "phi" and phi_t.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t, local_idx=local_idx)
        elif (
            mode_only
            and fit_key == "density"
            and density_np is not None
            and density_np.ndim <= 3
        ):
            signal = _extract_mode_only_signal(
                density_np,
                local_idx=local_idx,
                species_index=density_species_index,
            )
        else:
            sel_local = ModeSelection(
                ky_index=local_idx,
                kx_index=0,
                z_index=_midplane_index(batch.grid),
            )
            signal = _select_fit_signal(
                phi_t,
                density_np,
                sel_local,
                fit_signal=fit_key,
                mode_method=mode_method,
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=batch.batch_start + local_idx,
            dt=batch.dt,
            stride=stride,
            params=setup.params,
            diagnostic_norm=setup.diagnostic_norm,
        )
        output.gammas.append(gamma)
        output.omegas.append(omega)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _append_kinetic_krylov_result(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    krylov_cfg: KrylovConfig | None,
    output: _KineticScanOutput,
) -> None:
    gamma, omega = _run_kinetic_scan_krylov(batch, setup, krylov_cfg)
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(batch.ky_slice[0]))


def _append_kinetic_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    Nl: int,
    Nm: int,
    output: _KineticScanOutput,
) -> None:
    batch = _prepare_kinetic_scan_batch(
        setup,
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        dt=run_options.dt,
        steps=run_options.steps,
        Nl=Nl,
        Nm=Nm,
        init_species_index=run_options.init_species_index,
    )
    if run_options.solver_key == "krylov":
        _append_kinetic_krylov_result(
            batch, setup, krylov_cfg=run_options.krylov_cfg, output=output
        )
        return

    time_cfg_i = _kinetic_scan_time_config(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and run_options.streaming_fit:
        _append_kinetic_streaming_results(
            batch,
            setup,
            time_cfg=time_cfg_i,
            fit_key=run_options.fit_key,
            mode_method=run_options.mode_method,
            tmin=fit_options.tmin,
            tmax=fit_options.tmax,
            start_fraction=fit_options.start_fraction,
            window_fraction=fit_options.window_fraction,
            streaming_amp_floor=run_options.streaming_amp_floor,
            density_species_index=run_options.density_species_index,
            output=output,
        )
        return

    phi_t, density_t, stride = _integrate_kinetic_scan_history(
        batch,
        setup,
        time_cfg=time_cfg_i,
        method=run_options.method,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        sample_stride=run_options.sample_stride,
        density_species_index=run_options.density_species_index,
        show_progress=run_options.show_progress,
    )
    _append_kinetic_sampled_results(
        batch,
        setup,
        phi_t=phi_t,
        density_t=density_t,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        fit_policy=fit_options.fit_policy,
        density_species_index=run_options.density_species_index,
        stride=stride,
        output=output,
    )


def _build_kinetic_scan_fit_policy(
    *,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> ScanFitWindowPolicy:
    """Pack kinetic-scan growth-window options once for all batches."""

    return ScanFitWindowPolicy(
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )


def _build_kinetic_scan_run_options(
    *,
    ky_values: np.ndarray,
    time_cfg: TimeConfig | None,
    solver_key: str,
    krylov_cfg: KrylovConfig | None,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    use_batch: bool,
    show_progress: bool,
) -> _KineticScanRunOptions:
    """Pack kinetic scan runtime controls for batch execution."""

    return _KineticScanRunOptions(
        ky_values=ky_values,
        time_cfg=time_cfg,
        solver_key=solver_key,
        krylov_cfg=krylov_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        use_batch=use_batch,
        show_progress=show_progress,
    )


def _run_kinetic_scan_batches(
    *,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    n_laguerre: int,
    n_hermite: int,
) -> _KineticScanOutput:
    """Execute all kinetic ky batches and collect scan rows."""

    output = _KineticScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_kinetic_scan_batches(run_options):
        _append_kinetic_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            setup=setup,
            run_options=run_options,
            fit_options=fit_options,
            Nl=n_laguerre,
            Nm=n_hermite,
            output=output,
        )
    return output


def _kinetic_scan_result(output: _KineticScanOutput) -> LinearScanResult:
    """Pack kinetic scan rows into the public scan result."""

    return LinearScanResult(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )


def _prepare_kinetic_scan_controls(
    *,
    ky_values: np.ndarray,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    params: LinearParams | None,
    cfg: KineticElectronBaseCase | None,
    time_cfg: TimeConfig | None,
    solver: str,
    krylov_cfg: KrylovConfig | None,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    terms: LinearTerms | None,
    sample_stride: int | None,
    fit_signal: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    show_progress: bool,
) -> _KineticScanControls:
    """Resolve setup, execution, and fitting controls for one kinetic scan."""

    setup = _resolve_kinetic_scan_setup(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        Nm=Nm,
    )
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    resolved_mode_method = resolve_scan_mode_method(mode_method, mode_only=mode_only)
    use_batch = should_use_ky_batch(
        ky_batch=ky_batch,
        solver_key=solver_key,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
    )
    fit_policy = _build_kinetic_scan_fit_policy(
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )

    ky_values_arr = np.asarray(ky_values, dtype=float)
    _validate_kinetic_species_indices(
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    run_options = _build_kinetic_scan_run_options(
        ky_values=ky_values_arr,
        time_cfg=time_cfg,
        solver_key=solver_key,
        krylov_cfg=krylov_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=resolved_mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        use_batch=use_batch,
        show_progress=show_progress,
    )
    fit_options = _KineticScanFitOptions(
        tmin=tmin,
        tmax=tmax,
        start_fraction=start_fraction,
        window_fraction=window_fraction,
        fit_policy=fit_policy,
    )
    return _KineticScanControls(
        setup=setup,
        run_options=run_options,
        fit_options=fit_options,
    )


def run_kinetic_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "density",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearScanResult:
    """Run a kinetic-electron ITG/TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    controls = _prepare_kinetic_scan_controls(
        ky_values=ky_values,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        solver=solver,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        mode_method=mode_method,
        mode_only=mode_only,
        terms=terms,
        sample_stride=sample_stride,
        fit_signal=fit_signal,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        show_progress=show_progress,
    )
    output = _run_kinetic_scan_batches(
        setup=controls.setup,
        run_options=controls.run_options,
        fit_options=controls.fit_options,
        n_laguerre=Nl,
        n_hermite=Nm,
    )
    return _kinetic_scan_result(output)


__all__ = ["run_kinetic_scan"]
