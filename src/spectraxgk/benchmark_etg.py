"""ETG benchmark runners behind the public :mod:`spectraxgk.benchmarks` facade."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    fit_growth_rate,
    fit_growth_rate_auto,
    gx_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.benchmark_defaults import (
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
)
from spectraxgk.benchmark_helpers import (
    LinearRunResult,
    LinearScanResult,
    _build_initial_condition,
    _electron_only_params,
    _extract_mode_only_signal,
    _iter_ky_batches,
    _midplane_index,
    _normalize_growth_rate,
    _resolve_streaming_window,
    _select_fit_signal,
    _select_fit_signal_auto,
    _two_species_params,
)
from spectraxgk.benchmark_scan import (
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
from spectraxgk.diffrax_integrators import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


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
    gx_growth: bool = False,
    gx_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    cfg = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
    if terms is None:
        # The ETG benchmark contract is electrostatic for both the adiabatic-ion
        # and two-species variants. Keep the default ETG wrappers aligned with
        # the tracked ETG asset-generation tools.
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    charge = np.atleast_1d(np.asarray(params.charge_sign))
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
        init_cfg=cfg.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)

    G0_jax = jnp.asarray(G0)
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    if fit_key == "auto" and streaming_fit:
        streaming_fit = False
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "krylov"

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    if solver_key == "krylov":
        krylov_cfg = krylov_cfg or ETG_KRYLOV_DEFAULT
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            krylov_dim=krylov_cfg.krylov_dim,
            restarts=krylov_cfg.restarts,
            omega_min_factor=krylov_cfg.omega_min_factor,
            omega_target_factor=krylov_cfg.omega_target_factor,
            omega_cap_factor=krylov_cfg.omega_cap_factor,
            omega_sign=krylov_cfg.omega_sign,
            method=krylov_cfg.method,
            power_iters=krylov_cfg.power_iters,
            power_dt=krylov_cfg.power_dt,
            shift=krylov_cfg.shift,
            shift_source=krylov_cfg.shift_source,
            shift_tol=krylov_cfg.shift_tol,
            shift_maxiter=krylov_cfg.shift_maxiter,
            shift_restart=krylov_cfg.shift_restart,
            shift_solve_method=krylov_cfg.shift_solve_method,
            shift_preconditioner=krylov_cfg.shift_preconditioner,
            shift_selection=krylov_cfg.shift_selection,
            mode_family=krylov_cfg.mode_family,
            fallback_method=krylov_cfg.fallback_method,
            fallback_real_floor=krylov_cfg.fallback_real_floor,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_np = np.asarray(phi)[None, ...]
        t = np.array([0.0])
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        if krylov_cfg.omega_sign != 0:
            omega = float(np.sign(krylov_cfg.omega_sign)) * abs(omega)
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        if auto_solver and not _is_valid_growth(gamma, omega):
            solver_key = "time"

    if solver_key != "krylov":
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
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_key in {"density", "auto"}:
                if streaming_fit and time_cfg_use.use_diffrax:
                    t_total = float(dt * steps)
                    tmin_i, tmax_i = _resolve_streaming_window(
                        t_total, tmin, tmax, start_fraction, window_fraction, 1.0
                    )
                    G_last, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.diffrax_solver,
                        cache=cache,
                        terms=terms,
                        adaptive=False,
                        rtol=time_cfg_use.diffrax_rtol,
                        atol=time_cfg_use.diffrax_atol,
                        max_steps=time_cfg_use.diffrax_max_steps,
                        show_progress=show_progress,
                        progress_bar=time_cfg_use.progress_bar,
                        checkpoint=time_cfg_use.checkpoint,
                        tmin=tmin_i,
                        tmax=tmax_i,
                        fit_signal="density",
                        mode_ky_indices=np.array([0], dtype=int),
                        mode_kx_index=0,
                        mode_z_index=_midplane_index(grid),
                        mode_method=mode_method,
                        amp_floor=streaming_amp_floor,
                        density_species_index=electron_index,
                        return_state=True,
                    )
                    gamma = float(np.asarray(gamma_vals)[0])
                    omega = float(np.asarray(omega_vals)[0])
                    gamma, omega = _normalize_growth_rate(
                        gamma, omega, params, diagnostic_norm
                    )
                    if G_last is not None and G_last.ndim == 7:
                        G_last = G_last[0]
                    term_cfg = linear_terms_to_term_config(terms)
                    if G_last is None:
                        raise ValueError(
                            "Expected final state from streaming fit; got None."
                        )
                    phi_last = compute_fields_cached(
                        G_last, cache, params, terms=term_cfg
                    ).phi
                    phi_t = jnp.asarray(phi_last)[None, ...]
                    density_t = None
                    stride = time_cfg_use.sample_stride
                    phi_t_np = np.asarray(phi_t)
                    t = np.array([tmax_i])
                    return LinearRunResult(
                        t=t,
                        phi_t=phi_t_np,
                        gamma=gamma,
                        omega=omega,
                        ky=float(grid.ky[sel.ky_index]),
                        selection=sel,
                    )
                if time_cfg_use.use_diffrax:
                    _, saved = integrate_linear_diffrax(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.diffrax_solver,
                        cache=cache,
                        terms=terms,
                        adaptive=time_cfg_use.diffrax_adaptive,
                        rtol=time_cfg_use.diffrax_rtol,
                        atol=time_cfg_use.diffrax_atol,
                        max_steps=time_cfg_use.diffrax_max_steps,
                        show_progress=show_progress,
                        progress_bar=time_cfg_use.progress_bar,
                        checkpoint=time_cfg_use.checkpoint,
                        sample_stride=time_cfg_use.sample_stride,
                        return_state=time_cfg_use.save_state,
                        save_field="phi+density",
                        density_species_index=electron_index,
                    )
                    phi_t, density_t = saved
                else:
                    _diag = integrate_linear_diagnostics(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.method,
                        cache=cache,
                        terms=terms,
                        sample_stride=time_cfg_use.sample_stride,
                        species_index=electron_index,
                    )
                    phi_t = _diag[1]
                    density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    time_cfg_use,
                    cache=cache,
                    terms=terms,
                    show_progress=show_progress,
                )
                density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if fit_key in {"density", "auto"}:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    species_index=electron_index,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear(
                    G0_jax,
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
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t = np.arange(phi_t_np.shape[0]) * dt * stride
        density_np = None if density_t is None else np.asarray(density_t)
        if gx_growth and fit_key == "phi":
            gamma, omega, _gamma_t, _omega_t, _t_mid = gx_growth_rate_from_phi(
                phi_t_np,
                t,
                sel,
                navg_fraction=gx_navg_fraction,
                mode_method=mode_method,
            )
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            return LinearRunResult(
                t=t,
                phi_t=phi_t_np,
                gamma=gamma,
                omega=omega,
                ky=float(grid.ky[sel.ky_index]),
                selection=sel,
            )
        if fit_key == "auto":
            signal, _name, gamma, omega = _select_fit_signal_auto(
                t,
                phi_t_np,
                density_np,
                sel,
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
            )
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            return LinearRunResult(
                t=t,
                phi_t=phi_t_np,
                gamma=gamma,
                omega=omega,
                ky=float(grid.ky[sel.ky_index]),
                selection=sel,
            )

        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=fit_key,
            mode_method=mode_method,
        )

        use_auto = auto_window and tmin is None and tmax is None
        if not use_auto and not scan_window_valid(t, tmin, tmax):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_etg_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
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
    max_fraction: float = 0.8,
    end_fraction: float = 1.0,
    max_amp_fraction: float = 1.0,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.0,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
    window_method: str = "loglinear",
    mode_method: str = "project",
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    gx_growth: bool = False,
    gx_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearScanResult:
    """Run an ETG linear benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
    if terms is None:
        # Keep the ETG scan helper on the same electrostatic benchmark contract
        # as the single-ky ETG wrapper and the tracked ETG figure builders.
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    need_density = fit_key in {"density", "auto"}
    gammas = []
    omegas = []
    ky_out = []

    mode_method = resolve_scan_mode_method(mode_method, mode_only=mode_only)
    use_batch = should_use_ky_batch(
        ky_batch=ky_batch,
        solver_key=solver_key,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
    )
    fit_policy = ScanFitWindowPolicy(
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
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )

    def _fit_signal(
        signal: np.ndarray, idx: int, dt_i: float, stride: int
    ) -> tuple[float, float]:
        return fit_policy.fit_signal(
            signal,
            idx=idx,
            dt=dt_i,
            stride=stride,
            params=params,
            diagnostic_norm=diagnostic_norm,
        )

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    prev_vec: jnp.ndarray | None = None
    prev_eig: complex | None = None
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel: ModeSelection | ModeSelectionBatch

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [
                select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice
            ]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        charge = np.atleast_1d(np.asarray(params.charge_sign))
        ns = int(charge.size)
        electron_index = int(np.argmin(charge))
        G0 = np.zeros(
            (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        )
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver_key == "krylov":
            cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
            use_cont = bool(cfg_use.continuation)
            v0_use = G0_jax
            v_ref = None
            shift_override = cfg_use.shift
            shift_selection_use = cfg_use.shift_selection
            if use_cont and prev_vec is not None and prev_vec.shape == G0_jax.shape:
                v0_use = prev_vec
                v_ref = prev_vec
                if (
                    cfg_use.method.strip().lower() == "shift_invert"
                    and prev_eig is not None
                ):
                    if shift_override is None:
                        shift_override = prev_eig
                        # When continuation carries an explicit previous eigenvalue
                        # as the shift, select the closest shifted branch first and
                        # let overlap tracking keep the mode family coherent.
                        shift_selection_use = "shift"
            select_overlap = (
                use_cont
                and v_ref is not None
                and (cfg_use.continuation_selection.strip().lower() == "overlap")
            )
            eig, vec = dominant_eigenpair(
                v0_use,
                cache,
                params,
                terms=terms,
                v_ref=v_ref,
                select_overlap=select_overlap,
                krylov_dim=cfg_use.krylov_dim,
                restarts=cfg_use.restarts,
                omega_min_factor=cfg_use.omega_min_factor,
                omega_target_factor=cfg_use.omega_target_factor,
                omega_cap_factor=cfg_use.omega_cap_factor,
                omega_sign=cfg_use.omega_sign,
                method=cfg_use.method,
                power_iters=cfg_use.power_iters,
                power_dt=cfg_use.power_dt,
                shift=shift_override,
                shift_source=cfg_use.shift_source,
                shift_tol=cfg_use.shift_tol,
                shift_maxiter=cfg_use.shift_maxiter,
                shift_restart=cfg_use.shift_restart,
                shift_solve_method=cfg_use.shift_solve_method,
                shift_preconditioner=cfg_use.shift_preconditioner,
                shift_selection=shift_selection_use,
                mode_family=cfg_use.mode_family,
                fallback_method=cfg_use.fallback_method,
                fallback_real_floor=cfg_use.fallback_real_floor,
            )
            if use_cont:
                eig_host = complex(np.asarray(eig))
                if np.isfinite(eig_host.real) and np.isfinite(eig_host.imag):
                    prev_vec = vec
                    prev_eig = eig_host
                else:
                    prev_vec = None
                    prev_eig = None
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            if cfg_use.omega_sign != 0:
                omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total,
                indexed_float_value(tmin, batch_start),
                indexed_float_value(tmax, batch_start),
                start_fraction,
                window_fraction,
                1.0,
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt_i,
                steps=steps_i,
                method=time_cfg_i.diffrax_solver,
                cache=cache,
                terms=terms,
                adaptive=time_cfg_i.diffrax_adaptive,
                rtol=time_cfg_i.diffrax_rtol,
                atol=time_cfg_i.diffrax_atol,
                max_steps=time_cfg_i.diffrax_max_steps,
                progress_bar=time_cfg_i.progress_bar,
                checkpoint=time_cfg_i.checkpoint,
                tmin=tmin_i,
                tmax=tmax_i,
                fit_signal=fit_key,
                mode_ky_indices=np.arange(valid_count, dtype=int),
                mode_kx_index=0,
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                density_species_index=electron_index if fit_key == "density" else None,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gamma_i, omega_i = _normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params_use,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            save_field = (
                "phi+density"
                if fit_key == "auto"
                else ("density" if fit_key == "density" else "phi")
            )
            save_mode = (
                None
                if fit_key == "auto"
                else (sel if (mode_only and fit_key == "phi") else None)
            )
            _, saved = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=electron_index if need_density else None,
                show_progress=show_progress,
            )
            if fit_key == "auto":
                phi_t, density_t = saved
            else:
                phi_t = saved
                density_t = None
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if need_density:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=1,
                    show_progress=show_progress,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    show_progress=show_progress,
                )
                phi_t = phi_out_time
                density_t = None

        phi_t_np = np.asarray(phi_t)
        density_np = None if density_t is None else np.asarray(density_t)
        if fit_key == "density" and density_np is None:
            density_np = phi_t_np
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride

        def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
            if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
                return False
            if require_positive and gamma_val <= 0.0:
                return False
            return True

        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if fit_key == "auto":
                sel_local = ModeSelection(
                    ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid)
                )
                _signal, _name, gamma, omega = _select_fit_signal_auto(
                    t,
                    phi_t_np,
                    density_np,
                    sel_local,
                    mode_method=mode_method,
                    tmin=indexed_float_value(tmin, batch_start + local_idx),
                    tmax=indexed_float_value(tmax, batch_start + local_idx),
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                    max_amp_fraction=max_amp_fraction,
                    window_method=window_method,
                    max_fraction=max_fraction,
                    end_fraction=end_fraction,
                    num_windows=8,
                    phase_weight=phase_weight,
                    length_weight=length_weight,
                    min_r2=min_r2,
                    late_penalty=late_penalty,
                    min_slope=min_slope,
                    min_slope_frac=min_slope_frac,
                    slope_var_weight=slope_var_weight,
                )
                gamma, omega = _normalize_growth_rate(
                    gamma, omega, params_use, diagnostic_norm
                )
                if auto_solver and not _is_valid_growth(gamma, omega):
                    res = run_etg_linear(
                        ky_target=float(ky_val),
                        cfg=cfg,
                        Nl=Nl,
                        Nm=Nm,
                        dt=dt_i,
                        steps=steps_i,
                        method=method,
                        params=params,
                        solver="krylov",
                        krylov_cfg=krylov_cfg,
                        diagnostic_norm=diagnostic_norm,
                        fit_signal="phi",
                        show_progress=show_progress,
                    )
                    gamma = float(res.gamma)
                    omega = float(res.omega)
                gammas.append(gamma)
                omegas.append(omega)
                ky_out.append(float(ky_val))
                continue

            if mode_only and fit_key == "phi" and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = ModeSelection(
                    ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid)
                )
                signal = _select_fit_signal(
                    phi_t_np,
                    density_np,
                    sel_local,
                    fit_signal=fit_key,
                    mode_method=mode_method,
                )
            if gx_growth and fit_key == "phi":
                sel_local = ModeSelection(
                    ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid)
                )
                gamma, omega, _gamma_t, _omega_t, _t_mid = gx_growth_rate_from_phi(
                    phi_t_np,
                    t,
                    sel_local,
                    navg_fraction=gx_navg_fraction,
                    mode_method=mode_method,
                )
                gamma, omega = _normalize_growth_rate(
                    gamma, omega, params_use, diagnostic_norm
                )
            else:
                gamma, omega = _fit_signal(
                    signal, batch_start + local_idx, dt_i, stride
                )
            if auto_solver and not _is_valid_growth(gamma, omega):
                res = run_etg_linear(
                    ky_target=float(ky_val),
                    cfg=cfg,
                    Nl=Nl,
                    Nm=Nm,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    params=params,
                    solver="krylov",
                    krylov_cfg=krylov_cfg,
                    diagnostic_norm=diagnostic_norm,
                    fit_signal="phi",
                    show_progress=show_progress,
                )
                gamma = float(res.gamma)
                omega = float(res.omega)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
