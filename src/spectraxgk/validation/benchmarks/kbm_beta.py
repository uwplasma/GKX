"""KBM fixed-ky beta-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.kbm_beta_solver_paths import (
    KBMBetaExplicitHooks,
    KBMBetaKrylovHooks,
    KBMBetaTimeHooks,
    fit_kbm_beta_explicit_time_sample,
    fit_kbm_beta_time_sample,
    solve_kbm_beta_krylov_sample,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.species import (
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
)
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


def run_kbm_beta_scan(
    betas: np.ndarray,
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    kbm_target_factors: Sequence[float] | None = (0.7, 1.5),
    kbm_beta_transition: float | None = None,
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
    fit_signal: str = "auto",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    if reference_aligned_use and diagnostic_norm == "none":
        diagnostic_norm = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )

    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )

    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    use_continuation = bool(getattr(krylov_cfg_use, "continuation", False))
    prev_vec = None
    prev_eig = None

    gammas = []
    omegas = []
    beta_out = []
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

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
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )
    explicit_hooks = KBMBetaExplicitHooks(
        integrate_linear_explicit_diagnostics=integrate_linear_explicit_diagnostics,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        windowed_growth_rate_from_omega_series=windowed_growth_rate_from_omega_series,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto=fit_growth_rate_auto,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_cfl_fac=resolve_cfl_fac,
    )
    krylov_hooks = KBMBetaKrylovHooks(
        dominant_eigenpair=dominant_eigenpair,
        use_multi_target_krylov=_kbm_use_multi_target_krylov,
        normalize_growth_rate=_normalize_growth_rate,
    )
    time_hooks = KBMBetaTimeHooks(
        integrate_linear_diffrax_streaming=integrate_linear_diffrax_streaming,
        integrate_linear_from_config=integrate_linear_from_config,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        integrate_linear=integrate_linear,
        resolve_streaming_window=_resolve_streaming_window,
        midplane_index=_midplane_index,
        select_fit_signal_auto=_select_fit_signal_auto,
        extract_mode_only_signal=_extract_mode_only_signal,
        select_fit_signal=_select_fit_signal,
        normalize_growth_rate=_normalize_growth_rate,
    )

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    for i, beta in enumerate(betas):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=float(beta),
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

        ns = 2
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
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        G0_jax = jnp.asarray(G0)
        solver_use = select_kbm_solver_auto(
            solver_key,
            ky_target=ky_target,
            reference_aligned=reference_aligned_use,
        )

        if solver_use == "explicit_time":
            gamma, omega = fit_kbm_beta_explicit_time_sample(
                G0_jax=G0_jax,
                grid=grid,
                cache=cache,
                params=params,
                geom=geom,
                terms=terms,
                dt_i=dt_i,
                steps_i=steps_i,
                time_cfg=time_cfg,
                sample_stride=sample_stride,
                mode_method=mode_method,
                sel=sel,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
                diagnostic_norm=diagnostic_norm,
                hooks=explicit_hooks,
            )
        elif solver_use == "krylov":
            krylov_result = solve_kbm_beta_krylov_sample(
                beta=float(beta),
                cfg=cfg,
                G0_jax=G0_jax,
                cache=cache,
                params=params,
                terms=terms,
                solver_key=solver_key,
                krylov_cfg_use=krylov_cfg_use,
                use_continuation=use_continuation,
                prev_vec=prev_vec,
                prev_eig=prev_eig,
                kbm_target_factors=kbm_target_factors,
                kbm_beta_transition=kbm_beta_transition,
                diagnostic_norm=diagnostic_norm,
                is_valid_growth=_is_valid_growth,
                hooks=krylov_hooks,
            )
            gamma = krylov_result.gamma
            omega = krylov_result.omega
            if krylov_result.fallback_to_time:
                solver_use = "time"
            else:
                prev_vec = krylov_result.prev_vec
                prev_eig = krylov_result.prev_eig

        if solver_use not in {"krylov", "explicit_time"}:
            gamma, omega = fit_kbm_beta_time_sample(
                G0_jax=G0_jax,
                grid=grid,
                geom=geom,
                cache=cache,
                params=params,
                terms=terms,
                dt_i=dt_i,
                steps_i=steps_i,
                method=method,
                time_cfg=time_cfg,
                sample_stride=sample_stride,
                fit_key=fit_key,
                streaming_fit=streaming_fit,
                streaming_amp_floor=streaming_amp_floor,
                mode_only=mode_only,
                mode_method=mode_method,
                sel=sel,
                tmin=tmin,
                tmax=tmax,
                sample_index=i,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
                diagnostic_norm=diagnostic_norm,
                density_species_index=density_species_index,
                fit_policy=fit_policy,
                hooks=time_hooks,
            )

        gammas.append(gamma)
        omegas.append(omega)
        beta_out.append(float(beta))

    return LinearScanResult(
        ky=np.array(beta_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
