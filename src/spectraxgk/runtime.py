"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from spectraxgk.cetg import (
    build_cetg_model_params,
    integrate_cetg_gx_diagnostics_state,
    validate_cetg_runtime_config,
)
from spectraxgk.config import resolve_cfl_fac
from spectraxgk.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    select_ky_index,
)
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import apply_geometry_grid_defaults, FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.nonlinear import integrate_nonlinear_gx_diagnostics_state
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import apply_diagnostic_normalization, get_normalization_contract
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk import runtime_startup
from spectraxgk.runtime_diagnostics import (
    concat_gx_diagnostics,
    slice_gx_diagnostics,
    stride_gx_diagnostics,
    truncate_gx_diagnostics,
)
from spectraxgk.runtime_chunks import run_adaptive_gx_chunk_loop
from spectraxgk.runtime_results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
    build_runtime_nonlinear_result,
)
from spectraxgk.runtime_startup import (
    _build_gaussian_profile,
    _build_initial_condition,
    _enforce_full_ky_hermitian,
    _expand_ky,
    _gx_default_p_hyper_m,
    _require_full_gk_runtime_model,
    _resolve_runtime_hl_dims,
    _reshape_gx_state,
    _runtime_default_krylov_config,
    _runtime_model_key,
    _species_to_linear,
)
from spectraxgk.runners import integrate_linear_from_config, integrate_nonlinear_from_config
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.miller_eik import generate_runtime_miller_eik
from spectraxgk.vmec_eik import generate_runtime_vmec_eik

_GX_RAND_MAX = float((1 << 31) - 1)


def _normalize_linear_solver_name(solver: str) -> str:
    solver_key = solver.strip().lower()
    if solver_key == "explicit_time":
        return "gx_time"
    return solver_key


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


build_flux_tube_geometry = runtime_startup.build_flux_tube_geometry
load_gx_restart_state = runtime_startup.load_gx_restart_state


def build_runtime_geometry(cfg: RuntimeConfig) -> FluxTubeGeometryLike:
    """Resolve runtime geometry while preserving the runtime module patch surface."""

    model = cfg.geometry.model.strip().lower()
    if model == "vmec":
        eik_path = generate_runtime_vmec_eik(cfg)
        geom_cfg = replace(cfg.geometry, model="vmec-eik", geometry_file=str(eik_path))
        return build_flux_tube_geometry(geom_cfg)
    if model == "miller":
        eik_path = generate_runtime_miller_eik(cfg)
        geom_cfg = replace(cfg.geometry, model="gx-eik", geometry_file=str(eik_path))
        return build_flux_tube_geometry(geom_cfg)
    return build_flux_tube_geometry(cfg.geometry)


def build_runtime_linear_params(
    cfg: RuntimeConfig,
    *,
    Nm: int | None = None,
    geom: FluxTubeGeometryLike | None = None,
) -> LinearParams:
    """Build runtime linear parameters using the runtime module geometry surface."""

    if geom is None:
        geom = build_runtime_geometry(cfg)
    return runtime_startup.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)


def build_runtime_linear_terms(cfg: RuntimeConfig) -> LinearTerms:
    """Build runtime linear term toggles."""

    return runtime_startup.build_runtime_linear_terms(cfg)


def build_runtime_term_config(cfg: RuntimeConfig) -> TermConfig:
    """Build runtime nonlinear-ready term config."""

    return runtime_startup.build_runtime_term_config(cfg)


def _load_initial_state_from_file(
    path: Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    """Load an initial state while preserving the runtime module patch surface."""

    if path.suffix.lower() == ".nc":
        return load_gx_restart_state(
            path,
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=ny,
            nx=nx,
            nz=nz,
        )
    return runtime_startup._load_initial_state_from_file(
        path,
        nspecies=nspecies,
        Nl=Nl,
        Nm=Nm,
        ny=ny,
        nx=nx,
        nz=nz,
    )


def _gx_centered_random_pairs(seed: int, count: int) -> np.ndarray:
    """Return GX-style centered random pairs using glibc ``rand()`` semantics."""

    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)

    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(_GX_RAND_MAX))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)

    rand_vals = (state[344:] >> np.uint64(1)).astype(np.float64, copy=False)
    half = 0.5 * _GX_RAND_MAX
    inv = 1.0 / _GX_RAND_MAX
    pairs = np.empty((count, 2), dtype=np.float64)
    for i in range(count):
        pairs[i, 0] = (rand_vals[2 * i] - half) * inv
        pairs[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return pairs


def _gx_init_mode_pairs(grid: SpectralGrid) -> list[tuple[int, int]]:
    """Return the GX startup-loop ``(kx, ky)`` pairs for multimode initial conditions."""

    nx = int(np.asarray(grid.kx).size)
    ny = int(np.asarray(grid.ky).size)
    kx_max = 1 + (nx - 1) // 3
    ky_max = 1 + (ny - 1) // 3
    return [(int(kx_i), int(ky_i)) for kx_i in range(kx_max) for ky_i in range(1, ky_max)]


def _gx_periodic_zp(z: np.ndarray) -> float:
    """Return GX's periodic ``Zp`` from the discrete theta grid."""

    z_arr = np.asarray(z, dtype=float)
    if z_arr.size <= 1:
        return 1.0
    dz = float(z_arr[1] - z_arr[0])
    period = abs(dz) * float(z_arr.size)
    if period <= 0.0:
        return 1.0
    return period / (2.0 * np.pi)


def _select_nonlinear_mode_indices(
    grid: SpectralGrid,
    *,
    ky_target: float,
    kx_target: float | None,
    use_dealias_mask: bool,
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    kx_pick_target = 0.0 if kx_target is None else float(kx_target)
    if not use_dealias_mask:
        ky_pick = select_ky_index(ky, ky_target)
        kx_pick = int(np.argmin(np.abs(kx - kx_pick_target)))
        return ky_pick, kx_pick

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    ky_candidates = np.where(np.any(mask, axis=1))[0]
    if ky_candidates.size == 0:
        ky_candidates = np.arange(ky.size, dtype=int)
    ky_pick = ky_candidates[int(np.argmin(np.abs(ky[ky_candidates] - float(ky_target))))]
    kx_candidates = np.where(mask[ky_pick])[0]
    if kx_candidates.size == 0:
        kx_candidates = np.arange(kx.size, dtype=int)
    kx_pick = kx_candidates[int(np.argmin(np.abs(kx[kx_candidates] - kx_pick_target)))]
    return int(ky_pick), int(kx_pick)


def _infer_runtime_nonlinear_steps(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int | None,
) -> int:
    """Infer nonlinear explicit step counts with the same dt ceiling as the integrator."""

    if steps is not None:
        steps_val = int(steps)
    elif bool(cfg.time.fixed_dt):
        steps_val = int(np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12)))
    else:
        # Keep runtime inference aligned with GX-style adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _runtime_external_phi(cfg: RuntimeConfig) -> float | None:
    """Return a GX-style runtime external-phi source if requested."""

    source = str(cfg.expert.source).strip().lower()
    if source in {"", "default"}:
        return None
    if source != "phiext_full":
        raise ValueError(f"unsupported expert.source={cfg.expert.source!r}; expected 'default' or 'phiext_full'")
    return float(cfg.expert.phi_ext)


_slice_gx_diagnostics = slice_gx_diagnostics
_truncate_gx_diagnostics = truncate_gx_diagnostics
_stride_gx_diagnostics = stride_gx_diagnostics
_concat_gx_diagnostics = concat_gx_diagnostics


def run_runtime_linear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    def _resolved_fit_bounds(
        t_arr: np.ndarray,
        tmin_fit: float | None,
        tmax_fit: float | None,
    ) -> tuple[float | None, float | None]:
        if t_arr.size == 0:
            return None, None
        tmin_use = float(tmin_fit) if tmin_fit is not None else float(t_arr[0])
        tmax_use = float(tmax_fit) if tmax_fit is not None else float(t_arr[-1])
        return tmin_use, tmax_use

    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if _runtime_model_key(cfg) == "cetg":
        geom = build_runtime_geometry(cfg)
        validate_cetg_runtime_config(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        _status("building spectral grid")
        grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
        grid_full = build_spectral_grid(grid_cfg)
        ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
        grid = select_ky_grid(grid_full, ky_index)
        sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        _status(f"selected ky index {ky_index} at ky={float(grid.ky[0]):.4f}")
        _status("building initial condition")
        g0 = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=sel.ky_index,
            kx_index=sel.kx_index,
            Nl=Nl_use,
            Nm=Nm_use,
            nspecies=1,
        )
        cetg_terms = build_runtime_term_config(cfg)
        cetg_params = build_cetg_model_params(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        solver_key = _normalize_linear_solver_name(solver)
        if solver_key == "krylov":
            raise NotImplementedError("solver='krylov' is not implemented for physics.reduced_model='cetg'")
        if solver_key not in {"auto", "time", "gx_time"}:
            raise ValueError("solver must be one of {'auto', 'time', 'explicit_time', 'gx_time', 'krylov'}")
        dt_val = float(cfg.time.dt if dt is None else dt)
        if dt_val <= 0.0:
            raise ValueError("dt must be > 0")
        steps_val = int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_val))
        if steps_val < 1:
            raise ValueError("steps must be >= 1")
        sample_stride_use = int(cfg.time.sample_stride if sample_stride is None else sample_stride)
        _status(f"running cETG time integration over {steps_val} steps")
        _t, diag, G_final, _fields = integrate_cetg_gx_diagnostics_state(
            g0,
            grid,
            cetg_params,
            cetg_terms,
            dt=dt_val,
            steps=steps_val,
            method=str(method or cfg.time.method),
            sample_stride=sample_stride_use,
            diagnostics_stride=1,
            gx_real_fft=bool(cfg.time.gx_real_fft),
            omega_ky_index=0,
            omega_kx_index=0,
            fixed_dt=bool(cfg.time.fixed_dt),
            dt_min=float(cfg.time.dt_min),
            dt_max=cfg.time.dt_max,
            cfl=float(cfg.time.cfl),
            cfl_fac=cfg.time.cfl_fac,
        )
        signal = np.asarray(diag.phi_mode_t if diag.phi_mode_t is not None else np.zeros_like(np.asarray(diag.t)))
        t_arr = np.asarray(diag.t, dtype=float)
        fit_window_tmin: float | None = None
        fit_window_tmax: float | None = None
        _status(f"integration complete; fitting growth rate from {t_arr.size} saved samples")
        if t_arr.size < 2:
            gamma = float(np.asarray(diag.gamma_t)[-1]) if np.asarray(diag.gamma_t).size else 0.0
            omega = float(np.asarray(diag.omega_t)[-1]) if np.asarray(diag.omega_t).size else 0.0
        elif auto_window:
            gamma, omega, fit_window_tmin, fit_window_tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
            fit_window_tmin, fit_window_tmax = _resolved_fit_bounds(t_arr, tmin, tmax)
        _status(f"fit complete: gamma={float(gamma):.6f} omega={float(omega):.6f}")
        return RuntimeLinearResult(
            ky=float(grid.ky[0]),
            gamma=float(gamma),
            omega=float(omega),
            selection=sel,
            t=t_arr,
            signal=np.asarray(signal),
            state=np.asarray(G_final) if return_state else None,
            fit_window_tmin=fit_window_tmin,
            fit_window_tmax=fit_window_tmax,
            fit_signal_used="phi",
        )

    geom = build_runtime_geometry(cfg)
    _status("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = build_spectral_grid(grid_cfg)
    _status("building runtime linear parameters")
    params = build_runtime_linear_params(cfg, Nm=Nm_use, geom=geom)
    terms = build_runtime_linear_terms(cfg)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[sel.ky_index]):.4f}")
    _status("building initial condition")
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl_use,
        Nm=Nm_use,
        nspecies=max(len([s for s in cfg.species if s.kinetic]), 1),
    )

    solver_key = _normalize_linear_solver_name(solver)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    def _run_krylov() -> tuple[float, float]:
        _status("starting Krylov solve")
        kcfg = krylov_cfg or _runtime_default_krylov_config(cfg)
        _status("building linear cache")
        cache = build_linear_cache(grid, geom, params, Nl_use, Nm_use)
        eig, _vec = dominant_eigenpair(
            g0,
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
            shift=kcfg.shift,
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
            status_callback=_status,
        )
        gamma = float(jnp.real(eig))
        omega = float(-jnp.imag(eig))
        gamma, omega = apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        _status(f"Krylov solve complete: gamma={gamma:.6f} omega={omega:.6f}")
        return gamma, omega

    def _run_time() -> RuntimeLinearResult:
        _status(f"starting time integration path with fit_signal={fit_key}")
        tcfg = cfg.time
        if method is not None:
            tcfg = replace(tcfg, method=str(method))
        if dt is not None:
            tcfg = replace(tcfg, dt=float(dt))
        if steps is not None:
            tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
        if sample_stride is not None:
            tcfg = replace(tcfg, sample_stride=int(sample_stride))
        if return_state and solver_key == "gx_time":
            raise ValueError("return_state is not supported with solver='gx_time'")
        if return_state:
            tcfg = replace(tcfg, save_state=True)

        need_density = fit_key in {"density", "auto"}
        g_last = None
        if tcfg.use_diffrax:
            _status(
                f"running diffrax integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
            )
            save_field = "phi+density" if need_density else "phi"
            # Keep the full field history on the diffrax path. The downstream
            # runtime fitting and eigenfunction extraction logic expects
            # ``phi_t`` / ``density_t`` with shape ``(t, ky, kx, z)``, while the
            # diffrax mode-save path only supports scalar mode traces for
            # ``z_index`` / ``max`` extraction.
            save_mode = None
            g_last, saved = integrate_linear_from_config(
                g0,
                grid,
                geom,
                params,
                tcfg,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
                show_progress=show_progress,
            )
            if need_density:
                phi_t, density_t = saved
            else:
                phi_t, density_t = saved, None
        else:
            if need_density:
                _status(
                    f"running diagnostics integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
                )
                _diag = integrate_linear_diagnostics(
                    g0,
                    grid,
                    geom,
                    params,
                    dt=tcfg.dt,
                    steps=int(round(tcfg.t_max / tcfg.dt)),
                    method=tcfg.method,
                    terms=terms,
                    sample_stride=tcfg.sample_stride,
                    species_index=0,
                    record_hl_energy=False,
                    show_progress=show_progress,
                )
                g_last = _diag[0]
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _status(
                    f"running cached linear integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
                )
                g_last, phi_t = integrate_linear_from_config(
                    g0,
                    grid,
                    geom,
                    params,
                    tcfg,
                    terms=terms,
                    save_mode=sel,
                    mode_method=mode_method,
                    save_field="phi",
                    show_progress=show_progress,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = float(tcfg.dt) * float(tcfg.sample_stride) * (
            np.arange(phi_t_np.shape[0], dtype=float) + 1.0
        )
        density_np = None if density_t is None else np.asarray(density_t)
        _status(f"integration complete; fitting growth rate from {t_arr.size} saved samples")

        signal_out: np.ndarray | None = None
        z_out: np.ndarray | None = np.asarray(grid.z, dtype=float)
        eigenfunction_out: np.ndarray | None = None
        fit_window_tmin: float | None = None
        fit_window_tmax: float | None = None
        fit_signal_used: str | None = None
        if fit_key == "auto":
            phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            gamma_phi, omega_phi, phi_tmin, phi_tmax, r2_phi, r2p_phi = fit_growth_rate_auto_with_stats(
                t_arr,
                phi_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            best_gamma, best_omega = gamma_phi, omega_phi
            signal_out = np.asarray(phi_signal)
            fit_window_tmin, fit_window_tmax = phi_tmin, phi_tmax
            fit_signal_used = "phi"
            best_score = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            if density_np is not None:
                dens_signal = extract_mode_time_series(density_np, sel, method=mode_method)
                gamma_den, omega_den, den_tmin, den_tmax, r2_den, r2p_den = fit_growth_rate_auto_with_stats(
                    t_arr,
                    dens_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
                score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
                if score_den > best_score:
                    best_gamma, best_omega = gamma_den, omega_den
                    signal_out = np.asarray(dens_signal)
                    fit_window_tmin, fit_window_tmax = den_tmin, den_tmax
                    fit_signal_used = "density"
            gamma, omega = best_gamma, best_omega
            _status(f"automatic fit selected signal '{fit_signal_used}'")
        else:
            signal = extract_mode_time_series(
                density_np if fit_key == "density" and density_np is not None else phi_t_np,
                sel,
                method=mode_method,
            )
            signal_out = np.asarray(signal)
            fit_signal_used = "density" if fit_key == "density" and density_np is not None else "phi"
            if auto_window:
                gamma, omega, fit_window_tmin, fit_window_tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            else:
                gamma, omega = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
                fit_window_tmin, fit_window_tmax = _resolved_fit_bounds(t_arr, tmin, tmax)
        try:
            eigenfunction_out = np.asarray(
                extract_eigenfunction(
                    phi_t_np,
                    t_arr,
                    sel,
                    z=z_out,
                    method="svd",
                    tmin=fit_window_tmin,
                    tmax=fit_window_tmax,
                )
            )
        except Exception:
            eigenfunction_out = None
        gamma, omega = apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        _status(f"fit complete: gamma={gamma:.6f} omega={omega:.6f}")
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]),
            gamma=float(gamma),
            omega=float(omega),
            selection=sel,
            t=t_arr,
            signal=signal_out,
            state=None if g_last is None or not return_state else np.asarray(g_last),
            z=z_out if eigenfunction_out is not None else None,
            eigenfunction=eigenfunction_out,
            fit_window_tmin=fit_window_tmin,
            fit_window_tmax=fit_window_tmax,
            fit_signal_used=fit_signal_used,
        )

    if solver_key == "krylov":
        gamma, omega = _run_krylov()
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel
        )
    if solver_key == "auto":
        result = _run_time()
        if not _is_valid_growth(result.gamma, result.omega):
            _status("time-path result rejected; falling back to Krylov solve")
            gamma, omega = _run_krylov()
            return RuntimeLinearResult(
                ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel
            )
        return result

    return _run_time()


def run_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: Sequence[float],
    *,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    batch_ky: bool = False,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    show_progress: bool = False,
) -> RuntimeLinearScanResult:
    """Run a ky scan using the unified runtime config path.

    When ``batch_ky`` is enabled, all ky points are integrated together using
    the time integrator (Krylov is not supported in this mode).
    """

    ky_arr = np.asarray(ky_values, dtype=float)
    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    solver_key = _normalize_linear_solver_name(solver)
    if batch_ky and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if batch_ky:
        return _run_runtime_scan_batch(
            cfg,
            ky_arr,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            fit_signal=fit_signal,
            show_progress=show_progress,
        )
    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    for i, ky in enumerate(ky_arr):
        res = run_runtime_linear(
            cfg,
            ky_target=float(ky),
            Nl=Nl_use,
            Nm=Nm_use,
            solver=solver,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            krylov_cfg=krylov_cfg,
            mode_method=mode_method,
            fit_signal=fit_signal,
            show_progress=show_progress,
        )
        gamma[i] = float(res.gamma)
        omega[i] = float(res.omega)
    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


def _run_runtime_scan_batch(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
) -> RuntimeLinearScanResult:
    """Batch a ky scan using one time integration over the full grid."""

    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = build_runtime_linear_terms(cfg)

    ky_indices = np.asarray([select_ky_index(np.asarray(grid.ky), ky) for ky in ky_arr], dtype=int)
    nspecies = max(len([s for s in cfg.species if s.kinetic]), 1)

    g0 = None
    for ky_idx in ky_indices:
        g0_local = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=int(ky_idx),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            nspecies=nspecies,
        )
        g0 = g0_local if g0 is None else g0 + g0_local
    if g0 is None:
        raise ValueError("No ky values provided for batch scan")

    tcfg = cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))

    steps_val = int(round(tcfg.t_max / tcfg.dt))
    diag = integrate_linear_diagnostics(
        g0,
        grid,
        geom,
        params,
        dt=tcfg.dt,
        steps=steps_val,
        method=tcfg.method,
        terms=terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
        show_progress=show_progress,
    )
    phi_t = diag[1]
    density_t = diag[2]
    phi_t_np = np.asarray(phi_t)
    dens_t_np = np.asarray(density_t)
    t_arr = float(tcfg.dt) * float(tcfg.sample_stride) * (
        np.arange(phi_t_np.shape[0], dtype=float) + 1.0
    )

    gamma = np.zeros_like(ky_arr, dtype=float)
    omega = np.zeros_like(ky_arr, dtype=float)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    for i, ky_idx in enumerate(ky_indices):
        sel = ModeSelection(ky_index=int(ky_idx), kx_index=0, z_index=_midplane_index(grid))
        if fit_key == "auto":
            phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = fit_growth_rate_auto_with_stats(
                t_arr,
                phi_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            dens_signal = extract_mode_time_series(dens_t_np, sel, method=mode_method)
            gamma_den, omega_den, _, _, r2_den, r2p_den = fit_growth_rate_auto_with_stats(
                t_arr,
                dens_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            score_phi = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
            g_val, o_val = (gamma_phi, omega_phi) if score_phi >= score_den else (gamma_den, omega_den)
        else:
            signal = extract_mode_time_series(
                dens_t_np if fit_key == "density" else phi_t_np, sel, method=mode_method
            )
            if auto_window:
                g_val, o_val, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            else:
                g_val, o_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)

        g_val, o_val = apply_diagnostic_normalization(
            g_val,
            o_val,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        gamma[i] = float(g_val)
        omega[i] = float(o_val)

    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


def run_runtime_nonlinear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeNonlinearResult:
    """Run a nonlinear point using the unified runtime config path."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    progress_kw = {"show_progress": True} if show_progress else {}
    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if _runtime_model_key(cfg) == "cetg":
        geom = build_runtime_geometry(cfg)
        validate_cetg_runtime_config(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        _status("building spectral grid")
        grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
        grid = build_spectral_grid(grid_cfg)
        ky_index, kx_index = _select_nonlinear_mode_indices(
            grid,
            ky_target=ky_target,
            kx_target=kx_target,
            use_dealias_mask=bool(cfg.time.nonlinear_dealias),
        )
        _status(
            f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} kx={float(np.asarray(grid.kx[kx_index])):.6g}"
        )
        _status("building initial condition")
        G0 = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=Nl_use,
            Nm=Nm_use,
            nspecies=1,
        )
        dt_val = float(cfg.time.dt if dt is None else dt)
        if dt_val <= 0.0:
            raise ValueError("dt must be > 0")
        cetg_params = build_cetg_model_params(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        cetg_term_cfg = build_runtime_term_config(cfg)
        sample_stride_use = cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        diag_stride = cfg.time.diagnostics_stride if diagnostics_stride is None else int(diagnostics_stride)
        diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
        _status(
            f"nonlinear diagnostics={'on' if diagnostics_on else 'off'} sample_stride={int(sample_stride_use)} diagnostics_stride={int(diag_stride)}"
        )
        adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
        if adaptive_chunked:
            chunk_steps = 1024
            G_chunk = G0

            def _run_cetg_chunk(chunk_show_progress: bool):
                nonlocal G_chunk
                kwargs: dict[str, Any] = dict(
                    dt=dt_val,
                    steps=chunk_steps,
                    method=str(method or cfg.time.method),
                    sample_stride=1,
                    diagnostics_stride=1,
                    gx_real_fft=bool(cfg.time.gx_real_fft),
                    omega_ky_index=int(ky_index),
                    omega_kx_index=int(kx_index),
                    fixed_dt=False,
                    dt_min=float(cfg.time.dt_min),
                    dt_max=cfg.time.dt_max,
                    cfl=float(cfg.time.cfl),
                    cfl_fac=cfg.time.cfl_fac,
                )
                if chunk_show_progress:
                    kwargs["show_progress"] = True
                t_chunk, diag_chunk, G_next, fields_next = integrate_cetg_gx_diagnostics_state(
                    G_chunk,
                    grid,
                    cetg_params,
                    cetg_term_cfg,
                    **kwargs,
                )
                G_chunk = G_next
                return t_chunk, diag_chunk, G_next, fields_next

            chunk_result = run_adaptive_gx_chunk_loop(
                integrate_chunk=_run_cetg_chunk,
                t_max=float(cfg.time.t_max),
                chunk_steps=chunk_steps,
                label="cETG",
                show_progress=show_progress,
                status_callback=_status,
            )
            diag = chunk_result.diagnostics
            G_final = chunk_result.state
            cetg_fields_final = chunk_result.fields
            return build_runtime_nonlinear_result(
                t=np.asarray(diag.t),
                diagnostics=diag,
                fields=cetg_fields_final,
                state=np.asarray(G_final) if return_state else None,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
                summarize_fields=diagnostics_on is False,
            )

        steps_val = int(round(float(cfg.time.t_max) / dt_val)) if steps is None else int(steps)
        if steps_val < 1:
            raise ValueError("steps must be >= 1")
        _status(f"running cETG nonlinear integration over {steps_val} steps with dt={dt_val:.6g}")
        _t, diag, G_final, cetg_fields_final = integrate_cetg_gx_diagnostics_state(
            G0,
            grid,
            cetg_params,
            cetg_term_cfg,
            dt=dt_val,
            steps=steps_val,
            method=str(method or cfg.time.method),
            sample_stride=int(sample_stride_use),
            diagnostics_stride=int(diag_stride),
            gx_real_fft=bool(cfg.time.gx_real_fft),
            omega_ky_index=int(ky_index),
            omega_kx_index=int(kx_index),
            fixed_dt=bool(cfg.time.fixed_dt),
            dt_min=float(cfg.time.dt_min),
            dt_max=cfg.time.dt_max,
            cfl=float(cfg.time.cfl),
            cfl_fac=cfg.time.cfl_fac,
            **progress_kw,
        )
        if diagnostics_on is False:
            _status("diagnostics disabled; returning final cETG state summary")
        return build_runtime_nonlinear_result(
            t=np.asarray(diag.t),
            diagnostics=diag,
            fields=cetg_fields_final,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
            summarize_fields=diagnostics_on is False,
        )

    geom = build_runtime_geometry(cfg)
    _status("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    _status("building runtime nonlinear parameters")
    params = build_runtime_linear_params(cfg, Nm=Nm_use, geom=geom)
    term_cfg = build_runtime_term_config(cfg)

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    _status(
        f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} kx={float(np.asarray(grid.kx[kx_index])):.6g}"
    )
    _status("building initial condition")
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl_use,
        Nm=Nm_use,
        nspecies=len(_species_to_linear(cfg.species)),
    )

    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
    steps_val = _infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)

    fixed_mode_on = bool(cfg.expert.fixed_mode)
    fixed_ky_index = cfg.expert.iky_fixed
    fixed_kx_index = cfg.expert.ikx_fixed
    external_phi = _runtime_external_phi(cfg)
    source_on = external_phi is not None
    fixed_ky_index_use: int | None = None
    fixed_kx_index_use: int | None = None
    if fixed_mode_on:
        if fixed_ky_index is None or fixed_kx_index is None:
            raise ValueError("expert.iky_fixed and expert.ikx_fixed must be set when expert.fixed_mode=true")
        fixed_ky_index_use = int(fixed_ky_index)
        fixed_kx_index_use = int(fixed_kx_index)

    diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
    _status(
        f"nonlinear diagnostics={'on' if diagnostics_on else 'off'} fixed_mode={'on' if fixed_mode_on else 'off'} source={cfg.expert.source}"
    )
    if diagnostics_on or fixed_mode_on or return_state or adaptive_chunked or source_on:
        sample_stride_use = cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        diag_stride = cfg.time.diagnostics_stride if diagnostics_stride is None else int(diagnostics_stride)
        laguerre_mode_use = cfg.time.laguerre_nonlinear_mode if laguerre_mode is None else str(laguerre_mode)
        _status(
            f"sample_stride={int(sample_stride_use)} diagnostics_stride={int(diag_stride)} laguerre_mode={laguerre_mode_use}"
        )
        if adaptive_chunked:
            chunk_steps = min(steps_val, 1024)
            G_chunk = G0

            def _run_nonlinear_chunk(chunk_show_progress: bool):
                nonlocal G_chunk
                kwargs: dict[str, Any] = dict(
                    dt=dt_val,
                    steps=chunk_steps,
                    method=str(method or cfg.time.method),
                    terms=term_cfg,
                    sample_stride=1,
                    diagnostics_stride=1,
                    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                    laguerre_mode=laguerre_mode_use,
                    omega_ky_index=int(ky_index),
                    omega_kx_index=int(kx_index),
                    flux_scale=float(cfg.normalization.flux_scale),
                    wphi_scale=float(cfg.normalization.wphi_scale),
                    fixed_dt=False,
                    dt_min=float(cfg.time.dt_min),
                    dt_max=cfg.time.dt_max,
                    cfl=float(cfg.time.cfl),
                    cfl_fac=resolve_cfl_fac(str(method or cfg.time.method), cfg.time.cfl_fac),
                    collision_split=bool(cfg.time.collision_split),
                    collision_scheme=str(cfg.time.collision_scheme),
                    implicit_restart=int(cfg.time.implicit_restart),
                    implicit_solve_method=str(cfg.time.implicit_solve_method),
                    implicit_preconditioner=cfg.time.implicit_preconditioner,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                    external_phi=external_phi,
                )
                if chunk_show_progress:
                    kwargs["show_progress"] = True
                t_chunk, diag_chunk, G_next, fields_next = integrate_nonlinear_gx_diagnostics_state(
                    G_chunk,
                    grid,
                    geom,
                    params,
                    **kwargs,
                )
                G_chunk = G_next
                return t_chunk, diag_chunk, G_next, fields_next

            chunk_result = run_adaptive_gx_chunk_loop(
                integrate_chunk=_run_nonlinear_chunk,
                t_max=float(cfg.time.t_max),
                chunk_steps=chunk_steps,
                label="nonlinear",
                show_progress=show_progress,
                status_callback=_status,
                diagnostics_stride=max(int(sample_stride_use), int(diag_stride), 1),
            )
            diag = chunk_result.diagnostics
            t = jnp.asarray(diag.t)
            G_final = chunk_result.state
            fields_final = chunk_result.fields
        else:
            _status(f"running nonlinear diagnostics integrator over {steps_val} steps with dt={dt_val:.6g}")
            if show_progress:
                t, diag, G_final, fields_final = integrate_nonlinear_gx_diagnostics_state(
                    G0,
                    grid,
                    geom,
                    params,
                    dt=dt_val,
                    steps=steps_val,
                    method=str(method or cfg.time.method),
                    terms=term_cfg,
                    sample_stride=int(sample_stride_use),
                    diagnostics_stride=int(diag_stride),
                    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                    laguerre_mode=laguerre_mode_use,
                    omega_ky_index=int(ky_index),
                    omega_kx_index=int(kx_index),
                    flux_scale=float(cfg.normalization.flux_scale),
                    wphi_scale=float(cfg.normalization.wphi_scale),
                    fixed_dt=bool(cfg.time.fixed_dt),
                    dt_min=float(cfg.time.dt_min),
                    dt_max=cfg.time.dt_max,
                    cfl=float(cfg.time.cfl),
                    cfl_fac=resolve_cfl_fac(str(method or cfg.time.method), cfg.time.cfl_fac),
                    collision_split=bool(cfg.time.collision_split),
                    collision_scheme=str(cfg.time.collision_scheme),
                    implicit_restart=int(cfg.time.implicit_restart),
                    implicit_solve_method=str(cfg.time.implicit_solve_method),
                    implicit_preconditioner=cfg.time.implicit_preconditioner,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                    external_phi=external_phi,
                    show_progress=True,
                )
            else:
                t, diag, G_final, fields_final = integrate_nonlinear_gx_diagnostics_state(
                    G0,
                    grid,
                    geom,
                    params,
                    dt=dt_val,
                    steps=steps_val,
                    method=str(method or cfg.time.method),
                    terms=term_cfg,
                    sample_stride=int(sample_stride_use),
                    diagnostics_stride=int(diag_stride),
                    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                    laguerre_mode=laguerre_mode_use,
                    omega_ky_index=int(ky_index),
                    omega_kx_index=int(kx_index),
                    flux_scale=float(cfg.normalization.flux_scale),
                    wphi_scale=float(cfg.normalization.wphi_scale),
                    fixed_dt=bool(cfg.time.fixed_dt),
                    dt_min=float(cfg.time.dt_min),
                    dt_max=cfg.time.dt_max,
                    cfl=float(cfg.time.cfl),
                    cfl_fac=resolve_cfl_fac(str(method or cfg.time.method), cfg.time.cfl_fac),
                    collision_split=bool(cfg.time.collision_split),
                    collision_scheme=str(cfg.time.collision_scheme),
                    implicit_restart=int(cfg.time.implicit_restart),
                    implicit_solve_method=str(cfg.time.implicit_solve_method),
                    implicit_preconditioner=cfg.time.implicit_preconditioner,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                    external_phi=external_phi,
                )
        if diagnostics_on:
            _status(f"completed nonlinear run with {int(np.asarray(t).size)} saved samples")
            state_out = np.asarray(G_final) if return_state else None
            return build_runtime_nonlinear_result(
                t=np.asarray(t),
                diagnostics=diag,
                fields=fields_final,
                state=state_out,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
                summarize_fields=False,
            )
        if fields_final is None:
            raise RuntimeError("adaptive nonlinear runtime did not produce final fields")
        _status("diagnostics disabled; returning final nonlinear field summary")
        return build_runtime_nonlinear_result(
            t=np.asarray([]),
            diagnostics=None,
            fields=fields_final,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
            summarize_fields=True,
        )

    # Diagnostics disabled: use the config-driven integrator for final state.
    _status(f"diagnostics disabled; running final-state nonlinear integrator over {steps_val} steps with dt={dt_val:.6g}")
    t_cfg = replace(cfg.time, dt=dt_val, t_max=dt_val * steps_val)
    if show_progress:
        G_final, fields = integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
            show_progress=True,
        )
    else:
        G_final, fields = integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
        )
    _status("completed nonlinear final-state integration")
    return build_runtime_nonlinear_result(
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=np.asarray(G_final) if return_state else None,
        ky_selected=float(np.asarray(grid.ky[ky_index])),
        kx_selected=float(np.asarray(grid.kx[kx_index])),
        summarize_fields=True,
    )


_RUNTIME_CASE_FIT_KEYS = {
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
    "fit_signal",
}


def run_linear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    show_progress: bool = True,
) -> int:
    """Run a linear case from a runtime TOML with optional overrides."""

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.runtime_artifacts import write_runtime_linear_artifacts

    cfg, raw = load_runtime_from_toml(config_path)
    run_cfg = dict(raw.get("run", {}))
    fit_cfg = {k: v for k, v in raw.get("fit", {}).items() if k in _RUNTIME_CASE_FIT_KEYS}

    result = run_runtime_linear(
        cfg,
        ky_target=float(ky if ky is not None else run_cfg.get("ky", 0.3)),
        Nl=int(Nl if Nl is not None else run_cfg.get("Nl", 24)),
        Nm=int(Nm if Nm is not None else run_cfg.get("Nm", 12)),
        solver=str(solver if solver is not None else run_cfg.get("solver", "auto")),
        method=method if method is not None else run_cfg.get("method", None),
        dt=dt if dt is not None else run_cfg.get("dt", None),
        steps=steps if steps is not None else run_cfg.get("steps", None),
        sample_stride=sample_stride if sample_stride is not None else raw.get("time", {}).get("sample_stride", None),
        show_progress=show_progress,
        **fit_cfg,
    )
    if cfg.output.path:
        paths = write_runtime_linear_artifacts(cfg.output.path, result)
        if "summary" in paths:
            print(f"saved {paths['summary']}")
    print(f"ky={result.ky:.6f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
    return 0


def run_nonlinear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    show_progress: bool = True,
) -> int:
    """Run a nonlinear case from a runtime TOML with optional overrides."""

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.runtime_artifacts import run_runtime_nonlinear_with_artifacts, write_runtime_nonlinear_artifacts

    cfg, raw = load_runtime_from_toml(config_path)
    run_cfg = dict(raw.get("run", {}))
    time_cfg = dict(raw.get("time", {}))

    def _status(message: str) -> None:
        print(f"runtime: {message}")

    ky_target = float(ky if ky is not None else run_cfg.get("ky", 0.3))
    Nl_use = int(Nl if Nl is not None else run_cfg.get("Nl", 4))
    Nm_use = int(Nm if Nm is not None else run_cfg.get("Nm", 8))
    method_use = method if method is not None else run_cfg.get("method", None)
    dt_use = dt if dt is not None else time_cfg.get("dt", None)
    steps_use = steps if steps is not None else run_cfg.get("steps", None)
    sample_stride_use = sample_stride if sample_stride is not None else time_cfg.get("sample_stride", None)
    diagnostics_stride_use = diagnostics_stride if diagnostics_stride is not None else time_cfg.get("diagnostics_stride", None)

    if cfg.output.path:
        result, paths = run_runtime_nonlinear_with_artifacts(
            cfg,
            out=cfg.output.path,
            ky_target=ky_target,
            Nl=Nl_use,
            Nm=Nm_use,
            dt=dt_use,
            steps=steps_use,
            method=method_use,
            sample_stride=sample_stride_use,
            diagnostics_stride=diagnostics_stride_use,
            diagnostics=True,
            show_progress=show_progress,
            status_callback=_status,
        )
        if "summary" in paths:
            print(f"saved {paths['summary']}")
    else:
        result = run_runtime_nonlinear(
            cfg,
            ky_target=ky_target,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method_use,
            dt=dt_use,
            steps=steps_use,
            sample_stride=sample_stride_use,
            diagnostics_stride=diagnostics_stride_use,
            diagnostics=True,
            show_progress=show_progress,
            status_callback=_status,
        )
    if result.diagnostics is None or result.ky_selected is None:
        print("completed without streamed diagnostics")
        return 0
    diag = result.diagnostics
    print(
        "ky={:.6f} Wg={:.8e} Wphi={:.8e} heat={:.8e} pflux={:.8e}".format(
            float(result.ky_selected),
            float(diag.Wg_t[-1]),
            float(diag.Wphi_t[-1]),
            float(diag.heat_flux_t[-1]),
            float(diag.particle_flux_t[-1]),
        )
    )
    return 0
