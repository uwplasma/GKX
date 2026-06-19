"""Solver-path policies for the Cyclone single-mode linear benchmark."""

from __future__ import annotations

from dataclasses import replace
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
from spectraxgk.validation.benchmarks.fit_signals import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.solvers.time.explicit import ExplicitTimeConfig, integrate_linear_explicit
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import linear_terms_to_term_config
from spectraxgk.solvers.linear.krylov import dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached

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
    gamma_seed = 0.0
    omega_seed = 0.0
    seed_ok = False
    omega_ok = False
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    if kcfg.shift is None:
        try:
            status("estimating frequency seed with short explicit time march")
            t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
            time_cfg = ExplicitTimeConfig(
                dt=float(kcfg.power_dt),
                t_max=t_seed,
                sample_stride=1,
                fixed_dt=True,
            )
            t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
                fresh_G0(),
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
                sel,
                navg_fraction=0.5,
                mode_method="z_index",
            )
            omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
            seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
        except Exception:
            seed_ok = False
            omega_ok = False

        if not seed_ok:
            try:
                status("primary seed failed; retrying reduced Hermite-Laguerre seed")
                Nl_seed = min(Nl, 16)
                Nm_seed = min(Nm, 12)
                cache_seed = build_linear_cache(grid, geom, params, Nl_seed, Nm_seed)
                G0_seed = _build_initial_condition(
                    grid,
                    geom,
                    ky_index=sel.ky_index,
                    kx_index=sel.kx_index,
                    Nl=Nl_seed,
                    Nm=Nm_seed,
                    init_cfg=init_cfg,
                )
                t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
                time_cfg = ExplicitTimeConfig(
                    dt=float(kcfg.power_dt),
                    t_max=t_seed,
                    sample_stride=1,
                    fixed_dt=True,
                )
                t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
                    jnp.asarray(np.asarray(G0_seed)),
                    grid,
                    cache_seed,
                    params,
                    geom,
                    time_cfg,
                    terms=terms,
                    mode_method="z_index",
                    show_progress=show_progress,
                )
                sel_seed = ModeSelection(
                    ky_index=0,
                    kx_index=0,
                    z_index=_midplane_index(grid),
                )
                gamma_seed, omega_seed, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
                    phi_t,
                    t_short,
                    sel_seed,
                    navg_fraction=0.5,
                    mode_method="z_index",
                )
                omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
            except Exception:
                seed_ok = False
                omega_ok = False

    shift = None
    if omega_ok:
        shift = complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
    status("running dominant eigenpair solve")
    eig, vec = dominant_eigenpair(
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
    term_cfg = linear_terms_to_term_config(terms)
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    phi_t_out = np.asarray(phi)[None, ...]
    t_out = np.array([0.0])
    gamma_out = float(np.real(eig))
    omega_out = float(-np.imag(eig))
    if seed_ok:
        seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
        if seed_strong:
            omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
            gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
            use_seed = (
                not np.isfinite(gamma_out)
                or not np.isfinite(omega_out)
                or (gamma_seed > 0.0 and gamma_out < 0.0)
                or abs(omega_out - omega_seed) > omega_tol
                or abs(gamma_out - gamma_seed) > gamma_tol
            )
            if use_seed:
                gamma_out = float(gamma_seed)
                omega_out = float(omega_seed)
    if kcfg.omega_sign != 0:
        omega_out = float(np.sign(kcfg.omega_sign)) * abs(omega_out)
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    status(f"Krylov solve complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return gamma_out, omega_out, phi_t_out, t_out


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

    status(f"starting time integration path with fit_signal={fit_key}")
    method_key = method.lower()
    time_cfg_use = None
    if time_cfg is not None:
        time_cfg_use = replace(time_cfg, dt=float(dt), t_max=float(dt) * int(steps))
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    elif cfg.time.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        time_cfg_use = replace(cfg.time, dt=float(dt), t_max=float(dt) * int(steps))
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)

    phi_t: Any
    if reference_aligned:
        status("running reference-aligned explicit integrator")
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

    density_t: Any | None
    if time_cfg_use is not None:
        status(
            f"running runtime-configured integrator over {int(steps)} steps with sample_stride={int(time_cfg_use.sample_stride)}"
        )
        if need_density:
            status("saving phi and density diagnostics for automatic fit selection")
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
        else:
            _, phi_t = integrate_linear_from_config(
                fresh_G0(),
                grid,
                geom,
                params,
                time_cfg_use,
                terms=terms,
                show_progress=show_progress,
            )
            density_t = None
        stride = time_cfg_use.sample_stride
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        if need_density or not use_jit:
            status(
                f"running explicit diagnostics integrator over {int(steps)} steps with sample_stride={stride}"
            )
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
        else:
            status(
                f"running cached linear integrator over {int(steps)} steps with sample_stride={stride}"
            )
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
            phi_t = phi_out_time
            density_t = None

    phi_t_np = np.asarray(phi_t)
    t_arr = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    status(f"integration complete; fitting growth rate from {phi_t_np.shape[0]} saved samples")
    auto_fit_kwargs: dict[str, Any] = {
        "tmin": tmin,
        "tmax": tmax,
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
        "max_amp_fraction": max_amp_fraction,
        "window_method": window_method,
        "max_fraction": max_fraction,
        "end_fraction": end_fraction,
        "num_windows": 8,
        "phase_weight": phase_weight,
        "length_weight": length_weight,
        "min_r2": min_r2,
        "late_penalty": late_penalty,
        "min_slope": min_slope,
        "min_slope_frac": min_slope_frac,
        "slope_var_weight": slope_var_weight,
    }
    if fit_key == "auto":
        _signal, name, gamma_out, omega_out = _select_fit_signal_auto(
            t_arr,
            phi_t_np,
            density_np,
            sel,
            mode_method=mode_method,
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
            fit_signal=fit_key,
            mode_method=mode_method,
        )
        if auto_window and tmin is None and tmax is None:
            gamma_out, omega_out, _tmin, _tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                **auto_fit_kwargs,
            )
        else:
            gamma_out, omega_out = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    status(f"time integration fit complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return float(gamma_out), float(omega_out), phi_t_np, t_arr
