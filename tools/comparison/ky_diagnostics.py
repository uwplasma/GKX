"""Multi-method diagnostics for a single ky mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gkx.diagnostics.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate_auto,
    fit_growth_rate_with_stats,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from gkx.benchmarking.shared import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    _apply_reference_hypercollisions,
    _build_initial_condition,
    _two_species_params,
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
)
from gkx.config import (
    CycloneBaseCase,
    InitializationConfig,
    KBMBaseCase,
)
from gkx.solvers.time.diffrax_linear import integrate_linear_diffrax
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers.linear.integrators import integrate_linear_diagnostics
from gkx.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from gkx.terms.assembly import compute_fields_cached
from gkx.terms.config import TermConfig
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-method diagnostics for a single ky."
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=["cyclone", "kinetic", "etg", "kbm", "tem"],
        help="Benchmark case to run.",
    )
    parser.add_argument("--ky", type=float, required=True, help="Target ky.")
    parser.add_argument("--beta", type=float, default=None, help="Beta for KBM.")
    parser.add_argument("--Nl", type=int, default=48, help="Hermite resolution.")
    parser.add_argument("--Nm", type=int, default=16, help="Laguerre resolution.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument("--steps", type=int, default=15000, help="Number of steps.")
    parser.add_argument("--sample-stride", type=int, default=5, help="Sampling stride.")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method labels or 'all'.",
    )
    parser.add_argument(
        "--mode-method",
        choices=["z_index", "max", "project"],
        default="project",
        help="Mode extraction for fits.",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="Z index for mode extraction (default: mid-plane).",
    )
    parser.add_argument(
        "--window-start-fraction",
        type=float,
        default=0.1,
        help="Fit window start fraction.",
    )
    parser.add_argument(
        "--window-max-fraction",
        type=float,
        default=0.7,
        help="Fit window max fraction.",
    )
    parser.add_argument(
        "--window-end-fraction",
        type=float,
        default=1.0,
        help="Fit window end fraction.",
    )
    parser.add_argument(
        "--window-min-amp-fraction",
        type=float,
        default=0.05,
        help="Fit window min amplitude fraction.",
    )
    parser.add_argument(
        "--window-max-amp-fraction",
        type=float,
        default=0.9,
        help="Fit window max amplitude fraction.",
    )
    parser.add_argument(
        "--window-growth-weight",
        type=float,
        default=0.0,
        help="Fit window growth-rate weight.",
    )
    parser.add_argument(
        "--window-late-penalty",
        type=float,
        default=0.1,
        help="Fit window late penalty.",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.0,
        help="Minimum R2 for window selection.",
    )
    parser.add_argument(
        "--navg-fraction",
        type=float,
        default=0.5,
        help="Fraction of samples to skip for GX averaging.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "docs" / "_static" / "diagnostics"),
        help="Output directory.",
    )
    return parser.parse_args()


def _midplane_index(z: np.ndarray) -> int:
    if z.size <= 1:
        return 0
    idx = int(z.size // 2 + 1)
    return min(idx, int(z.size) - 1)


def _load_reference(case: str):
    if case == "cyclone":
        return load_cyclone_reference()
    if case == "kinetic":
        return load_cyclone_reference_kinetic()
    if case == "etg":
        return load_etg_reference()
    if case == "kbm":
        return load_kbm_reference()
    if case == "tem":
        return load_tem_reference()
    raise ValueError(f"Unknown case '{case}'")


def _reference_at(case: str, ky: float) -> tuple[float, float]:
    data = _load_reference(case)
    idx = int(np.argmin(np.abs(np.asarray(data.ky) - ky)))
    return float(data.gamma[idx]), float(data.omega[idx])


def _build_problem(case: str, ky: float, beta: float | None, Nl: int, Nm: int):
    init_species_index = 0
    if case == "cyclone":
        cfg = CycloneBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = LinearParams(
            fprim=cfg.model.fprim,
            tprim=cfg.model.tprim_i,
            tprim_e=cfg.model.tprim_e,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=REFERENCE_DAMP_ENDS_AMP,
            damp_ends_widthfrac=REFERENCE_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_reference_hypercollisions(params, nhermite=Nm)
        terms = LinearTerms()
    elif case == "kinetic":
        cfg, _raw = load_runtime_from_toml(
            ROOT
            / "examples"
            / "linear"
            / "axisymmetric"
            / "runtime_kinetic_electron.toml"
        )
        grid_full = build_spectral_grid(cfg.grid)
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
        terms = build_runtime_linear_terms(cfg)
        init_species_index = 1
    elif case == "etg":
        cfg, _ = load_runtime_from_toml(ROOT / "examples/linear/axisymmetric/etg.toml")
        grid_full = build_spectral_grid(cfg.grid)
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
        terms = build_runtime_linear_terms(cfg)
        init_species_index = 0
    elif case == "kbm":
        cfg = KBMBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        beta_val = float(beta) if beta is not None else 0.015
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=beta_val,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
        terms = LinearTerms(bpar=0.0)
        init_species_index = 1
    elif case == "tem":
        cfg, _raw = load_runtime_from_toml(
            ROOT / "examples" / "linear" / "axisymmetric" / "runtime_tem.toml"
        )
        grid_full = build_spectral_grid(cfg.grid)
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
        terms = build_runtime_linear_terms(cfg)
        init_species_index = 1
    else:
        raise ValueError(f"Unknown case '{case}'")

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky)
    grid = select_ky_grid(grid_full, ky_index)
    init_cfg = getattr(cfg, "init", None) or InitializationConfig()
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
    )
    ns = int(np.atleast_1d(np.asarray(params.charge_sign)).shape[0])
    if ns == 1:
        G0 = G0_single
    else:
        G0 = np.zeros(
            (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return cfg, grid, geom, params, terms, G0


def _energy_timeseries(field_t: np.ndarray) -> np.ndarray:
    if field_t is None:
        return None
    return np.mean(np.abs(field_t) ** 2, axis=(1, 2, 3))


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, 1.0e-300))


def _plot_timeseries(
    outdir: Path,
    prefix: str,
    t: np.ndarray,
    phi_energy: np.ndarray,
    dens_energy: np.ndarray | None,
    window: tuple[float, float] | None,
    ref: tuple[float, float] | None,
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, phi_energy, label=r"$|\phi|^2$")
    if dens_energy is not None:
        ax[0].plot(t, dens_energy, label=r"$|n|^2$")
    if window is not None:
        ax[0].axvspan(
            window[0], window[1], color="orange", alpha=0.25, label="fit window"
        )
    ax[0].set_ylabel("energy")
    ax[0].legend(loc="best")

    ax[1].plot(t, _safe_log(phi_energy), label=r"$\log|\phi|^2$")
    if dens_energy is not None:
        ax[1].plot(t, _safe_log(dens_energy), label=r"$\log|n|^2$")
    if window is not None:
        ax[1].axvspan(window[0], window[1], color="orange", alpha=0.25)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("log energy")
    ax[1].legend(loc="best")

    title = prefix
    if ref is not None:
        title += f" | ref γ={ref[0]:.4g} ω={ref[1]:.4g}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_timeseries.png", dpi=140)
    plt.close(fig)


def _plot_eigenfunction(
    outdir: Path,
    prefix: str,
    z: np.ndarray,
    eigenfunction: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(z, eigenfunction.real, label="Re")
    ax.plot(z, eigenfunction.imag, label="Im", linestyle="--")
    ax.set_xlabel("theta")
    ax.set_ylabel("eigenfunction")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_eigenfunction.png", dpi=140)
    plt.close(fig)


def _fit_signal(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    start_fraction: float,
    max_fraction: float,
    end_fraction: float,
    min_amp_fraction: float,
    max_amp_fraction: float,
    growth_weight: float,
    late_penalty: float,
    min_r2: float,
):
    gamma, omega, tmin, tmax = fit_growth_rate_auto(
        t,
        signal,
        window_method="loglinear",
        window_fraction=0.3,
        min_points=40,
        start_fraction=start_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        growth_weight=growth_weight,
        phase_weight=0.2,
        length_weight=0.05,
        min_r2=min_r2,
        late_penalty=late_penalty,
        require_positive=False,
        slope_var_weight=0.3,
        min_slope_frac=0.2,
    )
    gamma2, omega2, r2_log, r2_phase = fit_growth_rate_with_stats(
        t,
        signal,
        tmin=tmin,
        tmax=tmax,
    )
    return {
        "gamma": float(gamma2),
        "omega": float(omega2),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "r2_log": float(r2_log),
        "r2_phase": float(r2_phase),
    }


def _run_time_method(
    outdir: Path,
    label: str,
    G0: np.ndarray,
    grid,
    geom,
    params,
    terms,
    *,
    dt: float,
    steps: int,
    sample_stride: int,
    mode_method: str,
    z_index: int,
    fit_cfg: dict,
    navg_fraction: float,
    ref: tuple[float, float] | None,
):
    params_use = params
    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        grid,
        geom,
        params_use,
        dt=dt,
        steps=steps,
        method=label.replace("time-", ""),
        terms=terms,
        sample_stride=sample_stride,
        record_hl_energy=True,
    )
    phi_t_np = np.asarray(phi_t)
    dens_t_np = np.asarray(density_t)
    t = np.arange(phi_t_np.shape[0]) * dt * sample_stride

    sel = ModeSelection(ky_index=0, kx_index=0, z_index=z_index)
    phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    dens_signal = extract_mode_time_series(dens_t_np, sel, method=mode_method)
    phi_fit = _fit_signal(t, phi_signal, **fit_cfg)
    dens_fit = _fit_signal(t, dens_signal, **fit_cfg)

    gx_gamma, gx_omega, gx_gamma_t, gx_omega_t, gx_t = (
        instantaneous_growth_rate_from_phi(
            phi_t_np,
            t,
            sel,
            navg_fraction=navg_fraction,
            mode_method="z_index",
        )
    )

    best = phi_fit if phi_fit["r2_log"] >= dens_fit["r2_log"] else dens_fit
    eigen = extract_eigenfunction(
        phi_t_np,
        t,
        sel,
        z=np.asarray(grid.z),
        method="svd",
        tmin=best["tmin"],
        tmax=best["tmax"],
    )

    phi_energy = _energy_timeseries(phi_t_np)
    dens_energy = _energy_timeseries(dens_t_np)

    prefix = label
    np.save(outdir / f"{prefix}_t.npy", t)
    np.save(outdir / f"{prefix}_phi_t.npy", phi_t_np)
    np.save(outdir / f"{prefix}_density_t.npy", dens_t_np)
    np.save(outdir / f"{prefix}_phi_energy.npy", phi_energy)
    np.save(outdir / f"{prefix}_density_energy.npy", dens_energy)
    np.save(outdir / f"{prefix}_phi_log_energy.npy", _safe_log(phi_energy))
    np.save(outdir / f"{prefix}_density_log_energy.npy", _safe_log(dens_energy))
    np.save(outdir / f"{prefix}_hl_energy.npy", np.asarray(hl_t))
    np.save(outdir / f"{prefix}_gx_gamma_t.npy", gx_gamma_t)
    np.save(outdir / f"{prefix}_gx_omega_t.npy", gx_omega_t)
    np.save(outdir / f"{prefix}_gx_t.npy", gx_t)
    np.save(outdir / f"{prefix}_eigenfunction.npy", eigen)

    _plot_timeseries(
        outdir, prefix, t, phi_energy, dens_energy, (best["tmin"], best["tmax"]), ref
    )
    _plot_eigenfunction(outdir, prefix, np.asarray(grid.z), eigen)

    summary = {
        "method": label,
        "dt": dt,
        "steps": steps,
        "sample_stride": sample_stride,
        "phi_fit": phi_fit,
        "density_fit": dens_fit,
        "gx_avg": {"gamma": float(gx_gamma), "omega": float(gx_omega)},
    }
    return summary


def _run_diffrax_method(
    outdir: Path,
    label: str,
    G0: np.ndarray,
    grid,
    geom,
    params,
    terms,
    *,
    dt: float,
    steps: int,
    sample_stride: int,
    mode_method: str,
    z_index: int,
    fit_cfg: dict,
    navg_fraction: float,
    adaptive: bool,
    ref: tuple[float, float] | None,
):
    params_use = params
    _, phi_t = integrate_linear_diffrax(
        G0,
        grid,
        geom,
        params_use,
        dt=dt,
        steps=steps,
        method=label.replace("diffrax-", ""),
        adaptive=adaptive,
        rtol=1.0e-5,
        atol=1.0e-7,
        max_steps=max(steps * 10, 10000),
        progress_bar=False,
        sample_stride=sample_stride,
        terms=terms,
        save_field="phi",
    )
    _, dens_t = integrate_linear_diffrax(
        G0,
        grid,
        geom,
        params_use,
        dt=dt,
        steps=steps,
        method=label.replace("diffrax-", ""),
        adaptive=adaptive,
        rtol=1.0e-5,
        atol=1.0e-7,
        max_steps=max(steps * 10, 10000),
        progress_bar=False,
        sample_stride=sample_stride,
        terms=terms,
        save_field="density",
    )

    phi_t_np = np.asarray(phi_t)
    dens_t_np = np.asarray(dens_t)
    t = np.arange(phi_t_np.shape[0]) * dt * sample_stride
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=z_index)
    phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    dens_signal = extract_mode_time_series(dens_t_np, sel, method=mode_method)
    phi_fit = _fit_signal(t, phi_signal, **fit_cfg)
    dens_fit = _fit_signal(t, dens_signal, **fit_cfg)

    gx_gamma, gx_omega, gx_gamma_t, gx_omega_t, gx_t = (
        instantaneous_growth_rate_from_phi(
            phi_t_np,
            t,
            sel,
            navg_fraction=navg_fraction,
            mode_method="z_index",
        )
    )

    best = phi_fit if phi_fit["r2_log"] >= dens_fit["r2_log"] else dens_fit
    eigen = extract_eigenfunction(
        phi_t_np,
        t,
        sel,
        z=np.asarray(grid.z),
        method="svd",
        tmin=best["tmin"],
        tmax=best["tmax"],
    )

    phi_energy = _energy_timeseries(phi_t_np)
    dens_energy = _energy_timeseries(dens_t_np)

    prefix = label + ("-adapt" if adaptive else "-fixed")
    np.save(outdir / f"{prefix}_t.npy", t)
    np.save(outdir / f"{prefix}_phi_t.npy", phi_t_np)
    np.save(outdir / f"{prefix}_density_t.npy", dens_t_np)
    np.save(outdir / f"{prefix}_phi_energy.npy", phi_energy)
    np.save(outdir / f"{prefix}_density_energy.npy", dens_energy)
    np.save(outdir / f"{prefix}_phi_log_energy.npy", _safe_log(phi_energy))
    np.save(outdir / f"{prefix}_density_log_energy.npy", _safe_log(dens_energy))
    np.save(outdir / f"{prefix}_gx_gamma_t.npy", gx_gamma_t)
    np.save(outdir / f"{prefix}_gx_omega_t.npy", gx_omega_t)
    np.save(outdir / f"{prefix}_gx_t.npy", gx_t)
    np.save(outdir / f"{prefix}_eigenfunction.npy", eigen)

    _plot_timeseries(
        outdir, prefix, t, phi_energy, dens_energy, (best["tmin"], best["tmax"]), ref
    )
    _plot_eigenfunction(outdir, prefix, np.asarray(grid.z), eigen)

    summary = {
        "method": prefix,
        "dt": dt,
        "steps": steps,
        "sample_stride": sample_stride,
        "adaptive": adaptive,
        "phi_fit": phi_fit,
        "density_fit": dens_fit,
        "gx_avg": {"gamma": float(gx_gamma), "omega": float(gx_omega)},
    }
    return summary


def _run_krylov_method(
    outdir: Path,
    label: str,
    G0: np.ndarray,
    grid,
    geom,
    params,
    terms,
    *,
    dt: float,
    steps: int,
    fit_cfg: dict,
    ref: tuple[float, float] | None,
):
    krylov_cfg = KrylovConfig(method=label.replace("krylov-", ""))
    if G0.ndim != 5:
        raise ValueError(
            "Expected G0 shape (Nl, Nm, Ny, Nx, Nz) for Krylov diagnostics"
        )
    Nl, Nm = G0.shape[0], G0.shape[1]
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    eig, vec = dominant_eigenpair(
        G0,
        cache,
        params,
        terms=terms,
        krylov_dim=krylov_cfg.krylov_dim,
        restarts=krylov_cfg.restarts,
        omega_min_factor=krylov_cfg.omega_min_factor,
        omega_target_factor=krylov_cfg.omega_target_factor,
        omega_cap_factor=krylov_cfg.omega_cap_factor,
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
    )
    term_cfg = TermConfig(
        streaming=terms.streaming,
        mirror=terms.mirror,
        curvature=terms.curvature,
        gradb=terms.gradb,
        diamagnetic=terms.diamagnetic,
        collisions=terms.collisions,
        hypercollisions=terms.hypercollisions,
        end_damping=terms.end_damping,
        apar=terms.apar,
        bpar=terms.bpar,
        nonlinear=0.0,
    )
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    phi_t_np = np.asarray(phi)[None, ...]
    t = np.array([0.0])
    sel = ModeSelection(
        ky_index=0, kx_index=0, z_index=_midplane_index(np.asarray(grid.z))
    )
    eigen = extract_eigenfunction(
        phi_t_np,
        t,
        sel,
        z=np.asarray(grid.z),
        method="snapshot",
    )

    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))

    prefix = label
    np.save(outdir / f"{prefix}_eigenfunction.npy", eigen)
    _plot_eigenfunction(outdir, prefix, np.asarray(grid.z), eigen)

    summary = {
        "method": label,
        "gamma": gamma,
        "omega": omega,
    }
    if ref is not None:
        summary["ref"] = {"gamma": ref[0], "omega": ref[1]}
    return summary


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, grid, geom, params, terms, G0 = _build_problem(
        args.case, args.ky, args.beta, args.Nl, args.Nm
    )

    ky = float(grid.ky[0])
    z_index = (
        args.z_index
        if args.z_index is not None
        else _midplane_index(np.asarray(grid.z))
    )
    mode_method = args.mode_method
    if mode_method not in {"z_index", "max", "project"}:
        mode_method = "z_index"

    ref = _reference_at(args.case, ky)

    method_specs = {
        "time-rk4": ("time", "rk4"),
        "time-imex2": ("time", "imex2"),
        "diffrax-dopri8": ("diffrax", "Dopri8"),
        "diffrax-kencarp4": ("diffrax", "KenCarp4"),
        "krylov-propagator": ("krylov", "propagator"),
        "krylov-power": ("krylov", "power"),
    }
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.methods.lower() == "all":
        requested = list(method_specs.keys())

    fit_cfg = {
        "start_fraction": args.window_start_fraction,
        "max_fraction": args.window_max_fraction,
        "end_fraction": args.window_end_fraction,
        "min_amp_fraction": args.window_min_amp_fraction,
        "max_amp_fraction": args.window_max_amp_fraction,
        "growth_weight": args.window_growth_weight,
        "late_penalty": args.window_late_penalty,
        "min_r2": args.min_r2,
    }
    navg_fraction = args.navg_fraction

    summary_all = {
        "case": args.case,
        "ky": ky,
        "Nl": args.Nl,
        "Nm": args.Nm,
        "dt": args.dt,
        "steps": args.steps,
        "sample_stride": args.sample_stride,
        "reference": {"gamma": ref[0], "omega": ref[1]},
        "methods": {},
    }

    run_dir = outdir / f"{args.case}_ky{ky:.4g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for label in requested:
        if label not in method_specs:
            print(f"Skipping unknown method '{label}'")
            continue
        kind, method_name = method_specs[label]
        print(f"[{args.case}] ky={ky:.4g} running {label} ({kind})")
        if kind == "time":
            summary = _run_time_method(
                run_dir,
                label,
                G0,
                grid,
                geom,
                params,
                terms,
                dt=args.dt,
                steps=args.steps,
                sample_stride=args.sample_stride,
                mode_method=mode_method,
                z_index=z_index,
                fit_cfg=fit_cfg,
                navg_fraction=navg_fraction,
                ref=ref,
            )
        elif kind == "diffrax":
            summary = _run_diffrax_method(
                run_dir,
                label,
                G0,
                grid,
                geom,
                params,
                terms,
                dt=args.dt,
                steps=args.steps,
                sample_stride=args.sample_stride,
                mode_method=mode_method,
                z_index=z_index,
                fit_cfg=fit_cfg,
                navg_fraction=navg_fraction,
                adaptive=False,
                ref=ref,
            )
            summary_adapt = _run_diffrax_method(
                run_dir,
                label,
                G0,
                grid,
                geom,
                params,
                terms,
                dt=args.dt,
                steps=args.steps,
                sample_stride=args.sample_stride,
                mode_method=mode_method,
                z_index=z_index,
                fit_cfg=fit_cfg,
                navg_fraction=navg_fraction,
                adaptive=True,
                ref=ref,
            )
            summary = {"fixed": summary, "adaptive": summary_adapt}
        else:
            summary = _run_krylov_method(
                run_dir,
                label,
                G0,
                grid,
                geom,
                params,
                terms,
                dt=args.dt,
                steps=args.steps,
                fit_cfg=fit_cfg,
                ref=ref,
            )
        summary_all["methods"][label] = summary

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"Wrote diagnostics to {run_dir}")


if __name__ == "__main__":
    main()
