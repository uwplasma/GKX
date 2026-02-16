"""Benchmark utilities for linear Cyclone base case comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from importlib import resources

import jax.numpy as jnp

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_ky_index,
)
from spectraxgk.config import CycloneBaseCase, ETGBaseCase, MTMBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache, integrate_linear


CYCLONE_OMEGA_D_SCALE = 1.0
CYCLONE_OMEGA_STAR_SCALE = 1.0
CYCLONE_RHO_STAR = 1.0

ETG_OMEGA_D_SCALE = 1.0
ETG_OMEGA_STAR_SCALE = 1.0
ETG_RHO_STAR = 1.0

MTM_OMEGA_D_SCALE = 1.0
MTM_OMEGA_STAR_SCALE = 1.0
MTM_RHO_STAR = 1.0

@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class CycloneRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


@dataclass(frozen=True)
class CycloneScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class CycloneComparison:
    ky: float
    gamma: float
    omega: float
    gamma_ref: float
    omega_ref: float
    rel_gamma: float
    rel_omega: float


@dataclass(frozen=True)
class ETGRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


@dataclass(frozen=True)
class ETGScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class MTMRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


@dataclass(frozen=True)
class MTMScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


def load_cyclone_reference() -> CycloneReference:
    """Load Cyclone base case reference data (adiabatic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_reference_adiabatic.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def _electron_gyro_params(Te_over_Ti: float, mass_ratio: float) -> tuple[float, float, float, float]:
    """Return (tau_e, vth, rho, tz) for kinetic electrons."""

    temp_ratio = float(Te_over_Ti)
    mass_ratio = float(mass_ratio)
    if temp_ratio <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    tau_e = 1.0 / temp_ratio
    vth = np.sqrt(temp_ratio * mass_ratio)
    rho = np.sqrt(temp_ratio / mass_ratio)
    tz = -tau_e
    return tau_e, vth, rho, tz


def run_cyclone_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    cfg = cfg or CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = params or LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
    )

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, operator=operator
    )

    phi_t_np = np.asarray(phi_t)
    t = np.arange(steps) * dt
    signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_cyclone_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> CycloneScanResult:
    """Run the linear Cyclone benchmark for a list of ky values."""

    cfg = cfg or CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = params or LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    t = np.arange(steps) * dt
    for ky in ky_values:
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        _, phi_t = integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            cache=cache,
            operator=operator,
        )

        phi_t_np = np.asarray(phi_t)
        signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
        if auto_window and tmin is None and tmax is None:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return CycloneScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def compare_cyclone_to_reference(
    result: CycloneRunResult, reference: CycloneReference
) -> CycloneComparison:
    """Compare a Cyclone run result against the reference data set."""

    idx = int(np.argmin(np.abs(reference.ky - result.ky)))
    gamma_ref = float(reference.gamma[idx])
    omega_ref = float(reference.omega[idx])
    rel_gamma = (result.gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
    rel_omega = (result.omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
    return CycloneComparison(
        ky=float(reference.ky[idx]),
        gamma=result.gamma,
        omega=result.omega,
        gamma_ref=gamma_ref,
        omega_ref=omega_ref,
        rel_gamma=rel_gamma,
        rel_omega=rel_omega,
    )


def run_etg_linear(
    ky_target: float = 3.0,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: ETGBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> ETGRunResult:
    """Run a reduced ETG linear benchmark and extract growth rate."""

    cfg = cfg or ETGBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        tau_e, vth, rho, tz = _electron_gyro_params(cfg.model.Te_over_Ti, cfg.model.mass_ratio)
        params = LinearParams(
            charge_sign=-1.0,
            tau_e=tau_e,
            vth=vth,
            rho=rho,
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=-cfg.model.R_over_LTe,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            tz=tz,
            kpar_scale=float(geom.gradpar()),
        )

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, operator=operator
    )

    phi_t_np = np.asarray(phi_t)
    t = np.arange(steps) * dt
    signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return ETGRunResult(
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
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: ETGBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> ETGScanResult:
    """Run a reduced ETG linear benchmark for a list of ky values."""

    cfg = cfg or ETGBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        tau_e, vth, rho, tz = _electron_gyro_params(cfg.model.Te_over_Ti, cfg.model.mass_ratio)
        params = LinearParams(
            charge_sign=-1.0,
            tau_e=tau_e,
            vth=vth,
            rho=rho,
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=-cfg.model.R_over_LTe,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            tz=tz,
            kpar_scale=float(geom.gradpar()),
        )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    t = np.arange(steps) * dt
    for ky in ky_values:
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        _, phi_t = integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            cache=cache,
            operator=operator,
        )

        phi_t_np = np.asarray(phi_t)
        signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
        if auto_window and tmin is None and tmax is None:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return ETGScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_mtm_linear(
    ky_target: float = 3.0,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: MTMBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> MTMRunResult:
    """Run a reduced MTM linear benchmark and extract growth rate."""

    cfg = cfg or MTMBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        tau_e, vth, rho, tz = _electron_gyro_params(cfg.model.Te_over_Ti, cfg.model.mass_ratio)
        params = LinearParams(
            charge_sign=-1.0,
            tau_e=tau_e,
            vth=vth,
            rho=rho,
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=-cfg.model.R_over_LTe,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=MTM_OMEGA_D_SCALE,
            omega_star_scale=MTM_OMEGA_STAR_SCALE,
            rho_star=MTM_RHO_STAR,
            tz=tz,
            nu=cfg.model.nu,
            kpar_scale=float(geom.gradpar()),
        )

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, operator=operator
    )

    phi_t_np = np.asarray(phi_t)
    t = np.arange(steps) * dt
    signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return MTMRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_mtm_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: MTMBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    mode_method: str = "svd",
    operator: str = "full",
) -> MTMScanResult:
    """Run a reduced MTM linear benchmark for a list of ky values."""

    cfg = cfg or MTMBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        tau_e, vth, rho, tz = _electron_gyro_params(cfg.model.Te_over_Ti, cfg.model.mass_ratio)
        params = LinearParams(
            charge_sign=-1.0,
            tau_e=tau_e,
            vth=vth,
            rho=rho,
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=-cfg.model.R_over_LTe,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=MTM_OMEGA_D_SCALE,
            omega_star_scale=MTM_OMEGA_STAR_SCALE,
            rho_star=MTM_RHO_STAR,
            tz=tz,
            nu=cfg.model.nu,
            kpar_scale=float(geom.gradpar()),
        )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    t = np.arange(steps) * dt
    for ky in ky_values:
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        _, phi_t = integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            cache=cache,
            operator=operator,
        )

        phi_t_np = np.asarray(phi_t)
        signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
        if auto_window and tmin is None and tmax is None:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return MTMScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))
