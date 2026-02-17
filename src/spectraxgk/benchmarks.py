"""Benchmark utilities for linear Cyclone base case comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
    TimeConfig,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, integrate_linear
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.species import Species, build_linear_params


CYCLONE_OMEGA_D_SCALE = 1.0
CYCLONE_OMEGA_STAR_SCALE = 1.0
CYCLONE_RHO_STAR = 1.0

ETG_OMEGA_D_SCALE = 1.0
ETG_OMEGA_STAR_SCALE = 1.0
ETG_RHO_STAR = 1.0

Kinetic_OMEGA_D_SCALE = 1.0
Kinetic_OMEGA_STAR_SCALE = 1.0
Kinetic_RHO_STAR = 1.0

TEM_OMEGA_D_SCALE = 1.0
TEM_OMEGA_STAR_SCALE = 1.0
TEM_RHO_STAR = 1.0

KBM_OMEGA_D_SCALE = 1.0
KBM_OMEGA_STAR_SCALE = 1.0
KBM_RHO_STAR = 1.0

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
class LinearRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


@dataclass(frozen=True)
class LinearScanResult:
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


def load_cyclone_reference_kinetic() -> CycloneReference:
    """Load Cyclone base case reference data (kinetic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_reference_kinetic.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_kbm_reference() -> CycloneReference:
    """Load KBM reference data (finite beta, kinetic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "kbm_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_etg_reference() -> CycloneReference:
    """Load ETG reference data digitized from the GX paper."""

    data_path = resources.files("spectraxgk").joinpath("data", "etg_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_tem_reference() -> CycloneReference:
    """Load TEM reference data digitized from the literature."""

    data_path = resources.files("spectraxgk").joinpath("data", "tem_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def _two_species_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
) -> LinearParams:
    """Build LinearParams for a two-species kinetic model (ions + electrons)."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")

    nu_i = float(getattr(model, "nu_i", 0.0))
    nu_e = float(getattr(model, "nu_e", 0.0))
    beta = float(getattr(model, "beta", 1.0e-5))
    if beta_override is not None:
        beta = float(beta_override)

    ion = Species(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=float(getattr(model, "R_over_LTi", model.R_over_LTe)),
        fprim=float(model.R_over_Ln),
        nu=nu_i,
    )
    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.R_over_LTe),
        fprim=float(model.R_over_Ln),
        nu=nu_e,
    )
    params = build_linear_params(
        [ion, electron],
        tau_e=0.0,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=1.0 if beta > 0.0 else 0.0,
    )
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params


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
    min_amp_fraction: float = 0.0,
    mode_method: str = "svd",
    terms: LinearTerms | None = None,
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
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, terms=terms
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
            min_amp_fraction=min_amp_fraction,
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
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
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
    min_amp_fraction: float = 0.0,
    mode_method: str = "svd",
    terms: LinearTerms | None = None,
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
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    for i, ky in enumerate(ky_values):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
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
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
        )

        phi_t_np = np.asarray(phi_t)
        t = np.arange(steps_i) * dt_i
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
                min_amp_fraction=min_amp_fraction,
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
    time_cfg: TimeConfig | None = None,
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
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    cfg = cfg or ETGBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    ns = 2
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    if time_cfg is not None:
        dt = float(time_cfg.dt)
        steps = int(round(time_cfg.t_max / time_cfg.dt))
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        _, phi_t = integrate_linear_from_config(
            G0_jax,
            grid,
            geom,
            params,
            time_cfg,
            cache=cache,
            terms=terms,
        )
    else:
        _, phi_t = integrate_linear(
            G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, terms=terms
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
            min_amp_fraction=min_amp_fraction,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

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
    min_amp_fraction: float = 0.0,
    mode_method: str = "project",
    terms: LinearTerms | None = None,
) -> LinearScanResult:
    """Run an ETG linear benchmark for a list of ky values."""

    cfg = cfg or ETGBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    for i, ky in enumerate(ky_values):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        _, phi_t = integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
        )

        phi_t_np = np.asarray(phi_t)
        t = np.arange(steps_i) * dt_i
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
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_kinetic_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
) -> LinearRunResult:
    """Run a kinetic-electron ITG/TEM benchmark and extract growth rate."""

    cfg = cfg or KineticElectronBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    ns = 2
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, terms=terms
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
            min_amp_fraction=min_amp_fraction,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_kinetic_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
) -> LinearScanResult:
    """Run a kinetic-electron ITG/TEM benchmark for a list of ky values."""

    cfg = cfg or KineticElectronBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(hypercollisions=0.0)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    for i, ky in enumerate(ky_values):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        _, phi_t = integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
        )

        phi_t_np = np.asarray(phi_t)
        t = np.arange(steps_i) * dt_i
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
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_tem_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: TEMBaseCase | None = None,
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
) -> LinearRunResult:
    """Run the TEM benchmark and extract growth rate."""

    cfg = cfg or TEMBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE / float(cfg.geometry.R0),
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0, hypercollisions=0.0)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    ns = 2
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(
        G0_jax, grid, geom, params, dt=dt, steps=steps, method=method, terms=terms
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
            min_amp_fraction=min_amp_fraction,
        )
    else:
        gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_tem_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: TEMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
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
) -> LinearScanResult:
    """Run the TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or TEMBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE / float(cfg.geometry.R0),
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0, hypercollisions=0.0)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    gammas = []
    omegas = []
    ky_out = []
    for i, ky in enumerate(ky_values):
        if time_cfg is not None:
            dt_i = float(time_cfg.dt)
            steps_i = int(round(time_cfg.t_max / time_cfg.dt))
        else:
            dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        ky_index = select_ky_index(np.asarray(grid.ky), float(ky))
        sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        if time_cfg is not None:
            _, phi_t = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg,
                cache=cache,
                terms=terms,
            )
        else:
            _, phi_t = integrate_linear(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt_i,
                steps=steps_i,
                method=method,
                cache=cache,
                terms=terms,
            )

        phi_t_np = np.asarray(phi_t)
        t = np.arange(steps_i) * dt_i
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
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(grid.ky[sel.ky_index]))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_kbm_beta_scan(
    betas: np.ndarray,
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "rk4",
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
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
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KBMBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if terms is None:
        terms = LinearTerms(bpar=0.0, hypercollisions=0.0)

    gammas = []
    omegas = []
    beta_out = []
    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    for i, beta in enumerate(betas):
        if time_cfg is not None:
            dt_i = float(time_cfg.dt)
            steps_i = int(round(time_cfg.t_max / time_cfg.dt))
        else:
            dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=float(beta),
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0[1, 0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

        G0_jax = jnp.asarray(G0)
        if time_cfg is not None:
            _, phi_t = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg,
                cache=cache,
                terms=terms,
            )
        else:
            _, phi_t = integrate_linear(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt_i,
                steps=steps_i,
                method=method,
                cache=cache,
                terms=terms,
            )

        phi_t_np = np.asarray(phi_t)
        t = np.arange(steps_i) * dt_i
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
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

        gammas.append(gamma)
        omegas.append(omega)
        beta_out.append(float(beta))

    return LinearScanResult(ky=np.array(beta_out), gamma=np.array(gammas), omega=np.array(omegas))
