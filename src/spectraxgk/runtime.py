"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import ModeSelection, extract_mode_time_series, fit_growth_rate, fit_growth_rate_auto, select_ky_index
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import apply_diagnostic_normalization, get_normalization_contract
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import TermConfig


@dataclass(frozen=True)
class RuntimeLinearResult:
    """Result container for runtime linear runs."""

    ky: float
    gamma: float
    omega: float
    selection: ModeSelection
    t: np.ndarray | None = None
    signal: np.ndarray | None = None


@dataclass(frozen=True)
class RuntimeLinearScanResult:
    """Result container for runtime linear ky scans."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _species_to_linear(species_cfg: Sequence[RuntimeSpeciesConfig]) -> list[Species]:
    kinetic = [s for s in species_cfg if bool(s.kinetic)]
    if not kinetic:
        raise ValueError("RuntimeConfig.species must include at least one kinetic species")
    return [
        Species(
            charge=float(s.charge),
            mass=float(s.mass),
            density=float(s.density),
            temperature=float(s.temperature),
            tprim=float(s.tprim),
            fprim=float(s.fprim),
            nu=float(s.nu),
        )
        for s in kinetic
    ]


def build_runtime_linear_params(cfg: RuntimeConfig) -> LinearParams:
    """Build ``LinearParams`` from a unified runtime config."""

    geom = SAlphaGeometry.from_config(cfg.geometry)
    contract = get_normalization_contract(cfg.normalization.contract)
    rho_star = contract.rho_star if cfg.normalization.rho_star is None else float(cfg.normalization.rho_star)
    omega_d_scale = (
        contract.omega_d_scale if cfg.normalization.omega_d_scale is None else float(cfg.normalization.omega_d_scale)
    )
    omega_star_scale = (
        contract.omega_star_scale
        if cfg.normalization.omega_star_scale is None
        else float(cfg.normalization.omega_star_scale)
    )

    species = _species_to_linear(cfg.species)
    has_kinetic_electron = any(float(s.charge) < 0.0 for s in species)
    if cfg.physics.adiabatic_electrons and has_kinetic_electron:
        raise ValueError("adiabatic_electrons=True conflicts with kinetic electron species")

    tau_e = float(cfg.physics.tau_e) if cfg.physics.adiabatic_electrons else 0.0
    beta = float(cfg.physics.beta) if cfg.physics.electromagnetic else 0.0
    fapar = 1.0 if (cfg.physics.electromagnetic and cfg.physics.use_apar and beta > 0.0) else 0.0

    params = build_linear_params(
        species,
        tau_e=tau_e,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=float(omega_d_scale),
        omega_star_scale=float(omega_star_scale),
        rho_star=float(rho_star),
        beta=beta,
        fapar=fapar,
        nu_hyper=float(cfg.collisions.nu_hyper),
        p_hyper=float(cfg.collisions.p_hyper),
        hypercollisions_const=float(cfg.collisions.hypercollisions_const),
        hypercollisions_kz=float(cfg.collisions.hypercollisions_kz),
    )
    return replace(
        params,
        nu_hermite=float(cfg.collisions.nu_hermite),
        nu_laguerre=float(cfg.collisions.nu_laguerre),
        damp_ends_amp=float(cfg.collisions.damp_ends_amp),
        damp_ends_widthfrac=float(cfg.collisions.damp_ends_widthfrac),
    )


def build_runtime_linear_terms(cfg: RuntimeConfig) -> LinearTerms:
    """Build ``LinearTerms`` from unified toggles."""

    em_on = bool(cfg.physics.electromagnetic)
    use_apar = em_on and bool(cfg.physics.use_apar)
    use_bpar = em_on and bool(cfg.physics.use_bpar)
    collisions_on = bool(cfg.physics.collisions)
    hyper_on = bool(cfg.physics.hypercollisions)
    return LinearTerms(
        streaming=float(cfg.terms.streaming),
        mirror=float(cfg.terms.mirror),
        curvature=float(cfg.terms.curvature),
        gradb=float(cfg.terms.gradb),
        diamagnetic=float(cfg.terms.diamagnetic),
        collisions=float(cfg.terms.collisions if collisions_on else 0.0),
        hypercollisions=float(cfg.terms.hypercollisions if hyper_on else 0.0),
        end_damping=float(cfg.terms.end_damping),
        apar=float(cfg.terms.apar if use_apar else 0.0),
        bpar=float(cfg.terms.bpar if use_bpar else 0.0),
    )


def build_runtime_term_config(cfg: RuntimeConfig) -> TermConfig:
    """Build nonlinear-ready ``TermConfig`` from unified toggles."""

    lin_terms = build_runtime_linear_terms(cfg)
    nonlinear_on = float(cfg.terms.nonlinear if cfg.physics.nonlinear else 0.0)
    return TermConfig(
        streaming=lin_terms.streaming,
        mirror=lin_terms.mirror,
        curvature=lin_terms.curvature,
        gradb=lin_terms.gradb,
        diamagnetic=lin_terms.diamagnetic,
        collisions=lin_terms.collisions,
        hypercollisions=lin_terms.hypercollisions,
        end_damping=lin_terms.end_damping,
        apar=lin_terms.apar,
        bpar=lin_terms.bpar,
        nonlinear=nonlinear_on,
    )


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    width: float,
    envelope_constant: float,
    envelope_sine: float,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    env = envelope_constant + envelope_sine * np.sin(z - theta0)
    return env * np.exp(-((z - theta0) / width) ** 2)


def _build_initial_condition(
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    cfg: RuntimeConfig,
    *,
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
    nspecies: int,
) -> jnp.ndarray:
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    init_field = cfg.init.init_field.lower()
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )
    if cfg.init.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be > 0")

    g0 = np.zeros((nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(cfg.init.init_amp)
    ky_val = float(grid.ky[ky_index])
    if ky_val == 0.0:
        return jnp.asarray(g0)

    if cfg.init.gaussian_init:
        profile = _build_gaussian_profile(
            np.asarray(grid.z),
            kx=float(grid.kx[kx_index]),
            ky=ky_val,
            s_hat=float(geom.s_hat),
            width=float(cfg.init.gaussian_width),
            envelope_constant=float(cfg.init.gaussian_envelope_constant),
            envelope_sine=float(cfg.init.gaussian_envelope_sine),
        )
        vals = amp * profile * (1.0 + 1.0j)
    else:
        vals = amp * (1.0 + 1.0j) * np.ones_like(np.asarray(grid.z))

    species_index = 0 if nspecies == 1 else nspecies - 1
    if init_field == "all":
        for l_idx, m_idx in field_map.values():
            if l_idx < Nl and m_idx < Nm:
                g0[species_index, l_idx, m_idx, ky_index, kx_index, :] = vals
    else:
        l_idx, m_idx = field_map[init_field]
        if l_idx >= Nl or m_idx >= Nm:
            raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
        g0[species_index, l_idx, m_idx, ky_index, kx_index, :] = vals
    return jnp.asarray(g0)


def run_runtime_linear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    Nl: int = 24,
    Nm: int = 12,
    solver: str = "krylov",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
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
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_runtime_linear_params(cfg)
    terms = build_runtime_linear_terms(cfg)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=max(len([s for s in cfg.species if s.kinetic]), 1),
    )

    solver_key = solver.strip().lower()
    if solver_key == "krylov":
        kcfg = krylov_cfg or KrylovConfig()
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
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
        )
        gamma = float(jnp.real(eig))
        omega = float(-jnp.imag(eig))
        gamma, omega = apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        return RuntimeLinearResult(ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel)

    tcfg = cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))

    out = integrate_linear_from_config(
        g0,
        grid,
        geom,
        params,
        tcfg,
        terms=terms,
        save_mode=sel,
        mode_method=mode_method,
        save_field="phi",
    )
    saved = np.asarray(out[1])
    if saved.ndim == 1:
        signal = saved
    else:
        signal = extract_mode_time_series(saved, sel, method=mode_method)
    t = float(tcfg.dt) * float(tcfg.sample_stride) * (np.arange(signal.shape[0], dtype=float) + 1.0)
    if auto_window:
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
    gamma, omega = apply_diagnostic_normalization(
        gamma,
        omega,
        rho_star=float(np.asarray(params.rho_star)),
        diagnostic_norm=cfg.normalization.diagnostic_norm,
    )
    return RuntimeLinearResult(
        ky=float(grid.ky[sel.ky_index]),
        gamma=float(gamma),
        omega=float(omega),
        selection=sel,
        t=t,
        signal=np.asarray(signal),
    )


def run_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: Sequence[float],
    *,
    Nl: int = 24,
    Nm: int = 12,
    solver: str = "krylov",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
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
) -> RuntimeLinearScanResult:
    """Run a ky scan using the unified runtime config path."""

    ky_arr = np.asarray(ky_values, dtype=float)
    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    for i, ky in enumerate(ky_arr):
        res = run_runtime_linear(
            cfg,
            ky_target=float(ky),
            Nl=Nl,
            Nm=Nm,
            solver=solver,
            method=method,
            dt=dt,
            steps=steps,
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
        )
        gamma[i] = float(res.gamma)
        omega[i] = float(res.omega)
    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)
