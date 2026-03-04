"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    select_ky_index,
)
from spectraxgk.diagnostics import GXDiagnostics
from spectraxgk.geometry import SAlphaGeometry, gx_twist_shift_params
from spectraxgk.grids import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.nonlinear import integrate_nonlinear_gx_diagnostics
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import apply_diagnostic_normalization, get_normalization_contract
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.runners import integrate_linear_from_config, integrate_nonlinear_from_config
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import FieldState, TermConfig


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


@dataclass(frozen=True)
class RuntimeNonlinearResult:
    """Result container for runtime nonlinear runs."""

    t: np.ndarray
    diagnostics: GXDiagnostics | None
    phi2: np.ndarray | None = None
    fields: FieldState | None = None


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
        nu_hyper_l=float(cfg.collisions.nu_hyper_l),
        nu_hyper_m=float(cfg.collisions.nu_hyper_m),
        nu_hyper_lm=float(cfg.collisions.nu_hyper_lm),
        p_hyper_l=float(cfg.collisions.p_hyper_l),
        p_hyper_m=float(cfg.collisions.p_hyper_m),
        p_hyper_lm=float(cfg.collisions.p_hyper_lm),
        D_hyper=float(cfg.collisions.D_hyper),
        p_hyper_kperp=float(cfg.collisions.p_hyper_kperp),
        hypercollisions_const=float(cfg.collisions.hypercollisions_const),
        hypercollisions_kz=float(cfg.collisions.hypercollisions_kz),
    )
    return replace(
        params,
        nu_hermite=float(cfg.collisions.nu_hermite),
        nu_laguerre=float(cfg.collisions.nu_laguerre),
        damp_ends_amp=(
            float(cfg.collisions.damp_ends_amp) / float(cfg.time.dt)
            if cfg.collisions.damp_ends_scale_by_dt and float(cfg.time.dt) != 0.0
            else float(cfg.collisions.damp_ends_amp)
        ),
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
        hyperdiffusion=float(cfg.terms.hyperdiffusion),
        end_damping=float(cfg.terms.end_damping),
        apar=float(cfg.terms.apar if use_apar else 0.0),
        bpar=float(cfg.terms.bpar if use_bpar else 0.0),
    )


def build_runtime_term_config(cfg: RuntimeConfig) -> TermConfig:
    """Build nonlinear-ready ``TermConfig`` from unified toggles."""

    lin_terms = build_runtime_linear_terms(cfg)
    nonlinear_on = float(cfg.terms.nonlinear if cfg.physics.nonlinear else 0.0)
    return linear_terms_to_term_config(lin_terms, nonlinear=nonlinear_on)


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


def _reshape_gx_state(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    nR = nyc * nx * nz
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr = arr[..., idxyz.ravel()]
    return arr.reshape((nspec, nl, nm, nyc, nx, nz))


def _expand_ky(arr: np.ndarray, *, nyc: int) -> np.ndarray:
    ny_full = 2 * (nyc - 1)
    if ny_full <= 0 or arr.shape[-3] == ny_full:
        return arr
    if nyc <= 2:
        return arr
    pos = arr
    neg = np.conj(pos[..., 1 : nyc - 1, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


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

    if cfg.init.init_file is not None:
        path = Path(cfg.init.init_file)
        raw = np.fromfile(path, dtype=np.complex64)
        ny = grid.ky.size
        nx = grid.kx.size
        nz = grid.z.size
        nyc = ny // 2 + 1
        expected_nyc = nspecies * Nl * Nm * nyc * nx * nz
        expected_full = nspecies * Nl * Nm * ny * nx * nz
        if raw.size == expected_nyc:
            arr = _reshape_gx_state(raw, nspec=nspecies, nl=Nl, nm=Nm, nyc=nyc, nx=nx, nz=nz)
            arr = _expand_ky(arr, nyc=nyc)
            return jnp.asarray(arr)
        if raw.size == expected_full:
            arr = raw.reshape((nspecies, Nl, Nm, ny, nx, nz))
            return jnp.asarray(arr)
        raise ValueError(
            f"init_file size {raw.size} does not match expected {expected_nyc} (nyc) or {expected_full} (full)"
        )

    g0 = np.zeros((nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(cfg.init.init_amp)
    ky_val = float(grid.ky[ky_index])
    if ky_val == 0.0:
        return jnp.asarray(g0)

    z = np.asarray(grid.z)
    if cfg.init.gaussian_init:
        profile = _build_gaussian_profile(
            z,
            kx=float(grid.kx[kx_index]),
            ky=ky_val,
            s_hat=float(geom.s_hat),
            width=float(cfg.init.gaussian_width),
            envelope_constant=float(cfg.init.gaussian_envelope_constant),
            envelope_sine=float(cfg.init.gaussian_envelope_sine),
        )
        vals = amp * profile * (1.0 + 1.0j)
    else:
        vals = amp * (1.0 + 1.0j) * np.ones_like(z)

    species_index = 0 if nspecies == 1 else nspecies - 1
    if cfg.init.gaussian_init and not cfg.init.init_single:
        ny = grid.ky.size
        nx = grid.kx.size
        dealias = np.asarray(grid.dealias_mask)
        ky_indices = np.where(np.asarray(grid.ky) > 0.0)[0]
        kx_pos = np.where(np.asarray(grid.kx) >= 0.0)[0]

        def _set_mode(l_idx: int, m_idx: int, ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
            if l_idx >= Nl or m_idx >= Nm:
                return
            g0[species_index, l_idx, m_idx, ky_i, kx_i, :] = vals_k

        for ky_i in ky_indices:
            ky_k = float(grid.ky[ky_i])
            if ky_k == 0.0:
                continue
            for kx_i in kx_pos:
                if not dealias[ky_i, kx_i]:
                    continue
                kx_k = float(grid.kx[kx_i])
                profile_k = _build_gaussian_profile(
                    z,
                    kx=abs(kx_k),
                    ky=ky_k,
                    s_hat=float(geom.s_hat),
                    width=float(cfg.init.gaussian_width),
                    envelope_constant=float(cfg.init.gaussian_envelope_constant),
                    envelope_sine=float(cfg.init.gaussian_envelope_sine),
                )
                vals_k = amp * profile_k * (1.0 + 1.0j)
                if init_field == "all":
                    for l_idx, m_idx in field_map.values():
                        _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)
                else:
                    l_idx, m_idx = field_map[init_field]
                    _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)

                if kx_i == 0:
                    continue
                kx_neg = int(np.argmin(np.abs(np.asarray(grid.kx) + kx_k)))
                if kx_neg == kx_i:
                    continue
                if init_field == "all":
                    for l_idx, m_idx in field_map.values():
                        _set_mode(l_idx, m_idx, ky_i, kx_neg, vals_k)
                else:
                    l_idx, m_idx = field_map[init_field]
                    _set_mode(l_idx, m_idx, ky_i, kx_neg, vals_k)
    elif not cfg.init.init_single and not cfg.init.gaussian_init:
        rng = np.random.default_rng(int(cfg.init.random_seed))
        z_min = float(z.min())
        z_max = float(z.max())
        Zp = (z_max - z_min) / (2.0 * np.pi) if z_max > z_min else 1.0
        kpar = float(cfg.init.kpar_init)
        z_phase = np.cos(kpar * z / Zp)
        ny = grid.ky.size
        nx = grid.kx.size
        ky_mask = np.asarray(grid.ky) > 0.0
        kx_mask = np.asarray(grid.kx) >= 0.0
        dealias = np.asarray(grid.dealias_mask)
        ky_indices = np.where(ky_mask)[0]
        kx_indices = np.where(kx_mask)[0]
        l_idx, m_idx = field_map[init_field]
        if l_idx >= Nl or m_idx >= Nm:
            raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
        for ky_i in ky_indices:
            for kx_i in kx_indices:
                if not dealias[ky_i, kx_i]:
                    continue
                ra = amp * (rng.random() - 0.5)
                rb = amp * (rng.random() - 0.5)
                vals_k = (ra + 1j * rb) * z_phase
                g0[species_index, l_idx, m_idx, ky_i, kx_i, :] = vals_k
                if kx_i != 0:
                    kx_neg = nx - kx_i
                    vals_neg = (rb + 1j * ra) * z_phase
                    g0[species_index, l_idx, m_idx, ky_i, kx_neg, :] = vals_neg
    else:
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
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_cfg = cfg.grid
    if grid_cfg.boundary == "linked" and not grid_cfg.non_twist:
        jtwist, x0 = gx_twist_shift_params(geom, grid_cfg)
        grid_cfg = replace(grid_cfg, Lx=2.0 * np.pi * x0, jtwist=jtwist)
    grid_full = build_spectral_grid(grid_cfg)
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
        return gamma, omega

    def _run_time() -> RuntimeLinearResult:
        tcfg = cfg.time
        if method is not None:
            tcfg = replace(tcfg, method=str(method))
        if dt is not None:
            tcfg = replace(tcfg, dt=float(dt))
        if steps is not None:
            tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
        if sample_stride is not None:
            tcfg = replace(tcfg, sample_stride=int(sample_stride))

        need_density = fit_key in {"density", "auto"}
        if tcfg.use_diffrax:
            save_field = "phi+density" if need_density else "phi"
            save_mode = None if need_density else sel
            _g_last, saved = integrate_linear_from_config(
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
            )
            if need_density:
                phi_t, density_t = saved
            else:
                phi_t, density_t = saved, None
        else:
            if need_density:
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
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _g_last, phi_t = integrate_linear_from_config(
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
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = float(tcfg.dt) * float(tcfg.sample_stride) * (
            np.arange(phi_t_np.shape[0], dtype=float) + 1.0
        )
        density_np = None if density_t is None else np.asarray(density_t)

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
            best_gamma, best_omega = gamma_phi, omega_phi
            best_score = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            if density_np is not None:
                dens_signal = extract_mode_time_series(density_np, sel, method=mode_method)
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
                score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
                if score_den > best_score:
                    best_gamma, best_omega = gamma_den, omega_den
            gamma, omega = best_gamma, best_omega
        else:
            signal = extract_mode_time_series(
                density_np if fit_key == "density" and density_np is not None else phi_t_np,
                sel,
                method=mode_method,
            )
            if auto_window:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
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
            t=t_arr,
            signal=None,
        )

    if solver_key == "krylov":
        gamma, omega = _run_krylov()
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel
        )
    if solver_key == "auto":
        result = _run_time()
        if not _is_valid_growth(result.gamma, result.omega):
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
    Nl: int = 24,
    Nm: int = 12,
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
) -> RuntimeLinearScanResult:
    """Run a ky scan using the unified runtime config path.

    When ``batch_ky`` is enabled, all ky points are integrated together using
    the time integrator (Krylov is not supported in this mode).
    """

    ky_arr = np.asarray(ky_values, dtype=float)
    solver_key = solver.strip().lower()
    if batch_ky and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if batch_ky:
        return _run_runtime_scan_batch(
            cfg,
            ky_arr,
            Nl=Nl,
            Nm=Nm,
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
        )
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
) -> RuntimeLinearScanResult:
    """Batch a ky scan using one time integration over the full grid."""

    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_cfg = cfg.grid
    if grid_cfg.boundary == "linked" and not grid_cfg.non_twist:
        jtwist, x0 = gx_twist_shift_params(geom, grid_cfg)
        grid_cfg = replace(grid_cfg, Lx=2.0 * np.pi * x0, jtwist=jtwist)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg)
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
    Nl: int = 24,
    Nm: int = 12,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
) -> RuntimeNonlinearResult:
    """Run a nonlinear point using the unified runtime config path."""

    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_cfg = cfg.grid
    if grid_cfg.boundary == "linked" and not grid_cfg.non_twist:
        jtwist, x0 = gx_twist_shift_params(geom, grid_cfg)
        grid_cfg = replace(grid_cfg, Lx=2.0 * np.pi * x0, jtwist=jtwist)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg)
    term_cfg = build_runtime_term_config(cfg)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    kx_index = 0
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=len(_species_to_linear(cfg.species)),
    )

    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    if steps is None:
        if not cfg.time.fixed_dt:
            dt_cap = cfg.time.dt_max if cfg.time.dt_max is not None else dt_val * 5.0
            steps_val = int(np.ceil(cfg.time.t_max / max(dt_cap, 1.0e-12)))
        else:
            steps_val = int(round(cfg.time.t_max / cfg.time.dt))
    else:
        steps_val = int(steps)
    if steps_val < 1:
        raise ValueError("steps must be >= 1")

    diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
    if diagnostics_on:
        sample_stride_use = cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        diag_stride = cfg.time.diagnostics_stride if diagnostics_stride is None else int(diagnostics_stride)
        laguerre_mode_use = cfg.time.laguerre_nonlinear_mode if laguerre_mode is None else str(laguerre_mode)
        t, diag = integrate_nonlinear_gx_diagnostics(
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
            cfl_fac=float(cfg.time.cfl_fac),
            collision_split=bool(cfg.time.collision_split),
            collision_scheme=str(cfg.time.collision_scheme),
            implicit_restart=int(cfg.time.implicit_restart),
            implicit_solve_method=str(cfg.time.implicit_solve_method),
            implicit_preconditioner=cfg.time.implicit_preconditioner,
        )
        return RuntimeNonlinearResult(
            t=np.asarray(t),
            diagnostics=diag,
            phi2=None,
            fields=None,
        )

    # Diagnostics disabled: use the config-driven integrator for final state.
    t_cfg = replace(cfg.time, dt=dt_val, t_max=dt_val * steps_val)
    G_final, fields = integrate_nonlinear_from_config(
        G0,
        grid,
        geom,
        params,
        t_cfg,
        terms=term_cfg,
    )
    phi2 = np.asarray(jnp.mean(jnp.abs(fields.phi) ** 2))
    return RuntimeNonlinearResult(
        t=np.asarray([]),
        diagnostics=None,
        phi2=phi2,
        fields=fields,
    )
