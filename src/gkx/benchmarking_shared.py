"""Shared benchmark reference data, normalization, and scan policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import resources
from typing import Sequence, TypeVar

import jax.numpy as jnp
import numpy as np

from gkx.config import InitializationConfig
from gkx.core_grid import SpectralGrid
from gkx.operators.linear.params import Species, build_linear_params
from gkx.diagnostics.modes import ModeSelection
from gkx.diagnostics.normalization import (
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
)
from gkx.geometry import FluxTubeGeometryLike
from gkx.operators.linear.params import LinearParams
from gkx.solvers_linear_krylov import KrylovConfig

VALID_FIT_SIGNALS = frozenset({"phi", "density", "auto"})
_Record = TypeVar("_Record")


CYCLONE_OMEGA_D_SCALE = CYCLONE_NORMALIZATION.omega_d_scale

CYCLONE_OMEGA_STAR_SCALE = CYCLONE_NORMALIZATION.omega_star_scale

CYCLONE_RHO_STAR = CYCLONE_NORMALIZATION.rho_star

ETG_OMEGA_D_SCALE = ETG_NORMALIZATION.omega_d_scale

ETG_OMEGA_STAR_SCALE = ETG_NORMALIZATION.omega_star_scale

ETG_RHO_STAR = ETG_NORMALIZATION.rho_star

KINETIC_OMEGA_D_SCALE = KINETIC_NORMALIZATION.omega_d_scale

KINETIC_OMEGA_STAR_SCALE = KINETIC_NORMALIZATION.omega_star_scale

KINETIC_RHO_STAR = KINETIC_NORMALIZATION.rho_star

TEM_OMEGA_D_SCALE = TEM_NORMALIZATION.omega_d_scale

TEM_OMEGA_STAR_SCALE = TEM_NORMALIZATION.omega_star_scale

TEM_RHO_STAR = TEM_NORMALIZATION.rho_star

KBM_OMEGA_D_SCALE = KBM_NORMALIZATION.omega_d_scale

KBM_OMEGA_STAR_SCALE = KBM_NORMALIZATION.omega_star_scale

KBM_RHO_STAR = KBM_NORMALIZATION.rho_star

REFERENCE_NU_HYPER_L = 0.0

REFERENCE_NU_HYPER_M = 1.0

REFERENCE_P_HYPER_L = 6.0

REFERENCE_P_HYPER_M = 20.0

REFERENCE_DAMP_ENDS_AMP = 0.1

REFERENCE_DAMP_ENDS_WIDTHFRAC = 1.0 / 8.0


def _reference_hypercollision_power(nhermite: int | None) -> float:
    if nhermite is None:
        return REFERENCE_P_HYPER_M
    return float(min(REFERENCE_P_HYPER_M, max(int(nhermite) // 2, 1)))


def _apply_reference_hypercollisions(
    params: LinearParams, *, nhermite: int | None = None
) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=REFERENCE_NU_HYPER_L,
        nu_hyper_m=REFERENCE_NU_HYPER_M,
        p_hyper_l=REFERENCE_P_HYPER_L,
        p_hyper_m=_reference_hypercollision_power(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )


def _two_species_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
    nhermite: int | None = None,
) -> LinearParams:
    """Build ``LinearParams`` for a two-species kinetic model."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    ion_fprim_raw = getattr(model, "fprim_i", None)
    ele_fprim_raw = getattr(model, "fprim_e", None)
    ion_fprim = float(model.fprim) if ion_fprim_raw is None else float(ion_fprim_raw)
    ele_fprim = float(model.fprim) if ele_fprim_raw is None else float(ele_fprim_raw)

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
        tprim=float(getattr(model, "tprim_i", model.tprim_e)),
        fprim=ion_fprim,
        nu=nu_i,
    )
    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.tprim_e),
        fprim=ele_fprim,
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
        apar_beta_scale=0.5 if apar_beta_scale is None else float(apar_beta_scale),
        ampere_g0_scale=0.5 if ampere_g0_scale is None else float(ampere_g0_scale),
        bpar_beta_scale=0.5 if bpar_beta_scale is None else float(bpar_beta_scale),
    )
    params = _apply_reference_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params


KBM_EXPLICIT_SOLVER_LOCK: tuple[tuple[float, str], ...] = (
    (0.10, "explicit_time"),
    (0.30, "explicit_time"),
    (0.40, "explicit_time"),
)

KBM_EXPLICIT_SOLVER_LOCK_TOL = 0.03


def _midplane_index(grid: SpectralGrid) -> int:
    """Return reference midplane index for growth-rate diagnostics."""

    if grid.z.size <= 1:
        return 0
    idx = int(grid.z.size // 2 + 1)
    return min(idx, int(grid.z.size) - 1)


CYCLONE_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_target_factor=0.3,
    power_iters=60,
    power_dt=0.001,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
    shift_preconditioner="auto",
    omega_sign=1,
    mode_family="cyclone",
    fallback_method="propagator",
)

KINETIC_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.05,
    omega_cap_factor=0.8,
    omega_target_factor=0.3,
    omega_sign=1,
    power_iters=60,
    power_dt=0.001,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="auto",
    mode_family="cyclone",
    fallback_method="propagator",
)

KINETIC_KRYLOV_REFERENCE_ALIGNED = replace(
    KINETIC_KRYLOV_DEFAULT, shift_source="history"
)

# No generator uses this config: ledger row L-lin-etg is time-integrated. On
# benchmarks/cases/etg_linear.toml (Nl=24, Nm=8, ky=10/20/30) its raw propagator returned wrong branches
# at residual 0.98-0.99 (Q12, 2026-09-13), so it now fails closed rather than
# opting out of the certification gate.
ETG_KRYLOV_DEFAULT = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_target_factor=0.3,
    omega_cap_factor=0.6,
    omega_sign=-1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
    mode_family="etg",
    fallback_method="arnoldi",
    continuation=True,
    continuation_selection="overlap",
)

KBM_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_cap_factor=2.0,
    omega_target_factor=1.5,
    omega_sign=1,
    power_iters=60,
    power_dt=0.005,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="auto",
    shift_selection="targeted",
    mode_family="kbm",
    fallback_method="propagator",
    continuation=False,
)

TEM_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.05,
    omega_cap_factor=0.6,
    omega_target_factor=0.25,
    omega_sign=-1,
    power_iters=60,
    power_dt=0.005,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="auto",
    mode_family="tem",
    fallback_method="propagator",
)


@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class CycloneScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class LinearRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection
    gamma_t: np.ndarray | None = None
    omega_t: np.ndarray | None = None


@dataclass(frozen=True)
class LinearScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


def _load_csv_reference(filename: str) -> CycloneReference:
    data_path = resources.files("gkx").joinpath("data", filename)
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_cyclone_reference() -> CycloneReference:
    """Load Cyclone base case reference data (adiabatic electrons)."""

    return _load_csv_reference("cyclone_reference_adiabatic.csv")


def load_cyclone_reference_kinetic() -> CycloneReference:
    """Load Cyclone base case reference data (kinetic electrons)."""

    return _load_csv_reference("cyclone_reference_kinetic.csv")


def load_kbm_reference() -> CycloneReference:
    """Load KBM reference data (finite beta, kinetic electrons)."""

    return _load_csv_reference("kbm_reference.csv")


def load_etg_reference() -> CycloneReference:
    """Load ETG reference data for the tracked two-species ETG lane."""

    return _load_csv_reference("etg_reference.csv")


def load_tem_reference() -> CycloneReference:
    """Load the provisional TEM reference digitized from the literature.

    This lane remains an extended stress case while the literature case
    definition is being reconstructed.
    """

    return _load_csv_reference("tem_reference.csv")


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    envelope = (
        init_cfg.gaussian_envelope_constant
        + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    )
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-(((z - theta0) / width) ** 2))


def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    *,
    ky_index: int | Sequence[int] | np.ndarray,
    kx_index: int,
    Nl: int,
    Nm: int,
    init_cfg: InitializationConfig,
) -> jnp.ndarray:
    init_field = init_cfg.init_field.lower()
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    # Moment-normalized initializer amplitudes for init_field="all".
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(init_cfg.init_amp)
    ky_idx = np.atleast_1d(np.asarray(ky_index, dtype=int))
    for ky_i in ky_idx:
        if init_cfg.gaussian_init:
            profile = _build_gaussian_profile(
                np.asarray(grid.z),
                kx=float(grid.kx[kx_index]),
                ky=float(grid.ky[ky_i]),
                s_hat=geom.s_hat,
                init_cfg=init_cfg,
            )
            init_vals = amp * profile * (1.0 + 1.0j)
        else:
            init_vals = amp * (1.0 + 1.0j) * np.ones_like(grid.z)
        if grid.ky[ky_i] != 0.0:
            if init_field == "all":
                for field_name, (l_idx, m_idx) in field_map.items():
                    if l_idx < Nl and m_idx < Nm:
                        scale = all_scales.get(field_name, 1.0)
                        G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals * scale
            else:
                l_idx, m_idx = field_map[init_field]
                if l_idx >= Nl or m_idx >= Nm:
                    raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
                G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals
    return jnp.asarray(G0)


__all__ = [
    "resources",
    "VALID_FIT_SIGNALS",
    "CYCLONE_OMEGA_D_SCALE",
    "CYCLONE_OMEGA_STAR_SCALE",
    "CYCLONE_RHO_STAR",
    "ETG_OMEGA_D_SCALE",
    "ETG_OMEGA_STAR_SCALE",
    "ETG_RHO_STAR",
    "KINETIC_OMEGA_D_SCALE",
    "KINETIC_OMEGA_STAR_SCALE",
    "KINETIC_RHO_STAR",
    "TEM_OMEGA_D_SCALE",
    "TEM_OMEGA_STAR_SCALE",
    "TEM_RHO_STAR",
    "KBM_OMEGA_D_SCALE",
    "KBM_OMEGA_STAR_SCALE",
    "KBM_RHO_STAR",
    "REFERENCE_NU_HYPER_L",
    "REFERENCE_NU_HYPER_M",
    "REFERENCE_P_HYPER_L",
    "REFERENCE_P_HYPER_M",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "_reference_hypercollision_power",
    "_apply_reference_hypercollisions",
    "_two_species_params",
    "KBM_EXPLICIT_SOLVER_LOCK",
    "KBM_EXPLICIT_SOLVER_LOCK_TOL",
    "_midplane_index",
    "CYCLONE_KRYLOV_DEFAULT",
    "KINETIC_KRYLOV_DEFAULT",
    "KINETIC_KRYLOV_REFERENCE_ALIGNED",
    "ETG_KRYLOV_DEFAULT",
    "KBM_KRYLOV_DEFAULT",
    "TEM_KRYLOV_DEFAULT",
    "CycloneReference",
    "CycloneScanResult",
    "LinearRunResult",
    "LinearScanResult",
    "_load_csv_reference",
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_kbm_reference",
    "load_etg_reference",
    "load_tem_reference",
    "_build_gaussian_profile",
    "_build_initial_condition",
]
