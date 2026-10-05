"""Shared benchmark reference data, normalization, and scan policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import resources

import numpy as np

from gkx.core_grid import SpectralGrid
from gkx.diagnostics.modes import ModeSelection
from gkx.diagnostics.normalization import (
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
)
from gkx.solvers_linear_krylov import KrylovConfig

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

REFERENCE_DAMP_ENDS_AMP = 0.1

REFERENCE_DAMP_ENDS_WIDTHFRAC = 1.0 / 8.0


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


__all__ = [
    "resources",
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
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
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
]
