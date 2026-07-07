"""Default normalization and Krylov policies for shipped benchmark lanes.

The runner functions in :mod:`spectraxgk.benchmarks` re-export these constants,
but benchmark policy lives here so it is separated from long-running scan
orchestration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.solvers.linear.krylov import KrylovConfig
from spectraxgk.diagnostics.normalization import (
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
)

__all__ = [
    "CYCLONE_KRYLOV_DEFAULT",
    "CYCLONE_OMEGA_D_SCALE",
    "CYCLONE_OMEGA_STAR_SCALE",
    "CYCLONE_RHO_STAR",
    "ETG_KRYLOV_DEFAULT",
    "ETG_OMEGA_D_SCALE",
    "ETG_OMEGA_STAR_SCALE",
    "ETG_RHO_STAR",
    "KBM_KRYLOV_DEFAULT",
    "select_kbm_solver_auto",
    "_midplane_index",
    "_kbm_use_multi_target_krylov",
    "KBM_EXPLICIT_SOLVER_LOCK_TOL",
    "KBM_EXPLICIT_SOLVER_LOCK",
    "KBM_OMEGA_D_SCALE",
    "KBM_OMEGA_STAR_SCALE",
    "KBM_RHO_STAR",
    "KINETIC_KRYLOV_DEFAULT",
    "KINETIC_KRYLOV_REFERENCE_ALIGNED",
    "KINETIC_OMEGA_D_SCALE",
    "KINETIC_OMEGA_STAR_SCALE",
    "KINETIC_RHO_STAR",
    "TEM_KRYLOV_DEFAULT",
    "TEM_OMEGA_D_SCALE",
    "TEM_OMEGA_STAR_SCALE",
    "TEM_RHO_STAR",
]


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


def select_kbm_solver_auto(
    solver: str,
    *,
    ky_target: float,
    reference_aligned: bool | None = None,
) -> str:
    """Return deterministic KBM solver choice for auto mode."""

    solver_key = solver.strip().lower()
    if solver_key != "auto":
        return solver_key
    if not bool(True if reference_aligned is None else reference_aligned):
        return "time"
    ky_abs = abs(float(ky_target))
    for ky_ref, solver_ref in KBM_EXPLICIT_SOLVER_LOCK:
        if abs(ky_abs - ky_ref) <= KBM_EXPLICIT_SOLVER_LOCK_TOL:
            return solver_ref
    return "explicit_time"


def _kbm_use_multi_target_krylov(
    kcfg: KrylovConfig,
    targets: Sequence[float] | None,
    *,
    shift: complex | None,
) -> bool:
    """Return whether KBM benchmark helpers should sweep target factors."""

    if targets is None:
        return False
    if kcfg.mode_family.strip().lower() != "kbm":
        return False
    if kcfg.method.strip().lower() != "shift_invert":
        return False
    if shift is not None:
        return False
    if kcfg.shift_selection.strip().lower() == "shift":
        return False
    return True


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
    shift_preconditioner="hermite-line",
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
    shift_preconditioner="hermite-line",
    mode_family="cyclone",
    fallback_method="propagator",
)

KINETIC_KRYLOV_REFERENCE_ALIGNED = replace(
    KINETIC_KRYLOV_DEFAULT, shift_source="history"
)

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
    omega_sign=-1,
    power_iters=60,
    power_dt=0.005,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="hermite-line",
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
    shift_preconditioner="hermite-line",
    mode_family="tem",
    fallback_method="propagator",
)
