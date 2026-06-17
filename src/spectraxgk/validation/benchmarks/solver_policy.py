"""Benchmark solver-selection and mode-index policies."""

from __future__ import annotations

from typing import Sequence

from spectraxgk.grids import SpectralGrid
from spectraxgk.solvers.linear.krylov import KrylovConfig


__all__ = [
    "KBM_EXPLICIT_SOLVER_LOCK",
    "KBM_EXPLICIT_SOLVER_LOCK_TOL",
    "_kbm_use_multi_target_krylov",
    "_midplane_index",
    "select_kbm_solver_auto",
]
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
    gx_reference: bool | None = None,
) -> str:
    """Return deterministic KBM solver choice for auto mode."""

    solver_key = solver.strip().lower()
    if solver_key != "auto":
        return solver_key
    if gx_reference is not None:
        reference_aligned = gx_reference
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
