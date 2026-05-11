"""Pure runtime policy helpers shared by runtime runners and tests."""

from __future__ import annotations

import numpy as np

from spectraxgk.analysis import select_ky_index
from spectraxgk.grids import SpectralGrid
from spectraxgk.runtime_config import RuntimeConfig

__all__ = [
    "_infer_runtime_nonlinear_steps",
    "_midplane_index",
    "_normalize_linear_solver_name",
    "_parallel_requests_combined_ky_scan",
    "_runtime_external_phi",
    "_select_nonlinear_mode_indices",
    "_zero_kx_index",
]


def _normalize_linear_solver_name(solver: str) -> str:
    solver_key = solver.strip().lower()
    if solver_key == "explicit_time":
        return "gx_time"
    return solver_key


def _parallel_requests_combined_ky_scan(cfg: RuntimeConfig) -> bool:
    """Return whether runtime parallel config requests the combined-ky scan path."""

    parallel = getattr(cfg, "parallel", None)
    if parallel is None:
        return False
    return str(getattr(parallel, "strategy", "serial")).lower() == "combined_ky" and str(
        getattr(parallel, "axis", "ky")
    ).lower() == "ky"


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


def _select_nonlinear_mode_indices(
    grid: SpectralGrid,
    *,
    ky_target: float,
    kx_target: float | None,
    use_dealias_mask: bool,
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    kx_pick_target = 0.0 if kx_target is None else float(kx_target)
    if not use_dealias_mask:
        ky_pick = select_ky_index(ky, ky_target)
        kx_pick = int(np.argmin(np.abs(kx - kx_pick_target)))
        return ky_pick, kx_pick

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    ky_candidates = np.where(np.any(mask, axis=1))[0]
    if ky_candidates.size == 0:
        ky_candidates = np.arange(ky.size, dtype=int)
    ky_pick = ky_candidates[int(np.argmin(np.abs(ky[ky_candidates] - float(ky_target))))]
    kx_candidates = np.where(mask[ky_pick])[0]
    if kx_candidates.size == 0:
        kx_candidates = np.arange(kx.size, dtype=int)
    kx_pick = kx_candidates[int(np.argmin(np.abs(kx[kx_candidates] - kx_pick_target)))]
    return int(ky_pick), int(kx_pick)


def _infer_runtime_nonlinear_steps(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int | None,
) -> int:
    """Infer nonlinear explicit step counts with the same dt ceiling as the integrator."""

    if steps is not None:
        steps_val = int(steps)
    elif bool(cfg.time.fixed_dt):
        steps_val = int(np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12)))
    else:
        # Keep runtime inference aligned with GX-style adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _runtime_external_phi(cfg: RuntimeConfig) -> float | None:
    """Return a GX-style runtime external-phi source if requested."""

    source = str(cfg.expert.source).strip().lower()
    if source in {"", "default"}:
        return None
    if source != "phiext_full":
        raise ValueError(
            f"unsupported expert.source={cfg.expert.source!r}; expected 'default' or 'phiext_full'"
        )
    return float(cfg.expert.phi_ext)
