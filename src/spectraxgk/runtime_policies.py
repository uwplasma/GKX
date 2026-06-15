"""Pure runtime policy helpers shared by runtime runners and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spectraxgk.analysis import select_ky_index
from spectraxgk.grids import SpectralGrid
from spectraxgk.runtime_config import RuntimeConfig

__all__ = [
    "RuntimeIndependentParallelPlan",
    "_infer_runtime_nonlinear_steps",
    "_midplane_index",
    "_normalize_linear_solver_name",
    "_parallel_requests_combined_ky_scan",
    "_runtime_external_phi",
    "_runtime_independent_parallel_plan",
    "_select_nonlinear_mode_indices",
    "_zero_kx_index",
]


@dataclass(frozen=True)
class RuntimeIndependentParallelPlan:
    """Resolved independent-worker policy for runtime scan workloads."""

    requested_workers: int
    effective_workers: int
    executor: str
    strategy: str
    axis: str
    source: str
    problem_size: int

    @property
    def enabled(self) -> bool:
        """Whether the resolved plan uses more than one independent worker."""

        return self.effective_workers > 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly policy payload for runtime artifacts."""

        payload = asdict(self)
        payload["enabled"] = self.enabled
        return payload


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
    return (
        str(getattr(parallel, "strategy", "serial")).lower() == "combined_ky"
        and str(getattr(parallel, "axis", "ky")).lower() == "ky"
    )


def _normalize_independent_executor(backend: str, fallback: str) -> str:
    backend_key = str(backend).strip().lower().replace("-", "_")
    fallback_key = str(fallback).strip().lower().replace("-", "_")
    aliases = {
        "thread": "thread",
        "threads": "thread",
        "process": "process",
        "processes": "process",
    }
    if backend_key in {"", "auto"}:
        try:
            return aliases[fallback_key]
        except KeyError as exc:
            raise ValueError("parallel_executor must be 'thread' or 'process'") from exc
    try:
        return aliases[backend_key]
    except KeyError as exc:
        raise ValueError(
            "runtime [parallel] backend for independent scans must be "
            "'auto', 'thread', or 'process'"
        ) from exc


def _runtime_independent_parallel_plan(
    cfg: RuntimeConfig,
    *,
    problem_size: int,
    workers: int,
    executor: str,
) -> RuntimeIndependentParallelPlan:
    """Resolve independent ``k_y`` worker policy from arguments and config."""

    size = int(problem_size)
    if size < 0:
        raise ValueError("problem_size must be non-negative")
    requested = int(workers)
    if requested < 1:
        raise ValueError("workers must be >= 1")
    executor_key = _normalize_independent_executor("auto", executor)
    source = "arguments"
    strategy = "serial"
    axis = "ky"

    parallel = getattr(cfg, "parallel", None)
    if parallel is not None:
        strategy = str(getattr(parallel, "strategy", "serial")).strip().lower()
        axis = str(getattr(parallel, "axis", "ky")).strip().lower()
        if requested == 1 and strategy == "batch":
            if axis != "ky":
                raise ValueError(
                    "runtime [parallel] strategy='batch' is supported only for axis='ky'"
                )
            configured_workers = getattr(parallel, "num_devices", None)
            if configured_workers is None:
                configured_workers = getattr(parallel, "batch_size", None)
            requested = max(int(configured_workers or 1), 1)
            executor_key = _normalize_independent_executor(
                str(getattr(parallel, "backend", "auto")), executor
            )
            source = "runtime_config"

    effective = 0 if size == 0 else min(requested, size)
    return RuntimeIndependentParallelPlan(
        requested_workers=requested,
        effective_workers=effective,
        executor=executor_key,
        strategy=strategy,
        axis=axis,
        source=source,
        problem_size=size,
    )


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


def _nearest_index_from_candidates(
    values: np.ndarray,
    target: float,
    candidates: np.ndarray,
) -> int:
    """Return the candidate index nearest to ``target`` in physical coordinates."""

    values_arr = np.asarray(values, dtype=float)
    candidate_arr = np.asarray(candidates, dtype=int)
    if values_arr.size == 0:
        raise ValueError("values must be non-empty")
    if candidate_arr.size == 0:
        raise ValueError("candidate indices must be non-empty")
    return int(
        candidate_arr[int(np.argmin(np.abs(values_arr[candidate_arr] - float(target))))]
    )


def _validate_dealias_mask_shape(
    mask: Any,
    *,
    ky_size: int,
    kx_size: int,
) -> np.ndarray:
    """Return a boolean dealias mask after validating it matches ky/kx axes."""

    mask_arr = np.asarray(mask, dtype=bool)
    expected = (int(ky_size), int(kx_size))
    if mask_arr.shape != expected:
        raise ValueError(
            "dealias_mask shape must match (ky, kx) grid sizes; "
            f"got {mask_arr.shape}, expected {expected}"
        )
    return mask_arr


def _active_ky_indices(mask: np.ndarray, ky_size: int) -> np.ndarray:
    """Return ky rows with at least one retained kx, falling back to all ky."""

    candidates = np.where(np.any(mask, axis=1))[0]
    if candidates.size == 0:
        return np.arange(int(ky_size), dtype=int)
    return candidates


def _active_kx_indices(mask: np.ndarray, ky_index: int, kx_size: int) -> np.ndarray:
    """Return retained kx entries for ``ky_index``, falling back to all kx."""

    candidates = np.where(mask[int(ky_index)])[0]
    if candidates.size == 0:
        return np.arange(int(kx_size), dtype=int)
    return candidates


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
        kx_pick = _nearest_index_from_candidates(
            kx, kx_pick_target, np.arange(kx.size, dtype=int)
        )
        return ky_pick, kx_pick

    mask = _validate_dealias_mask_shape(
        grid.dealias_mask,
        ky_size=ky.size,
        kx_size=kx.size,
    )
    ky_pick = _nearest_index_from_candidates(
        ky, ky_target, _active_ky_indices(mask, ky.size)
    )
    kx_pick = _nearest_index_from_candidates(
        kx, kx_pick_target, _active_kx_indices(mask, ky_pick, kx.size)
    )
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
        steps_val = int(
            np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12))
        )
    else:
        # Keep runtime inference aligned with adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _runtime_external_phi(cfg: RuntimeConfig) -> float | None:
    """Return a runtime external-phi source if requested."""

    source = str(cfg.expert.source).strip().lower()
    if source in {"", "default"}:
        return None
    if source != "phiext_full":
        raise ValueError(
            f"unsupported expert.source={cfg.expert.source!r}; expected 'default' or 'phiext_full'"
        )
    return float(cfg.expert.phi_ext)
