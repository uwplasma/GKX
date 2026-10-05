"""Public runtime solve facade: one entry point per solve, plus the
two-stage secondary-instability workflow that composes them.

The solve owners live in :mod:`gkx.workflows.linear`,
:mod:`gkx.workflows.nonlinear` and
:mod:`gkx.workflows.runtime.orchestration_scan`; this module only re-exports
them and sits above them, so nothing below imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from jax.typing import ArrayLike

from gkx.artifacts.io import write_netcdf_restart_state
from gkx.config import Case, RuntimeConfig, RuntimeExpertConfig
from gkx.core_grid import build_spectral_grid
from gkx.core_ky_layout import source_ny_full
from gkx.diagnostics.analysis import fit_growth_rate
from gkx.geometry import apply_geometry_grid_defaults, build_flux_tube_geometry
from gkx.workflows.linear import run_runtime_linear
from gkx.workflows.nonlinear import (
    _runtime_external_phi,
    _select_nonlinear_mode_indices,
    build_runtime_nonlinear_diagnostics_kwargs,
    prepare,
    run_runtime_nonlinear,
)
from gkx.workflows.runtime.orchestration_scan import (
    RuntimeIndependentParallelPlan,
    run_runtime_scan,
)
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
)
from gkx.workflows.runtime.startup import (
    _build_initial_condition,
    _load_initial_state_from_file,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    build_runtime_term_config,
)


def solve(case: Case, **options: Any) -> RuntimeLinearResult | RuntimeNonlinearResult:
    """Solve one case through its existing linear or nonlinear runtime owner."""
    if case.physics.nonlinear:
        return run_runtime_nonlinear(case, **options)
    if case.physics.linear:
        return run_runtime_linear(case, **options)
    raise ValueError("case must enable linear or nonlinear physics")


@dataclass(frozen=True)
class SecondaryModeResult:
    """Late-time growth and frequency for one secondary-instability mode."""

    ky: float
    kx: float
    gamma: float
    omega: float


def _leading_finite_prefix(
    t: ArrayLike,
    signal: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    sig_arr = np.asarray(signal, dtype=np.complex128)
    finite = np.isfinite(sig_arr)
    if not np.any(finite):
        return t_arr[:0], sig_arr[:0]
    first_bad = np.where(~finite)[0]
    stop = int(first_bad[0]) if first_bad.size else int(sig_arr.size)
    return t_arr[:stop], sig_arr[:stop]


def _tail_mean_pair(
    gamma_t: ArrayLike,
    omega_t: ArrayLike,
    *,
    tail_fraction: float | None,
) -> tuple[float, float] | None:
    gamma_arr = np.asarray(gamma_t, dtype=float)
    omega_arr = np.asarray(omega_t, dtype=float)
    finite = np.isfinite(gamma_arr) & np.isfinite(omega_arr)
    if not np.any(finite):
        return None
    gamma_finite = gamma_arr[finite]
    omega_finite = omega_arr[finite]
    if tail_fraction is None:
        return float(gamma_finite[-1]), float(omega_finite[-1])
    istart = int(len(gamma_finite) * (1.0 - float(tail_fraction)))
    istart = max(0, min(istart, len(gamma_finite) - 1))
    return float(np.mean(gamma_finite[istart:])), float(np.mean(omega_finite[istart:]))


def write_restart_state(
    path: str | Path, state: np.ndarray, *, ny_full: int | None = None
) -> Path:
    """Write a complex restart state in the runtime NetCDF layout.

    ``ny_full`` is the length of the two-sided ``ky`` axis, and is what lets
    the writer keep a half-spectrum state out of the one file size whose
    meaning is already taken; see :func:`gkx.artifacts.io.write_netcdf_restart_state`.
    """

    return write_netcdf_restart_state(path, state, ny_full=ny_full)


def _embed_linear_seed_on_full_grid(
    cfg: RuntimeConfig,
    state: np.ndarray,
    *,
    ky_target: float,
) -> np.ndarray:
    geom = build_flux_tube_geometry(cfg.geometry)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    full_shape = (
        state.shape[0],
        state.shape[1],
        state.shape[2],
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    if tuple(state.shape) == full_shape:
        return np.asarray(state, dtype=np.complex64)
    if state.ndim != 6 or state.shape[3] != 1:
        raise ValueError(
            f"expected selected-ky linear state with shape (..., 1, Nx, Nz), got {state.shape}"
        )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid.ky, dtype=float) - float(ky_target))))
    full_state = np.zeros(full_shape, dtype=np.complex64)
    full_state[..., ky_idx : ky_idx + 1, :, :] = np.asarray(state, dtype=np.complex64)
    return full_state


def run_secondary_seed(
    cfg: RuntimeConfig,
    *,
    restart_path: str | Path,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float = 1.0,
    steps: int = 2,
    method: str = "sspx3",
    solver: str = "time",
) -> Path:
    """Run the linear seed stage and write its final full-grid restart state."""

    result = run_runtime_linear(
        cfg,
        ky_target=ky_target,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        return_state=True,
    )
    if result.state is None:
        raise RuntimeError("Secondary seed run did not return a final state.")
    state_full = _embed_linear_seed_on_full_grid(cfg, result.state, ky_target=ky_target)
    geom = build_flux_tube_geometry(cfg.geometry)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    return write_restart_state(restart_path, state_full, ny_full=source_ny_full(grid))


def build_secondary_stage2_config(
    cfg: RuntimeConfig,
    *,
    restart_file: str | Path,
    restart_scale: float = 500.0,
    init_amp: float = 1.0e-5,
    dt: float = 0.01,
    t_max: float = 100.0,
    method: str = "sspx3",
    iky_fixed: int = 1,
    ikx_fixed: int = 0,
) -> RuntimeConfig:
    """Build the nonlinear stage that evolves a saved primary-mode seed."""

    return replace(
        cfg,
        time=replace(
            cfg.time,
            t_max=float(t_max),
            dt=float(dt),
            method=str(method),
            fixed_dt=True,
        ),
        init=replace(
            cfg.init,
            init_amp=float(init_amp),
            init_single=False,
            init_file=str(restart_file),
            init_file_scale=float(restart_scale),
            init_file_mode="add",
        ),
        physics=replace(cfg.physics, linear=False, nonlinear=True),
        terms=replace(cfg.terms, nonlinear=1.0),
        expert=RuntimeExpertConfig(
            fixed_mode=True,
            iky_fixed=int(iky_fixed),
            ikx_fixed=int(ikx_fixed),
        ),
    )


def run_secondary_modes(
    cfg: RuntimeConfig,
    *,
    modes: Sequence[tuple[float, float]],
    Nl: int,
    Nm: int,
    steps: int | None = None,
    sample_stride: int = 100,
    fit_fraction: float | None = 0.5,
) -> list[SecondaryModeResult]:
    """Run one nonlinear secondary stage per requested diagnostic mode."""

    rows: list[SecondaryModeResult] = []
    for ky_target, kx_target in modes:
        result = run_runtime_nonlinear(
            cfg,
            ky_target=float(ky_target),
            kx_target=float(kx_target),
            Nl=Nl,
            Nm=Nm,
            steps=steps,
            sample_stride=sample_stride,
        )
        if result.diagnostics is None:
            raise RuntimeError("Secondary nonlinear run did not produce diagnostics.")
        tail_mean = _tail_mean_pair(
            result.diagnostics.gamma_t,
            result.diagnostics.omega_t,
            tail_fraction=fit_fraction,
        )
        gamma = float(tail_mean[0]) if tail_mean is not None else 0.0
        omega = float(tail_mean[1]) if tail_mean is not None else 0.0
        phi_mode_t = result.diagnostics.phi_mode_t
        if fit_fraction is not None and phi_mode_t is not None:
            t, signal = _leading_finite_prefix(result.diagnostics.t, phi_mode_t)
            if t.size >= 2 and np.max(np.abs(signal)) > 0.0:
                span = float(t[-1] - t[0])
                tmin = float(t[0] + (1.0 - fit_fraction) * span) if span > 0.0 else None
                try:
                    gamma_fit, omega_fit = fit_growth_rate(t, signal, tmin=tmin)
                    gamma = float(gamma_fit)
                    if tail_mean is None:
                        omega = float(omega_fit)
                except ValueError:
                    pass
        rows.append(
            SecondaryModeResult(
                ky=float(ky_target),
                kx=float(kx_target),
                gamma=gamma,
                omega=omega,
            )
        )
    return rows


__all__ = [
    "RuntimeIndependentParallelPlan",
    "RuntimeLinearResult",
    "RuntimeLinearScanResult",
    "RuntimeNonlinearResult",
    "SecondaryModeResult",
    "_build_initial_condition",
    "_load_initial_state_from_file",
    "_runtime_external_phi",
    "_select_nonlinear_mode_indices",
    "build_runtime_geometry",
    "build_runtime_linear_params",
    "build_runtime_linear_terms",
    "build_runtime_nonlinear_diagnostics_kwargs",
    "build_runtime_term_config",
    "build_secondary_stage2_config",
    "prepare",
    "run_runtime_linear",
    "run_runtime_nonlinear",
    "run_runtime_scan",
    "run_secondary_modes",
    "run_secondary_seed",
    "solve",
    "write_restart_state",
]
