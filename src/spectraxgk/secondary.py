"""Helpers for GX-style secondary-instability staged workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from jax.typing import ArrayLike

from spectraxgk.analysis import fit_growth_rate
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, build_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimePhysicsConfig,
    RuntimeTermsConfig,
)


@dataclass(frozen=True)
class SecondaryModeResult:
    """Per-mode nonlinear secondary diagnostic summary."""

    ky: float
    kx: float
    gamma: float
    omega: float


def _leading_finite_prefix(
    t: ArrayLike,
    signal: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the longest leading segment where the complex mode remains finite."""

    t_arr = np.asarray(t, dtype=float)
    sig_arr = np.asarray(signal, dtype=np.complex128)
    finite = np.isfinite(sig_arr)
    if not np.any(finite):
        return t_arr[:0], sig_arr[:0]
    first_bad = np.where(~finite)[0]
    stop = int(first_bad[0]) if first_bad.size else int(sig_arr.size)
    return t_arr[:stop], sig_arr[:stop]


def write_restart_state(path: str | Path, state: np.ndarray) -> Path:
    """Write a complex restart state in the runtime binary layout."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(state, dtype=np.complex64).tofile(out)
    return out


def _embed_linear_seed_on_full_grid(
    cfg: RuntimeConfig,
    state: np.ndarray,
    *,
    ky_target: float,
) -> np.ndarray:
    """Embed a single-ky linear state back onto the full runtime grid."""

    geom = build_flux_tube_geometry(cfg.geometry)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
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
        raise ValueError(f"expected selected-ky linear state with shape (..., 1, Nx, Nz), got {state.shape}")
    ky = np.asarray(grid.ky, dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
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
    """Run the linear seed stage and write its final state as a restart file."""

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
    return write_restart_state(restart_path, state_full)


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
    """Return a nonlinear stage-2 config matching GX's ``kh01a`` contract."""

    time_cfg = replace(
        cfg.time,
        t_max=float(t_max),
        dt=float(dt),
        method=str(method),
        use_diffrax=False,
        fixed_dt=True,
    )
    init_cfg = replace(
        cfg.init,
        init_amp=float(init_amp),
        init_single=False,
        init_file=str(restart_file),
        init_file_scale=float(restart_scale),
        init_file_mode="add",
    )
    physics_cfg = replace(cfg.physics, linear=False, nonlinear=True)
    terms_cfg = replace(cfg.terms, nonlinear=1.0)
    expert_cfg = RuntimeExpertConfig(fixed_mode=True, iky_fixed=int(iky_fixed), ikx_fixed=int(ikx_fixed))
    return replace(
        cfg,
        time=time_cfg,
        init=init_cfg,
        physics=physics_cfg,
        terms=terms_cfg,
        expert=expert_cfg,
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
    """Run the nonlinear stage once per requested diagnostic mode."""

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
        gamma_t = np.asarray(result.diagnostics.gamma_t, dtype=float)
        omega_t = np.asarray(result.diagnostics.omega_t, dtype=float)
        gamma = float(gamma_t[-1])
        omega = float(omega_t[-1])
        phi_mode_t = result.diagnostics.phi_mode_t
        if fit_fraction is not None and phi_mode_t is not None:
            t, signal = _leading_finite_prefix(result.diagnostics.t, phi_mode_t)
            if t.size >= 2 and np.max(np.abs(signal)) > 0.0:
                t_span = float(t[-1] - t[0]) if t.size > 1 else 0.0
                tmin = float(t[0] + (1.0 - float(fit_fraction)) * t_span) if t_span > 0.0 else None
                try:
                    gamma, omega = fit_growth_rate(t, signal, tmin=tmin)
                except ValueError:
                    gamma = float(gamma_t[-1])
                    omega = float(omega_t[-1])
        rows.append(
            SecondaryModeResult(
                ky=float(ky_target),
                kx=float(kx_target),
                gamma=float(gamma),
                omega=float(omega),
            )
        )
    return rows
