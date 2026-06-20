"""Configuration policies for linear eigenmode solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KrylovConfig:
    """Controls for the Krylov-based eigen solver."""

    krylov_dim: int = 24
    restarts: int = 2
    omega_min_factor: float = 0.0
    omega_target_factor: float = 0.0
    omega_cap_factor: float = 2.0
    omega_sign: int = 0
    method: str = "propagator"
    power_iters: int = 200
    power_dt: float = 0.01
    shift: complex | None = None
    shift_source: str = "propagator"
    shift_tol: float = 1.0e-4
    shift_maxiter: int = 50
    shift_restart: int = 20
    shift_solve_method: str = "batched"
    shift_preconditioner: str | None = "damping"
    shift_selection: str = "targeted"
    mode_family: str = "auto"
    fallback_method: str = "propagator"
    fallback_real_floor: float = -1.0e-6
    continuation: bool = False
    continuation_selection: str = "overlap"


__all__ = ["KrylovConfig"]
