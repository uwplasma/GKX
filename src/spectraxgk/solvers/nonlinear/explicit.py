"""Explicit nonlinear time-step policies.

This module keeps the RK/SSP/K10 one-step formulas outside the public nonlinear
facade.  The integrator still injects the RHS and projection functions so the
step policy remains pure, small, and directly testable.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

RhsFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, object]]
ProjectFn = Callable[[jnp.ndarray], jnp.ndarray]

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def advance_explicit_nonlinear_state(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    method: str,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
    state_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Advance one explicit nonlinear step with a static method string."""

    if method == "euler":
        G_new = G + dt_local * dG
    elif method == "rk2":
        k1 = dG
        G_half = project_state(G + 0.5 * dt_local * k1)
        k2, _ = rhs_fn(G_half)
        G_new = G + dt_local * k2
    elif method == "rk3_classic":
        k1 = dG
        G1 = project_state(G + dt_local * k1)
        k2, _ = rhs_fn(G1)
        G2 = project_state(0.75 * G + 0.25 * (G1 + dt_local * k2))
        k3, _ = rhs_fn(G2)
        G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_local * k3)
    elif method in {"rk3", "rk3_heun"}:
        k1 = dG
        G1 = project_state(G + (dt_local / 3.0) * k1)
        k2, _ = rhs_fn(G1)
        G2 = project_state(G + (2.0 * dt_local / 3.0) * k2)
        k3, _ = rhs_fn(G2)
        G3 = project_state(G + 0.75 * dt_local * k3)
        G_new = G3 + 0.25 * dt_local * k1
    elif method == "rk4":
        k1 = dG
        G2 = project_state(G + 0.5 * dt_local * k1)
        k2, _ = rhs_fn(G2)
        G3 = project_state(G + 0.5 * dt_local * k2)
        k3, _ = rhs_fn(G3)
        G4 = project_state(G + dt_local * k3)
        k4, _ = rhs_fn(G4)
        G_new = G + (dt_local / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    elif method == "sspx3":

        def _sspx3_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
            dG_state, _fields_state = rhs_fn(G_state)
            return project_state(G_state + (_SSPX3_ADT * dt_local) * dG_state)

        G1 = _sspx3_euler_step(G)
        G2_euler = _sspx3_euler_step(G1)
        G2 = project_state((1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler)
        G3 = _sspx3_euler_step(G2)
        G_new = (
            (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
            + _SSPX3_W3 * G1
            + (_SSPX3_W2 - 1.0) * G2
            + G3
        )
    elif method == "k10":

        def _k10_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
            dG_state, _ = rhs_fn(G_state)
            return project_state(G_state + (dt_local / 6.0) * dG_state)

        G_q1 = G
        G_q2 = G
        for _ in range(5):
            G_q1 = _k10_euler_step(G_q1)

        G_q2 = 0.04 * G_q2 + 0.36 * G_q1
        G_q1 = 15.0 * G_q2 - 5.0 * G_q1

        for _ in range(4):
            G_q1 = _k10_euler_step(G_q1)

        dG_final, _ = rhs_fn(G_q1)
        G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_local * dG_final
    else:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', "
            "'rk3_heun', 'rk4', 'k10', 'sspx3'}"
        )
    G_new = project_state(G_new)
    return jnp.asarray(G_new, dtype=state_dtype)


def checkpoint_explicit_step(step: Callable[..., object], checkpoint: bool):
    """Apply JAX checkpointing to an explicit scan step when requested."""

    return jax.checkpoint(step) if checkpoint else step


__all__ = ["advance_explicit_nonlinear_state", "checkpoint_explicit_step"]
