"""Time integration helpers for nonlinear gyrokinetic evolution."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState

RHSFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]]

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


@partial(
    jax.jit,
    static_argnames=("rhs_fn", "steps", "method", "checkpoint", "project_state"),
    donate_argnums=(1,),
)
def integrate_nonlinear(
    rhs_fn: RHSFn,
    G0: jnp.ndarray,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    checkpoint: bool = False,
    project_state: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate a nonlinear RHS using lax.scan for kernel fusion."""

    if method not in {"euler", "rk2", "rk3", "rk3_gx", "rk4", "k10", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
        )

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    projector = project_state if project_state is not None else (lambda G: G)

    def step(G, _):
        G = jnp.asarray(projector(G), dtype=state_dtype)
        dG, _fields = rhs_fn(G)
        dG = jnp.asarray(dG, dtype=state_dtype)
        if method == "euler":
            G_new = G + dt_val * dG
        elif method == "rk2":
            k1 = dG
            G_half = jnp.asarray(projector(G + 0.5 * dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G_half)
            G_new = G + dt_val * k2
        elif method == "rk3":
            k1 = dG
            G1 = jnp.asarray(projector(G + dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G1)
            G2 = jnp.asarray(projector(0.75 * G + 0.25 * (G1 + dt_val * k2)), dtype=state_dtype)
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)
        elif method == "rk3_gx":
            k1 = dG
            G1 = jnp.asarray(projector(G + (dt_val / 3.0) * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G1)
            G2 = jnp.asarray(projector(G + (2.0 * dt_val / 3.0) * k2), dtype=state_dtype)
            k3, _ = rhs_fn(G2)
            G3 = jnp.asarray(projector(G + 0.75 * dt_val * k3), dtype=state_dtype)
            G_new = G3 + 0.25 * dt_val * k1
        elif method == "rk4":
            k1 = dG
            G2 = jnp.asarray(projector(G + 0.5 * dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G2)
            G3 = jnp.asarray(projector(G + 0.5 * dt_val * k2), dtype=state_dtype)
            k3, _ = rhs_fn(G3)
            G4 = jnp.asarray(projector(G + dt_val * k3), dtype=state_dtype)
            k4, _ = rhs_fn(G4)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif method == "sspx3":
            def _sspx3_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _ = rhs_fn(G_state)
                dG_state = jnp.asarray(dG_state, dtype=state_dtype)
                trial = G_state + (_SSPX3_ADT * dt_val) * dG_state
                return jnp.asarray(projector(trial), dtype=state_dtype)

            G1 = _sspx3_euler_step(G)
            G2_euler = _sspx3_euler_step(G1)
            G2 = (
                (1.0 - _SSPX3_W1) * G
                + (_SSPX3_W1 - 1.0) * G1
                + G2_euler
            )
            G2 = jnp.asarray(projector(G2), dtype=state_dtype)
            G3 = _sspx3_euler_step(G2)
            G_new = (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        else:
            # Ketcheson 10-stage (K10,4) scheme as used in GX.
            def _k10_euler_step(G_state):
                dG_state, _ = rhs_fn(G_state)
                dG_state = jnp.asarray(dG_state, dtype=state_dtype)
                trial = G_state + (dt_val / 6.0) * dG_state
                return jnp.asarray(projector(trial), dtype=state_dtype)

            G_q1 = G
            G_q2 = G
            for _ in range(5):
                G_q1 = _k10_euler_step(G_q1)

            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1

            for _ in range(4):
                G_q1 = _k10_euler_step(G_q1)

            dG_final, _ = rhs_fn(G_q1)
            dG_final = jnp.asarray(dG_final, dtype=state_dtype)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final
        G_new = jnp.asarray(projector(G_new), dtype=state_dtype)
        _dG_new, fields_new = rhs_fn(G_new)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    return jax.lax.scan(step_fn, G0, None, length=steps)
