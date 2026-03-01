"""Time integration helpers for nonlinear gyrokinetic evolution."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState

RHSFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]]


@partial(
    jax.jit,
    static_argnames=("rhs_fn", "steps", "method", "checkpoint"),
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
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate a nonlinear RHS using lax.scan for kernel fusion."""

    if method not in {"euler", "rk2", "rk3", "rk3_gx", "rk4", "k10"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_gx', 'rk4', 'k10'}"
        )

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    def step(G, _):
        dG, _fields = rhs_fn(G)
        if method == "euler":
            G_new = G + dt_val * dG
        elif method == "rk2":
            k1 = dG
            k2, _ = rhs_fn(G + 0.5 * dt_val * k1)
            G_new = G + dt_val * k2
        elif method == "rk3":
            k1 = dG
            G1 = G + dt_val * k1
            k2, _ = rhs_fn(G1)
            G2 = 0.75 * G + 0.25 * (G1 + dt_val * k2)
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)
        elif method == "rk3_gx":
            k1 = dG
            G1 = G + (dt_val / 3.0) * k1
            k2, _ = rhs_fn(G1)
            G2 = G + (2.0 * dt_val / 3.0) * k2
            k3, _ = rhs_fn(G2)
            G3 = G + 0.75 * dt_val * k3
            G_new = G3 + 0.25 * dt_val * k1
        elif method == "rk4":
            k1 = dG
            k2, _ = rhs_fn(G + 0.5 * dt_val * k1)
            k3, _ = rhs_fn(G + 0.5 * dt_val * k2)
            k4, _ = rhs_fn(G + dt_val * k3)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            # Ketcheson 10-stage (K10,4) scheme as used in GX.
            def _euler_step(G_state):
                dG_state, _ = rhs_fn(G_state)
                return G_state + (dt_val / 6.0) * dG_state

            G_q1 = G
            G_q2 = G
            for _ in range(5):
                G_q1 = _euler_step(G_q1)

            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1

            for _ in range(4):
                G_q1 = _euler_step(G_q1)

            dG_final, _ = rhs_fn(G_q1)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final
        _dG_new, fields_new = rhs_fn(G_new)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    return jax.lax.scan(step_fn, G0, None, length=steps)
