"""Time integration helpers for nonlinear gyrokinetic evolution."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState

RHSFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, FieldState]]


@partial(jax.jit, static_argnames=("rhs_fn", "steps", "method", "checkpoint"))
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

    if method not in {"euler", "rk2", "rk4"}:
        raise ValueError("method must be one of {'euler', 'rk2', 'rk4'}")

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
        else:
            k1 = dG
            k2, _ = rhs_fn(G + 0.5 * dt_val * k1)
            k3, _ = rhs_fn(G + 0.5 * dt_val * k2)
            k4, _ = rhs_fn(G + dt_val * k3)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _dG_new, fields_new = rhs_fn(G_new)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    return jax.lax.scan(step_fn, G0, None, length=steps)
