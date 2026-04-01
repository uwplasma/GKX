"""Time integration helpers for nonlinear gyrokinetic evolution."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState, RHSFn


# Keep the SSPX3 coefficients in one importable location because the cETG and
# runtime paths reuse the same low-storage update.
_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


@partial(
    jax.jit,
    static_argnames=("rhs_fn", "steps", "method", "checkpoint", "project_state", "show_progress"),
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
    show_progress: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate a nonlinear RHS using lax.scan for kernel fusion."""

    if method not in {"euler", "rk2", "rk3", "rk3_classic", "rk3_gx", "rk4", "k10", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
        )

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    projector = project_state if project_state is not None else (lambda G: G)

    def step(G, idx):
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback  # type: ignore[import-untyped]
            G = print_callback(G, idx, steps, 0.0, 0.0, 0.0, 0.0)
        G = jnp.asarray(projector(G), dtype=state_dtype)
        dG, _fields = rhs_fn(G)
        dG = jnp.asarray(dG, dtype=state_dtype)
        if method == "euler":
            G_new = G + dt_val * dG
        elif method == "rk2":
            k1 = dG
            G_half = jnp.asarray(projector(G + 0.5 * dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G_half)
            G_new = G + dt_val * jnp.asarray(k2, dtype=state_dtype)
        elif method == "rk3_classic":
            k1 = dG
            G1 = jnp.asarray(projector(G + dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G1)
            G2 = jnp.asarray(projector(0.75 * G + 0.25 * (G1 + dt_val * jnp.asarray(k2, dtype=state_dtype))), dtype=state_dtype)
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * jnp.asarray(k3, dtype=state_dtype))
        elif method in {"rk3", "rk3_gx"}:
            k1 = dG
            G1 = jnp.asarray(projector(G + (dt_val / 3.0) * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G1)
            G2 = jnp.asarray(projector(G + (2.0 * dt_val / 3.0) * jnp.asarray(k2, dtype=state_dtype)), dtype=state_dtype)
            k3, _ = rhs_fn(G2)
            G3 = jnp.asarray(projector(G + 0.75 * dt_val * jnp.asarray(k3, dtype=state_dtype)), dtype=state_dtype)
            G_new = G3 + 0.25 * dt_val * k1
        elif method == "rk4":
            k1 = dG
            G2 = jnp.asarray(projector(G + 0.5 * dt_val * k1), dtype=state_dtype)
            k2, _ = rhs_fn(G2)
            G3 = jnp.asarray(projector(G + 0.5 * dt_val * jnp.asarray(k2, dtype=state_dtype)), dtype=state_dtype)
            k3, _ = rhs_fn(G3)
            G4 = jnp.asarray(projector(G + dt_val * jnp.asarray(k3, dtype=state_dtype)), dtype=state_dtype)
            k4, _ = rhs_fn(G4)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * jnp.asarray(k2, dtype=state_dtype) + 2.0 * jnp.asarray(k3, dtype=state_dtype) + jnp.asarray(k4, dtype=state_dtype))
        elif method == "sspx3":
            k1, _ = rhs_fn(G)
            G1 = jnp.asarray(projector(G + dt_val * jnp.asarray(k1, dtype=state_dtype)), dtype=state_dtype)
            k2, _ = rhs_fn(G1)
            G2 = jnp.asarray(projector(0.75 * G + 0.25 * (G1 + dt_val * jnp.asarray(k2, dtype=state_dtype))), dtype=state_dtype)
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * jnp.asarray(k3, dtype=state_dtype))
        elif method == "k10":
            def _k10_euler_step(G_state):
                dG_state, _ = rhs_fn(G_state)
                return jnp.asarray(projector(G_state + (dt_val / 6.0) * jnp.asarray(dG_state, dtype=state_dtype)), dtype=state_dtype)

            G_q1 = G
            G_q2 = G
            for _ in range(5):
                G_q1 = _k10_euler_step(G_q1)

            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1

            for _ in range(4):
                G_q1 = _k10_euler_step(G_q1)

            dG_final, _ = rhs_fn(G_q1)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_val * jnp.asarray(dG_final, dtype=state_dtype)
        else:
            raise ValueError(f"Unsupported method '{method}'")

        G_new = jnp.asarray(projector(G_new), dtype=state_dtype)
        _dG_new, fields_new = rhs_fn(G_new)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    return jax.lax.scan(step_fn, G0, indices)
