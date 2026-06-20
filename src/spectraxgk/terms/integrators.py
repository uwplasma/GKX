"""Time integration helpers for nonlinear gyrokinetic evolution."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState, RHSFn


_EXPLICIT_METHODS = {"euler", "rk2", "rk3", "rk3_classic", "rk3_heun", "rk4", "k10", "sspx3"}

# Keep the SSPX3 coefficients in one importable location because the cETG and
# runtime paths reuse the same low-storage update.
_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _validate_explicit_method(method: str) -> None:
    if method not in _EXPLICIT_METHODS:
        raise ValueError(f"method must be one of {sorted(_EXPLICIT_METHODS)!r}")


def _project(
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    return jnp.asarray(projector(G), dtype=dtype)


def _rhs(
    rhs_fn: RHSFn,
    G: jnp.ndarray,
    *,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, FieldState]:
    dG, fields = rhs_fn(G)
    return jnp.asarray(dG, dtype=dtype), fields


def _maybe_emit_progress(
    G: jnp.ndarray,
    idx: jnp.ndarray,
    *,
    steps: int,
    dt_val: jnp.ndarray,
    real_dtype: jnp.dtype,
    show_progress: bool,
) -> jnp.ndarray:
    if not show_progress:
        return G
    from spectraxgk.utils.callbacks import print_callback, should_emit_progress  # type: ignore[import-untyped]

    sim_time = (idx + 1) * dt_val
    sim_total = jnp.asarray(steps, dtype=real_dtype) * dt_val
    return jax.lax.cond(
        should_emit_progress(idx, steps),
        lambda state: print_callback(
            state, idx, steps, 0.0, 0.0, 0.0, 0.0, sim_time, sim_total
        ),
        lambda state: state,
        G,
    )


def _rk2_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    G_half = _project(projector, G + 0.5 * dt_val * k1, dtype=dtype)
    k2, _ = _rhs(rhs_fn, G_half, dtype=dtype)
    return G + dt_val * k2


def _rk3_classic_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    G1 = _project(projector, G + dt_val * k1, dtype=dtype)
    k2, _ = _rhs(rhs_fn, G1, dtype=dtype)
    G2 = _project(projector, 0.75 * G + 0.25 * (G1 + dt_val * k2), dtype=dtype)
    k3, _ = _rhs(rhs_fn, G2, dtype=dtype)
    return (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)


def _rk3_heun_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    G1 = _project(projector, G + (dt_val / 3.0) * k1, dtype=dtype)
    k2, _ = _rhs(rhs_fn, G1, dtype=dtype)
    G2 = _project(projector, G + (2.0 * dt_val / 3.0) * k2, dtype=dtype)
    k3, _ = _rhs(rhs_fn, G2, dtype=dtype)
    G3 = _project(projector, G + 0.75 * dt_val * k3, dtype=dtype)
    return G3 + 0.25 * dt_val * k1


def _rk4_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    G2 = _project(projector, G + 0.5 * dt_val * k1, dtype=dtype)
    k2, _ = _rhs(rhs_fn, G2, dtype=dtype)
    G3 = _project(projector, G + 0.5 * dt_val * k2, dtype=dtype)
    k3, _ = _rhs(rhs_fn, G3, dtype=dtype)
    G4 = _project(projector, G + dt_val * k3, dtype=dtype)
    k4, _ = _rhs(rhs_fn, G4, dtype=dtype)
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _k10_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    def _k10_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        dG_state, _ = _rhs(rhs_fn, G_state, dtype=dtype)
        return _project(projector, G_state + (dt_val / 6.0) * dG_state, dtype=dtype)

    G_q1 = G
    G_q2 = G
    for _ in range(5):
        G_q1 = _k10_euler_step(G_q1)

    G_q2 = 0.04 * G_q2 + 0.36 * G_q1
    G_q1 = 15.0 * G_q2 - 5.0 * G_q1

    for _ in range(4):
        G_q1 = _k10_euler_step(G_q1)

    dG_final, _ = _rhs(rhs_fn, G_q1, dtype=dtype)
    return G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final


def _explicit_method_update(
    rhs_fn: RHSFn,
    projector: Callable[[jnp.ndarray], jnp.ndarray],
    G: jnp.ndarray,
    k1: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    dtype: jnp.dtype,
    method: str,
) -> jnp.ndarray:
    if method == "euler":
        return G + dt_val * k1
    if method == "rk2":
        return _rk2_update(rhs_fn, projector, G, k1, dt_val=dt_val, dtype=dtype)
    if method == "rk3_classic" or method == "sspx3":
        return _rk3_classic_update(
            rhs_fn, projector, G, k1, dt_val=dt_val, dtype=dtype
        )
    if method == "rk3" or method == "rk3_heun":
        return _rk3_heun_update(rhs_fn, projector, G, k1, dt_val=dt_val, dtype=dtype)
    if method == "rk4":
        return _rk4_update(rhs_fn, projector, G, k1, dt_val=dt_val, dtype=dtype)
    if method == "k10":
        return _k10_update(rhs_fn, projector, G, dt_val=dt_val, dtype=dtype)
    raise ValueError(f"Unsupported method '{method}'")


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

    _validate_explicit_method(method)

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    projector = project_state if project_state is not None else (lambda G: G)

    def step(G, idx):
        G = _maybe_emit_progress(
            G,
            idx,
            steps=steps,
            dt_val=dt_val,
            real_dtype=real_dtype,
            show_progress=show_progress,
        )
        G = _project(projector, G, dtype=state_dtype)
        dG, _fields = _rhs(rhs_fn, G, dtype=state_dtype)
        G_new = _explicit_method_update(
            rhs_fn,
            projector,
            G,
            dG,
            dt_val=dt_val,
            dtype=state_dtype,
            method=method,
        )
        G_new = _project(projector, G_new, dtype=state_dtype)
        _dG_new, fields_new = _rhs(rhs_fn, G_new, dtype=state_dtype)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    return jax.lax.scan(step_fn, G0, indices)
