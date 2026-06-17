"""Explicit nonlinear time-step policies.

This module keeps the RK/SSP/K10 one-step formulas outside the public nonlinear
facade.  The integrator still injects the RHS and projection functions so the
step policy remains pure, small, and directly testable.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

RhsFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, object]]
ProjectFn = Callable[[jnp.ndarray], jnp.ndarray]
ScanFn = Callable[..., tuple[jnp.ndarray, Any]]
DiagnosticFn = Callable[..., Any]
FieldSolveFn = Callable[..., Any]
CollisionSplitFn = Callable[[jnp.ndarray, Any, jnp.ndarray, str], jnp.ndarray]
DiagnosticStepFn = Callable[
    [tuple[Any, Any, Any, Any, Any, Any], Any],
    tuple[tuple[Any, Any, Any, Any, Any, Any], tuple[Any, Any, Any]],
]
SampledDiagnosticScanFn = Callable[
    ...,
    tuple[tuple[Any, Any, Any, Any, Any, Any], tuple[Any, Any, Any]],
]

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


def integrate_cached_explicit_scan(
    G0: jnp.ndarray,
    dt: float,
    steps: int,
    *,
    method: str,
    rhs_fn: RhsFn,
    scan_fn: ScanFn,
    checkpoint: bool = False,
    project_state: ProjectFn | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, Any]:
    """Run a cached explicit nonlinear scan with injected RHS and projection."""

    return scan_fn(
        rhs_fn,
        G0,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        project_state=project_state,
        show_progress=show_progress,
    )


def make_explicit_diagnostic_step(
    *,
    rhs_fn: RhsFn,
    method: str,
    project_state: ProjectFn,
    state_dtype: jnp.dtype,
    real_dtype: jnp.dtype,
    time_step_policy: Any,
    compute_fields_fn: FieldSolveFn,
    cache: Any,
    params: Any,
    term_cfg: Any,
    external_phi: jnp.ndarray | float | None,
    compute_diag_from_state: DiagnosticFn,
    diagnostics_stride: int,
    select_diagnostics_fn: Callable[..., Any],
    show_progress: bool,
    steps: int,
    emit_progress_fn: Callable[..., jnp.ndarray],
    use_collision_split: bool = False,
    damping: Any | None = None,
    collision_scheme: str = "implicit",
    apply_collision_split_fn: CollisionSplitFn | None = None,
) -> DiagnosticStepFn:
    """Build one explicit diagnostic scan step with injected runtime seams."""

    def step(
        carry: tuple[Any, Any, Any, Any, Any, Any],
        idx: Any,
    ) -> tuple[tuple[Any, Any, Any, Any, Any, Any], tuple[Any, Any, Any]]:
        G, G_prev_step, fields_prev_step, diag_prev, t_prev, dt_prev = carry
        dG, fields = rhs_fn(G)
        dt_local = jnp.asarray(
            time_step_policy.update_dt(fields, dt_prev), dtype=real_dtype
        )
        G_new = advance_explicit_nonlinear_state(
            G,
            dG,
            dt_local,
            method=method,
            rhs_fn=rhs_fn,
            project_state=project_state,
            state_dtype=state_dtype,
        )
        if use_collision_split and damping is not None:
            if apply_collision_split_fn is None:
                raise ValueError(
                    "apply_collision_split_fn is required when collision split is active"
                )
            G_new = apply_collision_split_fn(
                G_new, damping, dt_local, collision_scheme
            )
        G_new = project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_local, dtype=real_dtype)
        fields_new = compute_fields_fn(
            G_new, cache, params, terms=term_cfg, external_phi=external_phi
        )

        def _compute_diag():
            return compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_local
            )

        diag = select_diagnostics_fn(
            idx,
            diagnostics_stride=diagnostics_stride,
            diag_prev=diag_prev,
            compute_diag_fn=_compute_diag,
        )
        G_new = emit_progress_fn(
            G_new,
            show_progress=show_progress,
            diag=diag,
            idx=idx,
            steps=steps,
            t_new=t_new,
            progress_total=time_step_policy.progress_total,
        )
        return (G_new, G_new, fields_new, diag, t_new, dt_local), (
            diag,
            t_new,
            dt_local,
        )

    return step


def run_explicit_diagnostic_scan(
    step_fn: DiagnosticStepFn,
    initial_carry: tuple[Any, Any, Any, Any, Any, Any],
    *,
    steps: int,
    stride: int,
    sampled_scan: bool,
    checkpoint: bool,
    sampled_scan_fn: SampledDiagnosticScanFn,
) -> tuple[jnp.ndarray, tuple[Any, Any, Any]]:
    """Run the explicit diagnostic scan using sampled or dense retention."""

    scan_step = jax.checkpoint(step_fn) if checkpoint else step_fn
    if sampled_scan:
        (
            (
                G_final,
                _G_prev_last,
                _fields_prev_last,
                _diag_last,
                _t_last,
                _dt_last,
            ),
            scan_diag_out,
        ) = sampled_scan_fn(
            scan_step,
            initial_carry,
            steps=steps,
            stride=stride,
        )
        return G_final, scan_diag_out

    idx = jnp.arange(steps, dtype=jnp.int32)
    (
        (G_final, _G_prev_last, _fields_prev_last, _diag_last, _t_last, _dt_last),
        scan_diag_out,
    ) = jax.lax.scan(
        scan_step,
        initial_carry,
        idx,
        length=steps,
    )
    return G_final, scan_diag_out


__all__ = [
    "advance_explicit_nonlinear_state",
    "checkpoint_explicit_step",
    "integrate_cached_explicit_scan",
    "make_explicit_diagnostic_step",
    "run_explicit_diagnostic_scan",
]
