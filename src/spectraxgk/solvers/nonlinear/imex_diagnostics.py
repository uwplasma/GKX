"""Diagnostic IMEX stepping policy for nonlinear scans."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from spectraxgk.solvers.nonlinear.explicit import (
    _SSPX3_ADT,
    _SSPX3_W1,
    _SSPX3_W2,
    _SSPX3_W3,
)

FieldSolveFn = Callable[..., object]
NonlinearTermFn = Callable[[jnp.ndarray], jnp.ndarray]
ProjectFn = Callable[[jnp.ndarray], jnp.ndarray]
SolveStepFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DiagnosticFn = Callable[..., Any]
CollisionSplitFn = Callable[[jnp.ndarray, Any, jnp.ndarray, str], jnp.ndarray]
DiagnosticStepFn = Callable[
    [tuple[Any, Any, Any, Any, Any], Any],
    tuple[tuple[Any, Any, Any, Any, Any], tuple[Any, Any]],
]
DiagnosticScanOutput = tuple[jnp.ndarray, tuple[Any, Any]]


def advance_imex_nonlinear_state(
    G: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: SolveStepFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    """Advance one IMEX nonlinear step with optional SSPX3 stage composition."""

    if method == "sspx3":

        def _euler_step(G_state: jnp.ndarray, dt_stage: jnp.ndarray) -> jnp.ndarray:
            rhs_stage = G_state + dt_stage * nonlinear_term(G_state)
            return solve_step(G_state, rhs_stage)

        G1 = _euler_step(G, _SSPX3_ADT * dt_val)
        G2_euler = _euler_step(G1, _SSPX3_ADT * dt_val)
        G2 = project_state(
            (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
        )
        G3 = _euler_step(G2, _SSPX3_ADT * dt_val)
        return (
            (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
            + _SSPX3_W3 * G1
            + (_SSPX3_W2 - 1.0) * G2
            + G3
        )

    rhs = G + dt_val * nonlinear_term(G)
    return solve_step(G, rhs)


def make_imex_diagnostic_step(
    *,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: SolveStepFn,
    project_state: ProjectFn,
    state_dtype: jnp.dtype,
    real_dtype: jnp.dtype,
    dt_val: jnp.ndarray,
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
    progress_total: jnp.ndarray,
    emit_progress_fn: Callable[..., jnp.ndarray],
    use_collision_split: bool = False,
    damping: Any | None = None,
    collision_scheme: str = "implicit",
    apply_collision_split_fn: CollisionSplitFn | None = None,
) -> DiagnosticStepFn:
    """Build one IMEX diagnostic scan step with injected runtime seams."""

    def step(
        carry: tuple[Any, Any, Any, Any, Any],
        idx: Any,
    ) -> tuple[tuple[Any, Any, Any, Any, Any], tuple[Any, Any]]:
        G, G_prev_step, fields_prev_step, diag_prev, t_prev = carry
        G_new = advance_imex_nonlinear_state(
            G,
            dt_val=dt_val,
            method=method,
            nonlinear_term=nonlinear_term,
            solve_step=solve_step,
            project_state=project_state,
        )
        if use_collision_split and damping is not None:
            if apply_collision_split_fn is None:
                raise ValueError(
                    "apply_collision_split_fn is required when collision split is active"
                )
            G_new = apply_collision_split_fn(G_new, damping, dt_val, collision_scheme)
        G_new = project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_val, dtype=real_dtype)
        fields_new = compute_fields_fn(
            G_new, cache, params, terms=term_cfg, external_phi=external_phi
        )

        def _compute_diag():
            return compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_val
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
            progress_total=progress_total,
        )
        return (G_new, G_new, fields_new, diag, t_new), (diag, t_new)

    return step


def run_imex_diagnostic_scan(
    step_fn: DiagnosticStepFn,
    initial_carry: tuple[Any, Any, Any, Any, Any],
    *,
    steps: int,
    checkpoint: bool,
) -> DiagnosticScanOutput:
    """Run the fixed-step IMEX diagnostic scan."""

    scan_step = jax.checkpoint(step_fn) if checkpoint else step_fn
    idx = jnp.arange(steps, dtype=jnp.int32)
    (
        (G_final, _G_prev_last, _fields_prev_last, _diag_last, _t_last),
        scan_diag_out,
    ) = jax.lax.scan(
        scan_step,
        initial_carry,
        idx,
        length=steps,
    )
    return G_final, scan_diag_out


__all__ = [
    "advance_imex_nonlinear_state",
    "make_imex_diagnostic_step",
    "run_imex_diagnostic_scan",
]
