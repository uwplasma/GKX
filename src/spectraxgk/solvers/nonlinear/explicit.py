"""Explicit nonlinear time-step policies.

This module keeps the RK/SSP/K10 one-step formulas outside the public nonlinear
facade.  The integrator still injects the RHS and projection functions so the
step policy remains pure, small, and directly testable.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

RhsFn = Callable[..., tuple[jnp.ndarray, object]]
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


def _rhs_value(rhs_fn: RhsFn, state: jnp.ndarray) -> jnp.ndarray:
    dG, _fields = rhs_fn(state)
    return dG


def _explicit_rk3_classic_state(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    G1 = project_state(G + dt_local * dG)
    k2 = _rhs_value(rhs_fn, G1)
    G2 = project_state(0.75 * G + 0.25 * (G1 + dt_local * k2))
    k3 = _rhs_value(rhs_fn, G2)
    return (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_local * k3)


def _explicit_rk3_heun_state(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    G1 = project_state(G + (dt_local / 3.0) * dG)
    k2 = _rhs_value(rhs_fn, G1)
    G2 = project_state(G + (2.0 * dt_local / 3.0) * k2)
    k3 = _rhs_value(rhs_fn, G2)
    G3 = project_state(G + 0.75 * dt_local * k3)
    return G3 + 0.25 * dt_local * dG


def _explicit_rk4_state(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    G2 = project_state(G + 0.5 * dt_local * dG)
    k2 = _rhs_value(rhs_fn, G2)
    G3 = project_state(G + 0.5 * dt_local * k2)
    k3 = _rhs_value(rhs_fn, G3)
    G4 = project_state(G + dt_local * k3)
    k4 = _rhs_value(rhs_fn, G4)
    return G + (dt_local / 6.0) * (dG + 2.0 * k2 + 2.0 * k3 + k4)


def _explicit_sspx3_state(
    G: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        dG_state = _rhs_value(rhs_fn, G_state)
        return project_state(G_state + (_SSPX3_ADT * dt_local) * dG_state)

    G1 = euler_step(G)
    G2_euler = euler_step(G1)
    G2 = project_state((1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler)
    G3 = euler_step(G2)
    return (
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
        + _SSPX3_W3 * G1
        + (_SSPX3_W2 - 1.0) * G2
        + G3
    )


def _explicit_k10_state(
    G: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        return project_state(G_state + (dt_local / 6.0) * _rhs_value(rhs_fn, G_state))

    G_q1 = G
    G_q2 = G
    for _ in range(5):
        G_q1 = euler_step(G_q1)
    G_q2 = 0.04 * G_q2 + 0.36 * G_q1
    G_q1 = 15.0 * G_q2 - 5.0 * G_q1
    for _ in range(4):
        G_q1 = euler_step(G_q1)
    dG_final = _rhs_value(rhs_fn, G_q1)
    return G_q2 + 0.6 * G_q1 + 0.1 * dt_local * dG_final


def _explicit_stage_update(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    method: str,
    rhs_fn: RhsFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    """Advance one nonlinear explicit method before final projection/dtype cast."""

    if method == "euler":
        return G + dt_local * dG
    if method == "rk2":
        G_half = project_state(G + 0.5 * dt_local * dG)
        return G + dt_local * _rhs_value(rhs_fn, G_half)
    if method == "rk3_classic":
        return _explicit_rk3_classic_state(
            G, dG, dt_local, rhs_fn=rhs_fn, project_state=project_state
        )
    if method in {"rk3", "rk3_heun"}:
        return _explicit_rk3_heun_state(
            G, dG, dt_local, rhs_fn=rhs_fn, project_state=project_state
        )
    if method == "rk4":
        return _explicit_rk4_state(
            G, dG, dt_local, rhs_fn=rhs_fn, project_state=project_state
        )
    if method == "sspx3":
        return _explicit_sspx3_state(
            G, dt_local, rhs_fn=rhs_fn, project_state=project_state
        )
    if method == "k10":
        return _explicit_k10_state(
            G, dt_local, rhs_fn=rhs_fn, project_state=project_state
        )
    raise ValueError(
        "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', "
        "'rk3_heun', 'rk4', 'k10', 'sspx3'}"
    )


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

    G_new = _explicit_stage_update(
        G,
        dG,
        dt_local,
        method=method,
        rhs_fn=rhs_fn,
        project_state=project_state,
    )
    G_new = project_state(G_new)
    return jnp.asarray(G_new, dtype=state_dtype)


def checkpoint_explicit_step(step: Callable[..., object], checkpoint: bool):
    """Apply JAX checkpointing to an explicit scan step when requested."""

    return jax.checkpoint(step) if checkpoint else step


def _maybe_emit_progress(
    G: jnp.ndarray,
    idx: jnp.ndarray,
    *,
    steps: int,
    dt_val: jnp.ndarray,
    real_dtype: jnp.dtype,
    show_progress: bool,
) -> jnp.ndarray:
    """Emit executable progress without changing the traced state."""

    if not show_progress:
        return G
    from spectraxgk.utils.callbacks import (  # type: ignore[import-untyped]
        print_callback,
        should_emit_progress,
    )

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


@partial(
    jax.jit,
    static_argnames=(
        "rhs_fn",
        "rhs_static_args",
        "steps",
        "method",
        "checkpoint",
        "project_state",
        "show_progress",
        "return_fields",
    ),
    donate_argnums=(1,),
)
def integrate_nonlinear_scan(
    rhs_fn: RhsFn,
    G0: jnp.ndarray,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    checkpoint: bool = False,
    project_state: ProjectFn | None = None,
    show_progress: bool = False,
    return_fields: bool = True,
    rhs_args: tuple[Any, ...] = (),
    rhs_static_args: tuple[Any, ...] = (),
) -> tuple[jnp.ndarray, Any] | jnp.ndarray:
    """Integrate a cached nonlinear RHS using the explicit solver scan policy."""

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    projector = project_state if project_state is not None else (lambda G: G)

    def bound_rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, object]:
        return rhs_fn(state, *rhs_args, *rhs_static_args)

    def step(G: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
        G = _maybe_emit_progress(
            G,
            idx,
            steps=steps,
            dt_val=dt_val,
            real_dtype=real_dtype,
            show_progress=show_progress,
        )
        G = jnp.asarray(projector(G), dtype=state_dtype)
        dG, _fields = bound_rhs(G)
        dG = jnp.asarray(dG, dtype=state_dtype)
        G_new = advance_explicit_nonlinear_state(
            G,
            dG,
            dt_val,
            method=method,
            rhs_fn=bound_rhs,
            project_state=projector,
            state_dtype=state_dtype,
        )
        if not return_fields:
            return G_new, None
        _dG_new, fields_new = bound_rhs(G_new)
        return G_new, fields_new

    step_fn = checkpoint_explicit_step(step, checkpoint)
    G_final, fields_t = jax.lax.scan(step_fn, G0, jnp.arange(steps))
    if return_fields:
        return G_final, fields_t
    return G_final


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
    return_fields: bool = True,
    rhs_args: tuple[Any, ...] = (),
    rhs_static_args: tuple[Any, ...] = (),
) -> tuple[jnp.ndarray, Any] | jnp.ndarray:
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
        return_fields=return_fields,
        rhs_args=rhs_args,
        rhs_static_args=rhs_static_args,
    )


def _apply_explicit_collision_split(
    G_new: jnp.ndarray,
    dt_local: jnp.ndarray,
    *,
    use_collision_split: bool,
    damping: Any | None,
    collision_scheme: str,
    apply_collision_split_fn: CollisionSplitFn | None,
) -> jnp.ndarray:
    """Apply the optional post-step collision split used by diagnostics."""

    if not use_collision_split or damping is None:
        return G_new
    if apply_collision_split_fn is None:
        raise ValueError(
            "apply_collision_split_fn is required when collision split is active"
        )
    return apply_collision_split_fn(G_new, damping, dt_local, collision_scheme)


def _advance_explicit_diagnostic_state(
    G: jnp.ndarray,
    dt_prev: jnp.ndarray,
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
    use_collision_split: bool,
    damping: Any | None,
    collision_scheme: str,
    apply_collision_split_fn: CollisionSplitFn | None,
) -> tuple[jnp.ndarray, Any, jnp.ndarray]:
    """Advance one explicit diagnostic state and compute its new fields."""

    dG, fields = rhs_fn(G)
    dt_local = jnp.asarray(time_step_policy.update_dt(fields, dt_prev), dtype=real_dtype)
    G_new = advance_explicit_nonlinear_state(
        G,
        dG,
        dt_local,
        method=method,
        rhs_fn=rhs_fn,
        project_state=project_state,
        state_dtype=state_dtype,
    )
    G_new = _apply_explicit_collision_split(
        G_new,
        dt_local,
        use_collision_split=use_collision_split,
        damping=damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=apply_collision_split_fn,
    )
    G_new = jnp.asarray(project_state(G_new), dtype=state_dtype)
    fields_new = compute_fields_fn(
        G_new, cache, params, terms=term_cfg, external_phi=external_phi
    )
    return G_new, fields_new, dt_local


def _select_explicit_diagnostic(
    idx: Any,
    *,
    diagnostics_stride: int,
    diag_prev: Any,
    G_new: jnp.ndarray,
    fields_new: Any,
    G_prev_step: jnp.ndarray,
    fields_prev_step: Any,
    dt_local: jnp.ndarray,
    compute_diag_from_state: DiagnosticFn,
    select_diagnostics_fn: Callable[..., Any],
) -> Any:
    """Select or reuse the diagnostic payload for one explicit scan sample."""

    def compute_diag():
        return compute_diag_from_state(
            G_new, fields_new, G_prev_step, fields_prev_step, dt_local
        )

    return select_diagnostics_fn(
        idx,
        diagnostics_stride=diagnostics_stride,
        diag_prev=diag_prev,
        compute_diag_fn=compute_diag,
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
        G_new, fields_new, dt_local = _advance_explicit_diagnostic_state(
            G,
            dt_prev,
            rhs_fn=rhs_fn,
            method=method,
            project_state=project_state,
            state_dtype=state_dtype,
            real_dtype=real_dtype,
            time_step_policy=time_step_policy,
            compute_fields_fn=compute_fields_fn,
            cache=cache,
            params=params,
            term_cfg=term_cfg,
            external_phi=external_phi,
            use_collision_split=use_collision_split,
            damping=damping,
            collision_scheme=collision_scheme,
            apply_collision_split_fn=apply_collision_split_fn,
        )
        diag = _select_explicit_diagnostic(
            idx,
            diagnostics_stride=diagnostics_stride,
            diag_prev=diag_prev,
            G_new=G_new,
            fields_new=fields_new,
            G_prev_step=G_prev_step,
            fields_prev_step=fields_prev_step,
            dt_local=dt_local,
            compute_diag_from_state=compute_diag_from_state,
            select_diagnostics_fn=select_diagnostics_fn,
        )
        t_new = jnp.asarray(t_prev + dt_local, dtype=real_dtype)
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
    "integrate_nonlinear_scan",
    "make_explicit_diagnostic_step",
    "run_explicit_diagnostic_scan",
]
