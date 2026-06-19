"""Diagnostic IMEX stepping policy for nonlinear scans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp


from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.config import TermConfig

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


@dataclass(frozen=True)
class IMEXNonlinearDiagnosticsDeps:
    """Patchable kernels used by the IMEX diagnostic integrator."""

    ensure_geometry_fn: Callable[..., Any]
    build_cache_fn: Callable[..., Any]
    quadrature_weights_fn: Callable[..., Any]
    omega_mask_fn: Callable[..., Any]
    midplane_index_fn: Callable[..., Any]
    linear_rhs_for_terms_fn: Callable[..., Any]
    build_diagnostic_setup_fn: Callable[..., Any]
    build_imex_operator_fn: Callable[..., Any]
    build_collision_split_policy_fn: Callable[..., Any]
    collision_damping_fn: Callable[..., Any]
    make_imex_nonlinear_term_fn: Callable[..., Any]
    make_imex_solve_step_fn: Callable[..., Any]
    solve_imex_step_fn: Callable[..., Any]
    make_diagnostic_tuple_fn: Callable[..., Any]
    make_imex_step_fn: Callable[..., Any]
    run_imex_scan_fn: Callable[..., Any]
    finalize_scan_diagnostics_fn: Callable[..., Any]
    select_step_diagnostics_fn: Callable[..., Any]
    emit_progress_fn: Callable[..., Any]
    apply_collision_split_fn: Callable[..., Any]
    compute_fields_fn: Callable[..., Any]
    nonlinear_term_fn: Callable[..., Any]
    nonlinear_contribution_fn: Callable[..., Any]
    diagnostic_kernels_fn: Callable[..., Any]


def integrate_imex_nonlinear_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    method: str = "imex",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, SimulationDiagnostics]:
    """Integrate an IMEX nonlinear run and return diagnostics."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)
    linear_rhs_fn = deps.linear_rhs_for_terms_fn(linear_cfg)

    setup = deps.build_diagnostic_setup_fn(
        G0,
        grid,
        geom,
        params,
        cache=cache,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        ensure_geometry_fn=deps.ensure_geometry_fn,
        build_cache_fn=deps.build_cache_fn,
        quadrature_weights_fn=deps.quadrature_weights_fn,
        omega_mask_fn=deps.omega_mask_fn,
        midplane_index_fn=deps.midplane_index_fn,
    )
    cache = setup.cache
    project_state = setup.project_state

    initial_state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=initial_state_dtype)
    G0 = project_state(G0)

    implicit_operator = deps.build_imex_operator_fn(
        G0,
        cache,
        params,
        dt,
        terms=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
    )

    # Keep the scan carry in the same dtype as the implicit operator, especially
    # under x64 where the operator promotes complex64 inputs to complex128.
    state_dtype = implicit_operator.state_dtype
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    progress_total = jnp.asarray(float(steps) * float(dt), dtype=real_dtype)

    squeeze_species = implicit_operator.squeeze_species
    if squeeze_species and G0.ndim == len(implicit_operator.shape) - 1:
        G0 = G0[None, ...]
    collision_policy = deps.build_collision_split_policy_fn(
        cache,
        params,
        term_cfg,
        real_dtype,
        squeeze_species=squeeze_species,
        collision_split=collision_split,
        collision_damping_fn=deps.collision_damping_fn,
    )

    nonlinear_term = deps.make_imex_nonlinear_term_fn(
        cache,
        params,
        term_cfg,
        real_dtype=real_dtype,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        fields_fn=deps.compute_fields_fn,
        nonlinear_term_fn=deps.nonlinear_term_fn,
        nonlinear_contribution_fn=deps.nonlinear_contribution_fn,
    )
    solve_step = deps.make_imex_solve_step_fn(
        linear_rhs_fn=linear_rhs_fn,
        cache=cache,
        params=params,
        linear_cfg=linear_cfg,
        external_phi=external_phi,
        dt_val=dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        matvec=implicit_operator.matvec,
        shape=implicit_operator.shape,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        precond_op=implicit_operator.precond_op,
        solve_step_fn=deps.solve_imex_step_fn,
    )

    compute_diag_from_state = deps.make_diagnostic_tuple_fn(
        grid=grid,
        cache=cache,
        params=params,
        vol_fac=setup.vol_fac,
        flux_fac=setup.flux_fac,
        mask=setup.mask,
        z_idx=setup.z_idx,
        use_dealias=setup.use_dealias,
        real_dtype=real_dtype,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        resolved_diagnostics=True,
        kernels=deps.diagnostic_kernels_fn(),
    )

    fields0 = deps.compute_fields_fn(
        G0, cache, params, terms=term_cfg, external_phi=external_phi
    )

    step = deps.make_imex_step_fn(
        method=method,
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=project_state,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        dt_val=dt_val,
        compute_fields_fn=deps.compute_fields_fn,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        external_phi=external_phi,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=diagnostics_stride,
        select_diagnostics_fn=deps.select_step_diagnostics_fn,
        show_progress=show_progress,
        steps=steps,
        progress_total=progress_total,
        emit_progress_fn=deps.emit_progress_fn,
        use_collision_split=collision_policy.active,
        damping=collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=deps.apply_collision_split_fn,
    )

    diag_zero = compute_diag_from_state(G0, fields0, G0, fields0, dt_val)
    _G_final, scan_diag_out = deps.run_imex_scan_fn(
        step,
        (
            G0,
            G0,
            fields0,
            diag_zero,
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        steps=steps,
        checkpoint=checkpoint,
    )

    diag, t = scan_diag_out
    dt_series = jnp.ones_like(t) * dt_val

    stride = int(max(sample_stride, diagnostics_stride, 1))
    diag_out = deps.finalize_scan_diagnostics_fn(
        diag,
        t=t,
        dt_series=dt_series,
        stride=stride,
        resolved_diagnostics=True,
        resolved_to_numpy=True,
    )
    return jnp.asarray(diag_out.t), diag_out


__all__ = [
    "IMEXNonlinearDiagnosticsDeps",
    "advance_imex_nonlinear_state",
    "make_imex_diagnostic_step",
    "integrate_imex_nonlinear_diagnostics_impl",
    "run_imex_diagnostic_scan",
]
