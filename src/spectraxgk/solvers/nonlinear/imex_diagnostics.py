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


@dataclass(frozen=True)
class _IMEXPreparedState:
    term_cfg: TermConfig
    linear_cfg: TermConfig
    setup: Any
    cache: Any
    project_state: ProjectFn
    implicit_operator: Any
    G0: jnp.ndarray
    state_dtype: Any
    real_dtype: Any
    dt_val: jnp.ndarray
    progress_total: jnp.ndarray


@dataclass(frozen=True)
class _IMEXRuntimeOperators:
    collision_policy: Any
    nonlinear_term: NonlinearTermFn
    solve_step: SolveStepFn


@dataclass(frozen=True)
class _IMEXPreparationOptions:
    cache: LinearCache | None
    terms: TermConfig | None
    collision_split: bool
    implicit_preconditioner: str | None
    compressed_real_fft: bool
    use_dealias_mask: bool
    z_index: int | None
    fixed_mode_ky_index: int | None
    fixed_mode_kx_index: int | None


@dataclass(frozen=True)
class _IMEXRuntimeOptions:
    collision_split: bool
    external_phi: jnp.ndarray | float | None
    compressed_real_fft: bool
    laguerre_mode: str
    implicit_iters: int
    implicit_relax: float
    implicit_tol: float
    implicit_maxiter: int
    implicit_restart: int
    implicit_solve_method: str


@dataclass(frozen=True)
class _IMEXDiagnosticOptions:
    omega_ky_index: int | None
    omega_kx_index: int | None
    flux_scale: float
    wphi_scale: float


@dataclass(frozen=True)
class _IMEXScanOptions:
    method: str
    steps: int
    checkpoint: bool
    sample_stride: int
    diagnostics_stride: int
    external_phi: jnp.ndarray | float | None
    show_progress: bool
    collision_scheme: str


@dataclass(frozen=True)
class _IMEXOptionBundle:
    preparation: _IMEXPreparationOptions
    runtime: _IMEXRuntimeOptions
    diagnostics: _IMEXDiagnosticOptions
    scan: _IMEXScanOptions


@dataclass(frozen=True)
class _IMEXScanContext:
    prepared: _IMEXPreparedState
    step: DiagnosticStepFn
    compute_diag_from_state: DiagnosticFn


def _prepare_imex_diagnostic_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    cache: LinearCache | None,
    terms: TermConfig | None,
    collision_split: bool,
    implicit_preconditioner: str | None,
    compressed_real_fft: bool,
    use_dealias_mask: bool,
    z_index: int | None,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
) -> _IMEXPreparedState:
    """Prepare fixed-step IMEX state, linear operator, and dtype policy."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)

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
    initial_state_dtype = jnp.result_type(G0, jnp.complex64)
    G0_projected = setup.project_state(jnp.asarray(G0, dtype=initial_state_dtype))
    implicit_operator = deps.build_imex_operator_fn(
        G0_projected,
        setup.cache,
        params,
        dt,
        terms=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
    )
    state_dtype = implicit_operator.state_dtype
    G0_projected = jnp.asarray(G0_projected, dtype=state_dtype)
    if (
        implicit_operator.squeeze_species
        and G0_projected.ndim == len(implicit_operator.shape) - 1
    ):
        G0_projected = G0_projected[None, ...]
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    progress_total = jnp.asarray(float(steps) * float(dt), dtype=real_dtype)
    return _IMEXPreparedState(
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        setup=setup,
        cache=setup.cache,
        project_state=setup.project_state,
        implicit_operator=implicit_operator,
        G0=G0_projected,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        dt_val=dt_val,
        progress_total=progress_total,
    )


def _build_imex_runtime_operators(
    prepared: _IMEXPreparedState,
    params: LinearParams,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    linear_rhs_fn: Callable[..., Any],
    collision_split: bool,
    external_phi: jnp.ndarray | float | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
    implicit_iters: int,
    implicit_relax: float,
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
    implicit_solve_method: str,
) -> _IMEXRuntimeOperators:
    """Build collision, nonlinear, and linear solve operators for IMEX scans."""

    collision_policy = deps.build_collision_split_policy_fn(
        prepared.cache,
        params,
        prepared.term_cfg,
        prepared.real_dtype,
        squeeze_species=prepared.implicit_operator.squeeze_species,
        collision_split=collision_split,
        collision_damping_fn=deps.collision_damping_fn,
    )
    nonlinear_term = deps.make_imex_nonlinear_term_fn(
        prepared.cache,
        params,
        prepared.term_cfg,
        real_dtype=prepared.real_dtype,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        fields_fn=deps.compute_fields_fn,
        nonlinear_term_fn=deps.nonlinear_term_fn,
        nonlinear_contribution_fn=deps.nonlinear_contribution_fn,
    )
    solve_step = deps.make_imex_solve_step_fn(
        linear_rhs_fn=linear_rhs_fn,
        cache=prepared.cache,
        params=params,
        linear_cfg=prepared.linear_cfg,
        external_phi=external_phi,
        dt_val=prepared.dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        matvec=prepared.implicit_operator.matvec,
        shape=prepared.implicit_operator.shape,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        precond_op=prepared.implicit_operator.precond_op,
        solve_step_fn=deps.solve_imex_step_fn,
    )
    return _IMEXRuntimeOperators(
        collision_policy=collision_policy,
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
    )


def _make_imex_diagnostic_callable(
    prepared: _IMEXPreparedState,
    grid: SpectralGrid,
    params: LinearParams,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
) -> DiagnosticFn:
    """Return the state-to-diagnostic tuple closure for fixed-step IMEX scans."""

    return deps.make_diagnostic_tuple_fn(
        grid=grid,
        cache=prepared.cache,
        params=params,
        vol_fac=prepared.setup.vol_fac,
        flux_fac=prepared.setup.flux_fac,
        mask=prepared.setup.mask,
        z_idx=prepared.setup.z_idx,
        use_dealias=prepared.setup.use_dealias,
        real_dtype=prepared.real_dtype,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        resolved_diagnostics=True,
        kernels=deps.diagnostic_kernels_fn(),
    )


def _make_imex_scan_step(
    prepared: _IMEXPreparedState,
    runtime_ops: _IMEXRuntimeOperators,
    compute_diag_from_state: DiagnosticFn,
    params: LinearParams,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    method: str,
    diagnostics_stride: int,
    show_progress: bool,
    steps: int,
    external_phi: jnp.ndarray | float | None,
    collision_scheme: str,
) -> DiagnosticStepFn:
    """Build the fixed-step IMEX diagnostic scan step."""

    return deps.make_imex_step_fn(
        method=method,
        nonlinear_term=runtime_ops.nonlinear_term,
        solve_step=runtime_ops.solve_step,
        project_state=prepared.project_state,
        state_dtype=prepared.state_dtype,
        real_dtype=prepared.real_dtype,
        dt_val=prepared.dt_val,
        compute_fields_fn=deps.compute_fields_fn,
        cache=prepared.cache,
        params=params,
        term_cfg=prepared.term_cfg,
        external_phi=external_phi,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=diagnostics_stride,
        select_diagnostics_fn=deps.select_step_diagnostics_fn,
        show_progress=show_progress,
        steps=steps,
        progress_total=prepared.progress_total,
        emit_progress_fn=deps.emit_progress_fn,
        use_collision_split=runtime_ops.collision_policy.active,
        damping=runtime_ops.collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=deps.apply_collision_split_fn,
    )


def _run_imex_diagnostic_scan_and_finalize(
    prepared: _IMEXPreparedState,
    step: DiagnosticStepFn,
    compute_diag_from_state: DiagnosticFn,
    params: LinearParams,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    steps: int,
    checkpoint: bool,
    sample_stride: int,
    diagnostics_stride: int,
    external_phi: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, SimulationDiagnostics]:
    """Run the fixed-step IMEX scan and finalize diagnostics."""

    fields0 = deps.compute_fields_fn(
        prepared.G0,
        prepared.cache,
        params,
        terms=prepared.term_cfg,
        external_phi=external_phi,
    )
    diag_zero = compute_diag_from_state(
        prepared.G0, fields0, prepared.G0, fields0, prepared.dt_val
    )
    _G_final, scan_diag_out = deps.run_imex_scan_fn(
        step,
        (
            prepared.G0,
            prepared.G0,
            fields0,
            diag_zero,
            jnp.asarray(0.0, dtype=prepared.real_dtype),
        ),
        steps=steps,
        checkpoint=checkpoint,
    )

    diag, t = scan_diag_out
    dt_series = jnp.ones_like(t) * prepared.dt_val
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


def _build_imex_scan_context(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    preparation: _IMEXPreparationOptions,
    runtime: _IMEXRuntimeOptions,
    diagnostics: _IMEXDiagnosticOptions,
    scan: _IMEXScanOptions,
) -> _IMEXScanContext:
    prepared = _prepare_imex_diagnostic_state(
        G0,
        grid,
        geom,
        params,
        dt,
        scan.steps,
        deps=deps,
        cache=preparation.cache,
        terms=preparation.terms,
        collision_split=preparation.collision_split,
        implicit_preconditioner=preparation.implicit_preconditioner,
        compressed_real_fft=preparation.compressed_real_fft,
        use_dealias_mask=preparation.use_dealias_mask,
        z_index=preparation.z_index,
        fixed_mode_ky_index=preparation.fixed_mode_ky_index,
        fixed_mode_kx_index=preparation.fixed_mode_kx_index,
    )
    linear_rhs_fn = deps.linear_rhs_for_terms_fn(prepared.linear_cfg)
    runtime_ops = _build_imex_runtime_operators(
        prepared,
        params,
        deps=deps,
        linear_rhs_fn=linear_rhs_fn,
        collision_split=runtime.collision_split,
        external_phi=runtime.external_phi,
        compressed_real_fft=runtime.compressed_real_fft,
        laguerre_mode=runtime.laguerre_mode,
        implicit_iters=runtime.implicit_iters,
        implicit_relax=runtime.implicit_relax,
        implicit_tol=runtime.implicit_tol,
        implicit_maxiter=runtime.implicit_maxiter,
        implicit_restart=runtime.implicit_restart,
        implicit_solve_method=runtime.implicit_solve_method,
    )
    compute_diag_from_state = _make_imex_diagnostic_callable(
        prepared,
        grid,
        params=params,
        deps=deps,
        omega_ky_index=diagnostics.omega_ky_index,
        omega_kx_index=diagnostics.omega_kx_index,
        flux_scale=diagnostics.flux_scale,
        wphi_scale=diagnostics.wphi_scale,
    )
    step = _make_imex_scan_step(
        prepared,
        runtime_ops,
        compute_diag_from_state,
        params,
        deps=deps,
        method=scan.method,
        diagnostics_stride=scan.diagnostics_stride,
        show_progress=scan.show_progress,
        steps=scan.steps,
        external_phi=scan.external_phi,
        collision_scheme=scan.collision_scheme,
    )
    return _IMEXScanContext(prepared, step, compute_diag_from_state)


def _integrate_imex_nonlinear_diagnostics_core(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    *,
    deps: IMEXNonlinearDiagnosticsDeps,
    preparation: _IMEXPreparationOptions,
    runtime: _IMEXRuntimeOptions,
    diagnostics: _IMEXDiagnosticOptions,
    scan: _IMEXScanOptions,
) -> tuple[jnp.ndarray, SimulationDiagnostics]:
    context = _build_imex_scan_context(
        G0,
        grid,
        geom,
        params,
        dt,
        deps=deps,
        preparation=preparation,
        runtime=runtime,
        diagnostics=diagnostics,
        scan=scan,
    )
    return _run_imex_diagnostic_scan_and_finalize(
        context.prepared,
        context.step,
        context.compute_diag_from_state,
        params,
        deps=deps,
        steps=scan.steps,
        checkpoint=scan.checkpoint,
        sample_stride=scan.sample_stride,
        diagnostics_stride=scan.diagnostics_stride,
        external_phi=scan.external_phi,
    )


def _imex_option_bundle(
    *,
    cache: LinearCache | None,
    terms: TermConfig | None,
    collision_split: bool,
    implicit_preconditioner: str | None,
    external_phi: jnp.ndarray | float | None,
    compressed_real_fft: bool,
    use_dealias_mask: bool,
    z_index: int | None,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
    laguerre_mode: str,
    implicit_iters: int,
    implicit_relax: float,
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
    implicit_solve_method: str,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
    method: str,
    steps: int,
    checkpoint: bool,
    sample_stride: int,
    diagnostics_stride: int,
    show_progress: bool,
    collision_scheme: str,
) -> _IMEXOptionBundle:
    return _IMEXOptionBundle(
        preparation=_IMEXPreparationOptions(
            cache=cache,
            terms=terms,
            collision_split=collision_split,
            implicit_preconditioner=implicit_preconditioner,
            compressed_real_fft=compressed_real_fft,
            use_dealias_mask=use_dealias_mask,
            z_index=z_index,
            fixed_mode_ky_index=fixed_mode_ky_index,
            fixed_mode_kx_index=fixed_mode_kx_index,
        ),
        runtime=_IMEXRuntimeOptions(
            collision_split=collision_split,
            external_phi=external_phi,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
        ),
        diagnostics=_IMEXDiagnosticOptions(
            omega_ky_index=omega_ky_index, omega_kx_index=omega_kx_index,
            flux_scale=flux_scale, wphi_scale=wphi_scale,
        ),
        scan=_IMEXScanOptions(
            method=method,
            steps=steps, checkpoint=checkpoint,
            sample_stride=sample_stride, diagnostics_stride=diagnostics_stride,
            external_phi=external_phi, show_progress=show_progress,
            collision_scheme=collision_scheme,
        ),
    )


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
    options = _imex_option_bundle(
        cache=cache,
        terms=terms,
        collision_split=collision_split,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        external_phi=external_phi,
        laguerre_mode=laguerre_mode,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        method=method,
        steps=steps,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        show_progress=show_progress,
        collision_scheme=collision_scheme,
    )
    return _integrate_imex_nonlinear_diagnostics_core(
        G0, grid, geom, params, dt, deps=deps,
        preparation=options.preparation, runtime=options.runtime,
        diagnostics=options.diagnostics, scan=options.scan,
    )


__all__ = [
    "IMEXNonlinearDiagnosticsDeps",
    "advance_imex_nonlinear_state",
    "make_imex_diagnostic_step",
    "integrate_imex_nonlinear_diagnostics_impl",
    "run_imex_diagnostic_scan",
]
