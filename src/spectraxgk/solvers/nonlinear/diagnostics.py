"""Nonlinear diagnostic integration drivers.

This module owns the orchestration for explicit nonlinear diagnostic runs.  The
public :mod:`spectraxgk.nonlinear` facade injects the concrete kernels so tests
and downstream users can still patch facade-level seams without keeping the
large implementation body in the facade itself.
"""

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
from spectraxgk.terms.config import FieldState, TermConfig


@dataclass(frozen=True)
class ExplicitNonlinearDiagnosticsDeps:
    """Patchable kernels used by the explicit diagnostic integrator."""

    ensure_geometry_fn: Callable[..., Any]
    build_cache_fn: Callable[..., Any]
    quadrature_weights_fn: Callable[..., Any]
    omega_mask_fn: Callable[..., Any]
    midplane_index_fn: Callable[..., Any]
    resolve_cfl_fac_fn: Callable[..., Any]
    linear_frequency_bound_fn: Callable[..., Any]
    laguerre_velocity_max_fn: Callable[..., Any]
    cfl_frequency_components_fn: Callable[..., Any]
    collision_damping_fn: Callable[..., Any]
    nonlinear_rhs_fn: Callable[..., Any]
    compute_fields_fn: Callable[..., Any]
    diagnostic_kernels_fn: Callable[..., Any]
    build_diagnostic_setup_fn: Callable[..., Any]
    build_time_step_policy_fn: Callable[..., Any]
    build_collision_split_policy_fn: Callable[..., Any]
    make_diagnostic_tuple_fn: Callable[..., Any]
    make_explicit_step_fn: Callable[..., Any]
    run_explicit_scan_fn: Callable[..., Any]
    run_sampled_explicit_scan_fn: Callable[..., Any]
    finalize_scan_diagnostics_fn: Callable[..., Any]
    select_step_diagnostics_fn: Callable[..., Any]
    emit_progress_fn: Callable[..., Any]
    apply_collision_split_fn: Callable[..., Any]


def integrate_explicit_nonlinear_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    deps: ExplicitNonlinearDiagnosticsDeps,
    method: str = "rk3",
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
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
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
    resolved_diagnostics: bool = True,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, SimulationDiagnostics, jnp.ndarray, FieldState]:
    """Integrate an explicit nonlinear run and return diagnostics plus final state."""

    del implicit_tol
    del implicit_maxiter
    del implicit_iters
    del implicit_relax
    del implicit_restart
    del implicit_solve_method
    del implicit_preconditioner

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        raise ValueError(
            "Final-state runtime diagnostics helper only supports explicit methods"
        )
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
    geom_eff = setup.geom
    cache = setup.cache
    project_state = setup.project_state

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    G0 = project_state(G0)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    time_step_policy = deps.build_time_step_policy_fn(
        grid,
        geom_eff,
        params,
        cache,
        method=method,
        dt=dt,
        steps=steps,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        compressed_real_fft=compressed_real_fft,
        real_dtype=real_dtype,
        resolve_cfl_fac_fn=deps.resolve_cfl_fac_fn,
        linear_frequency_bound_fn=deps.linear_frequency_bound_fn,
        laguerre_velocity_max_fn=deps.laguerre_velocity_max_fn,
        cfl_frequency_components_fn=deps.cfl_frequency_components_fn,
    )
    collision_policy = deps.build_collision_split_policy_fn(
        cache,
        params,
        term_cfg,
        real_dtype,
        squeeze_species=G0.ndim == 5,
        collision_split=collision_split,
        collision_damping_fn=deps.collision_damping_fn,
    )

    def rhs_fn(G):
        return deps.nonlinear_rhs_fn(
            G,
            cache,
            params,
            collision_policy.rhs_terms,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            external_phi=external_phi,
        )

    fields0 = deps.compute_fields_fn(
        G0, cache, params, terms=term_cfg, external_phi=external_phi
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
        resolved_diagnostics=resolved_diagnostics,
        kernels=deps.diagnostic_kernels_fn(),
    )

    step = deps.make_explicit_step_fn(
        rhs_fn=rhs_fn,
        method=method,
        project_state=project_state,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        time_step_policy=time_step_policy,
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
        emit_progress_fn=deps.emit_progress_fn,
        use_collision_split=collision_policy.active,
        damping=collision_policy.damping,
        collision_scheme=collision_scheme,
        apply_collision_split_fn=deps.apply_collision_split_fn,
    )

    dt0 = jnp.asarray(
        time_step_policy.update_dt(fields0, time_step_policy.dt_init), dtype=real_dtype
    )
    diag_zero = compute_diag_from_state(G0, fields0, G0, fields0, dt0)

    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    G_final, scan_diag_out = deps.run_explicit_scan_fn(
        step,
        (
            G0,
            G0,
            fields0,
            diag_zero,
            jnp.asarray(0.0, dtype=real_dtype),
            dt0,
        ),
        steps=steps,
        stride=stride,
        sampled_scan=sampled_scan,
        checkpoint=checkpoint,
        sampled_scan_fn=deps.run_sampled_explicit_scan_fn,
    )

    diag, t, dt_series = scan_diag_out
    diag_out = deps.finalize_scan_diagnostics_fn(
        diag,
        t=t,
        dt_series=dt_series,
        stride=stride,
        sampled_scan=sampled_scan,
        resolved_diagnostics=resolved_diagnostics,
    )
    fields_final = deps.compute_fields_fn(
        G_final, cache, params, terms=term_cfg, external_phi=external_phi
    )
    return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


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
