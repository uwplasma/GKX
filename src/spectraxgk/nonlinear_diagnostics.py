"""Diagnostic nonlinear integration entry points and dependency wiring."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import LinearCache, build_linear_cache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution
from spectraxgk.diagnostics import (
    SimulationDiagnostics,
    heat_flux_species,
    heat_flux_resolved_species,
    heat_flux_channel_resolved_species,
    particle_flux_species,
    particle_flux_resolved_species,
    particle_flux_channel_resolved_species,
    phi2_resolved,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
    turbulent_heating_species,
    turbulent_heating_resolved_species,
    fieldline_quadrature_weights,
    magnetic_vector_potential_energy,
    magnetic_vector_potential_energy_resolved,
    distribution_free_energy,
    distribution_free_energy_resolved,
    electrostatic_field_energy,
    electrostatic_field_energy_resolved,
)
from spectraxgk.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    make_nonlinear_diagnostic_tuple_fn,
)
from spectraxgk.operators.nonlinear.diagnostics import (
    finalize_nonlinear_scan_diagnostics,
    maybe_emit_nonlinear_progress,
    run_sampled_explicit_diagnostic_scan,
    select_nonlinear_step_diagnostics,
)
from spectraxgk.operators.nonlinear.policies import (
    _apply_collision_split,
    _collision_damping,
    _diagnostic_omega_mode_mask,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_collision_split_policy,
    build_nonlinear_diagnostic_setup,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from spectraxgk.operators.nonlinear.rhs import nonlinear_em_term_cached_impl
from spectraxgk.solvers.nonlinear.diagnostics import (
    ExplicitNonlinearDiagnosticsDeps,
    IMEXNonlinearDiagnosticsDeps,
    integrate_explicit_nonlinear_diagnostics_impl,
    integrate_imex_nonlinear_diagnostics_impl,
)
from spectraxgk.solvers.nonlinear.explicit import (
    make_explicit_diagnostic_step,
    run_explicit_diagnostic_scan,
)
from spectraxgk.solvers.nonlinear.imex import (
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    run_imex_diagnostic_scan,
    solve_imex_step,
)
from spectraxgk.solvers.time.explicit import (
    _diagnostic_midplane_index,
    _instantaneous_growth_rate_step,
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from spectraxgk.nonlinear_core import _linear_rhs_jit_for_terms, nonlinear_rhs_cached

_EXPLICIT_DIAGNOSTIC_OPTION_KEYS = (
    "method", "cache", "terms", "checkpoint", "sample_stride",
    "diagnostics_stride", "use_dealias_mask", "z_index", "compressed_real_fft",
    "laguerre_mode", "omega_ky_index", "omega_kx_index", "flux_scale",
    "wphi_scale", "fixed_dt", "dt_min", "dt_max", "cfl", "cfl_fac",
    "collision_split", "collision_scheme", "implicit_tol", "implicit_maxiter",
    "implicit_iters", "implicit_relax", "implicit_restart",
    "implicit_solve_method", "implicit_preconditioner", "fixed_mode_ky_index",
    "fixed_mode_kx_index", "external_phi", "resolved_diagnostics",
    "show_progress",
)
_IMEX_DIAGNOSTIC_OPTION_KEYS = tuple(
    key
    for key in _EXPLICIT_DIAGNOSTIC_OPTION_KEYS
    if key not in {"fixed_dt", "dt_min", "dt_max", "cfl", "cfl_fac", "resolved_diagnostics"}
)


def _options_from_scope(scope: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: scope[key] for key in keys}


def _nonlinear_diagnostic_kernels() -> NonlinearDiagnosticKernels:
    """Return diagnostic kernels for dependency-injected nonlinear diagnostics."""

    return NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_instantaneous_growth_rate_step,
        phi2_resolved=phi2_resolved,
        zonal_phi_mode_kxt=zonal_phi_mode_kxt,
        zonal_phi_line_kxt=zonal_phi_line_kxt,
        distribution_free_energy=distribution_free_energy,
        distribution_free_energy_resolved=distribution_free_energy_resolved,
        electrostatic_field_energy=electrostatic_field_energy,
        electrostatic_field_energy_resolved=electrostatic_field_energy_resolved,
        magnetic_vector_potential_energy=magnetic_vector_potential_energy,
        magnetic_vector_potential_energy_resolved=magnetic_vector_potential_energy_resolved,
        heat_flux_species=heat_flux_species,
        heat_flux_resolved_species=heat_flux_resolved_species,
        heat_flux_channel_resolved_species=heat_flux_channel_resolved_species,
        particle_flux_species=particle_flux_species,
        particle_flux_resolved_species=particle_flux_resolved_species,
        particle_flux_channel_resolved_species=particle_flux_channel_resolved_species,
        turbulent_heating_species=turbulent_heating_species,
        turbulent_heating_resolved_species=turbulent_heating_resolved_species,
    )


def _explicit_nonlinear_diagnostics_deps() -> ExplicitNonlinearDiagnosticsDeps:
    """Collect dependencies for explicit diagnostic integration."""

    return ExplicitNonlinearDiagnosticsDeps(
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
        resolve_cfl_fac_fn=resolve_cfl_fac,
        linear_frequency_bound_fn=_linear_frequency_bound,
        laguerre_velocity_max_fn=_laguerre_velocity_max,
        cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
        collision_damping_fn=_collision_damping,
        nonlinear_rhs_fn=nonlinear_rhs_cached,
        compute_fields_fn=compute_fields_cached,
        diagnostic_kernels_fn=_nonlinear_diagnostic_kernels,
        build_diagnostic_setup_fn=build_nonlinear_diagnostic_setup,
        build_time_step_policy_fn=build_nonlinear_time_step_policy,
        build_collision_split_policy_fn=build_nonlinear_collision_split_policy,
        make_diagnostic_tuple_fn=make_nonlinear_diagnostic_tuple_fn,
        make_explicit_step_fn=make_explicit_diagnostic_step,
        run_explicit_scan_fn=run_explicit_diagnostic_scan,
        run_sampled_explicit_scan_fn=run_sampled_explicit_diagnostic_scan,
        finalize_scan_diagnostics_fn=finalize_nonlinear_scan_diagnostics,
        select_step_diagnostics_fn=select_nonlinear_step_diagnostics,
        emit_progress_fn=maybe_emit_nonlinear_progress,
        apply_collision_split_fn=_apply_collision_split,
    )


def _integrate_nonlinear_explicit_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
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
    """Integrate nonlinear system and return runtime diagnostics plus final state."""

    return integrate_explicit_nonlinear_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        deps=_explicit_nonlinear_diagnostics_deps(),
        method=method,
        cache=cache,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        collision_split=collision_split,
        collision_scheme=collision_scheme,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        implicit_preconditioner=implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        external_phi=external_phi,
        resolved_diagnostics=resolved_diagnostics,
        show_progress=show_progress,
    )


def integrate_nonlinear_explicit_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
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
) -> tuple[jnp.ndarray, SimulationDiagnostics]:
    """Integrate nonlinear system and return runtime diagnostics."""

    if method in {"imex", "semi-implicit"}:
        return integrate_nonlinear_imex_diagnostics(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            **_options_from_scope(locals(), _IMEX_DIAGNOSTIC_OPTION_KEYS),
        )

    t, diag_out, _G_final, _fields_final = _integrate_nonlinear_explicit_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        **_options_from_scope(locals(), _EXPLICIT_DIAGNOSTIC_OPTION_KEYS),
    )
    return t, diag_out


def integrate_nonlinear_explicit_diagnostics_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
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
    """Integrate nonlinear system and return runtime diagnostics plus the final state."""

    if method in {"imex", "semi-implicit"}:
        raise ValueError(
            "integrate_nonlinear_explicit_diagnostics_state only supports explicit methods"
        )

    return _integrate_nonlinear_explicit_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        **_options_from_scope(locals(), _EXPLICIT_DIAGNOSTIC_OPTION_KEYS),
    )


def _imex_nonlinear_diagnostics_deps() -> IMEXNonlinearDiagnosticsDeps:
    """Collect dependencies for IMEX diagnostic integration."""

    return IMEXNonlinearDiagnosticsDeps(
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
        linear_rhs_for_terms_fn=_linear_rhs_jit_for_terms,
        build_diagnostic_setup_fn=build_nonlinear_diagnostic_setup,
        build_imex_operator_fn=build_nonlinear_imex_operator,
        build_collision_split_policy_fn=build_nonlinear_collision_split_policy,
        collision_damping_fn=_collision_damping,
        make_imex_nonlinear_term_fn=make_imex_nonlinear_term,
        make_imex_solve_step_fn=make_imex_solve_step,
        solve_imex_step_fn=solve_imex_step,
        make_diagnostic_tuple_fn=make_nonlinear_diagnostic_tuple_fn,
        make_imex_step_fn=make_imex_diagnostic_step,
        run_imex_scan_fn=run_imex_diagnostic_scan,
        finalize_scan_diagnostics_fn=finalize_nonlinear_scan_diagnostics,
        select_step_diagnostics_fn=select_nonlinear_step_diagnostics,
        emit_progress_fn=maybe_emit_nonlinear_progress,
        apply_collision_split_fn=_apply_collision_split,
        compute_fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
        diagnostic_kernels_fn=_nonlinear_diagnostic_kernels,
    )


def integrate_nonlinear_imex_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
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
    """IMEX nonlinear integrator with runtime diagnostics."""

    return integrate_imex_nonlinear_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        deps=_imex_nonlinear_diagnostics_deps(),
        method=method,
        cache=cache,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        collision_split=collision_split,
        collision_scheme=collision_scheme,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        implicit_preconditioner=implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        external_phi=external_phi,
        show_progress=show_progress,
    )


__all__ = [
    "_EXPLICIT_DIAGNOSTIC_OPTION_KEYS",
    "_IMEX_DIAGNOSTIC_OPTION_KEYS",
    "_explicit_nonlinear_diagnostics_deps",
    "_imex_nonlinear_diagnostics_deps",
    "_integrate_nonlinear_explicit_diagnostics_impl",
    "_nonlinear_diagnostic_kernels",
    "_options_from_scope",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_diagnostics",
]
