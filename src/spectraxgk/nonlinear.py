"""Nonlinear gyrokinetic drivers built on term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.grids import SpectralGrid
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache import LinearCache, build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
)
from spectraxgk.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import integrate_nonlinear as integrate_nonlinear_scan
from spectraxgk.terms.nonlinear import nonlinear_em_contribution
from spectraxgk.explicit_time_integrators import (
    _instantaneous_growth_rate_step,
    _laguerre_velocity_max,
    _linear_frequency_bound,
    _diagnostic_midplane_index,
)
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
from spectraxgk.nonlinear_diagnostics import (
    _pack_resolved_diagnostics,  # noqa: F401 - compatibility re-export
    _sample_axis0,  # noqa: F401 - compatibility re-export
    _sample_indices_with_final,
    build_nonlinear_simulation_diagnostics,
    maybe_emit_nonlinear_progress,
    run_sampled_explicit_diagnostic_scan,
    sampled_scan_intervals,  # noqa: F401 - compatibility re-export
    select_nonlinear_step_diagnostics,
)
from spectraxgk.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from spectraxgk.solvers.nonlinear.explicit import advance_explicit_nonlinear_state
from spectraxgk.solvers.nonlinear.imex import (
    advance_imex_nonlinear_state,
    integrate_cached_imex_scan,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    solve_imex_step,
)
from spectraxgk.nonlinear_helpers import (
    IMEXLinearOperator,
    NonlinearDiagnosticSetup,  # noqa: F401 - compatibility re-export
    NonlinearTimeStepPolicy,  # noqa: F401 - compatibility re-export
    _apply_collision_split,
    _collision_damping,
    _nonlinear_cfl_frequency_components,
    _diagnostic_omega_mode_mask,
    _make_fixed_mode_projector,  # noqa: F401 - compatibility re-export
    _make_hermitian_projector,
    _make_nonlinear_state_projector,  # noqa: F401 - compatibility re-export
    build_nonlinear_diagnostic_setup,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)

def _nonlinear_diagnostic_kernels() -> NonlinearDiagnosticKernels:
    """Return facade-level diagnostic kernels for compatibility monkeypatch seams."""

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


def _linear_rhs_jit_for_terms(term_cfg: TermConfig):
    """Return the narrowest compiled linear RHS path compatible with ``term_cfg``."""

    return linear_rhs_jit_for_terms_impl(
        term_cfg,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
    )


def nonlinear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Compute the assembled nonlinear RHS and electromagnetic field state."""

    return nonlinear_rhs_cached_impl(
        G,
        cache,
        params,
        terms,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )


def integrate_nonlinear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        return integrate_nonlinear_imex_cached(
            G0,
            cache,
            params,
            dt,
            steps,
            terms=term_cfg,
            checkpoint=checkpoint,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            show_progress=show_progress,
        )

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )

    project_state = None
    if compressed_real_fft:
        project_state = _make_hermitian_projector(
            np.asarray(cache.ky), int(np.asarray(cache.kx).size)
        )

    return integrate_nonlinear_scan(
        rhs_fn,
        G0,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        project_state=project_state,
        show_progress=show_progress,
    )


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using built-in cache construction."""

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)
    return integrate_nonlinear_cached(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        show_progress=show_progress,
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

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        raise ValueError(
            "Final-state runtime diagnostics helper only supports explicit methods"
        )
    setup = build_nonlinear_diagnostic_setup(
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
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
    )
    geom_eff = setup.geom
    cache = setup.cache
    _project_state = setup.project_state

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    G0 = _project_state(G0)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    time_step_policy = build_nonlinear_time_step_policy(
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
        resolve_cfl_fac_fn=resolve_cfl_fac,
        linear_frequency_bound_fn=_linear_frequency_bound,
        laguerre_velocity_max_fn=_laguerre_velocity_max,
        cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
    )
    squeeze_species = G0.ndim == 5
    use_collision_split = bool(collision_split) and (
        float(term_cfg.collisions) != 0.0 or float(term_cfg.hypercollisions) != 0.0
    )
    rhs_term_cfg = (
        replace(term_cfg, collisions=0.0, hypercollisions=0.0)
        if use_collision_split
        else term_cfg
    )
    damping = None
    if use_collision_split:
        damping = _collision_damping(
            cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species
        )

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            cache,
            params,
            rhs_term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            external_phi=external_phi,
        )

    fields0 = compute_fields_cached(
        G0, cache, params, terms=term_cfg, external_phi=external_phi
    )

    _compute_diag_from_state = make_nonlinear_diagnostic_tuple_fn(
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
        kernels=_nonlinear_diagnostic_kernels(),
    )

    def step(carry, idx):
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
            project_state=_project_state,
            state_dtype=state_dtype,
        )
        if use_collision_split and damping is not None:
            G_new = _apply_collision_split(G_new, damping, dt_local, collision_scheme)
        G_new = _project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_local, dtype=real_dtype)
        fields_new = compute_fields_cached(
            G_new, cache, params, terms=term_cfg, external_phi=external_phi
        )

        def _compute_diag():
            return _compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_local
            )

        diag = select_nonlinear_step_diagnostics(
            idx,
            diagnostics_stride=diagnostics_stride,
            diag_prev=diag_prev,
            compute_diag_fn=_compute_diag,
        )
        G_new = maybe_emit_nonlinear_progress(
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

    step_fn = jax.checkpoint(step) if checkpoint else step
    dt0 = jnp.asarray(
        time_step_policy.update_dt(fields0, time_step_policy.dt_init), dtype=real_dtype
    )
    diag_zero = _compute_diag_from_state(G0, fields0, G0, fields0, dt0)

    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
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
        ) = run_sampled_explicit_diagnostic_scan(
            step_fn,
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
        )
    else:
        idx = jnp.arange(steps, dtype=jnp.int32)
        (
            (G_final, _G_prev_last, _fields_prev_last, _diag_last, _t_last, _dt_last),
            scan_diag_out,
        ) = jax.lax.scan(
            step_fn,
            (
                G0,
                G0,
                fields0,
                diag_zero,
                jnp.asarray(0.0, dtype=real_dtype),
                dt0,
            ),
            idx,
            length=steps,
        )

    diag, t, dt_series = scan_diag_out
    output_sample_idx = None
    if stride > 1 and not sampled_scan:
        output_sample_idx = _sample_indices_with_final(int(t.shape[0]), stride)
    diag_out = build_nonlinear_simulation_diagnostics(
        diag,
        t=t,
        dt_series=dt_series,
        resolved_diagnostics=resolved_diagnostics,
        sample_indices=output_sample_idx,
    )
    fields_final = compute_fields_cached(
        G_final, cache, params, terms=term_cfg, external_phi=external_phi
    )
    return jnp.asarray(diag_out.t), diag_out, G_final, fields_final


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

    t, diag_out, _G_final, _fields_final = (
        _integrate_nonlinear_explicit_diagnostics_impl(
            G0,
            grid,
            geom,
            params,
            dt,
            steps,
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

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)

    setup = build_nonlinear_diagnostic_setup(
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
        ensure_geometry_fn=ensure_flux_tube_geometry_data,
        build_cache_fn=build_linear_cache,
        quadrature_weights_fn=fieldline_quadrature_weights,
        omega_mask_fn=_diagnostic_omega_mode_mask,
        midplane_index_fn=_diagnostic_midplane_index,
    )
    cache = setup.cache
    _project_state = setup.project_state

    initial_state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=initial_state_dtype)
    G0 = _project_state(G0)

    implicit_operator = build_nonlinear_imex_operator(
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
    use_collision_split = bool(collision_split) and (
        float(term_cfg.collisions) != 0.0 or float(term_cfg.hypercollisions) != 0.0
    )
    damping = None
    if use_collision_split:
        damping = _collision_damping(
            cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species
        )

    nonlinear_term = make_imex_nonlinear_term(
        cache,
        params,
        term_cfg,
        real_dtype=real_dtype,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )
    solve_step = make_imex_solve_step(
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
        solve_step_fn=solve_imex_step,
    )

    _compute_diag_from_state = make_nonlinear_diagnostic_tuple_fn(
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
        kernels=_nonlinear_diagnostic_kernels(),
    )

    fields0 = compute_fields_cached(
        G0, cache, params, terms=term_cfg, external_phi=external_phi
    )

    def step(carry, idx):
        G, G_prev_step, fields_prev_step, diag_prev, t_prev = carry
        G_new = advance_imex_nonlinear_state(
            G,
            dt_val=dt_val,
            method=method,
            nonlinear_term=nonlinear_term,
            solve_step=solve_step,
            project_state=_project_state,
        )
        if use_collision_split and damping is not None:
            G_new = _apply_collision_split(G_new, damping, dt_val, collision_scheme)
        G_new = _project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = t_prev + dt_val
        fields_new = compute_fields_cached(
            G_new, cache, params, terms=term_cfg, external_phi=external_phi
        )

        def _compute_diag():
            return _compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_val
            )

        diag = select_nonlinear_step_diagnostics(
            idx,
            diagnostics_stride=diagnostics_stride,
            diag_prev=diag_prev,
            compute_diag_fn=_compute_diag,
        )
        G_new = maybe_emit_nonlinear_progress(
            G_new,
            show_progress=show_progress,
            diag=diag,
            idx=idx,
            steps=steps,
            t_new=t_new,
            progress_total=progress_total,
        )
        return (G_new, G_new, fields_new, diag, t_new), (diag, t_new)

    step_fn = jax.checkpoint(step) if checkpoint else step
    diag_zero = _compute_diag_from_state(G0, fields0, G0, fields0, dt_val)
    idx = jnp.arange(steps, dtype=jnp.int32)
    (G_final, _G_prev_last, _fields_prev_last, _diag_last, _t_last), diag_out = (
        jax.lax.scan(
            step_fn,
            (
                G0,
                G0,
                fields0,
                diag_zero,
                jnp.asarray(0.0, dtype=real_dtype),
            ),
            idx,
            length=steps,
        )
    )

    diag, t = diag_out
    dt_series = jnp.ones_like(t) * dt_val

    stride = int(max(sample_stride, diagnostics_stride, 1))
    output_sample_idx = None
    if stride > 1:
        output_sample_idx = _sample_indices_with_final(int(t.shape[0]), stride)
    diag_out = build_nonlinear_simulation_diagnostics(
        diag,
        t=t,
        dt_series=dt_series,
        resolved_diagnostics=True,
        sample_indices=output_sample_idx,
        resolved_to_numpy=True,
    )
    return jnp.asarray(diag_out.t), diag_out


def integrate_nonlinear_imex_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    implicit_operator: IMEXLinearOperator | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    return integrate_cached_imex_scan(
        G0,
        cache,
        params,
        dt,
        steps,
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_nonlinear_imex_operator,
        build_implicit_operator_fn=_build_implicit_operator,
        fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
        checkpoint=checkpoint,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        implicit_preconditioner=implicit_preconditioner,
        implicit_operator=implicit_operator,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        show_progress=show_progress,
    )
