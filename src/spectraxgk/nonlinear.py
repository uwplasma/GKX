"""Nonlinear gyrokinetic drivers built on term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

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
    term_config_to_linear_terms,
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
    compute_nonlinear_diagnostic_tuple,
)
from spectraxgk.nonlinear_diagnostics import (
    _pack_resolved_diagnostics,  # noqa: F401 - compatibility re-export
    _sample_axis0,  # noqa: F401 - compatibility re-export
    _sample_indices_with_final,
    build_nonlinear_simulation_diagnostics,
)
from spectraxgk.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from spectraxgk.solvers.nonlinear.explicit import advance_explicit_nonlinear_state
from spectraxgk.solvers.nonlinear.imex import (
    advance_imex_nonlinear_state,
    solve_imex_step,
)
from spectraxgk.nonlinear_helpers import (
    IMEXLinearOperator,
    _apply_collision_split,
    _collision_damping,
    _nonlinear_cfl_frequency_components,
    _diagnostic_omega_mode_mask,
    _make_fixed_mode_projector,  # noqa: F401 - compatibility re-export
    _make_hermitian_projector,
    _make_nonlinear_state_projector,
    build_nonlinear_imex_operator,
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

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        raise ValueError(
            "Final-state runtime diagnostics helper only supports explicit methods"
        )
    vol_fac, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    mask = _diagnostic_omega_mode_mask(
        grid, cache, compressed_real_fft=compressed_real_fft
    )
    z_idx = _diagnostic_midplane_index(grid.z.size) if z_index is None else int(z_index)
    use_dealias = bool(use_dealias_mask)
    _project_state = _make_nonlinear_state_projector(
        G0,
        ky_vals=np.asarray(grid.ky),
        nx=int(grid.kx.size),
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    G0 = _project_state(G0)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    progress_total = (
        jnp.asarray(float(steps) * float(dt), dtype=real_dtype)
        if fixed_dt
        else jnp.asarray(jnp.nan, dtype=real_dtype)
    )
    dt_min_val = jnp.asarray(dt_min, dtype=real_dtype)
    # Explicit-time default behavior: when dt_max is unset, dt_max == dt.
    dt_max_val = jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype)
    cfl_val = jnp.asarray(cfl, dtype=real_dtype)
    cfl_fac_val = jnp.asarray(resolve_cfl_fac(method, cfl_fac), dtype=real_dtype)

    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx_np = np.asarray(cache.kx, dtype=float)
    ky_np = np.asarray(cache.ky, dtype=float)
    kx_max = float(abs(kx_np[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky_np[(ny - 1) // 3])) if ny > 1 else 0.0
    vtmax = float(np.max(np.abs(np.asarray(params.vth, dtype=float))))
    tzmax = float(np.max(np.abs(np.asarray(params.tz, dtype=float))))
    nl = int(cache.l.shape[0])
    nm = int(cache.m.shape[1])
    vpar_max = 2.0 * float(np.sqrt(max(nm, 1))) * vtmax
    muB_max = _laguerre_velocity_max(nl) * tzmax
    kxfac_val = float(np.asarray(cache.kxfac))
    linear_omega = jnp.asarray(
        _linear_frequency_bound(
            grid,
            geom_eff,
            params,
            nl,
            nm,
            include_diamagnetic_drive=False,
        ),
        dtype=real_dtype,
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

    def _update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return jnp.asarray(dt_prev, dtype=real_dtype)
        omega_nl_x, omega_nl_y = _nonlinear_cfl_frequency_components(
            fields_state,
            grid,
            cache,
            compressed_real_fft=compressed_real_fft,
            kx_max=kx_max,
            ky_max=ky_max,
            kxfac=kxfac_val,
            vpar_max=vpar_max,
            muB_max=muB_max,
        )
        wmax = (
            jnp.maximum(linear_omega[0], omega_nl_x)
            + jnp.maximum(linear_omega[1], omega_nl_y)
            + linear_omega[2]
        )
        dt_guess = jnp.where(wmax > 0.0, cfl_fac_val * cfl_val / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, dt_min_val, dt_max_val), dtype=real_dtype)

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

    diagnostic_kernels = _nonlinear_diagnostic_kernels()

    def _compute_diag_from_state(
        G_state: jnp.ndarray,
        fields_state: FieldState,
        G_prev_step: jnp.ndarray,
        fields_prev_step: FieldState,
        dt_step: jnp.ndarray,
    ):
        return compute_nonlinear_diagnostic_tuple(
            G_state,
            fields_state,
            G_prev_step,
            fields_prev_step,
            dt_step,
            grid=grid,
            cache=cache,
            params=params,
            vol_fac=vol_fac,
            flux_fac=flux_fac,
            mask=mask,
            z_idx=z_idx,
            use_dealias=use_dealias,
            real_dtype=real_dtype,
            omega_ky_index=omega_ky_index,
            omega_kx_index=omega_kx_index,
            flux_scale=flux_scale,
            wphi_scale=wphi_scale,
            resolved_diagnostics=resolved_diagnostics,
            kernels=diagnostic_kernels,
        )

    def step(carry, idx):
        G, G_prev_step, fields_prev_step, diag_prev, t_prev, dt_prev = carry
        dG, fields = rhs_fn(G)
        dt_local = jnp.asarray(_update_dt(fields, dt_prev), dtype=real_dtype)
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

        def _compute_diag(_):
            return _compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_local
            )

        def _reuse_diag(_):
            return diag_prev

        diag_stride = int(max(diagnostics_stride, 1))
        do_diag = (idx % diag_stride) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            gamma_cb, omega_cb = diag[0], diag[1]
            Wg_cb, Wphi_cb = diag[2], diag[3]
            G_new = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    gamma_cb,
                    omega_cb,
                    Wphi_cb,
                    Wg_cb,
                    t_new,
                    progress_total,
                ),
                lambda state: state,
                G_new,
            )
        return (G_new, G_new, fields_new, diag, t_new, dt_local), (
            diag,
            t_new,
            dt_local,
        )

    step_fn = jax.checkpoint(step) if checkpoint else step
    dt0 = jnp.asarray(_update_dt(fields0, dt_init), dtype=real_dtype)
    diag_zero = _compute_diag_from_state(G0, fields0, G0, fields0, dt0)

    stride = int(max(sample_stride, diagnostics_stride, 1))
    sampled_scan = stride > 1 and jax.default_backend() != "cpu"
    if sampled_scan:
        sample_idx_raw = _sample_indices_with_final(int(steps), stride)
        sampled_step_idx = np.asarray(
            sample_idx_raw
            if not isinstance(sample_idx_raw, slice)
            else np.arange(steps),
            dtype=np.int32,
        )
        sample_steps = sampled_step_idx + np.int32(1)
        intervals = np.diff(
            np.concatenate([np.asarray([0], dtype=np.int32), sample_steps])
        ).astype(np.int32)

        def sample_interval(carry, interval_steps):
            def run_one_step(_i, inner_carry):
                G_i, G_prev_i, fields_prev_i, diag_prev_i, t_i, dt_i, idx_i = (
                    inner_carry
                )
                next_carry, _diag_step = step_fn(
                    (G_i, G_prev_i, fields_prev_i, diag_prev_i, t_i, dt_i), idx_i
                )
                G_next, G_prev_next, fields_prev_next, diag_next, t_next, dt_next = (
                    next_carry
                )
                return (
                    G_next,
                    G_prev_next,
                    fields_prev_next,
                    diag_next,
                    t_next,
                    dt_next,
                    idx_i + 1,
                )

            carry_next = jax.lax.fori_loop(0, interval_steps, run_one_step, carry)
            (
                G_next,
                _G_prev_next,
                _fields_prev_next,
                diag_next,
                t_next,
                dt_next,
                _idx_next,
            ) = carry_next
            return carry_next, (diag_next, t_next, dt_next)

        (
            (
                G_final,
                _G_prev_last,
                _fields_prev_last,
                _diag_last,
                _t_last,
                _dt_last,
                _idx_last,
            ),
            diag_out,
        ) = jax.lax.scan(
            sample_interval,
            (
                G0,
                G0,
                fields0,
                diag_zero,
                jnp.asarray(0.0, dtype=real_dtype),
                dt0,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            jnp.asarray(intervals, dtype=jnp.int32),
            length=int(intervals.size),
        )
    else:
        idx = jnp.arange(steps, dtype=jnp.int32)
        (
            (G_final, _G_prev_last, _fields_prev_last, _diag_last, _t_last, _dt_last),
            diag_out,
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

    diag, t, dt_series = diag_out
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

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)

    vol_fac, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    mask = _diagnostic_omega_mode_mask(
        grid, cache, compressed_real_fft=compressed_real_fft
    )
    z_idx = _diagnostic_midplane_index(grid.z.size) if z_index is None else int(z_index)
    use_dealias = bool(use_dealias_mask)
    _project_state = _make_nonlinear_state_projector(
        G0,
        ky_vals=np.asarray(grid.ky),
        nx=int(grid.kx.size),
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )

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

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        return nonlinear_em_term_cached_impl(
            G_in,
            cache,
            params,
            term_cfg,
            real_dtype=real_dtype,
            external_phi=external_phi,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            fields_fn=compute_fields_cached,
            nonlinear_contribution_fn=nonlinear_em_contribution,
        )

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        return solve_imex_step(
            G_in,
            G_rhs,
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
        )

    diagnostic_kernels = _nonlinear_diagnostic_kernels()

    def _compute_diag_from_state(
        G_state: jnp.ndarray,
        fields_state: FieldState,
        G_prev_step: jnp.ndarray,
        fields_prev_step: FieldState,
        dt_step: jnp.ndarray,
    ):
        return compute_nonlinear_diagnostic_tuple(
            G_state,
            fields_state,
            G_prev_step,
            fields_prev_step,
            dt_step,
            grid=grid,
            cache=cache,
            params=params,
            vol_fac=vol_fac,
            flux_fac=flux_fac,
            mask=mask,
            z_idx=z_idx,
            use_dealias=use_dealias,
            real_dtype=real_dtype,
            omega_ky_index=omega_ky_index,
            omega_kx_index=omega_kx_index,
            flux_scale=flux_scale,
            wphi_scale=wphi_scale,
            resolved_diagnostics=True,
            kernels=diagnostic_kernels,
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

        def _compute_diag(_):
            return _compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_val
            )

        def _reuse_diag(_):
            return diag_prev

        diag_stride = int(max(diagnostics_stride, 1))
        do_diag = (idx % diag_stride) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            gamma_cb, omega_cb = diag[0], diag[1]
            Wg_cb, Wphi_cb = diag[2], diag[3]
            G_new = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    gamma_cb,
                    omega_cb,
                    Wphi_cb,
                    Wg_cb,
                    t_new,
                    progress_total,
                ),
                lambda state: state,
                G_new,
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

    linear_terms = term_config_to_linear_terms(linear_cfg)

    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    if implicit_operator is None:
        G, shape, _size, dt_val, precond_op, matvec, squeeze_species = (
            _build_implicit_operator(
                G0,
                cache,
                params,
                dt,
                linear_terms,
                implicit_preconditioner,
            )
        )
    else:
        shape = implicit_operator.shape
        dt_val = implicit_operator.dt_val
        precond_op = implicit_operator.precond_op
        matvec = implicit_operator.matvec
        squeeze_species = implicit_operator.squeeze_species
        G = jnp.asarray(G0, dtype=implicit_operator.state_dtype)
        if squeeze_species and G.ndim == len(shape) - 1:
            G = G[None, ...]
        if G.shape != shape:
            raise ValueError(
                "implicit_operator shape mismatch: "
                f"expected {shape}, got {tuple(G.shape)}"
            )

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        return nonlinear_em_term_cached_impl(
            G_in,
            cache,
            params,
            term_cfg,
            external_phi=external_phi,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            fields_fn=compute_fields_cached,
            nonlinear_contribution_fn=nonlinear_em_contribution,
        )

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        return solve_imex_step(
            G_in,
            G_rhs,
            linear_rhs_fn=linear_rhs_fn,
            cache=cache,
            params=params,
            linear_cfg=linear_cfg,
            external_phi=external_phi,
            dt_val=dt_val,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            matvec=matvec,
            shape=shape,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            precond_op=precond_op,
        )

    def step(G_in, _):
        rhs = G_in + dt_val * nonlinear_term(G_in)
        G_new = solve_step(G_in, rhs)
        _dG_new, fields_new = linear_rhs_fn(
            G_new, cache, params, linear_cfg, external_phi=external_phi
        )
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    G_out, fields_t = jax.lax.scan(step_fn, G, None, length=steps)
    G_out = G_out[0] if squeeze_species else G_out
    return G_out, fields_t
