"""Core nonlinear RHS and cached integrator drivers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.collision import CollisionOperator
from spectraxgk.diagnostics.transport import heat_flux_species
from spectraxgk.diagnostics.moments import fieldline_quadrature_weights
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from spectraxgk.operators.linear.params import LinearParams, _x64_enabled
from spectraxgk.operators.nonlinear.policies import (
    IMEXLinearOperator,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from spectraxgk.operators.nonlinear.projection import (
    _make_hermitian_projector,
    advance_shearing_coordinates,
)
from spectraxgk.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from spectraxgk.solvers.nonlinear.explicit import (
    integrate_cached_explicit_scan,
    integrate_nonlinear_scan,
)
from spectraxgk.solvers.nonlinear.imex import (
    integrate_cached_imex_scan,
    solve_imex_step,
)
from spectraxgk.solvers.time.explicit import (
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from spectraxgk.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution


class ShearedTransportTrace(NamedTuple):
    """Final state and compact heat-flux history from a sheared run."""

    final_state: jnp.ndarray
    time: jnp.ndarray
    heat_flux: jnp.ndarray


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
    collision_operator: CollisionOperator | None = None,
    radial_phase: jnp.ndarray | None = None,
    differentiable: bool = False,
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
        collision_operator=collision_operator,
        radial_phase=radial_phase,
        differentiable=differentiable,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )


def _nonlinear_rhs_scan(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
) -> tuple[jnp.ndarray, FieldState]:
    """Stable scan callable; arrays are dynamic while model switches are static."""

    return nonlinear_rhs_cached(
        G,
        cache,
        params,
        term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
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
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        if collision_operator is not None:
            raise NotImplementedError(
                "custom collision operators currently require explicit nonlinear integration"
            )
        result = integrate_nonlinear_imex_cached(
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
        return result if return_fields else result[0]

    project_state = None
    if compressed_real_fft:
        project_state = _make_hermitian_projector(
            np.asarray(cache.ky), int(np.asarray(cache.kx).size)
        )

    return integrate_cached_explicit_scan(
        G0,
        dt,
        steps,
        method=method,
        rhs_fn=_nonlinear_rhs_scan,
        rhs_args=(cache, params),
        rhs_static_args=(
            term_cfg,
            compressed_real_fft,
            laguerre_mode,
            collision_operator,
        ),
        scan_fn=integrate_nonlinear_scan,
        checkpoint=checkpoint,
        project_state=project_state,
        show_progress=show_progress,
        return_fields=return_fields,
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
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
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
        return_fields=return_fields,
        collision_operator=collision_operator,
    )


def _integrate_nonlinear_sheared_scan(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    record_transport: bool = False,
    return_fields: bool = True,
    flux_scale: float = 1.0,
    differentiable: bool = False,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
) -> tuple[jnp.ndarray, Any]:
    """Run the shared shearing-coordinate scan with optional transport output."""

    if str(grid.boundary).lower() not in {"periodic", "linked"} or bool(
        grid.non_twist
    ):
        raise NotImplementedError(
            "sheared integration requires a periodic or linked standard flux tube"
        )
    if steps < 1:
        raise ValueError("steps must be at least one")
    method_key = str(method).lower()
    if method_key not in {"euler", "rk2", "rk3", "imex"}:
        raise ValueError(
            "sheared integration method must be 'euler', 'rk2', 'rk3', or 'imex'"
        )
    if not fixed_dt and not record_transport:
        raise ValueError("adaptive sheared integration requires a transport trace")
    if method_key == "imex" and not fixed_dt:
        raise ValueError("sheared IMEX integration currently requires fixed_dt=True")
    if method_key == "imex" and collision_operator is not None:
        raise NotImplementedError(
            "sheared IMEX does not yet support custom collision operators"
        )
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            nl, nm = G0.shape[:2]
        elif G0.ndim == 6:
            nl, nm = G0.shape[1:3]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom_eff, params, int(nl), int(nm))

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    base_complex_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_complex_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_value = jnp.asarray(dt, dtype=real_dtype)
    initial_time_value = jnp.asarray(initial_time, dtype=real_dtype)
    initial_dt_value = jnp.asarray(
        dt if initial_dt is None else initial_dt, dtype=real_dtype
    )
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
    coordinate_kx = jnp.asarray(cache.kx, dtype=real_dtype)
    coordinate_ky = jnp.asarray(cache.ky, dtype=real_dtype)
    if int(coordinate_kx.size) > 1:
        coordinate_x0 = 1.0 / jnp.abs(coordinate_kx[1] - coordinate_kx[0])
    else:
        coordinate_x0 = jnp.asarray(grid.x0, dtype=real_dtype) / rho_star
    _, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    # Full-complex shearing coordinates must retain the real-field subspace.
    project_state = _make_hermitian_projector(
        np.asarray(grid.ky), int(np.asarray(grid.kx).size)
    )
    time_step_policy = None
    if not fixed_dt:
        time_step_policy = build_nonlinear_time_step_policy(
            grid,
            geom_eff,
            params,
            cache,
            method=method_key,
            dt=dt,
            steps=steps,
            fixed_dt=False,
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

    def coordinates(state: jnp.ndarray, time: jnp.ndarray, previous: jnp.ndarray):
        update = advance_shearing_coordinates(
            state,
            # The linked-boundary cache may adjust the radial spacing to close
            # the twist-shift chain. Working in the cache's normalized units
            # preserves both that adjustment and the periodic convention.
            kx=coordinate_kx,
            ky=coordinate_ky,
            x0=coordinate_x0,
            shear_rate=shear_rate,
            previous_time=previous,
            time=time,
            dealias_mask=grid.dealias_mask,
        )
        return update._replace(state=project_state(update.state))

    def cache_at(update):
        return update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom_eff,
            params,
            update.effective_kx,
        )

    def rhs_at(update):
        updated_cache = cache_at(update)
        derivative, fields = nonlinear_rhs_cached(
            update.state,
            updated_cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            radial_phase=update.phase,
            differentiable=differentiable,
        )
        return derivative, fields, updated_cache

    def sheared_fields(
        state,
        updated_cache,
        field_params,
        *,
        terms=None,
        external_phi=None,
    ):
        return compute_fields_cached(
            state,
            updated_cache,
            field_params,
            terms=terms,
            use_custom_vjp=not differentiable,
            external_phi=external_phi,
        )

    def fields_at(update):
        updated_cache = cache_at(update)
        fields = sheared_fields(
            update.state,
            updated_cache,
            params,
            terms=term_cfg,
        )
        return fields, updated_cache

    def advance(current, derivative, time, dt_local):
        new_time = time + dt_local
        if method_key == "imex":
            current_cache = cache_at(current)
            nonlinear_term = nonlinear_em_term_cached_impl(
                current.state,
                current_cache,
                params,
                term_cfg,
                external_phi=None,
                compressed_real_fft=compressed_real_fft,
                laguerre_mode=laguerre_mode,
                radial_phase=current.phase,
                fields_fn=sheared_fields,
                nonlinear_contribution_fn=nonlinear_em_contribution,
            )
            endpoint_guess = coordinates(current.state, new_time, time)
            endpoint_rhs = coordinates(
                current.state + dt_local * nonlinear_term,
                new_time,
                time,
            )
            endpoint_cache = cache_at(endpoint_guess)
            operator = build_nonlinear_imex_operator(
                endpoint_guess.state,
                endpoint_cache,
                params,
                dt_local,
                terms=linear_cfg,
                implicit_preconditioner=implicit_preconditioner,
                compressed_real_fft=compressed_real_fft,
                build_implicit_operator_fn=_build_implicit_operator,
            )
            guess = jnp.asarray(endpoint_guess.state, dtype=operator.state_dtype)
            rhs = jnp.asarray(endpoint_rhs.state, dtype=operator.state_dtype)
            if operator.squeeze_species:
                guess = guess[None, ...]
                rhs = rhs[None, ...]
            solution = solve_imex_step(
                guess,
                rhs,
                linear_rhs_fn=linear_rhs_fn,
                cache=endpoint_cache,
                params=params,
                linear_cfg=linear_cfg,
                external_phi=None,
                dt_val=operator.dt_val,
                implicit_iters=implicit_iters,
                implicit_relax=implicit_relax,
                matvec=operator.matvec,
                shape=operator.shape,
                implicit_tol=implicit_tol,
                implicit_maxiter=implicit_maxiter,
                implicit_restart=implicit_restart,
                precond_op=operator.precond_op,
            )
            if operator.squeeze_species:
                solution = solution[0]
            return endpoint_rhs._replace(state=project_state(solution)), new_time
        if method_key == "euler":
            trial = current.state + dt_local * derivative
        elif method_key == "rk2":
            midpoint_time = time + 0.5 * dt_local
            midpoint = coordinates(
                current.state + 0.5 * dt_local * derivative,
                midpoint_time,
                time,
            )
            midpoint_derivative, _, _ = rhs_at(midpoint)
            derivative_in_step_basis = coordinates(
                midpoint_derivative,
                time,
                midpoint_time,
            ).state
            trial = current.state + dt_local * derivative_in_step_basis
        else:
            stage1_time = time + dt_local / 3.0
            stage1 = coordinates(
                current.state + (dt_local / 3.0) * derivative,
                stage1_time,
                time,
            )
            stage1_derivative, _, _ = rhs_at(stage1)
            stage1_derivative_base = coordinates(
                stage1_derivative,
                time,
                stage1_time,
            ).state
            stage2_time = time + 2.0 * dt_local / 3.0
            stage2 = coordinates(
                current.state + (2.0 * dt_local / 3.0) * stage1_derivative_base,
                stage2_time,
                time,
            )
            stage2_derivative, _, _ = rhs_at(stage2)
            stage2_derivative_base = coordinates(
                stage2_derivative,
                time,
                stage2_time,
            ).state
            trial = (
                current.state
                + 0.25 * dt_local * derivative
                + 0.75 * dt_local * (stage2_derivative_base)
            )
        return coordinates(trial, new_time, time), new_time

    def local_dt(current_fields, dt_previous):
        if time_step_policy is None:
            return dt_value
        return time_step_policy.update_dt(current_fields, dt_previous)

    def state_only_step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], index: jnp.ndarray
    ):
        state, adaptive_time, dt_previous = carry
        fixed_time = (
            initial_time_value + jnp.asarray(index, dtype=real_dtype) * dt_value
        )
        time = fixed_time if fixed_dt else adaptive_time
        current = coordinates(state, time, time)
        if method_key == "imex":
            derivative = jnp.zeros_like(current.state)
            current_fields = None
        else:
            derivative, current_fields, _ = rhs_at(current)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time = advance(current, derivative, time, dt_local)
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
        )
        return next_carry, None

    def endpoint_step(carry, index: jnp.ndarray):
        state, adaptive_time, dt_previous, derivative, current_fields = carry
        del index
        time = adaptive_time
        current = coordinates(state, time, time)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time = advance(current, derivative, time, dt_local)
        if method_key == "imex":
            next_derivative = jnp.zeros_like(advanced.state)
            fields, advanced_cache = fields_at(advanced)
        else:
            next_derivative, fields, advanced_cache = rhs_at(advanced)
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
            jnp.asarray(next_derivative, dtype=state_dtype),
            fields,
        )
        if not record_transport:
            return next_carry, fields
        apar = jnp.zeros_like(fields.phi) if fields.apar is None else fields.apar
        bpar = jnp.zeros_like(fields.phi) if fields.bpar is None else fields.bpar
        heat_flux = heat_flux_species(
            advanced.state,
            fields.phi,
            apar,
            bpar,
            advanced_cache,
            grid,
            params,
            flux_fac,
            use_dealias=True,
            flux_scale=flux_scale,
        )
        return next_carry, (new_time, heat_flux)

    initial_state_carry = (
        jnp.asarray(project_state(G0), dtype=state_dtype),
        initial_time_value,
        initial_dt_value,
    )
    if not record_transport and not return_fields:
        state_final_carry, output = jax.lax.scan(
            state_only_step, initial_state_carry, jnp.arange(steps)
        )
        return state_final_carry[0], output

    initial_update = coordinates(
        initial_state_carry[0], initial_state_carry[1], initial_state_carry[1]
    )
    if method_key == "imex":
        initial_derivative = jnp.zeros_like(initial_update.state)
        initial_fields, _ = fields_at(initial_update)
    else:
        initial_derivative, initial_fields, _ = rhs_at(initial_update)
    # Some field-solve policies use a lower internal precision. Match the scan
    # carry to the requested state precision just as every subsequent step does.
    initial_endpoint_carry = initial_state_carry + (
        jnp.asarray(initial_derivative, dtype=state_dtype),
        initial_fields,
    )
    endpoint_final_carry, output = jax.lax.scan(
        endpoint_step, initial_endpoint_carry, jnp.arange(steps)
    )
    return endpoint_final_carry[0], output


def integrate_nonlinear_sheared(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    differentiable: bool = False,
    return_fields: bool = True,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the standard-flux-tube shearing-coordinate foundation.

    This research path supports fixed-step Euler, midpoint RK2, three-stage
    Heun RK3, and first-order IMEX. Stage states and derivatives are remapped to
    the stage coordinate basis before the RHS and back to the step basis before
    Runge--Kutta combinations. IMEX evaluates the explicit nonlinear term in the
    current basis and rebuilds the implicit linear operator in the endpoint
    basis. ``compressed_real_fft`` evaluates the nonlinear bracket in the
    equivalent canonical shearing-coordinate representation.
    """

    final_state, fields = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        differentiable=differentiable,
        return_fields=return_fields,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    if return_fields:
        return final_state, fields
    return final_state


def integrate_nonlinear_sheared_transport(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    flux_scale: float = 1.0,
    differentiable: bool = True,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
) -> ShearedTransportTrace:
    """Integrate a sheared run and record canonical heat flux at every step.

    With ``fixed_dt=False``, ``steps`` is the accepted-step budget and ``time``
    records the resulting nonuniform physical-time grid. ``initial_time`` and
    ``initial_dt`` continue a prior trace without resetting the shearing basis.
    """

    final_state, samples = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        record_transport=True,
        flux_scale=flux_scale,
        differentiable=differentiable,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        initial_time=initial_time,
        initial_dt=initial_dt,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    time, heat_flux = samples
    return ShearedTransportTrace(final_state, time, heat_flux)


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
        implicit_preconditioner=implicit_preconditioner,
        implicit_operator=implicit_operator,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        show_progress=show_progress,
    )


__all__ = [
    "_linear_rhs_jit_for_terms",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_sheared",
    "integrate_nonlinear_sheared_transport",
    "nonlinear_rhs_cached",
    "ShearedTransportTrace",
]
