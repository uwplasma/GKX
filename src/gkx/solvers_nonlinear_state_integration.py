"""Core nonlinear RHS and cached integrator drivers."""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gkx.config import resolve_cfl_fac
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.core_ky_layout import source_ny_full
from gkx.core_grid import SpectralGrid, _gyrokinetic_moment_shape
from gkx.operators.collision import CollisionOperator
from gkx.operators.fluxes import heat_flux_species, heat_flux_total
from gkx.operators.moments import fieldline_quadrature_weights
from gkx.solvers_linear_implicit import (
    ImplicitSolveStats,
    _build_implicit_operator,
    _empty_implicit_solve_stats,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.linked import mask_supplied_state
from gkx.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from gkx.operators.linear.params import LinearParams, _x64_enabled
from gkx.operators.nonlinear.policies import (
    IMEXLinearOperator,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from gkx.operators.nonlinear.projection import (
    _make_compressed_real_fft_projector,
    _make_hermitian_projector,
    advance_shearing_coordinates,
    hermitian_projector_for_signature,
    hermitian_projector_signature,
)
from gkx.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from gkx.solvers_nonlinear_explicit import (
    advance_explicit_nonlinear_state,
    checkpointed_explicit_scan,
    integrate_cached_explicit_scan,
    integrate_nonlinear_scan,
)
from gkx.solvers_nonlinear_imex import (
    integrate_cached_imex_scan,
    solve_imex_step_with_stats,
)
from gkx.solvers_time_explicit import (
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from gkx.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.nonlinear import nonlinear_em_contribution


#: Longest window whose discrete adjoint has been measured to still behave like
#: a gradient on the shipped saturated Cyclone case: the AD/FD ladder in
#: ``tools/campaigns/nonlinear_gradient_window.py`` tracks centered differences
#: through 1024 RK3 steps and departs between 1024 and 2048. It is a property of
#: that trajectory's Lyapunov time, not a solver tolerance, so it is a default to
#: warn against and remeasure -- not a hard limit. ``examples/optimization/
#: QA_optimization.py`` runs at exactly 1024, one rung below the departure.
DIVERGENCE_KNEE_STEPS = 1024


def _warn_if_window_exceeds_divergence_knee(
    steps: int, knee: int | None = DIVERGENCE_KNEE_STEPS
) -> None:
    """Warn when a differentiated window runs past the measured knee."""

    if knee is None or steps <= int(knee):
        return
    warnings.warn(
        f"differentiating a {steps}-step window, above the measured divergence "
        f"knee of {knee} steps: past it the windowed adjoint grows with the "
        "leading Lyapunov exponent and is large, reproducible, and not a "
        "descent direction. Remeasure the knee for this case with "
        "tools/campaigns/nonlinear_gradient_window.py, then pass "
        "divergence_knee_steps=<measured> or None.",
        RuntimeWarning,
        stacklevel=2,
    )


class ShearedTransportTrace(NamedTuple):
    """Final state and compact heat-flux history from a sheared run.

    ``solve_stats`` is the convergence summary of every implicit GMRES solve of
    a ``method="imex"`` run, and is populated only when the caller asked for it
    with ``return_solve_stats=True``; it stays ``None`` for explicit methods and
    for runs that did not ask, so a graph that does not read it is unchanged.
    """

    final_state: jnp.ndarray
    time: jnp.ndarray
    heat_flux: jnp.ndarray
    solve_stats: ImplicitSolveStats | None = None


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the nonlinear system using a cached geometry object.

    ``return_solve_stats=True`` appends the IMEX implicit solve summary
    (``None`` for explicit methods, which have no implicit solve).

    ``G0`` is a state this driver did not build, so on a linked (twist-shift)
    deck it is projected onto the chain cover before the run starts, exactly as
    the runtime does at intake (queue row Q19) and as the entry points above
    this one do (Q23). The projection is a ``where`` against a mask fixed by the
    deck's topology, applied once here and outside the compiled scan, so the
    scan's graph is unchanged and periodic or full-cover decks get the same
    array object back.
    """

    term_cfg = terms or TermConfig()
    G0 = mask_supplied_state(G0, cache)
    if method in {"imex", "semi-implicit"}:
        if collision_operator is not None:
            raise NotImplementedError(
                "custom collision operators currently require explicit nonlinear integration"
            )
        G_out, fields_t, solve_stats = integrate_nonlinear_imex_cached(
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
            return_solve_stats=True,
        )
        head = (G_out, fields_t) if return_fields else (G_out,)
        if return_solve_stats:
            return (*head, solve_stats)
        return head if return_fields else G_out

    project_state = None
    if compressed_real_fft:
        project_state = _make_compressed_real_fft_projector(
            ny_full=source_ny_full(cache),
            nx=int(cache.kx.size),
            rows=int(cache.ky.size),
        )

    result = integrate_cached_explicit_scan(
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
    if not return_solve_stats:
        return result
    return (*result, None) if return_fields else (result, None)


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the nonlinear system using built-in cache construction.

    A supplied ``G0`` is projected onto the linked chain cover by
    :func:`integrate_nonlinear_cached`, which this driver builds the cache for
    and then delegates to, so both raw drivers take the same view of a state
    they did not build.
    """

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        Nl, Nm = _gyrokinetic_moment_shape(G0)
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
        return_solve_stats=return_solve_stats,
    )


def _identity_state(state: jnp.ndarray) -> jnp.ndarray:
    """Use a stable function identity for the compiled window's projector."""

    return state


@partial(
    jax.jit,
    static_argnames=(
        "dt",
        "count",
        "tail",
        "method",
        "term_cfg",
        "compressed_real_fft",
        "laguerre_mode",
        "collision_operator",
        "checkpoint",
        "projector_signature",
    ),
)
def _nonlinear_heat_flux_window_total(
    initial_state: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_factor: jnp.ndarray,
    initial_heat: jnp.ndarray,
    indices: jnp.ndarray,
    *,
    dt: float,
    count: int,
    tail: int,
    method: str,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
    checkpoint: bool,
    projector_signature: tuple[int, bool, int] | None,
) -> jnp.ndarray:
    """Compile a reusable heat-flux sum for fixed shapes and static options.

    Arrays are operands, not captured geometry constants. A module-level JIT
    avoids rebuilding eager scan/cond executables on each objective call and
    permits reuse through ``value_and_grad``. Projector layout is static and
    has no derivative; configuration or topology changes may recompile.

    Keep ``total_heat / tail`` outside this graph: fusing the mean into the
    accumulation changed float32 rounding in the reference comparison.
    """

    project_state: Callable[[jnp.ndarray], jnp.ndarray] = (
        _identity_state
        if projector_signature is None
        else hermitian_projector_for_signature(projector_signature)
    )

    def rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        return nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            differentiable=True,
        )

    def advance(carry: tuple[jnp.ndarray, jnp.ndarray], index: jnp.ndarray):
        state, total_heat = carry
        state = project_state(state)
        derivative, _ = rhs(state)
        next_state = advance_explicit_nonlinear_state(
            state,
            derivative,
            jnp.asarray(dt, dtype=jnp.real(state).dtype),
            method=method,
            rhs_fn=rhs,
            project_state=project_state,
            state_dtype=state.dtype,
        )
        _, fields = rhs(next_state)
        zero = jnp.zeros_like(fields.phi)
        heat = heat_flux_total(
            next_state,
            fields.phi,
            zero if fields.apar is None else fields.apar,
            zero if fields.bpar is None else fields.bpar,
            cache,
            grid,
            params,
            flux_factor,
        )
        include = jnp.asarray(index >= count - tail, dtype=heat.dtype)
        return (next_state, total_heat + include * heat), None

    (_, total_heat), _ = checkpointed_explicit_scan(
        advance,
        (jax.lax.stop_gradient(initial_state), initial_heat),
        indices,
        checkpoint=checkpoint,
    )
    return total_heat


def nonlinear_heat_flux_window(
    saturated_state: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    method: str = "rk2",
    tail_steps: int | None = None,
    checkpoint: bool = True,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    divergence_knee_steps: int | None = DIVERGENCE_KNEE_STEPS,
) -> jnp.ndarray:
    r"""Mean physical heat flux over a differentiable nonlinear window.

    ``saturated_state`` is deliberately detached: first run GKX to saturation,
    then differentiate this finite window through ``geom`` and ``params``.
    Reverse-mode AD follows the exact discrete Runge--Kutta map. Block
    checkpointing retains :math:`O(\sqrt{N})` distribution states instead of
    :math:`O(N)` for a window of ``N`` steps.

    On a linked (twist-shift) deck ``saturated_state`` is projected onto the
    linked chain cover before the window starts (queue row Q23): it is a state
    this function did not build, and the ExB bracket takes the whole state to
    real space while dealiasing only its output, so off-chain rows would alias
    back onto the chain rows and change the flux this objective returns. The
    projection does not change the gradient contract, which is that
    ``saturated_state`` is detached and only ``geom`` and ``params`` carry
    derivatives. Periodic and full-cover decks are untouched.

    ``collision_operator`` is the same custom model
    :func:`integrate_nonlinear` accepts, and must be passed here too: a run
    saturated with a custom operator and then differentiated without one is a
    derivative of different physics from the trajectory it starts on.

    ``steps`` above ``divergence_knee_steps`` warns. Past the measured knee the
    windowed adjoint grows with the leading Lyapunov exponent and stops being a
    useful design direction; pass ``None`` to silence the check when the knee
    has been remeasured for the case at hand with
    ``tools/campaigns/nonlinear_gradient_window.py``.

    Host-side geometry/layout preparation stays here. The differentiated scan
    in :func:`_nonlinear_heat_flux_window_total` reuses its executable for
    matching shapes, topology and static options, with geometry arrays passed
    as operands. See ``docs/nonlinear_autodiff.rst`` for scope and evidence.
    """

    count = int(steps)
    tail = count if tail_steps is None else int(tail_steps)
    if count < 1 or not 1 <= tail <= count:
        raise ValueError("steps must be positive and tail_steps must lie within it")
    _warn_if_window_exceeds_divergence_knee(count, divergence_knee_steps)
    if saturated_state.ndim not in (5, 6):
        raise ValueError(
            "saturated_state must have shape (Nl, Nm, Ny, Nx, Nz) or "
            "(Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    offset = saturated_state.ndim - 5
    Nl, Nm = saturated_state.shape[offset : offset + 2]

    term_cfg = terms or TermConfig(nonlinear=1.0)
    geometry = ensure_flux_tube_geometry_data(geom, grid.z)
    cache = build_linear_cache(grid, geometry, params, Nl=Nl, Nm=Nm)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geometry, grid)
    projector_signature = (
        hermitian_projector_signature(
            np.asarray(grid.ky), int(np.asarray(grid.kx).size)
        )
        if compressed_real_fft
        else None
    )
    initial_state = jax.lax.stop_gradient(
        mask_supplied_state(jnp.asarray(saturated_state), cache)
    )
    heat_dtype = jnp.result_type(
        jnp.real(initial_state), flux_factor, *jax.tree_util.tree_leaves(params)
    )
    total_heat = _nonlinear_heat_flux_window_total(
        initial_state,
        cache,
        grid,
        params,
        flux_factor,
        jnp.zeros((), dtype=heat_dtype),
        jnp.arange(count),
        dt=dt,
        count=count,
        tail=tail,
        method=method,
        term_cfg=term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        checkpoint=checkpoint,
        projector_signature=projector_signature,
    )
    return total_heat / tail


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
) -> tuple[jnp.ndarray, Any, ImplicitSolveStats | None]:
    """Run the shared shearing-coordinate scan with optional transport output.

    Returns the final state, the scan output, and the implicit solve status of
    a ``method="imex"`` run (``None`` for the explicit methods, which take no
    implicit solve and so add no carry leaf).
    """

    if str(grid.boundary).lower() not in {"periodic", "linked"} or bool(grid.non_twist):
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
        nl, nm = _gyrokinetic_moment_shape(G0)
        cache = build_linear_cache(grid, geom_eff, params, nl, nm)

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

    def advance(current, derivative, time, dt_local, solve_stats=None):
        """Advance one sheared step, folding any implicit solve into the status.

        ``solve_stats`` rides through the scan carry because a traced step
        cannot raise, exactly as the cached IMEX and implicit linear scans do
        (queue row Q15). Explicit methods pass ``None`` straight back, so they
        add no carry leaf and compile unchanged.
        """

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
            solution, solve_stats = solve_imex_step_with_stats(
                guess,
                rhs,
                solve_stats,
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
            return (
                endpoint_rhs._replace(state=project_state(solution)),
                new_time,
                solve_stats,
            )
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
        return coordinates(trial, new_time, time), new_time, solve_stats

    def local_dt(current_fields, dt_previous):
        if time_step_policy is None:
            return dt_value
        return time_step_policy.update_dt(current_fields, dt_previous)

    # The status leaf exists only on the route that takes an implicit solve, so
    # the explicit sheared scans keep exactly the carry they had.
    carries_solve_stats = method_key == "imex"

    def _stats_of(carry, position: int):
        return carry[position] if carries_solve_stats else None

    def _stats_tail(solve_stats):
        return (solve_stats,) if carries_solve_stats else ()

    def state_only_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous = carry[:3]
        solve_stats = _stats_of(carry, 3)
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
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
        ) + _stats_tail(solve_stats)
        return next_carry, None

    def endpoint_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous, derivative, current_fields = carry[:5]
        solve_stats = _stats_of(carry, 5)
        del index
        time = adaptive_time
        current = coordinates(state, time, time)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
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
        ) + _stats_tail(solve_stats)
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
    initial_stats = (
        _empty_implicit_solve_stats(state_dtype) if carries_solve_stats else None
    )
    if not record_transport and not return_fields:
        state_final_carry, output = jax.lax.scan(
            state_only_step,
            initial_state_carry + _stats_tail(initial_stats),
            jnp.arange(steps),
        )
        return state_final_carry[0], output, _stats_of(state_final_carry, 3)

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
    initial_endpoint_carry = (
        initial_state_carry
        + (
            jnp.asarray(initial_derivative, dtype=state_dtype),
            initial_fields,
        )
        + _stats_tail(initial_stats)
    )
    endpoint_final_carry, output = jax.lax.scan(
        endpoint_step, initial_endpoint_carry, jnp.arange(steps)
    )
    return endpoint_final_carry[0], output, _stats_of(endpoint_final_carry, 5)


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the standard-flux-tube shearing-coordinate foundation.

    This research path supports fixed-step Euler, midpoint RK2, three-stage
    Heun RK3, and first-order IMEX. Stage states and derivatives are remapped to
    the stage coordinate basis before the RHS and back to the step basis before
    Runge--Kutta combinations. IMEX evaluates the explicit nonlinear term in the
    current basis and rebuilds the implicit linear operator in the endpoint
    basis. ``compressed_real_fft`` evaluates the nonlinear bracket in the
    equivalent canonical shearing-coordinate representation.

    ``return_solve_stats=True`` appends the implicit solve summary of a
    ``method="imex"`` run (:class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`,
    ``None`` for the explicit methods). The traced scan cannot raise on a
    starved inner budget, so this is the convergence channel; a host-side caller
    turns it into a refusal with
    :func:`~gkx.solvers_linear_implicit.require_converged_implicit_solves`.
    """

    final_state, fields, solve_stats = _integrate_nonlinear_sheared_scan(
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
    head: tuple[Any, ...] = (final_state, fields) if return_fields else (final_state,)
    if return_solve_stats:
        return (*head, solve_stats)
    return head if return_fields else final_state


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
    return_solve_stats: bool = False,
) -> ShearedTransportTrace:
    """Integrate a sheared run and record canonical heat flux at every step.

    With ``fixed_dt=False``, ``steps`` is the accepted-step budget and ``time``
    records the resulting nonuniform physical-time grid. ``initial_time`` and
    ``initial_dt`` continue a prior trace without resetting the shearing basis.

    ``return_solve_stats=True`` fills the trace's ``solve_stats`` field with the
    implicit solve summary of a ``method="imex"`` run; it stays ``None``
    otherwise, so a caller that does not ask reads and compiles exactly what it
    did before.
    """

    final_state, samples, solve_stats = _integrate_nonlinear_sheared_scan(
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
    return ShearedTransportTrace(
        final_state, time, heat_flux, solve_stats if return_solve_stats else None
    )


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term.

    Returns ``(G_out, fields_t)``, or ``(G_out, fields_t, stats)`` with
    ``return_solve_stats=True``; ``stats`` is the carried
    :class:`~gkx.solvers_linear_implicit.ImplicitSolveStats` of every solve.
    """

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
        return_solve_stats=return_solve_stats,
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
    "nonlinear_heat_flux_window",
    "nonlinear_rhs_cached",
    "ShearedTransportTrace",
]
