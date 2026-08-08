"""Core linear and quasilinear solver-objective evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.lax.linalg as lax_linalg
import jax.numpy as jnp
import numpy as np

from gkx.config import CycloneBaseCase, GridConfig
from gkx.diagnostics import (
    fieldline_quadrature_weights,
    heat_flux_species,
    particle_flux_species,
)
from gkx.diagnostics.quasilinear_transport import effective_kperp2, phi_norm2
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.objectives.autodiff_validation import explicit_complex_operator_matrix
from gkx.objectives.eigen import dominant_real_eigenvalue
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers.linear.krylov import adaptive_propagator_eigenpair
from gkx.solvers.linear.krylov_algorithms import _build_shift_invert_precond


SOLVER_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "linear_particle_flux_weight",
    "mixing_length_heat_flux_proxy",
)
SolverScalarObjective = Literal[
    "growth",
    "gamma",
    "frequency",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "linear_particle_flux_weight",
    "quasilinear_flux",
    "mixing_length_heat_flux_proxy",
]
LinearEigensolver = Literal["dense", "adaptive-propagator"]
_SOLVER_OBJECTIVE_INDEX = {
    name: index for index, name in enumerate(SOLVER_OBJECTIVE_NAMES)
}
_SOLVER_OBJECTIVE_ALIASES = {
    "growth": "gamma",
    "frequency": "omega",
    "quasilinear_flux": "mixing_length_heat_flux_proxy",
}


def _default_gradient_linear_params() -> LinearParams:
    # Cyclone drive in the units the operator consumes, a/L. These used to be
    # 6.9 and 2.2, which are the same case expressed as R/L: the objective was
    # running at R/a = 2.78 times the intended gradient.
    return LinearParams(
        fprim=0.8,
        tprim=2.49,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )


def _default_gradient_linear_terms() -> LinearTerms:
    return LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


@dataclass(frozen=True)
class _SolverGeometryContext:
    geometry: Any
    grid: Any
    linear_params: LinearParams
    linear_terms: LinearTerms
    cache: Any
    state_shape: tuple[int, ...]


@dataclass(frozen=True)
class _DominantLinearBranch:
    eigenvalue: jnp.ndarray
    state: jnp.ndarray
    phi: jnp.ndarray


@dataclass(frozen=True)
class _LinearTransportWeights:
    kperp_eff2: jnp.ndarray
    heat_flux_weight: jnp.ndarray
    particle_flux_weight: jnp.ndarray


@dataclass(frozen=True)
class AdaptiveLinearEigensolverConfig:
    """Fail-closed settings for the differentiable matrix-free objective path."""

    krylov_dim: int = 24
    restart_krylov_dim: int = 12
    candidate_count: int = 2
    max_restarts: int = 4
    tolerance: float = 1.0e-9
    chunk_horizon: float = 30.0
    stability_dimension: int = 12
    stability_probe_count: int = 2
    stability_safety: float = 0.9
    max_stability_retries: int = 2
    exponential_krylov_dim: int | None = None
    exponential_horizon: float = 5.0
    adjoint_krylov_dim: int = 24
    adjoint_max_restarts: int = 200
    sensitivity_rtol: float = 1.0e-8
    sensitivity_restart: int = 200
    sensitivity_max_restarts: int = 4
    sensitivity_solver: str = "propagator"
    sensitivity_krylov_dim: int = 16
    sensitivity_preconditioner: str = "field-corrected"
    condition_limit: float = 1.0e8
    branch_gap_floor: float = 1.0e-8

    def __post_init__(self) -> None:
        dimensions = (
            self.krylov_dim,
            self.restart_krylov_dim,
            self.stability_dimension,
            self.adjoint_krylov_dim,
        )
        if any(int(value) < 2 for value in dimensions):
            raise ValueError("all adaptive eigensolver dimensions must be at least two")
        if self.exponential_krylov_dim is not None and (
            self.exponential_krylov_dim < 2 or self.exponential_horizon <= 0.0
        ):
            raise ValueError(
                "exponential Krylov dimension and horizon must be positive"
            )
        if (
            self.max_restarts < 1
            or self.adjoint_max_restarts < 1
            or self.sensitivity_max_restarts < 1
            or self.sensitivity_krylov_dim < 1
        ):
            raise ValueError(
                "all primal, adjoint, and sensitivity limits must be positive"
            )
        if (
            not 1
            <= self.candidate_count
            <= min(
                self.krylov_dim,
                self.restart_krylov_dim,
            )
        ):
            raise ValueError("candidate_count must fit every Krylov subspace")
        if self.max_stability_retries < 0:
            raise ValueError("max_stability_retries must be non-negative")
        if self.tolerance <= 0.0 or self.sensitivity_rtol <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.chunk_horizon <= 0.0:
            raise ValueError("chunk_horizon must be positive")
        if self.stability_probe_count < 1:
            raise ValueError("stability_probe_count must be positive")
        if not 0.0 < self.stability_safety < 1.0:
            raise ValueError("stability_safety must lie in (0, 1)")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must exceed one")
        if self.branch_gap_floor < 0.0:
            raise ValueError("branch_gap_floor must be non-negative")
        if self.sensitivity_preconditioner not in {
            "field-corrected",
            "hermite-line",
            "damping",
        }:
            raise ValueError(
                "sensitivity_preconditioner must be field-corrected, "
                "hermite-line, or damping"
            )
        if self.sensitivity_solver not in {"propagator", "gmres"}:
            raise ValueError("sensitivity_solver must be propagator or gmres")


def _solver_geometry_context(
    geom: Any,
    *,
    selected_ky_index: int,
    n_laguerre: int,
    n_hermite: int,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    params_linear: LinearParams | None,
    terms: LinearTerms | None,
    spectral_grid: Any | None = None,
) -> _SolverGeometryContext:
    geometry = geom
    raw_ntheta = int(jnp.asarray(geometry.theta).shape[0])
    if raw_ntheta < 1:
        raise ValueError("geometry must expose at least one theta sample")
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    if spectral_grid is None:
        cfg = CycloneBaseCase(
            grid=GridConfig(
                Nx=int(nx),
                Ny=int(ny),
                Nz=raw_ntheta,
                Lx=float(lx),
                Ly=float(ly),
            )
        )
        full_grid = build_spectral_grid(cfg.grid)
        if not (0 <= int(selected_ky_index) < int(full_grid.ky.size)):
            raise ValueError("selected_ky_index is outside the ky grid")
        grid = select_ky_grid(full_grid, int(selected_ky_index))
    else:
        grid = spectral_grid
        if int(grid.ky.size) != 1:
            raise ValueError("spectral_grid must already select exactly one ky")
        if (
            bool(getattr(geometry, "theta_closed_interval", False))
            and raw_ntheta == int(grid.z.size) + 1
        ):
            geometry = geometry.trim_terminal_theta_point()
    ntheta = int(jnp.asarray(geometry.theta).shape[0])
    if int(grid.z.size) != ntheta:
        raise ValueError("spectral_grid z size must match the geometry theta size")
    linear_params = params_linear or _default_gradient_linear_params()
    linear_terms = terms or _default_gradient_linear_terms()
    cache = build_linear_cache(
        grid,
        geometry,
        linear_params,
        n_laguerre_int,
        n_hermite_int,
    )
    density = jnp.asarray(linear_params.density)
    if density.ndim > 1:
        raise ValueError("linear species parameters must be scalar or one-dimensional")
    species_shape = () if density.ndim == 0 else (int(density.size),)
    state_shape = species_shape + (
        n_laguerre_int,
        n_hermite_int,
        int(grid.ky.size),
        int(grid.kx.size),
        int(grid.z.size),
    )
    return _SolverGeometryContext(
        geometry=geometry,
        grid=grid,
        linear_params=linear_params,
        linear_terms=linear_terms,
        cache=cache,
        state_shape=state_shape,
    )


def _linear_rhs_phi(
    state_arr: jnp.ndarray,
    context: _SolverGeometryContext,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return linear_rhs_cached(
        state_arr,
        context.cache,
        context.linear_params,
        terms=context.linear_terms,
        use_jit=False,
        use_custom_vjp=False,
    )


def _solver_operator_matrix(context: _SolverGeometryContext) -> jnp.ndarray:
    return explicit_complex_operator_matrix(
        lambda state_arr: _linear_rhs_phi(state_arr, context)[0],
        context.state_shape,
    )


def _dominant_linear_branch(context: _SolverGeometryContext) -> _DominantLinearBranch:
    matrix = _solver_operator_matrix(context)
    # jnp.linalg.eig refuses to differentiate non-symmetric eigenvectors unless
    # the caller opts in (jax >= 0.11). The dominant ITG branch is a simple,
    # well-separated eigenvalue here, which is exactly the condition under which
    # the eigenvector derivative is well defined, so opt in explicitly -- without
    # it the whole objective vector is forward-only.
    eigenvalues, eigenvectors = lax_linalg.eig(
        matrix,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=True,
        enable_eigvec_derivs=True,
    )
    branch_index = jnp.argmax(jnp.real(eigenvalues))
    eigenvalue = eigenvalues[branch_index]
    eigenvector = eigenvectors[:, branch_index]
    state_arr = jnp.reshape(eigenvector, context.state_shape)
    _rhs, phi = _linear_rhs_phi(state_arr, context)
    return _DominantLinearBranch(eigenvalue=eigenvalue, state=state_arr, phi=phi)


def _matrix_free_dominant_linear_branch(
    context: _SolverGeometryContext,
    *,
    config: AdaptiveLinearEigensolverConfig,
) -> _DominantLinearBranch:
    """Return a residual-certified primal with an implicit eigenpair tangent."""

    try:
        from solvax import eigenpair_reverse, propagator_eigenpairs  # type: ignore[attr-defined]
    except ImportError as error:
        raise RuntimeError("SOLVAX differentiable eigenpair API is required") from error

    operator_size = int(np.prod(context.state_shape))
    if operator_size < 3:
        raise ValueError(
            "adaptive-propagator requires an operator of size at least three"
        )
    krylov_dim = min(config.krylov_dim, operator_size - 1)
    restart_krylov_dim = min(config.restart_krylov_dim, operator_size - 1)
    stability_dimension = min(config.stability_dimension, operator_size - 1)
    adjoint_krylov_dim = min(config.adjoint_krylov_dim, operator_size - 1)
    flat_index = jnp.arange(operator_size, dtype=jnp.float64)
    start = jnp.reshape(
        jnp.exp(1j * (flat_index + 1.0) * jnp.asarray(0.6180339887498948)),
        context.state_shape,
    )
    primal_solutions: list[Any] = []

    def build(cache: Any):
        def apply(state: jnp.ndarray) -> jnp.ndarray:
            return linear_rhs_cached(
                state,
                cache,
                context.linear_params,
                terms=context.linear_terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0]

        return apply

    def primal_solver(
        cache: Any,
        _apply: Any,
        initial: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        solution = adaptive_propagator_eigenpair(
            initial,
            cache,
            context.linear_params,
            terms=context.linear_terms,
            krylov_dim=krylov_dim,
            restart_krylov_dim=restart_krylov_dim,
            candidate_count=config.candidate_count,
            max_restarts=config.max_restarts,
            tol=config.tolerance,
            chunk_horizon=config.chunk_horizon,
            stability_dimension=stability_dimension,
            stability_probe_count=config.stability_probe_count,
            stability_safety=config.stability_safety,
            max_stability_retries=config.max_stability_retries,
            exponential_krylov_dim=config.exponential_krylov_dim,
            exponential_horizon=config.exponential_horizon,
        )
        if not bool(solution.stable) or not bool(solution.converged):
            raise RuntimeError(
                "adaptive dominant eigenpair did not certify the continuous "
                f"operator: stable={solution.stable}, "
                f"residual={float(np.asarray(solution.residual)):.3e}, "
                f"restarts={solution.restarts}"
            )
        growth_gap = float(np.asarray(solution.candidate_growth_gap))
        if growth_gap < config.branch_gap_floor:
            raise RuntimeError(
                "dominant-growth branch is not locally isolated: "
                f"certified candidate gap {growth_gap:.3e} is below "
                f"{config.branch_gap_floor:.3e}"
            )
        primal_solutions.append(solution)
        return solution.eigenvalue, solution.eigenvector

    def left_solver(
        _cache: Any,
        apply: Any,
        initial: jnp.ndarray,
        value: complex,
    ) -> jnp.ndarray:
        if not primal_solutions:
            raise RuntimeError("left solve requires a certified primal pair")
        primal = primal_solutions[-1]
        transpose = jax.linear_transpose(apply, initial)

        @jax.jit
        def adjoint(vector: jnp.ndarray) -> jnp.ndarray:
            return jnp.conj(transpose(jnp.conj(vector))[0])

        if config.exponential_krylov_dim is None:
            candidates = propagator_eigenpairs(
                adjoint,
                initial,
                dt=primal.filter_dt,
                steps=primal.filter_steps,
                krylov_dim=adjoint_krylov_dim,
                candidates=config.candidate_count,
                tol=config.tolerance,
            )
        else:
            from solvax import exponential_eigenpairs  # type: ignore[attr-defined]

            candidates = exponential_eigenpairs(
                adjoint,
                initial,
                horizon=config.exponential_horizon,
                inner_krylov_dim=min(config.exponential_krylov_dim, operator_size),
                outer_krylov_dim=adjoint_krylov_dim,
                candidates=config.candidate_count,
                tol=config.tolerance,
                restarts=config.max_restarts,
            )
        converged = np.asarray(candidates.converged, dtype=bool)
        if not np.any(converged):
            raise RuntimeError(
                "adjoint propagator did not certify the primal branch: "
                f"minimum residual={float(np.min(np.asarray(candidates.residuals))):.3e}"
            )
        values = np.asarray(candidates.eigenvalues)
        distances = np.where(converged, np.abs(values - np.conj(value)), np.inf)
        selected = int(np.argmin(distances))
        return candidates.eigenvectors[selected]

    term_config = linear_terms_to_term_config(context.linear_terms)

    def tangent_preconditioner(
        cache: Any,
        value: complex,
        right: jnp.ndarray,
        left: jnp.ndarray,
    ):
        _diagonal, apply_flat = _build_shift_invert_precond(
            right,
            cache,
            context.linear_params,
            term_config,
            jnp.asarray(value, dtype=right.dtype),
            config.sensitivity_preconditioner,
        )
        if apply_flat is None:
            raise RuntimeError("failed to construct the bordered preconditioner")

        def base_inverse(vector: jnp.ndarray) -> jnp.ndarray:
            return apply_flat(vector.reshape(right.size)).reshape(right.shape)

        inverse_right = base_inverse(right)
        denominator = 1.0 + jnp.vdot(left, inverse_right)
        safe_denominator = jnp.where(
            jnp.abs(denominator) > jnp.finfo(jnp.real(right).dtype).eps,
            denominator,
            1.0 + 0.0j,
        )

        def apply_preconditioner(vector: jnp.ndarray) -> jnp.ndarray:
            inverse_vector = base_inverse(vector)
            correction = jnp.vdot(left, inverse_vector) / safe_denominator
            return inverse_vector - inverse_right * correction

        return apply_preconditioner

    def transpose_propagator_solver(
        _cache: Any,
        _value: complex,
        right: jnp.ndarray,
        left: jnp.ndarray,
        transpose_bordered: Any,
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Invert the transposed reduced resolvent by stable time marching."""

        if not primal_solutions:
            raise RuntimeError("sensitivity solve requires a certified primal pair")
        primal = primal_solutions[-1]
        border_mode = jnp.conj(left)

        def bilinear_dot(left_vector: jnp.ndarray, right_vector: jnp.ndarray):
            return jnp.sum(left_vector * right_vector)

        def transpose_shifted(vector: jnp.ndarray) -> jnp.ndarray:
            border = border_mode * bilinear_dot(right, vector)
            return transpose_bordered(vector) - border

        border_coefficient = bilinear_dot(right, rhs)
        projected_rhs = rhs - border_mode * border_coefficient
        dt = jnp.asarray(primal.filter_dt, dtype=jnp.real(rhs).dtype)

        def project(vector: jnp.ndarray) -> jnp.ndarray:
            return vector - border_mode * bilinear_dot(right, vector)

        def homogeneous_step(_index: int, vector: jnp.ndarray) -> jnp.ndarray:
            k1 = transpose_shifted(vector)
            k2 = transpose_shifted(vector + 0.5 * dt * k1)
            k3 = transpose_shifted(vector + 0.5 * dt * k2)
            k4 = transpose_shifted(vector + dt * k3)
            return project(vector + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        def affine_step(_index: int, vector: jnp.ndarray) -> jnp.ndarray:
            def action(state: jnp.ndarray) -> jnp.ndarray:
                return transpose_shifted(state) - projected_rhs

            k1 = action(vector)
            k2 = action(vector + 0.5 * dt * k1)
            k3 = action(vector + 0.5 * dt * k2)
            k4 = action(vector + dt * k3)
            return project(vector + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        def propagate(vector: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.fori_loop(
                0,
                primal.filter_steps,
                homogeneous_step,
                vector,
            )

        affine_source = jax.lax.fori_loop(
            0,
            primal.filter_steps,
            affine_step,
            jnp.zeros_like(rhs),
        )

        # Solve the fixed point of one stable propagator chunk rather than
        # waiting for stationary iteration. GMRES chooses the optimal
        # polynomial in the time-stepper, accelerating weakly damped modes
        # while retaining an O(n) matrix-free operator.
        from solvax import gmres

        fixed_point = gmres(
            lambda vector: vector - propagate(vector),
            affine_source,
            restart=config.sensitivity_krylov_dim,
            rtol=0.1 * config.sensitivity_rtol,
            max_restarts=config.sensitivity_max_restarts,
        )
        reduced_solution = fixed_point.x
        rhs_norm = jnp.linalg.norm(rhs)
        threshold = config.sensitivity_rtol * rhs_norm
        solution = reduced_solution + border_coefficient * border_mode
        full_residual = jnp.linalg.norm(rhs - transpose_bordered(solution))
        converged = fixed_point.converged & (full_residual <= threshold)
        return jnp.where(converged, solution, jnp.full_like(solution, jnp.nan))

    eigenvalue, state_arr = eigenpair_reverse(
        context.cache,
        build,
        start,
        primal_solver=primal_solver,
        left_solver=left_solver,
        tangent_preconditioner=(
            tangent_preconditioner
            if config.sensitivity_solver == "gmres"
            or config.exponential_krylov_dim is not None
            else None
        ),
        transpose_tangent_solver=(
            transpose_propagator_solver
            if config.sensitivity_solver == "propagator"
            and config.exponential_krylov_dim is None
            else None
        ),
        sensitivity_rtol=config.sensitivity_rtol,
        sensitivity_restart=config.sensitivity_restart,
        sensitivity_max_restarts=config.sensitivity_max_restarts,
        condition_limit=config.condition_limit,
    )
    _rhs, phi = _linear_rhs_phi(state_arr, context)
    return _DominantLinearBranch(eigenvalue=eigenvalue, state=state_arr, phi=phi)


def _linear_transport_weights(
    geom: Any,
    context: _SolverGeometryContext,
    branch: _DominantLinearBranch,
) -> _LinearTransportWeights:
    zero_field = jnp.zeros_like(branch.phi)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, context.grid)
    norm2 = phi_norm2(branch.phi, context.cache, context.linear_params, vol_fac)
    kperp_eff = effective_kperp2(branch.phi, context.cache, vol_fac)
    heat_weight = jnp.real(
        jnp.sum(
            heat_flux_species(
                branch.state,
                branch.phi,
                zero_field,
                zero_field,
                context.cache,
                context.grid,
                context.linear_params,
                flux_fac,
            )
        )
        / norm2
    )
    particle_weight = jnp.real(
        jnp.sum(
            particle_flux_species(
                branch.state,
                branch.phi,
                zero_field,
                zero_field,
                context.cache,
                context.grid,
                context.linear_params,
                flux_fac,
            )
        )
        / norm2
    )
    return _LinearTransportWeights(
        kperp_eff2=kperp_eff,
        heat_flux_weight=heat_weight,
        particle_flux_weight=particle_weight,
    )


def solver_scalar_objective_from_vector(
    objective_vector: jnp.ndarray | np.ndarray,
    objective: SolverScalarObjective = "growth",
) -> jnp.ndarray:
    """Select one scalar objective from ``SOLVER_OBJECTIVE_NAMES``.

    This tiny selector keeps optimizer code honest about which scalar is being
    minimized. It also centralizes aliases used by the examples:
    ``growth -> gamma``, ``frequency -> omega``, and
    ``quasilinear_flux -> mixing_length_heat_flux_proxy``.
    """

    key = str(objective).strip()
    canonical = _SOLVER_OBJECTIVE_ALIASES.get(key, key)
    if canonical not in _SOLVER_OBJECTIVE_INDEX:
        valid = sorted(set(_SOLVER_OBJECTIVE_INDEX) | set(_SOLVER_OBJECTIVE_ALIASES))
        raise ValueError(
            f"unknown solver objective {objective!r}; expected one of {valid}"
        )
    vector = jnp.ravel(jnp.asarray(objective_vector))
    if int(vector.size) != len(SOLVER_OBJECTIVE_NAMES):
        raise ValueError(
            f"objective_vector must have length {len(SOLVER_OBJECTIVE_NAMES)}"
        )
    return vector[_SOLVER_OBJECTIVE_INDEX[canonical]]


def solver_linear_operator_matrix_from_geometry(
    geom: Any,
    *,
    spectral_grid: Any | None = None,
    selected_ky_index: int = 1,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 4,
    lx: float = 6.0,
    ly: float = 12.0,
    params_linear: LinearParams | None = None,
    terms: LinearTerms | None = None,
) -> jnp.ndarray:
    """Materialize the complex linear-RHS operator for one solver geometry.

    This helper exposes the exact matrix whose dominant eigenvalue is used by
    :func:`solver_growth_rate_from_geometry`. It is intended for branch
    locality and AD/finite-difference admission gates; production time
    integration should continue to call the RHS directly.
    """

    context = _solver_geometry_context(
        geom,
        spectral_grid=spectral_grid,
        selected_ky_index=selected_ky_index,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params_linear=params_linear,
        terms=terms,
    )
    return _solver_operator_matrix(context)


def solver_objective_vector_from_geometry(
    geom: Any,
    *,
    spectral_grid: Any | None = None,
    selected_ky_index: int = 1,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 4,
    lx: float = 6.0,
    ly: float = 12.0,
    params_linear: LinearParams | None = None,
    terms: LinearTerms | None = None,
    eigensolver: LinearEigensolver = "dense",
    adaptive_config: AdaptiveLinearEigensolverConfig | None = None,
) -> jnp.ndarray:
    """Evaluate dominant linear/quasilinear observables from geometry.

    This is a reusable value-level objective builder for optimization drivers
    and examples. It builds the production linear RHS on the supplied
    solver-ready flux-tube geometry, selects the maximum-growth eigenbranch,
    and returns the ordered ``SOLVER_OBJECTIVE_NAMES`` vector.

    ``eigensolver="adaptive-propagator"`` avoids materializing the dense
    operator, certifies the continuous residual, and differentiates the
    eigenpair implicitly. It remains explicit opt-in until branch-continuity
    gates admit a production optimization. The default dense path preserves
    established results. Pass a runtime ``spectral_grid`` that has already
    selected one ky to preserve linked stellarator boundary conditions;
    otherwise a periodic single-ky grid is built from the scalar grid options.
    One-dimensional species parameters add the leading multi-species state
    axis automatically.
    """

    context = _solver_geometry_context(
        geom,
        spectral_grid=spectral_grid,
        selected_ky_index=selected_ky_index,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params_linear=params_linear,
        terms=terms,
    )
    if eigensolver == "dense":
        branch = _dominant_linear_branch(context)
    elif eigensolver == "adaptive-propagator":
        branch = _matrix_free_dominant_linear_branch(
            context,
            config=adaptive_config or AdaptiveLinearEigensolverConfig(),
        )
    else:
        raise ValueError(
            f"eigensolver must be 'dense' or 'adaptive-propagator', got {eigensolver!r}"
        )
    weights = _linear_transport_weights(context.geometry, context, branch)
    gamma = jnp.real(branch.eigenvalue)
    ql_proxy = (
        gamma
        * weights.heat_flux_weight
        / jnp.maximum(
            weights.kperp_eff2,
            jnp.asarray(1.0e-12, dtype=weights.kperp_eff2.dtype),
        )
    )
    return jnp.asarray(
        [
            gamma,
            jnp.imag(branch.eigenvalue),
            weights.kperp_eff2,
            weights.heat_flux_weight,
            weights.particle_flux_weight,
            ql_proxy,
        ]
    )


def solver_growth_rate_from_geometry(
    geom: Any,
    *,
    spectral_grid: Any | None = None,
    selected_ky_index: int = 1,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 4,
    lx: float = 6.0,
    ly: float = 12.0,
    params_linear: LinearParams | None = None,
    terms: LinearTerms | None = None,
) -> jnp.ndarray:
    """Evaluate the dominant linear growth rate without eigenvector AD."""

    context = _solver_geometry_context(
        geom,
        spectral_grid=spectral_grid,
        selected_ky_index=selected_ky_index,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params_linear=params_linear,
        terms=terms,
    )
    return dominant_real_eigenvalue(_solver_operator_matrix(context))


__all__ = [
    "AdaptiveLinearEigensolverConfig",
    "LinearEigensolver",
    "SOLVER_OBJECTIVE_NAMES",
    "SolverScalarObjective",
    "_default_gradient_linear_params",
    "_default_gradient_linear_terms",
    "solver_growth_rate_from_geometry",
    "solver_linear_operator_matrix_from_geometry",
    "solver_objective_vector_from_geometry",
    "solver_scalar_objective_from_vector",
]
