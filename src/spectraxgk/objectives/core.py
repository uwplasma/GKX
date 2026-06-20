"""Core linear and quasilinear solver-objective evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import explicit_complex_operator_matrix
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import heat_flux_species, particle_flux_species, fieldline_quadrature_weights
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.diagnostics.quasilinear_transport import effective_kperp2, phi_norm2
from spectraxgk.objectives.eigen import dominant_real_eigenvalue


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
_SOLVER_OBJECTIVE_INDEX = {name: index for index, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
_SOLVER_OBJECTIVE_ALIASES = {
    "growth": "gamma",
    "frequency": "omega",
    "quasilinear_flux": "mixing_length_heat_flux_proxy",
}


def _default_gradient_linear_params() -> LinearParams:
    return LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
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
    grid: Any
    linear_params: LinearParams
    linear_terms: LinearTerms
    cache: Any
    state_shape: tuple[int, int, int, int, int]


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
) -> _SolverGeometryContext:
    ntheta = int(jnp.asarray(geom.theta).shape[0])
    if ntheta < 1:
        raise ValueError("geometry must expose at least one theta sample")
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=int(nx), Ny=int(ny), Nz=ntheta, Lx=float(lx), Ly=float(ly))
    )
    spectral_grid = build_spectral_grid(cfg.grid)
    if not (0 <= int(selected_ky_index) < int(spectral_grid.ky.size)):
        raise ValueError("selected_ky_index is outside the ky grid")
    grid = select_ky_grid(spectral_grid, int(selected_ky_index))
    linear_params = params_linear or _default_gradient_linear_params()
    linear_terms = terms or _default_gradient_linear_terms()
    cache = build_linear_cache(grid, geom, linear_params, n_laguerre_int, n_hermite_int)
    state_shape = (
        n_laguerre_int,
        n_hermite_int,
        int(grid.ky.size),
        int(grid.kx.size),
        int(grid.z.size),
    )
    return _SolverGeometryContext(
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
    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    branch_index = jnp.argmax(jnp.real(eigenvalues))
    eigenvalue = eigenvalues[branch_index]
    eigenvector = eigenvectors[:, branch_index]
    state_arr = jnp.reshape(eigenvector, context.state_shape)
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
        raise ValueError(f"unknown solver objective {objective!r}; expected one of {valid}")
    vector = jnp.ravel(jnp.asarray(objective_vector))
    if int(vector.size) != len(SOLVER_OBJECTIVE_NAMES):
        raise ValueError(f"objective_vector must have length {len(SOLVER_OBJECTIVE_NAMES)}")
    return vector[_SOLVER_OBJECTIVE_INDEX[canonical]]


def solver_linear_operator_matrix_from_geometry(
    geom: Any,
    *,
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
    """Evaluate dominant linear/quasilinear observables from geometry.

    This is a reusable value-level objective builder for optimization drivers
    and examples. It builds the production linear RHS on the supplied
    solver-ready flux-tube geometry, selects the maximum-growth eigenbranch,
    and returns the ordered ``SOLVER_OBJECTIVE_NAMES`` vector.

    Branch continuity and AD/finite-difference validation are deliberately
    handled by gate functions; this is the shared forward evaluator those
    production objectives can use after a branch has been selected and audited.
    """

    context = _solver_geometry_context(
        geom,
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
    branch = _dominant_linear_branch(context)
    weights = _linear_transport_weights(geom, context, branch)
    gamma = jnp.real(branch.eigenvalue)
    ql_proxy = gamma * weights.heat_flux_weight / jnp.maximum(
        weights.kperp_eff2,
        jnp.asarray(1.0e-12, dtype=weights.kperp_eff2.dtype),
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
    "SOLVER_OBJECTIVE_NAMES",
    "SolverScalarObjective",
    "_default_gradient_linear_params",
    "_default_gradient_linear_terms",
    "solver_growth_rate_from_geometry",
    "solver_linear_operator_matrix_from_geometry",
    "solver_objective_vector_from_geometry",
    "solver_scalar_objective_from_vector",
]
