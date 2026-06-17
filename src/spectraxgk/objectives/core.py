"""Core linear and quasilinear solver-objective evaluators."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import explicit_complex_operator_matrix
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import heat_flux_species, particle_flux_species, fieldline_quadrature_weights
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.quasilinear import effective_kperp2, phi_norm2
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

    ntheta = int(jnp.asarray(geom.theta).shape[0])
    if ntheta < 1:
        raise ValueError("geometry must expose at least one theta sample")
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=int(nx), Ny=int(ny), Nz=ntheta, Lx=float(lx), Ly=float(ly)))
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

    def rhs_phi(state_arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            linear_params,
            terms=linear_terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    return explicit_complex_operator_matrix(lambda state_arr: rhs_phi(state_arr)[0], state_shape)


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

    ntheta = int(jnp.asarray(geom.theta).shape[0])
    if ntheta < 1:
        raise ValueError("geometry must expose at least one theta sample")
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=int(nx), Ny=int(ny), Nz=ntheta, Lx=float(lx), Ly=float(ly)))
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

    def rhs_phi(state_arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            linear_params,
            terms=linear_terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    matrix = explicit_complex_operator_matrix(lambda state_arr: rhs_phi(state_arr)[0], state_shape)
    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    branch_index = jnp.argmax(jnp.real(eigenvalues))
    eigenvalue = eigenvalues[branch_index]
    eigenvector = eigenvectors[:, branch_index]
    state_arr = jnp.reshape(eigenvector, state_shape)
    _rhs, phi = rhs_phi(state_arr)
    zero_field = jnp.zeros_like(phi)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    norm2 = phi_norm2(phi, cache, linear_params, vol_fac)
    kperp_eff = effective_kperp2(phi, cache, vol_fac)
    heat_weight = jnp.real(
        jnp.sum(
            heat_flux_species(
                state_arr,
                phi,
                zero_field,
                zero_field,
                cache,
                grid,
                linear_params,
                flux_fac,
            )
        )
        / norm2
    )
    particle_weight = jnp.real(
        jnp.sum(
            particle_flux_species(
                state_arr,
                phi,
                zero_field,
                zero_field,
                cache,
                grid,
                linear_params,
                flux_fac,
            )
        )
        / norm2
    )
    gamma = jnp.real(eigenvalue)
    ql_proxy = gamma * heat_weight / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
    return jnp.asarray(
        [
            gamma,
            jnp.imag(eigenvalue),
            kperp_eff,
            heat_weight,
            particle_weight,
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

    ntheta = int(jnp.asarray(geom.theta).shape[0])
    if ntheta < 1:
        raise ValueError("geometry must expose at least one theta sample")
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=int(nx), Ny=int(ny), Nz=ntheta, Lx=float(lx), Ly=float(ly)))
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

    def rhs_phi(state_arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            linear_params,
            terms=linear_terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    matrix = explicit_complex_operator_matrix(lambda state_arr: rhs_phi(state_arr)[0], state_shape)
    return dominant_real_eigenvalue(matrix)


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
