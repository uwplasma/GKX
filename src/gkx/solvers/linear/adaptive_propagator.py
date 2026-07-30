"""Adaptive, residual-certified full-operator propagator eigensolve."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.solvers.linear.krylov_algorithms import (
    _apply_operator,
    dominant_eigenpair_propagator_cached,
)
from gkx.solvers.linear.krylov_propagator import (
    dominant_eigenpairs_propagator_cached,
)


class AdaptivePropagatorSolution(NamedTuple):
    """Certified pair plus the branch-cluster diagnostics from its last restart."""

    eigenvalue: jax.Array
    eigenvector: jax.Array
    residual: jax.Array
    converged: bool
    stable: bool
    restarts: int
    operator_applications: int
    filter_dt: float
    filter_steps: int
    filter_horizon: float
    filter_growth_defect: float
    candidate_eigenvalues: jax.Array
    candidate_residuals: jax.Array
    candidate_growth_gap: jax.Array


def adaptive_propagator_eigenpair(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    krylov_dim: int = 16,
    max_restarts: int = 4,
    tol: float = 1.0e-10,
    chunk_horizon: float = 30.0,
    stability_dimension: int = 12,
    stability_probe_count: int = 2,
    stability_safety: float = 0.9,
    max_stability_retries: int = 2,
    restart_krylov_dim: int | None = None,
    candidate_count: int = 1,
) -> AdaptivePropagatorSolution:
    """Adapt RK4 stability, horizon, and corrective-subspace cost."""

    try:
        from solvax import adaptive_eigenpair, estimate_rk4_timestep  # type: ignore
    except ImportError as error:
        raise RuntimeError("SOLVAX adaptive propagator API is required") from error
    restart_dimension = (
        krylov_dim if restart_krylov_dim is None else int(restart_krylov_dim)
    )
    if chunk_horizon <= 0.0 or max_stability_retries < 0:
        raise ValueError("chunk_horizon must be positive and retries non-negative")
    if krylov_dim < 2 or restart_dimension < 2:
        raise ValueError("Krylov dimensions must be at least two")
    if not 1 <= candidate_count <= min(krylov_dim, restart_dimension):
        raise ValueError("candidate_count must fit every Krylov subspace")
    term_cfg = linear_terms_to_term_config(terms)

    def apply(state: jnp.ndarray) -> jnp.ndarray:
        return _apply_operator(state, cache, params, term_cfg)

    estimate = estimate_rk4_timestep(
        apply,
        v0,
        dimension=min(max(stability_dimension, 2), v0.size - 1),
        probe_count=stability_probe_count,
        safety=stability_safety,
    )
    solution = None
    operator_applications = estimate.operator_applications
    candidate_records: list[tuple[jax.Array, jax.Array]] = []
    for retry in range(max_stability_retries + 1):
        dt_limit = estimate.dt / 2**retry
        steps = max(int(np.ceil(chunk_horizon / dt_limit)), 1)
        dt = chunk_horizon / steps
        restart_dimensions: list[int] = []

        def restart_once(vector: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            dimension = krylov_dim if not restart_dimensions else restart_dimension
            restart_dimensions.append(dimension)
            if candidate_count == 1:
                return dominant_eigenpair_propagator_cached(
                    vector,
                    v0,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=dimension,
                    restarts=1,
                    dt=dt,
                    propagator_steps=steps,
                    omega_min_factor=0.0,
                    omega_target_factor=0.0,
                    omega_cap_factor=2.0,
                    omega_sign=0,
                    select_overlap=False,
                )
            values, vectors, residuals = dominant_eigenpairs_propagator_cached(
                vector,
                cache,
                params,
                term_cfg,
                krylov_dim=dimension,
                dt=dt,
                propagator_steps=steps,
                candidates=candidate_count,
            )
            candidate_records.append((values, residuals))
            certified = residuals < tol
            growth = jnp.where(certified, jnp.real(values), -jnp.inf)
            selected = jnp.where(jnp.any(certified), jnp.argmax(growth), 0)
            return values[selected], vectors[selected]

        solution = adaptive_eigenpair(
            apply,
            restart_once,
            v0,
            tol=tol,
            max_restarts=max_restarts,
            filter_dt=dt,
            filter_steps=steps,
            applications_per_restart=0,
            base_operator_applications=0,
        )
        operator_applications += (
            solution.operator_applications
            + (candidate_count - 1) * len(restart_dimensions)
            + 4 * steps * sum(restart_dimensions)
        )
        if solution.stable and solution.converged:
            break
    assert solution is not None
    if candidate_records:
        candidate_values, candidate_residuals = candidate_records[-1]
    else:
        candidate_values = jnp.reshape(solution.eigenvalue, (1,))
        candidate_residuals = jnp.reshape(solution.residual, (1,))
    certified_growth = jnp.where(
        candidate_residuals < tol,
        jnp.real(candidate_values),
        -jnp.inf,
    )
    ordered_growth = jnp.sort(certified_growth)[::-1]
    growth_gap = (
        jnp.where(
            jnp.sum(jnp.isfinite(certified_growth)) >= 2,
            ordered_growth[0] - ordered_growth[1],
            jnp.inf,
        )
        if candidate_count >= 2
        else jnp.asarray(jnp.inf, dtype=jnp.real(solution.eigenvalue).dtype)
    )
    return AdaptivePropagatorSolution._make(
        (
            *solution._replace(operator_applications=operator_applications),
            candidate_values,
            candidate_residuals,
            growth_gap,
        )
    )


__all__ = ["AdaptivePropagatorSolution", "adaptive_propagator_eigenpair"]
