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
    candidate_overlaps: jax.Array
    selected_candidate_index: int
    continuation_overlap: jax.Array
    selected_spectral_gap: jax.Array
    continued: bool
    continuation_passed: bool


def _certified_candidates(
    values: jax.Array,
    vectors: jax.Array,
    residuals: jax.Array,
    tol: float,
) -> jax.Array:
    """Reject zero and non-finite vectors even when their residual is zero."""

    norms = jnp.linalg.norm(vectors.reshape((vectors.shape[0], -1)), axis=1)
    finite = (
        jnp.isfinite(jnp.real(values))
        & jnp.isfinite(jnp.imag(values))
        & jnp.isfinite(residuals)
        & jnp.isfinite(norms)
    )
    return finite & (norms > jnp.sqrt(jnp.finfo(norms.dtype).eps)) & (residuals < tol)


def _candidate_overlap_scores(
    reference: jax.Array,
    vectors: jax.Array,
    *,
    anchor: jax.Array | None,
) -> jax.Array:
    """Return phase-invariant right or biorthogonal continuation scores."""

    reference_flat = jnp.reshape(reference, (-1,))
    vectors_flat = jnp.reshape(vectors, (vectors.shape[0], -1))
    numerator = jnp.abs(jnp.einsum("n,kn->k", jnp.conj(reference_flat), vectors_flat))
    vector_norms = jnp.linalg.norm(vectors_flat, axis=1)
    tiny = jnp.finfo(jnp.real(vectors).dtype).tiny
    if anchor is None:
        denominator = jnp.linalg.norm(reference_flat) * vector_norms
        return numerator / jnp.maximum(denominator, tiny)
    anchor_flat = jnp.reshape(anchor, (-1,))
    anchor_projection = jnp.abs(jnp.vdot(reference_flat, anchor_flat))
    anchor_projection /= jnp.maximum(jnp.linalg.norm(anchor_flat), tiny)
    normalized_projection = numerator / jnp.maximum(vector_norms, tiny)
    return normalized_projection / jnp.maximum(anchor_projection, tiny)


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
    continuation_vector: jnp.ndarray | None = None,
    continuation_covector: jnp.ndarray | None = None,
    continuation_overlap_floor: float = 0.0,
    continuation_spectral_gap_floor: float = 0.0,
    exponential_krylov_dim: int | None = None,
    exponential_horizon: float = 5.0,
) -> AdaptivePropagatorSolution:
    """Adapt RK4 stability, horizon, and corrective-subspace cost.

    A supplied continuation covector selects the certified candidate maximizing
    ``|w_previous**H v_candidate|``. A right reference is the normalized-overlap
    fallback when a left vector is unavailable. This follows one physical mode
    through a growth-rate crossing instead of silently switching to the newly
    dominant branch.
    """

    try:
        from solvax import (  # type: ignore
            adaptive_eigenpair,
            estimate_rk4_timestep,
        )
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
    if continuation_vector is not None and continuation_vector.shape != v0.shape:
        raise ValueError("continuation_vector must have the same shape as v0")
    if continuation_covector is not None and continuation_covector.shape != v0.shape:
        raise ValueError("continuation_covector must have the same shape as v0")
    continued = continuation_covector is not None or continuation_vector is not None
    if continued and candidate_count < 2:
        raise ValueError("continuation requires at least two candidates")
    if continuation_overlap_floor < 0.0:
        raise ValueError("continuation_overlap_floor must be non-negative")
    if continuation_spectral_gap_floor < 0.0:
        raise ValueError("continuation_spectral_gap_floor must be non-negative")
    selection_reference = (
        continuation_covector
        if continuation_covector is not None
        else continuation_vector
    )
    term_cfg = linear_terms_to_term_config(terms)

    def apply(state: jnp.ndarray) -> jnp.ndarray:
        return _apply_operator(state, cache, params, term_cfg)

    if exponential_krylov_dim is not None:
        try:
            from solvax import exponential_eigenpairs  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "SOLVAX exponential propagator API is required"
            ) from error
        if continued:
            raise ValueError(
                "exponential cold discovery does not replace continuation selection"
            )
        inner_dimension = min(int(exponential_krylov_dim), v0.size)
        if inner_dimension < 2 or exponential_horizon <= 0.0:
            raise ValueError("exponential dimension and horizon must both be positive")
        modes = exponential_eigenpairs(
            apply,
            v0,
            horizon=exponential_horizon,
            inner_krylov_dim=inner_dimension,
            outer_krylov_dim=krylov_dim,
            candidates=candidate_count,
            tol=tol,
            restarts=max_restarts,
        )
        certified = _certified_candidates(
            modes.eigenvalues, modes.eigenvectors, modes.residuals, tol
        )
        selected_index = int(
            np.asarray(
                jnp.argmax(jnp.where(certified, jnp.real(modes.eigenvalues), -jnp.inf))
            )
        )
        selected_value = modes.eigenvalues[selected_index]
        ordered_growth = jnp.sort(
            jnp.where(certified, jnp.real(modes.eigenvalues), -jnp.inf)
        )[::-1]
        growth_gap = (
            jnp.where(
                jnp.sum(certified) >= 2,
                ordered_growth[0] - ordered_growth[1],
                jnp.inf,
            )
            if candidate_count >= 2
            else jnp.asarray(jnp.inf, dtype=jnp.real(selected_value).dtype)
        )
        indices = jnp.arange(candidate_count)
        spectral_gap = jnp.min(
            jnp.where(
                indices != selected_index,
                jnp.abs(modes.eigenvalues - selected_value),
                jnp.inf,
            )
        )
        return AdaptivePropagatorSolution(
            eigenvalue=selected_value,
            eigenvector=modes.eigenvectors[selected_index],
            residual=modes.residuals[selected_index],
            converged=bool(np.asarray(certified[selected_index])),
            stable=True,
            restarts=max_restarts,
            operator_applications=modes.operator_applications,
            filter_dt=exponential_horizon,
            filter_steps=1,
            filter_horizon=exponential_horizon,
            filter_growth_defect=0.0,
            candidate_eigenvalues=modes.eigenvalues,
            candidate_residuals=modes.residuals,
            candidate_growth_gap=growth_gap,
            candidate_overlaps=jnp.full(
                (candidate_count,),
                jnp.nan,
                dtype=jnp.real(selected_value).dtype,
            ),
            selected_candidate_index=selected_index,
            continuation_overlap=jnp.asarray(
                jnp.nan, dtype=jnp.real(selected_value).dtype
            ),
            selected_spectral_gap=spectral_gap,
            continued=False,
            continuation_passed=True,
        )

    estimate = estimate_rk4_timestep(
        apply,
        v0,
        dimension=min(max(stability_dimension, 2), v0.size - 1),
        probe_count=stability_probe_count,
        safety=stability_safety,
    )
    solution = None
    operator_applications = estimate.operator_applications
    candidate_records: list[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ] = []
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
            certified = _certified_candidates(values, vectors, residuals, tol)
            if selection_reference is None:
                scores = jnp.full(
                    (candidate_count,),
                    jnp.nan,
                    dtype=jnp.real(values).dtype,
                )
                ranking = jnp.real(values)
            else:
                scores = _candidate_overlap_scores(
                    selection_reference,
                    vectors,
                    anchor=(
                        continuation_vector
                        if continuation_covector is not None
                        else None
                    ),
                )
                ranking = scores
            admissible = jnp.where(certified, ranking, -jnp.inf)
            selected = jnp.where(jnp.any(certified), jnp.argmax(admissible), 0)
            candidate_records.append((values, residuals, scores, certified, selected))
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
        (
            candidate_values,
            candidate_residuals,
            candidate_overlaps,
            certified,
            selected_index_array,
        ) = candidate_records[-1]
        selected_index = int(np.asarray(selected_index_array))
    else:
        candidate_values = jnp.reshape(solution.eigenvalue, (1,))
        candidate_residuals = jnp.reshape(solution.residual, (1,))
        candidate_overlaps = jnp.full(
            (1,),
            jnp.nan,
            dtype=jnp.real(solution.eigenvalue).dtype,
        )
        certified = _certified_candidates(
            candidate_values,
            jnp.expand_dims(solution.eigenvector, 0),
            candidate_residuals,
            tol,
        )
        selected_index = 0
    certified_growth = jnp.where(
        certified,
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
    candidate_indices = jnp.arange(candidate_values.size)
    selected_value = candidate_values[selected_index]
    spectral_distances = jnp.where(
        candidate_indices != selected_index,
        jnp.abs(candidate_values - selected_value),
        jnp.inf,
    )
    selected_spectral_gap = jnp.min(spectral_distances)
    continuation_overlap = (
        candidate_overlaps[selected_index]
        if continued
        else jnp.asarray(jnp.nan, dtype=jnp.real(solution.eigenvalue).dtype)
    )
    continuation_passed = bool(
        not continued
        or (
            float(np.asarray(continuation_overlap)) >= continuation_overlap_floor
            and float(np.asarray(selected_spectral_gap))
            >= continuation_spectral_gap_floor
        )
    )
    solution_valid = bool(
        np.asarray(
            _certified_candidates(
                jnp.reshape(solution.eigenvalue, (1,)),
                jnp.expand_dims(solution.eigenvector, 0),
                jnp.reshape(solution.residual, (1,)),
                tol,
            )[0]
        )
    )
    solution = solution._replace(
        converged=bool(solution.converged) and continuation_passed and solution_valid
    )
    return AdaptivePropagatorSolution._make(
        (
            *solution._replace(operator_applications=operator_applications),
            candidate_values,
            candidate_residuals,
            growth_gap,
            candidate_overlaps,
            selected_index,
            continuation_overlap,
            selected_spectral_gap,
            continued,
            continuation_passed,
        )
    )


__all__ = ["AdaptivePropagatorSolution", "adaptive_propagator_eigenpair"]
