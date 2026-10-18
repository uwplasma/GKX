"""Adaptive, residual-certified full-operator propagator eigensolve.

The Hermite-banded shift-invert (:func:`banded_ritz_pairs`) is the direct route
to the same eigenpairs and shares the file's eigenpair contract.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from functools import partial

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.solvers_linear_krylov_algorithms import (
    _apply_operator,
    _linked_covered_mode_mask,
    _omega_scale,
    _project_to_linked_cover,
    _require_linked_cover_seed,
    dominant_eigenpair_propagator_cached,
    _advance_rk4,
    _arnoldi,
)
from gkx.terms.config import TermConfig


@partial(
    jax.jit,
    static_argnames=("krylov_dim", "propagator_steps", "candidates"),
)
def dominant_eigenpairs_propagator_cached(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    dt: float,
    propagator_steps: int,
    candidates: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract several leading-growth candidates from one filtered subspace.

    Near a branch crossing, roundoff in ``|exp(T lambda)|`` can reverse two
    almost equal growth rates even when both eigenvectors are already resolved.
    Returning the leading projected candidates and ranking their continuous
    Rayleigh values avoids another propagator pass and makes the cold path
    branch-cluster aware.
    """

    if not 1 <= candidates <= krylov_dim:
        raise ValueError("candidates must lie in [1, krylov_dim]")
    if propagator_steps < 1:
        raise ValueError("propagator_steps must be positive")
    dt_val = jnp.asarray(dt, dtype=jnp.real(v0).dtype)

    def apply_prop(x, cache, params, term_cfg):
        return jax.lax.fori_loop(
            0,
            propagator_steps,
            lambda _index, state: _advance_rk4(
                state,
                cache,
                params,
                term_cfg,
                dt_val,
            ),
            x,
        )

    basis, projected = _arnoldi(
        v0,
        apply_prop,
        cache,
        params,
        term_cfg,
        krylov_dim,
    )
    eigenvalues, coefficients = jnp.linalg.eig(
        projected[:krylov_dim, :krylov_dim],
    )
    candidate_indices = jnp.argsort(jnp.abs(eigenvalues))[-candidates:][::-1]
    # The only matrix-shaped contraction anywhere on the certified path, and it
    # produces the very vectors ``certify`` below and the adaptive solver's own
    # residual are measured against. XLA answers a matrix-shaped dot with TF32 on
    # Ampere and later NVIDIA GPUs, whose 10 mantissa bits carry ~4.9e-04 -- four
    # times ``certifiable_residual_tolerance``'s complex64 floor of 1.19e-04, so
    # unpinned the gate cannot certify a converged pair no matter how many
    # restarts it spends. On an RTX A4000 the Cyclone ky = 0.3 runtime scan
    # measured 3.37e-03 and was rejected after 19 minutes of restarts; pinned it
    # certifies on the first restart in 28.7 s at the CPU eigenvalue.
    lifted = jnp.tensordot(
        coefficients[:, candidate_indices].T,
        basis[:krylov_dim],
        axes=1,
        precision=jax.lax.Precision.HIGHEST,
    )
    flattened = lifted.reshape((candidates, -1))
    norms = jnp.linalg.norm(flattened, axis=1)
    safe_norms = jnp.where(norms > 0.0, norms, 1.0)
    vectors = (flattened / safe_norms[:, None]).reshape((candidates, *v0.shape))

    def certify(vector):
        image = _apply_operator(vector, cache, params, term_cfg)
        denominator = jnp.vdot(vector, vector)
        safe_denominator = jnp.where(
            denominator != 0.0,
            denominator,
            1.0 + 0.0j,
        )
        value = jnp.vdot(vector, image) / safe_denominator
        residual = jnp.linalg.norm(image - value * vector)
        residual = residual / jnp.maximum(
            jnp.abs(value) * jnp.linalg.norm(vector),
            jnp.finfo(jnp.real(vector).dtype).tiny,
        )
        return value, residual

    values, residuals = jax.vmap(certify)(vectors)
    return values, vectors, residuals


_RESIDUAL_NOISE_EPS = 1.0e3


def certifiable_residual_tolerance(tol: float, dtype: DTypeLike) -> float:
    """Raise a residual gate to the noise floor of the arithmetic feeding it.

    A relative residual bottoms out a few ``eps`` above zero at every precision,
    so an absolute gate that is comfortable in ``float64`` can sit below
    anything ``float32`` can ever produce. Such a gate certifies nothing and
    only burns restarts on an answer that is already as good as the dtype
    allows. A request the working precision can reach is honored unchanged.
    """

    return max(float(tol), _RESIDUAL_NOISE_EPS * float(np.finfo(np.dtype(dtype)).eps))


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

    ``tol`` gates a relative residual built from operator applications in the
    dtype of ``v0``, so it is raised to that dtype's noise floor. The default
    is a strict ``float64`` gate; the same call certifies in ``float32`` at the
    tightest residual single precision can actually deliver.
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
    covered = _linked_covered_mode_mask(cache)
    if covered is not None:
        # Modes outside every linked chain are decoupled neutral drift waves;
        # a seed component there is never part of a chain eigenvector.
        v0 = _project_to_linked_cover(v0, covered)
        _require_linked_cover_seed(v0)
    tol = certifiable_residual_tolerance(tol, v0.dtype)
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


# --- Hermite-banded shift-invert ---------------------------------------------
#
# Hermite-banded direct shift-invert for the single-ky linear operator.
#
# At one ky the operator couples Hermite m only to m +- 2 (streaming, mirror and
# A_par move m by one, the drifts by two, and the field response and collisions
# stay inside that band). Grouping m in pairs therefore makes it exactly block
# tridiagonal, with blocks of ``2 ns Nl Nz`` unknowns. The blocks are read off
# GKX's own operator in ``3 ceil(m / batch)`` batched products
# (:func:`solvax.block_tridiagonal_from_products`; the prototype matched the
# matrix-free operator to 4e-16), factored once per shift in complex64 by
# :func:`solvax.block_thomas_factor`, and one defect correction against the
# operator restores the working precision of each application. ARPACK runs
# shift-invert on that inverse; the caller certifies the Ritz pairs against the
# original operator.
#
# Without a shift, a dense eigensolve of a leading truncation (everything up to
# ``_DENSE_N`` unknowns, else Laguerre < ``_COARSE_NL`` of the first Hermite
# pairs) places it, and it follows the fastest-growing Ritz value while that
# moves. Dominance is that of the nearest ``_RITZ`` eigenvalues, not a proof.
_COARSE_NL = 3  # Laguerre moments kept in the dense shift estimate
_DENSE_N = 3072  # unknowns of the largest dense estimate
_RITZ = 12  # eigenvalues asked of ARPACK around the shift
_PASSES = 3  # shifts tried while the fastest-growing Ritz value keeps moving
_RECENTER = 0.05  # that movement, in omega_scale, that starts another pass
_REFINE = 2  # defect corrections per application
_DOMINANCE = 0.5  # excess growth, in units of best gamma*dt, that refuses the pair
_STEPS = 60  # backward-Euler steps of the dominance check
_RESTARTS = 100  # ARPACK restarts allowed per shift before the route gives up
_OFFSET = 1.0e-3  # step (in omega_scale) from the estimate to the shift
_BAND_TOL = (
    1.0e-4  # relative mismatch of the blocks against the operator that is refused
)


def _dense_estimate(lower, diag, upper, laguerre: np.ndarray, nl: int) -> complex:
    """Fastest-growing eigenvalue of a dense leading truncation of the blocks.

    The whole operator if it has at most ``_DENSE_N`` unknowns, else Laguerre
    < ``_COARSE_NL`` of as many leading Hermite pairs (at least two) as fit.
    ``laguerre`` holds each block unknown's Laguerre index, padding above ``nl``.
    """

    cut = nl if (laguerre < nl).sum() <= _DENSE_N else _COARSE_NL
    rows = [np.flatnonzero(row < cut) for row in laguerre]
    fit = np.searchsorted(np.cumsum([r.size for r in rows]), _DENSE_N, side="right")
    rows = rows[: max(int(fit), 2)]
    edge = np.cumsum([0] + [r.size for r in rows])
    dense = np.zeros((edge[-1], edge[-1]), np.complex128)
    for b, r in enumerate(rows):
        span = slice(edge[b], edge[b + 1])
        dense[span, span] = np.asarray(diag[b])[np.ix_(r, r)]
        if b:
            dense[span, edge[b - 1] : edge[b]] = np.asarray(lower[b])[
                np.ix_(r, rows[b - 1])
            ]
        if b + 1 < len(rows):
            dense[span, edge[b + 1] : edge[b + 2]] = np.asarray(upper[b])[
                np.ix_(r, rows[b + 1])
            ]
    values = np.linalg.eigvals(dense)
    return complex(values[np.argmax(values.real)])


def _growth_excess(step: Callable, v: np.ndarray, found: complex, dt: float) -> float:
    """Log growth per backward-Euler step of a random state, beyond that of ``found``.

    ``step`` applies ``(I - dt A)^{-1}``. A mode of rate ``lam`` is amplified by
    ``1 / |1 - lam dt|`` per step, so after ``_STEPS`` steps the last quarter's
    mean amplification exceeds that of ``found`` only if a faster mode is present
    (up to the transient growth of a non-normal operator).
    """

    ratios = []
    for _ in range(_STEPS):
        w = step(v)
        ratios.append(np.log(np.linalg.norm(w) / np.linalg.norm(v)))
        v = w / np.linalg.norm(w)
    return float(np.mean(ratios[-_STEPS // 4 :]) + np.log(abs(1 - found * dt)))


def banded_fits(shape: tuple[int, ...]) -> bool:
    """Whether the four complex64 block arrays fit in half the default device's memory."""

    nm = shape[-4]
    need = 8 * 4 * (nm // 2 + nm % 2) * (2 * int(np.prod(shape)) // nm) ** 2
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit") or os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_PHYS_PAGES"
    )
    return nm >= 4 and need <= limit / 2


def banded_ritz_pairs(*args, **kwargs):
    """Eigenpairs near a shift from the Hermite-banded factor; see ``_banded_ritz_pairs``."""

    # TF32 matmuls on GPU would put the factor's error above what one correction removes.
    with jax.default_matmul_precision("highest"):
        return _banded_ritz_pairs(*args, **kwargs)


def _banded_ritz_pairs(
    v0: jnp.ndarray,
    cache,
    params,
    term_cfg,
    shift: complex | None,
    status: Callable[[str], None] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, complex]:
    """Return ``_RITZ`` eigenpairs of the operator nearest ``shift`` and the shift.

    ``v0`` fixes the state shape ``(..., Nl, Nm, ky, kx, z)`` and dtype. On a
    linked cache only the chain modes are unknowns. Vectors come back in the
    full state shape; none is certified here.
    """

    from scipy.sparse.linalg import LinearOperator, eigs
    from solvax import (  # type: ignore[attr-defined]
        block_thomas_factor,
        block_thomas_solve,
        block_tridiag_matvec,
        block_tridiagonal_from_products,
        iterative_refinement,
    )

    note = status or (lambda _message: None)
    shape, size = v0.shape, int(v0.size)
    nm = shape[-4]
    if nm < 4:
        raise ValueError("banded shift-invert requires Nm >= 4")
    covered = _linked_covered_mode_mask(cache)
    keep = np.ones(shape, bool)
    if covered is not None:
        keep = np.broadcast_to(np.asarray(covered)[:, :, None], shape)
    columns = np.moveaxis(keep, -4, 0)[0].reshape(-1)  # unknowns kept, per m

    def pairs(index: np.ndarray, pad: int) -> np.ndarray:
        """Hermite-moment-major ``index`` as ``(n_blocks, 2 * unknowns)``, padded if Nm is odd."""
        rows = np.moveaxis(index, -4, 0).reshape(nm, -1)[:, columns]
        rows = np.vstack([rows, np.full_like(rows[:1], pad)]) if nm % 2 else rows
        return rows.reshape(rows.shape[0] // 2, -1)

    blocks = pairs(np.arange(size).reshape(shape), -1)
    laguerre = pairs(np.indices(shape)[-5], size)
    safe = jnp.asarray(np.where(blocks < 0, size, blocks))
    # Gathers, not scatters: an XLA scatter of complex values costs 10x the solve on GPU.
    position = np.full(size, blocks.size)
    position[blocks[blocks >= 0]] = np.flatnonzero(blocks >= 0)
    positions = jnp.asarray(position)
    mask = jnp.asarray(keep.reshape(-1))

    def apply(x):
        return (
            _apply_operator(x.reshape(shape), cache, params, term_cfg).reshape(-1)
            * mask
        )

    note(f"assembling {blocks.shape[0]} Hermite-pair blocks of {blocks.shape[1]}")
    lower, diag, upper = block_tridiagonal_from_products(
        apply, blocks, dtype=jnp.complex64
    )

    # The blocks are the operator only if it is Hermite-banded: compare on a random vector.
    noise = np.random.default_rng(0).standard_normal((2, size)) * keep.reshape(-1)
    probe = jnp.asarray(noise[0] + 1j * noise[1], v0.dtype)
    expected = jnp.append(apply(probe), 0)[safe]
    mismatch = jnp.linalg.norm(
        block_tridiag_matvec(lower, diag, upper, jnp.append(probe, 0)[safe]) - expected
    ) / jnp.linalg.norm(expected)
    if not mismatch < _BAND_TOL:
        raise ValueError(
            f"operator is not Hermite-banded (blocks miss it by {mismatch:.2g})"
        )

    scale = float(np.asarray(_omega_scale(cache, params)))
    auto_shift = shift is None
    if shift is None:
        # Dense eigenvalues of a leading truncation: everything if it is small, else
        # Laguerre < _COARSE_NL of as many Hermite pairs as fit in _DENSE_N unknowns.
        shift = _dense_estimate(lower, diag, upper, laguerre, shape[-5])
        note(f"banded shift from a dense truncation: sigma={shift:.5g}")
        shift += _OFFSET * scale
    sigma = complex(shift)

    @jax.jit  # factors are an argument: closed over, XLA would bake them in as constants
    def inverse(factors, b, sigma):
        def bare(r):
            x = block_thomas_solve(
                factors, jnp.append(r, 0)[safe].astype(jnp.complex64)
            )
            return jnp.append(x.reshape(-1), 0)[positions].astype(v0.dtype)

        residual = lambda x: apply(x) - sigma * x  # noqa: E731
        return iterative_refinement(residual, b, bare, iterations=_REFINE)[0].astype(
            v0.dtype
        )

    # A seeded random start: the seed state can be orthogonal to the fastest mode.
    start = np.asarray(probe)
    solves = [0]
    operator = LinearOperator(
        (size, size),
        matvec=lambda v: np.asarray(apply(jnp.asarray(v, v0.dtype))),
        dtype=np.complex128,
    )
    for _ in range(_PASSES):
        note(f"factoring the shifted blocks (complex64) at sigma={sigma:.5g}")
        factors = block_thomas_factor(lower, diag, upper, shift=sigma)  # type: ignore[call-arg]

        def solve(v, factors=factors, sigma=sigma):
            solves[0] += 1
            return np.asarray(inverse(factors, jnp.asarray(v, v0.dtype), sigma))

        values, vectors = eigs(
            operator,
            k=_RITZ,
            sigma=sigma,
            OPinv=LinearOperator((size, size), matvec=solve, dtype=np.complex128),
            v0=start,
            tol=1.0e-10,
            ncv=2 * _RITZ + 4,
            maxiter=_RESTARTS,
        )
        best = complex(values[np.argmax(values.real)])
        if abs(best - sigma) <= _RECENTER * scale:
            break
        sigma = (
            best + _OFFSET * scale
        )  # the fastest-growing Ritz value moved: follow it
    note(f"banded Arnoldi: {solves[0]} solves")
    if auto_shift:
        # Dominance check: the step is small against the fastest plausible frequency.
        dt = 0.25 / max(2 * scale, 1.5 * abs(best.imag))
        factors = block_thomas_factor(lower, diag, upper, shift=1 / dt)  # type: ignore[call-arg]
        excess = _growth_excess(
            lambda x: (
                -np.asarray(inverse(factors, jnp.asarray(x, v0.dtype), 1 / dt)) / dt
            ),
            start.astype(np.complex128),
            best,
            dt,
        )
        note(f"banded dominance check: excess growth {excess:.3g} per step")
        if excess > _DOMINANCE * max(best.real, 0.0) * dt + 1e-3:
            raise RuntimeError(
                f"a faster mode than {best:.4g} is present (growth excess {excess:.3g} per step)"
            )
    return (
        jnp.asarray(values, v0.dtype),
        jnp.asarray(vectors.T, v0.dtype).reshape((-1, *shape)),
        sigma,
    )


__all__ = [
    "AdaptivePropagatorSolution",
    "adaptive_propagator_eigenpair",
    "banded_fits",
    "banded_ritz_pairs",
    "certifiable_residual_tolerance",
    "dominant_eigenpairs_propagator_cached",
]
