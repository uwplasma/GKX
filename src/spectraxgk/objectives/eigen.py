"""Implicit eigenvalue objectives and branch-continuity diagnostics.

The routines in this module are intentionally small and JAX-native: they expose
an implicit left/right eigenpair VJP for locally isolated dominant-growth
branches, plus finite-difference diagnostics that verify the selected branch is
consistent before using the derivative in optimization gates.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np


def _select_dominant_eigen_triplet(
    matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the max-real eigenvalue and biorthogonal right/left vectors."""

    eigvals, eigvecs = jnp.linalg.eig(matrix)
    index = jnp.argmax(jnp.real(eigvals))
    eigenvalue = eigvals[index]
    right = eigvecs[:, index]
    left_vals, left_vecs = jnp.linalg.eig(jnp.conj(jnp.swapaxes(matrix, 0, 1)))
    left_index = jnp.argmin(jnp.abs(left_vals - jnp.conj(eigenvalue)))
    left = left_vecs[:, left_index]
    overlap = jnp.vdot(left, right)
    tiny = jnp.asarray(1.0e-30, dtype=jnp.real(overlap).dtype)
    safe_overlap = jnp.where(jnp.abs(overlap) > tiny, overlap, tiny + 0.0j)
    left = left / jnp.conj(safe_overlap)
    return eigenvalue, right, left


@jax.custom_vjp
def _dominant_real_eigenvalue_complex(matrix: jnp.ndarray) -> jnp.ndarray:
    eigenvalue, _right, _left = _select_dominant_eigen_triplet(matrix)
    return jnp.real(eigenvalue)


def _dominant_real_eigenvalue_complex_fwd(
    matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    eigenvalue, right, left = _select_dominant_eigen_triplet(matrix)
    return jnp.real(eigenvalue), (right, left)


def _dominant_real_eigenvalue_complex_bwd(
    residual: tuple[jnp.ndarray, jnp.ndarray],
    cotangent: jnp.ndarray,
) -> tuple[jnp.ndarray]:
    right, left = residual
    matrix_cotangent = jnp.asarray(cotangent) * jnp.outer(jnp.conj(left), right)
    return (matrix_cotangent,)


_dominant_real_eigenvalue_complex.defvjp(
    _dominant_real_eigenvalue_complex_fwd,
    _dominant_real_eigenvalue_complex_bwd,
)


def dominant_real_eigenvalue(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the dominant growth rate with an implicit left/right VJP.

    This helper treats the max-real eigenvalue branch selected at the primal
    point as locally isolated. Its reverse rule uses
    ``d lambda = w^H dA v`` with ``w^H v = 1`` instead of differentiating
    through non-Hermitian eigenvectors. Branch isolation is still a physics
    gate: callers that use this in optimization should keep finite-difference
    or branch-continuity checks enabled near accepted candidates.
    """

    matrix_arr = jnp.asarray(matrix)
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError("matrix must be square")
    if not jnp.iscomplexobj(matrix_arr):
        complex_dtype = jnp.complex128 if matrix_arr.dtype == jnp.float64 else jnp.complex64
        matrix_arr = matrix_arr.astype(complex_dtype)
    return _dominant_real_eigenvalue_complex(matrix_arr)


def _eigenvalues_for_branch_report(matrix: Any, *, name: str) -> np.ndarray:
    matrix_arr = jnp.asarray(matrix)
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"{name} must be square")
    eigs = np.asarray(jnp.linalg.eigvals(matrix_arr))
    if eigs.ndim != 1 or eigs.size == 0:
        raise ValueError(f"{name} must have at least one eigenvalue")
    if not np.all(np.isfinite(eigs)):
        raise ValueError(f"{name} eigenvalues must be finite")
    return eigs


def _branch_gap(eigs: np.ndarray, index: int) -> float:
    if eigs.size == 1:
        return float("inf")
    value = eigs[int(index)]
    return float(np.min(np.abs(np.delete(eigs, int(index)) - value)))


def _relative_slope_difference(a: float, b: float, *, floor: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), float(floor), 1.0e-300)


def dominant_eigenvalue_branch_locality_report(
    base_matrix: jnp.ndarray | np.ndarray,
    plus_matrix: jnp.ndarray | np.ndarray,
    minus_matrix: jnp.ndarray | np.ndarray,
    *,
    step: float,
    gap_floor: float = 1.0e-8,
    slope_rtol: float = 1.0e-2,
    slope_atol: float = 1.0e-8,
) -> dict[str, object]:
    """Report whether dominant-growth finite differences follow one branch.

    The dominant-growth custom VJP assumes the max-real eigenvalue selected at
    the primal point is locally isolated. This diagnostic compares two central
    finite-difference slopes:

    ``dominant_growth_fd_slope``
        Uses the max-real eigenvalue independently at the plus/minus points.
    ``nearest_branch_growth_fd_slope``
        Uses the eigenvalue nearest to the base dominant eigenvalue at the
        plus/minus points.

    When those branches differ, exact finite differences are measuring a
    branch switch rather than the local derivative used by the implicit
    eigenpair VJP.
    """

    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0.0:
        raise ValueError("step must be finite and positive")
    gap_floor_f = float(gap_floor)
    if gap_floor_f < 0.0:
        raise ValueError("gap_floor must be non-negative")
    if float(slope_rtol) < 0.0 or float(slope_atol) < 0.0:
        raise ValueError("slope tolerances must be non-negative")

    base_eigs = _eigenvalues_for_branch_report(base_matrix, name="base_matrix")
    plus_eigs = _eigenvalues_for_branch_report(plus_matrix, name="plus_matrix")
    minus_eigs = _eigenvalues_for_branch_report(minus_matrix, name="minus_matrix")
    if plus_eigs.size != base_eigs.size or minus_eigs.size != base_eigs.size:
        raise ValueError("base, plus, and minus matrices must have the same eigenvalue count")

    base_index = int(np.argmax(np.real(base_eigs)))
    base_value = base_eigs[base_index]
    base_gap = _branch_gap(base_eigs, base_index)
    rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    for label, eigs in (("minus", minus_eigs), ("plus", plus_eigs)):
        dominant_index = int(np.argmax(np.real(eigs)))
        nearest_index = int(np.argmin(np.abs(eigs - base_value)))
        dominant_value = eigs[dominant_index]
        nearest_value = eigs[nearest_index]
        row = {
            "side": label,
            "dominant_index": dominant_index,
            "nearest_index": nearest_index,
            "dominant_real": float(np.real(dominant_value)),
            "dominant_imag": float(np.imag(dominant_value)),
            "nearest_real": float(np.real(nearest_value)),
            "nearest_imag": float(np.imag(nearest_value)),
            "nearest_gap": _branch_gap(eigs, nearest_index),
            "dominant_matches_nearest": bool(dominant_index == nearest_index),
        }
        rows.append(row)
        selected[label] = row

    dominant_growth_fd_slope = (
        float(cast(float, selected["plus"]["dominant_real"]))
        - float(cast(float, selected["minus"]["dominant_real"]))
    ) / (2.0 * step_f)
    nearest_branch_growth_fd_slope = (
        float(cast(float, selected["plus"]["nearest_real"]))
        - float(cast(float, selected["minus"]["nearest_real"]))
    ) / (2.0 * step_f)
    slope_abs_diff = abs(dominant_growth_fd_slope - nearest_branch_growth_fd_slope)
    slope_rel_diff = _relative_slope_difference(
        dominant_growth_fd_slope,
        nearest_branch_growth_fd_slope,
        floor=float(slope_atol),
    )
    branch_rows_passed = all(
        bool(row["dominant_matches_nearest"]) and float(cast(float, row["nearest_gap"])) >= gap_floor_f
        for row in rows
    )
    base_isolated = bool(base_gap >= gap_floor_f)
    slope_passed = bool(slope_abs_diff <= float(slope_atol) or slope_rel_diff <= float(slope_rtol))
    passed = bool(base_isolated and branch_rows_passed and slope_passed)
    if not base_isolated:
        classification = "base_branch_underisolated"
    elif not branch_rows_passed:
        classification = "dominant_branch_differs_from_nearest_branch"
    elif not slope_passed:
        classification = "dominant_and_nearest_branch_slopes_differ"
    else:
        classification = "dominant_branch_locally_consistent"

    return {
        "kind": "dominant_eigenvalue_branch_locality_report",
        "passed": passed,
        "classification": classification,
        "step": step_f,
        "gap_floor": gap_floor_f,
        "slope_rtol": float(slope_rtol),
        "slope_atol": float(slope_atol),
        "base_selected_index": base_index,
        "base_eigenvalue_real": float(np.real(base_value)),
        "base_eigenvalue_imag": float(np.imag(base_value)),
        "base_eigenvalue_gap": base_gap,
        "dominant_growth_fd_slope": float(dominant_growth_fd_slope),
        "nearest_branch_growth_fd_slope": float(nearest_branch_growth_fd_slope),
        "slope_abs_difference": float(slope_abs_diff),
        "slope_relative_difference": float(slope_rel_diff),
        "branch_rows": rows,
        "next_action": (
            "dominant-growth finite differences are locally branch-consistent"
            if passed
            else (
                "do not use dominant-growth finite differences as a local derivative; "
                "reduce the perturbation, track the nearest branch explicitly, or "
                "regularize the branch before promotion"
            )
        ),
    }


__all__ = [
    "dominant_eigenvalue_branch_locality_report",
    "dominant_real_eigenvalue",
]
