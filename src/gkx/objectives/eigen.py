"""Implicit eigenvalue objectives and branch-continuity diagnostics.

The routines in this module are intentionally small and JAX-native: they expose
an implicit eigenpair derivative (forward and reverse) for locally isolated dominant-growth
branches, plus finite-difference diagnostics that verify the selected branch is
consistent before using the derivative in optimization gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np


def _dominant_pair(
    operator: Any, size: int, dtype: Any, params: Any
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Max-real eigenpair of ``A(params)``, with ``v`` scaled to ``v[k] = 1``."""

    columns = jax.vmap(lambda e: operator(params, e))(jnp.eye(size, dtype=dtype))
    matrix = jnp.swapaxes(columns, 0, 1)
    eigvals = jnp.linalg.eigvals(matrix)
    eigenvalue = eigvals[jnp.argmax(jnp.real(eigvals))]
    # Two inverse-iteration steps at a slightly offset shift give the right
    # vector; eigvals plus one LU costs about 60 % of a full ``eig``.
    offset = jnp.finfo(dtype).eps ** 0.75 * jnp.max(jnp.abs(matrix))
    eye = jnp.eye(size, dtype=dtype)
    factor = jax.scipy.linalg.lu_factor(matrix - (eigenvalue + offset) * eye)
    right = jax.scipy.linalg.lu_solve(factor, jnp.ones(size, dtype))
    right = jax.scipy.linalg.lu_solve(factor, right / jnp.linalg.norm(right))
    pivot = jnp.argmax(jnp.abs(right))
    return eigenvalue, right / right[pivot], pivot, matrix


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def dominant_eigenpair(
    operator: Any, size: int, dtype: Any, params: Any
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Max-real eigenpair of ``A(params)``, forward- and reverse-differentiable.

    ``operator(params, x)`` applies ``A(params)`` to a flat vector of length
    ``size``. The primal materializes ``A``, takes its ``eigvals`` and the
    right eigenvector by inverse iteration, scaled so its largest entry is one
    (the tangent gauge). The tangent solves Nelson's bordered system

    ``[[A - lambda I, -v], [e_k^T, 0]] [dv; dlambda] = [-(dA) v; 0]``

    whose right-hand side is one ``jax.jvp`` of ``operator`` at ``v``, so the
    tangent never materializes ``dA``. The bordered matrix depends on primals
    only: a batch of tangents (``jacfwd``, a vmapped ``jvp``) shares one ``eig``
    and one LU factorization. The tangent is linear in ``d params``, so JAX
    transposes it for reverse mode. Valid for a simple dominant eigenvalue; see
    :func:`dominant_eigenvalue_branch_locality_report`.
    """

    eigenvalue, right, _pivot, _matrix = _dominant_pair(operator, size, dtype, params)
    return eigenvalue, right


@dominant_eigenpair.defjvp
def _dominant_eigenpair_jvp(
    operator: Any, size: int, dtype: Any, primals: Any, tangents: Any
) -> tuple[Any, Any]:
    (params,), (params_dot,) = primals, tangents
    eigenvalue, right, pivot, matrix = _dominant_pair(operator, size, dtype, params)
    bordered = jnp.zeros((size + 1, size + 1), dtype)
    shifted = matrix - eigenvalue * jnp.eye(size, dtype=dtype)
    bordered = bordered.at[:size, :size].set(shifted).at[:size, size].set(-right)
    factor = jax.scipy.linalg.lu_factor(bordered.at[size, pivot].set(1.0))
    image = jax.jvp(lambda p: operator(p, right), (params,), (params_dot,))[1]
    rhs = jnp.concatenate([-image, jnp.zeros((1,), dtype)])
    solution = jax.scipy.linalg.lu_solve(factor, rhs)
    return (eigenvalue, right), (solution[size], solution[:size])


def dominant_real_eigenvalue(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the dominant growth rate with an implicit eigenpair derivative.

    The derivative is ``Re(dlambda)`` of :func:`dominant_eigenpair`: it works
    in forward mode (``jvp``, ``jacfwd``, VMEX's implicit Jacobian) and in
    reverse mode. It treats the max-real branch selected at the primal point
    as locally isolated; callers that use it in optimization should keep
    finite-difference or branch-continuity checks near accepted candidates.
    """

    matrix_arr = jnp.asarray(matrix)
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError("matrix must be square")
    dtype = jnp.result_type(matrix_arr.dtype, jnp.complex64)
    size = int(matrix_arr.shape[0])
    pair = dominant_eigenpair(jnp.matmul, size, dtype, matrix_arr.astype(dtype))
    return jnp.real(pair[0])


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
    return abs(float(a) - float(b)) / max(
        abs(float(a)), abs(float(b)), float(floor), 1.0e-300
    )


@dataclass(frozen=True)
class _BranchLocalityTolerances:
    """Validated tolerances for a dominant-eigenvalue branch locality check."""

    step: float
    gap_floor: float
    slope_rtol: float
    slope_atol: float


@dataclass(frozen=True)
class _BranchLocalitySpectrum:
    """Eigenvalue spectra and selected base branch for the locality diagnostic."""

    base_eigs: np.ndarray
    plus_eigs: np.ndarray
    minus_eigs: np.ndarray
    base_index: int
    base_value: np.complexfloating[Any, Any]
    base_gap: float


@dataclass(frozen=True)
class _BranchLocalitySlopes:
    """Central-difference slopes from dominant and nearest branch tracking."""

    dominant: float
    nearest: float
    abs_difference: float
    relative_difference: float


def _validate_branch_locality_tolerances(
    *,
    step: float,
    gap_floor: float,
    slope_rtol: float,
    slope_atol: float,
) -> _BranchLocalityTolerances:
    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0.0:
        raise ValueError("step must be finite and positive")
    gap_floor_f = float(gap_floor)
    if gap_floor_f < 0.0:
        raise ValueError("gap_floor must be non-negative")
    slope_rtol_f = float(slope_rtol)
    slope_atol_f = float(slope_atol)
    if slope_rtol_f < 0.0 or slope_atol_f < 0.0:
        raise ValueError("slope tolerances must be non-negative")
    return _BranchLocalityTolerances(
        step=step_f,
        gap_floor=gap_floor_f,
        slope_rtol=slope_rtol_f,
        slope_atol=slope_atol_f,
    )


def _branch_locality_spectrum(
    base_matrix: jnp.ndarray | np.ndarray,
    plus_matrix: jnp.ndarray | np.ndarray,
    minus_matrix: jnp.ndarray | np.ndarray,
) -> _BranchLocalitySpectrum:
    base_eigs = _eigenvalues_for_branch_report(base_matrix, name="base_matrix")
    plus_eigs = _eigenvalues_for_branch_report(plus_matrix, name="plus_matrix")
    minus_eigs = _eigenvalues_for_branch_report(minus_matrix, name="minus_matrix")
    if plus_eigs.size != base_eigs.size or minus_eigs.size != base_eigs.size:
        raise ValueError(
            "base, plus, and minus matrices must have the same eigenvalue count"
        )
    base_index = int(np.argmax(np.real(base_eigs)))
    base_value = base_eigs[base_index]
    return _BranchLocalitySpectrum(
        base_eigs=base_eigs,
        plus_eigs=plus_eigs,
        minus_eigs=minus_eigs,
        base_index=base_index,
        base_value=base_value,
        base_gap=_branch_gap(base_eigs, base_index),
    )


def _branch_locality_row(
    *,
    label: str,
    eigs: np.ndarray,
    base_value: complex,
) -> dict[str, object]:
    dominant_index = int(np.argmax(np.real(eigs)))
    nearest_index = int(np.argmin(np.abs(eigs - base_value)))
    dominant_value = eigs[dominant_index]
    nearest_value = eigs[nearest_index]
    return {
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


def _branch_locality_rows(
    spectrum: _BranchLocalitySpectrum,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    for label, eigs in (("minus", spectrum.minus_eigs), ("plus", spectrum.plus_eigs)):
        row = _branch_locality_row(
            label=label,
            eigs=eigs,
            base_value=complex(spectrum.base_value),
        )
        rows.append(row)
        selected[label] = row
    return rows, selected


def _selected_branch_float(
    selected: dict[str, dict[str, object]],
    *,
    side: str,
    key: str,
) -> float:
    return float(cast(float, selected[side][key]))


def _branch_locality_slopes(
    selected: dict[str, dict[str, object]],
    *,
    tolerances: _BranchLocalityTolerances,
) -> _BranchLocalitySlopes:
    dominant = (
        _selected_branch_float(selected, side="plus", key="dominant_real")
        - _selected_branch_float(selected, side="minus", key="dominant_real")
    ) / (2.0 * tolerances.step)
    nearest = (
        _selected_branch_float(selected, side="plus", key="nearest_real")
        - _selected_branch_float(selected, side="minus", key="nearest_real")
    ) / (2.0 * tolerances.step)
    abs_difference = abs(dominant - nearest)
    relative_difference = _relative_slope_difference(
        dominant,
        nearest,
        floor=tolerances.slope_atol,
    )
    return _BranchLocalitySlopes(
        dominant=dominant,
        nearest=nearest,
        abs_difference=abs_difference,
        relative_difference=relative_difference,
    )


def _branch_rows_passed(
    rows: list[dict[str, object]],
    *,
    gap_floor: float,
) -> bool:
    return all(
        bool(row["dominant_matches_nearest"])
        and float(cast(float, row["nearest_gap"])) >= gap_floor
        for row in rows
    )


def _branch_slope_passed(
    slopes: _BranchLocalitySlopes,
    *,
    tolerances: _BranchLocalityTolerances,
) -> bool:
    return bool(
        slopes.abs_difference <= tolerances.slope_atol
        or slopes.relative_difference <= tolerances.slope_rtol
    )


def _branch_locality_classification(
    *,
    base_isolated: bool,
    branch_rows_passed: bool,
    slope_passed: bool,
) -> str:
    if not base_isolated:
        return "base_branch_underisolated"
    if not branch_rows_passed:
        return "dominant_branch_differs_from_nearest_branch"
    if not slope_passed:
        return "dominant_and_nearest_branch_slopes_differ"
    return "dominant_branch_locally_consistent"


def _branch_locality_next_action(*, passed: bool) -> str:
    if passed:
        return "dominant-growth finite differences are locally branch-consistent"
    return (
        "do not use dominant-growth finite differences as a local derivative; "
        "reduce the perturbation, track the nearest branch explicitly, or "
        "regularize the branch before promotion"
    )


def _branch_locality_payload(
    *,
    spectrum: _BranchLocalitySpectrum,
    tolerances: _BranchLocalityTolerances,
    slopes: _BranchLocalitySlopes,
    rows: list[dict[str, object]],
    passed: bool,
    classification: str,
) -> dict[str, object]:
    return {
        "kind": "dominant_eigenvalue_branch_locality_report",
        "passed": passed,
        "classification": classification,
        "step": tolerances.step,
        "gap_floor": tolerances.gap_floor,
        "slope_rtol": tolerances.slope_rtol,
        "slope_atol": tolerances.slope_atol,
        "base_selected_index": spectrum.base_index,
        "base_eigenvalue_real": float(np.real(spectrum.base_value)),
        "base_eigenvalue_imag": float(np.imag(spectrum.base_value)),
        "base_eigenvalue_gap": spectrum.base_gap,
        "dominant_growth_fd_slope": float(slopes.dominant),
        "nearest_branch_growth_fd_slope": float(slopes.nearest),
        "slope_abs_difference": float(slopes.abs_difference),
        "slope_relative_difference": float(slopes.relative_difference),
        "branch_rows": rows,
        "next_action": _branch_locality_next_action(passed=passed),
    }


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

    tolerances = _validate_branch_locality_tolerances(
        step=step,
        gap_floor=gap_floor,
        slope_rtol=slope_rtol,
        slope_atol=slope_atol,
    )
    spectrum = _branch_locality_spectrum(base_matrix, plus_matrix, minus_matrix)
    rows, selected = _branch_locality_rows(spectrum)
    slopes = _branch_locality_slopes(selected, tolerances=tolerances)
    branch_rows_ok = _branch_rows_passed(rows, gap_floor=tolerances.gap_floor)
    base_isolated = bool(spectrum.base_gap >= tolerances.gap_floor)
    slope_ok = _branch_slope_passed(slopes, tolerances=tolerances)
    passed = bool(base_isolated and branch_rows_ok and slope_ok)
    classification = _branch_locality_classification(
        base_isolated=base_isolated,
        branch_rows_passed=branch_rows_ok,
        slope_passed=slope_ok,
    )
    return _branch_locality_payload(
        spectrum=spectrum,
        tolerances=tolerances,
        slopes=slopes,
        rows=rows,
        passed=passed,
        classification=classification,
    )


__all__ = [
    "dominant_eigenpair",
    "dominant_eigenvalue_branch_locality_report",
    "dominant_real_eigenvalue",
]
