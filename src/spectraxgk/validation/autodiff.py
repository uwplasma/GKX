"""Autodiff validation helpers for inverse and UQ examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff_finite_difference import (
    _jax_enable_x64,
    autodiff_finite_difference_report,
    central_finite_difference_jacobian,
    covariance_diagnostics,
)


@dataclass(frozen=True)
class _EigenBranchSelection:
    selector_key: str
    index: int
    eigenvalue: complex
    gap: float
    branch_isolated: bool


@dataclass(frozen=True)
class _ImplicitEigenpairData:
    matrix: jnp.ndarray
    eigenvalue: jnp.ndarray
    eigenvector: jnp.ndarray
    left_eigenvector: jnp.ndarray
    selection: _EigenBranchSelection
    left_index: int
    overlap_abs: float


@dataclass(frozen=True)
class _ImplicitSensitivityData:
    eigenvalue_sensitivity: jnp.ndarray
    eigenvector_sensitivity: jnp.ndarray
    jacobian_implicit: jnp.ndarray
    jacobian_fd: jnp.ndarray
    max_abs_error: float
    max_rel_error: float
    passed: bool


def _params_vector(params: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    return p


def _selector_key(selector: str) -> str:
    return selector.strip().lower()


def _selected_eigen_index(eig_np: np.ndarray, selector_key: str) -> int:
    if selector_key == "max_real":
        return int(np.argmax(np.real(eig_np)))
    if selector_key.startswith("index:"):
        index = int(selector_key.split(":", 1)[1])
        if index < 0 or index >= eig_np.size:
            raise ValueError(
                f"selector index {index} is out of bounds for {eig_np.size} eigenvalues"
            )
        return index
    raise ValueError("selector must be 'max_real' or 'index:N'")


def _eigen_gap(eig_np: np.ndarray, *, index: int, selected: complex) -> float:
    if eig_np.size == 1:
        return float("inf")
    others = np.delete(eig_np, index)
    return float(np.min(np.abs(selected - others)))


def _select_eigen_branch(
    eigvals: jnp.ndarray,
    *,
    selector: str,
    gap_floor: float,
) -> _EigenBranchSelection:
    eig_np = np.asarray(eigvals)
    if eig_np.ndim != 1 or eig_np.size == 0:
        raise ValueError(
            "matrix_fn must return a square matrix with at least one eigenvalue"
        )
    selector_key = _selector_key(selector)
    index = _selected_eigen_index(eig_np, selector_key)
    selected = complex(eig_np[index])
    gap = _eigen_gap(eig_np, index=index, selected=selected)
    return _EigenBranchSelection(
        selector_key=selector_key,
        index=index,
        eigenvalue=selected,
        gap=gap,
        branch_isolated=bool(gap >= float(gap_floor)),
    )


def _real_observable(
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    lam_i: jnp.ndarray,
    v_i: jnp.ndarray,
    p_i: jnp.ndarray,
) -> jnp.ndarray:
    obs = jnp.ravel(jnp.asarray(observable_fn(lam_i, v_i, p_i)))
    if jnp.iscomplexobj(obs):
        return jnp.concatenate([jnp.real(obs), jnp.imag(obs)])
    return jnp.real(obs)


def _selection_report_payload(
    selection: _EigenBranchSelection,
    *,
    gap_floor: float,
) -> dict[str, object]:
    return {
        "selector": selection.selector_key,
        "selected_index": selection.index,
        "eigenvalue_real": float(np.real(selection.eigenvalue)),
        "eigenvalue_imag": float(np.imag(selection.eigenvalue)),
        "eigenvalue_gap": selection.gap,
        "gap_floor": float(gap_floor),
        "branch_isolated": selection.branch_isolated,
    }


def _unsupported_eigen_ad_report(
    selection: _EigenBranchSelection,
    *,
    exc: NotImplementedError,
    step: float,
    rtol: float,
    atol: float,
    gap_floor: float,
) -> dict[str, object]:
    return {
        "passed": False,
        "ad_supported": False,
        "failure_reason": str(exc),
        "step": float(step),
        "rtol": float(rtol),
        "atol": float(atol),
        **_selection_report_payload(selection, gap_floor=gap_floor),
    }


def explicit_complex_operator_matrix(
    operator: Callable[[jnp.ndarray], Any],
    state_shape: tuple[int, ...],
    *,
    dtype: Any | None = None,
) -> jnp.ndarray:
    """Materialize a small complex linear operator as a dense matrix.

    This helper is intended for validation fixtures, not production solves. It
    applies ``operator`` to each basis vector of ``state_shape`` and returns a
    matrix whose columns are the flattened outputs. Small dense matrices make
    eigenvalue AD-vs-finite-difference gates easy to express while keeping the
    production code matrix-free.
    """

    if not state_shape or any(int(size) <= 0 for size in state_shape):
        raise ValueError("state_shape must contain positive dimensions")
    matrix_dtype = dtype or (jnp.complex128 if _jax_enable_x64() else jnp.complex64)
    size = int(np.prod(tuple(int(dim) for dim in state_shape)))
    eye = jnp.eye(size, dtype=matrix_dtype)

    def column(vec: jnp.ndarray) -> jnp.ndarray:
        out = jnp.asarray(operator(jnp.reshape(vec, state_shape)))
        if tuple(out.shape) != tuple(state_shape):
            raise ValueError("operator output shape must match state_shape")
        return jnp.ravel(out)

    columns = jax.vmap(column)(eye)
    return jnp.swapaxes(columns, 0, 1)


def isolated_eigenvalue_sensitivity_report(
    matrix_fn: Callable[[jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    selector: str = "max_real",
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    gap_floor: float = 1.0e-8,
) -> dict[str, object]:
    """Validate AD sensitivities of one isolated eigenvalue branch.

    The branch index is selected at the base point and then held fixed during
    the finite-difference comparison. This mirrors the branch-continuity
    assumption used for linear growth/frequency sensitivities.
    """

    p = _params_vector(params)
    eig_base = jnp.linalg.eigvals(jnp.asarray(matrix_fn(p)))
    selection = _select_eigen_branch(
        eig_base,
        selector=selector,
        gap_floor=gap_floor,
    )

    def branch_fn(x: jnp.ndarray) -> jnp.ndarray:
        value = jnp.linalg.eigvals(jnp.asarray(matrix_fn(x)))[selection.index]
        return jnp.asarray([jnp.real(value), jnp.imag(value)])

    try:
        report = autodiff_finite_difference_report(
            branch_fn,
            p,
            step=step,
            rtol=rtol,
            atol=atol,
        )
    except NotImplementedError as exc:
        return _unsupported_eigen_ad_report(
            selection,
            exc=exc,
            step=step,
            rtol=rtol,
            atol=atol,
            gap_floor=gap_floor,
        )
    return {
        **report,
        "passed": bool(report["passed"]) and selection.branch_isolated,
        "ad_supported": True,
        **_selection_report_payload(selection, gap_floor=gap_floor),
    }


def isolated_eigenpair_observable_sensitivity_report(
    matrix_fn: Callable[[jnp.ndarray], Any],
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    selector: str = "max_real",
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    gap_floor: float = 1.0e-8,
) -> dict[str, object]:
    """Validate AD sensitivities of an observable of one isolated eigenpair.

    ``observable_fn`` receives ``(eigenvalue, eigenvector, params)`` for the
    branch selected at the base point. The selected index is held fixed during
    finite differences, so this gate is appropriate for branch-continuous,
    phase-invariant quantities such as ``gamma / <k_perp^2>``.
    """

    p = _params_vector(params)
    eig_base, vec_base = jnp.linalg.eig(jnp.asarray(matrix_fn(p)))
    selection = _select_eigen_branch(
        eig_base,
        selector=selector,
        gap_floor=gap_floor,
    )
    eig_np = np.asarray(eig_base)
    if np.asarray(vec_base).shape[1] != eig_np.size:
        raise ValueError("eigenvector matrix shape is inconsistent with eigenvalues")

    def branch_fn(x: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = jnp.linalg.eig(jnp.asarray(matrix_fn(x)))
        return _real_observable(
            observable_fn,
            eigvals[selection.index],
            eigvecs[:, selection.index],
            x,
        )

    try:
        report = autodiff_finite_difference_report(
            branch_fn,
            p,
            step=step,
            rtol=rtol,
            atol=atol,
        )
    except NotImplementedError as exc:
        return _unsupported_eigen_ad_report(
            selection,
            exc=exc,
            step=step,
            rtol=rtol,
            atol=atol,
            gap_floor=gap_floor,
        )
    return {
        **report,
        "passed": bool(report["passed"]) and selection.branch_isolated,
        "ad_supported": True,
        **_selection_report_payload(selection, gap_floor=gap_floor),
    }


def _square_matrix_at_params(
    matrix_fn: Callable[[jnp.ndarray], Any],
    p: jnp.ndarray,
) -> jnp.ndarray:
    matrix = jnp.asarray(matrix_fn(p))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix_fn must return a square matrix")
    return matrix


def _normalized_left_eigenvector(
    matrix: jnp.ndarray,
    *,
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
) -> tuple[jnp.ndarray, int, float]:
    left_vals, left_vecs = jnp.linalg.eig(jnp.conj(jnp.swapaxes(matrix, 0, 1)))
    left_index = int(
        np.argmin(np.abs(np.asarray(left_vals) - np.conj(np.asarray(eigenvalue))))
    )
    left_eigenvector = left_vecs[:, left_index]
    overlap = jnp.vdot(left_eigenvector, eigenvector)
    overlap_abs = float(np.abs(np.asarray(overlap)))
    if overlap_abs <= 0.0 or not np.isfinite(overlap_abs):
        raise ValueError("left/right eigenvectors are biorthogonally singular")
    return left_eigenvector / jnp.conj(overlap), left_index, overlap_abs


def _implicit_eigenpair_data(
    matrix_fn: Callable[[jnp.ndarray], Any],
    p: jnp.ndarray,
    *,
    selector: str,
    gap_floor: float,
) -> _ImplicitEigenpairData:
    matrix = _square_matrix_at_params(matrix_fn, p)
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    selection = _select_eigen_branch(
        eigvals,
        selector=selector,
        gap_floor=gap_floor,
    )
    eigenvalue = eigvals[selection.index]
    eigenvector = eigvecs[:, selection.index]
    left_eigenvector, left_index, overlap_abs = _normalized_left_eigenvector(
        matrix,
        eigenvalue=eigenvalue,
        eigenvector=eigenvector,
    )
    return _ImplicitEigenpairData(
        matrix=matrix,
        eigenvalue=eigenvalue,
        eigenvector=eigenvector,
        left_eigenvector=left_eigenvector,
        selection=selection,
        left_index=left_index,
        overlap_abs=overlap_abs,
    )


def _matrix_parameter_jacobian(
    matrix_fn: Callable[[jnp.ndarray], Any],
    p: jnp.ndarray,
    matrix_shape: tuple[int, ...],
) -> jnp.ndarray:
    def flat_matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(jnp.asarray(matrix_fn(x)))

    dA_flat = jax.jacfwd(flat_matrix_fn)(p)
    return jnp.reshape(dA_flat, matrix_shape + (p.size,))


def _implicit_eigenpair_tangents(
    data: _ImplicitEigenpairData,
    dA: jnp.ndarray,
    parameter_count: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    matrix = data.matrix
    eigenvalue = data.eigenvalue
    eigenvector = data.eigenvector
    left_eigenvector = data.left_eigenvector
    n = int(matrix.shape[0])
    identity = jnp.eye(n, dtype=matrix.dtype)
    top = jnp.concatenate([matrix - eigenvalue * identity, -eigenvector[:, None]], axis=1)
    bottom = jnp.concatenate(
        [jnp.conj(left_eigenvector)[None, :], jnp.zeros((1, 1), dtype=matrix.dtype)],
        axis=1,
    )
    augmented = jnp.concatenate([top, bottom], axis=0)
    rhs = jnp.stack(
        [
            jnp.concatenate(
                [-dA[:, :, i] @ eigenvector, jnp.zeros((1,), dtype=matrix.dtype)]
            )
            for i in range(int(parameter_count))
        ],
        axis=1,
    )
    solution = jnp.linalg.solve(augmented, rhs)
    return solution[n, :], solution[:n, :]


def _packed_eigenpair_base(eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.asarray([jnp.real(eigenvalue), jnp.imag(eigenvalue)]),
            jnp.real(eigenvector),
            jnp.imag(eigenvector),
        ]
    )


def _packed_eigenpair_tangent(
    eigenvalue_sensitivity: jnp.ndarray,
    eigenvector_sensitivity: jnp.ndarray,
    *,
    index: int,
) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.asarray(
                [
                    jnp.real(eigenvalue_sensitivity[index]),
                    jnp.imag(eigenvalue_sensitivity[index]),
                ]
            ),
            jnp.real(eigenvector_sensitivity[:, index]),
            jnp.imag(eigenvector_sensitivity[:, index]),
        ]
    )


def _implicit_observable_jacobian(
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    p: jnp.ndarray,
    data: _ImplicitEigenpairData,
    *,
    eigenvalue_sensitivity: jnp.ndarray,
    eigenvector_sensitivity: jnp.ndarray,
) -> jnp.ndarray:
    n = int(data.matrix.shape[0])

    def observable_real_from_eigenpair(packed: jnp.ndarray) -> jnp.ndarray:
        lam_i = packed[0] + 1j * packed[1]
        v_real_start = 2
        v_imag_start = v_real_start + n
        v_i = packed[v_real_start:v_imag_start] + 1j * packed[v_imag_start:]
        return _real_observable(observable_fn, lam_i, v_i, p)

    def observable_real_from_params(p_i: jnp.ndarray) -> jnp.ndarray:
        return _real_observable(observable_fn, data.eigenvalue, data.eigenvector, p_i)

    eigenpair_base = _packed_eigenpair_base(data.eigenvalue, data.eigenvector)
    obs_jac_eigenpair = jax.jacfwd(observable_real_from_eigenpair)(eigenpair_base)
    obs_jac_params = jax.jacfwd(observable_real_from_params)(p)
    eye = jnp.eye(p.size, dtype=p.dtype)
    implicit_cols = [
        obs_jac_eigenpair
        @ _packed_eigenpair_tangent(
            eigenvalue_sensitivity,
            eigenvector_sensitivity,
            index=i,
        )
        + obs_jac_params @ eye[i]
        for i in range(int(p.size))
    ]
    return jnp.stack(implicit_cols, axis=1)


def _nearest_branch_observable(
    matrix_fn: Callable[[jnp.ndarray], Any],
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    reference_eigenvalue: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def branch_observable(x: jnp.ndarray) -> jnp.ndarray:
        eigvals_i, eigvecs_i = jnp.linalg.eig(jnp.asarray(matrix_fn(x)))
        branch_index = int(
            np.argmin(np.abs(np.asarray(eigvals_i) - np.asarray(reference_eigenvalue)))
        )
        return _real_observable(
            observable_fn,
            eigvals_i[branch_index],
            eigvecs_i[:, branch_index],
            x,
        )

    return branch_observable


def _jacobian_error_metrics(
    jacobian_implicit: jnp.ndarray,
    jacobian_fd: jnp.ndarray,
    *,
    atol: float,
) -> tuple[float, float]:
    err = np.asarray(jacobian_implicit - jacobian_fd, dtype=float)
    denom = np.maximum(np.asarray(np.abs(jacobian_fd), dtype=float), float(atol))
    rel = np.abs(err) / denom
    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    return max_abs, max_rel


def _implicit_sensitivity_data(
    matrix_fn: Callable[[jnp.ndarray], Any],
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    p: jnp.ndarray,
    data: _ImplicitEigenpairData,
    *,
    step: float,
    rtol: float,
    atol: float,
) -> _ImplicitSensitivityData:
    dA = _matrix_parameter_jacobian(matrix_fn, p, tuple(data.matrix.shape))
    dlam, dv = _implicit_eigenpair_tangents(data, dA, int(p.size))
    jac_implicit = _implicit_observable_jacobian(
        observable_fn,
        p,
        data,
        eigenvalue_sensitivity=dlam,
        eigenvector_sensitivity=dv,
    )
    jac_fd = central_finite_difference_jacobian(
        _nearest_branch_observable(matrix_fn, observable_fn, data.eigenvalue),
        p,
        step=step,
    )
    max_abs, max_rel = _jacobian_error_metrics(jac_implicit, jac_fd, atol=atol)
    passed = bool(
        data.selection.branch_isolated
        and (max_abs <= float(atol) or max_rel <= float(rtol))
    )
    return _ImplicitSensitivityData(
        eigenvalue_sensitivity=dlam,
        eigenvector_sensitivity=dv,
        jacobian_implicit=jac_implicit,
        jacobian_fd=jac_fd,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        passed=passed,
    )


def _pack_implicit_eigenpair_report(
    *,
    data: _ImplicitEigenpairData,
    sensitivities: _ImplicitSensitivityData,
    step: float,
    rtol: float,
    atol: float,
    gap_floor: float,
) -> dict[str, object]:
    dlam_np = np.asarray(sensitivities.eigenvalue_sensitivity, dtype=complex)
    return {
        "passed": sensitivities.passed,
        "ad_supported": True,
        "sensitivity_method": "implicit_left_right_eigenpair",
        "observable_chain_rule": "split_eigenpair_and_explicit_parameter",
        "step": float(step),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": sensitivities.max_abs_error,
        "max_rel_error": sensitivities.max_rel_error,
        "jacobian_implicit": np.asarray(
            sensitivities.jacobian_implicit, dtype=float
        ).tolist(),
        "jacobian_fd": np.asarray(sensitivities.jacobian_fd, dtype=float).tolist(),
        "eigenvalue_sensitivity_real": np.real(dlam_np).tolist(),
        "eigenvalue_sensitivity_imag": np.imag(dlam_np).tolist(),
        "selector": data.selection.selector_key,
        "selected_index": data.selection.index,
        "left_selected_index": data.left_index,
        "eigenvalue_real": float(np.real(np.asarray(data.eigenvalue))),
        "eigenvalue_imag": float(np.imag(np.asarray(data.eigenvalue))),
        "eigenvalue_gap": data.selection.gap,
        "gap_floor": float(gap_floor),
        "branch_isolated": data.selection.branch_isolated,
        "biorthogonal_overlap_abs": data.overlap_abs,
    }


def implicit_eigenpair_observable_sensitivity_report(
    matrix_fn: Callable[[jnp.ndarray], Any],
    observable_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    selector: str = "max_real",
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    gap_floor: float = 1.0e-8,
) -> dict[str, object]:
    """Validate implicit sensitivities of an isolated non-Hermitian eigenpair.

    JAX currently supports first derivatives of non-Hermitian eigenvalues but
    not eigenvectors. This helper avoids differentiating through
    ``jnp.linalg.eig``. It differentiates the matrix entries with JAX, solves
    the left/right eigenvector perturbation equation for each parameter, and
    compares the resulting observable Jacobian against nearest-branch central
    finite differences.

    The observable should be phase-invariant under ``v -> exp(i alpha) v``.
    The implicit solve fixes the gauge with ``w^H dv = 0`` using the left
    eigenvector ``w`` normalized by ``w^H v = 1``.
    """

    p = _params_vector(params)
    data = _implicit_eigenpair_data(
        matrix_fn,
        p,
        selector=selector,
        gap_floor=gap_floor,
    )
    sensitivities = _implicit_sensitivity_data(
        matrix_fn,
        observable_fn,
        p,
        data,
        step=step,
        rtol=rtol,
        atol=atol,
    )
    return _pack_implicit_eigenpair_report(
        data=data,
        sensitivities=sensitivities,
        step=step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )


__all__ = [
    "autodiff_finite_difference_report",
    "central_finite_difference_jacobian",
    "covariance_diagnostics",
    "explicit_complex_operator_matrix",
    "implicit_eigenpair_observable_sensitivity_report",
    "isolated_eigenpair_observable_sensitivity_report",
    "isolated_eigenvalue_sensitivity_report",
]
