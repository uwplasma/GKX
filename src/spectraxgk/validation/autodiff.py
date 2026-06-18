"""Autodiff validation helpers for inverse and UQ examples."""

from __future__ import annotations

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

    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    eig_base = jnp.linalg.eigvals(jnp.asarray(matrix_fn(p)))
    eig_np = np.asarray(eig_base)
    if eig_np.ndim != 1 or eig_np.size == 0:
        raise ValueError(
            "matrix_fn must return a square matrix with at least one eigenvalue"
        )
    selector_key = selector.strip().lower()
    if selector_key == "max_real":
        index = int(np.argmax(np.real(eig_np)))
    elif selector_key.startswith("index:"):
        index = int(selector_key.split(":", 1)[1])
        if index < 0 or index >= eig_np.size:
            raise ValueError(
                f"selector index {index} is out of bounds for {eig_np.size} eigenvalues"
            )
    else:
        raise ValueError("selector must be 'max_real' or 'index:N'")

    selected = eig_np[index]
    if eig_np.size == 1:
        gap = float("inf")
    else:
        others = np.delete(eig_np, index)
        gap = float(np.min(np.abs(selected - others)))

    def branch_fn(x: jnp.ndarray) -> jnp.ndarray:
        value = jnp.linalg.eigvals(jnp.asarray(matrix_fn(x)))[index]
        return jnp.asarray([jnp.real(value), jnp.imag(value)])

    branch_isolated = bool(gap >= float(gap_floor))
    try:
        report = autodiff_finite_difference_report(
            branch_fn,
            p,
            step=step,
            rtol=rtol,
            atol=atol,
        )
    except NotImplementedError as exc:
        return {
            "passed": False,
            "ad_supported": False,
            "failure_reason": str(exc),
            "step": float(step),
            "rtol": float(rtol),
            "atol": float(atol),
            "selector": selector_key,
            "selected_index": index,
            "eigenvalue_real": float(np.real(selected)),
            "eigenvalue_imag": float(np.imag(selected)),
            "eigenvalue_gap": gap,
            "gap_floor": float(gap_floor),
            "branch_isolated": branch_isolated,
        }
    return {
        **report,
        "passed": bool(report["passed"]) and branch_isolated,
        "ad_supported": True,
        "selector": selector_key,
        "selected_index": index,
        "eigenvalue_real": float(np.real(selected)),
        "eigenvalue_imag": float(np.imag(selected)),
        "eigenvalue_gap": gap,
        "gap_floor": float(gap_floor),
        "branch_isolated": branch_isolated,
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

    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    eig_base, vec_base = jnp.linalg.eig(jnp.asarray(matrix_fn(p)))
    eig_np = np.asarray(eig_base)
    if eig_np.ndim != 1 or eig_np.size == 0:
        raise ValueError(
            "matrix_fn must return a square matrix with at least one eigenvalue"
        )
    if np.asarray(vec_base).shape[1] != eig_np.size:
        raise ValueError("eigenvector matrix shape is inconsistent with eigenvalues")
    selector_key = selector.strip().lower()
    if selector_key == "max_real":
        index = int(np.argmax(np.real(eig_np)))
    elif selector_key.startswith("index:"):
        index = int(selector_key.split(":", 1)[1])
        if index < 0 or index >= eig_np.size:
            raise ValueError(
                f"selector index {index} is out of bounds for {eig_np.size} eigenvalues"
            )
    else:
        raise ValueError("selector must be 'max_real' or 'index:N'")

    selected = eig_np[index]
    if eig_np.size == 1:
        gap = float("inf")
    else:
        others = np.delete(eig_np, index)
        gap = float(np.min(np.abs(selected - others)))

    def branch_fn(x: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = jnp.linalg.eig(jnp.asarray(matrix_fn(x)))
        obs = jnp.ravel(
            jnp.asarray(observable_fn(eigvals[index], eigvecs[:, index], x))
        )
        if jnp.iscomplexobj(obs):
            obs = jnp.concatenate([jnp.real(obs), jnp.imag(obs)])
        return obs

    branch_isolated = bool(gap >= float(gap_floor))
    try:
        report = autodiff_finite_difference_report(
            branch_fn,
            p,
            step=step,
            rtol=rtol,
            atol=atol,
        )
    except NotImplementedError as exc:
        return {
            "passed": False,
            "ad_supported": False,
            "failure_reason": str(exc),
            "step": float(step),
            "rtol": float(rtol),
            "atol": float(atol),
            "selector": selector_key,
            "selected_index": index,
            "eigenvalue_real": float(np.real(selected)),
            "eigenvalue_imag": float(np.imag(selected)),
            "eigenvalue_gap": gap,
            "gap_floor": float(gap_floor),
            "branch_isolated": branch_isolated,
        }
    return {
        **report,
        "passed": bool(report["passed"]) and branch_isolated,
        "ad_supported": True,
        "selector": selector_key,
        "selected_index": index,
        "eigenvalue_real": float(np.real(selected)),
        "eigenvalue_imag": float(np.imag(selected)),
        "eigenvalue_gap": gap,
        "gap_floor": float(gap_floor),
        "branch_isolated": branch_isolated,
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

    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    A = jnp.asarray(matrix_fn(p))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix_fn must return a square matrix")
    eigvals, eigvecs = jnp.linalg.eig(A)
    eig_np = np.asarray(eigvals)
    if eig_np.size == 0:
        raise ValueError("matrix_fn must return at least one eigenvalue")

    selector_key = selector.strip().lower()
    if selector_key == "max_real":
        index = int(np.argmax(np.real(eig_np)))
    elif selector_key.startswith("index:"):
        index = int(selector_key.split(":", 1)[1])
        if index < 0 or index >= eig_np.size:
            raise ValueError(
                f"selector index {index} is out of bounds for {eig_np.size} eigenvalues"
            )
    else:
        raise ValueError("selector must be 'max_real' or 'index:N'")

    lam = eigvals[index]
    v = eigvecs[:, index]
    if eig_np.size == 1:
        gap = float("inf")
    else:
        gap = float(np.min(np.abs(np.delete(eig_np, index) - np.asarray(lam))))
    branch_isolated = bool(gap >= float(gap_floor))

    left_vals, left_vecs = jnp.linalg.eig(jnp.conj(jnp.swapaxes(A, 0, 1)))
    left_index = int(
        np.argmin(np.abs(np.asarray(left_vals) - np.conj(np.asarray(lam))))
    )
    w = left_vecs[:, left_index]
    overlap = jnp.vdot(w, v)
    overlap_abs = float(np.abs(np.asarray(overlap)))
    if overlap_abs <= 0.0 or not np.isfinite(overlap_abs):
        raise ValueError("left/right eigenvectors are biorthogonally singular")
    w = w / jnp.conj(overlap)

    def flat_matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(jnp.asarray(matrix_fn(x)))

    dA_flat = jax.jacfwd(flat_matrix_fn)(p)
    dA = jnp.reshape(dA_flat, A.shape + (p.size,))
    n = int(A.shape[0])
    identity = jnp.eye(n, dtype=A.dtype)
    top = jnp.concatenate([A - lam * identity, -v[:, None]], axis=1)
    bottom = jnp.concatenate(
        [jnp.conj(w)[None, :], jnp.zeros((1, 1), dtype=A.dtype)],
        axis=1,
    )
    augmented = jnp.concatenate([top, bottom], axis=0)
    rhs_columns = []
    for i in range(int(p.size)):
        rhs_columns.append(
            jnp.concatenate([-dA[:, :, i] @ v, jnp.zeros((1,), dtype=A.dtype)])
        )
    rhs = jnp.stack(rhs_columns, axis=1)
    solution = jnp.linalg.solve(augmented, rhs)
    dv = solution[:n, :]
    dlam = solution[n, :]

    def observable_real(
        lam_i: jnp.ndarray, v_i: jnp.ndarray, p_i: jnp.ndarray
    ) -> jnp.ndarray:
        obs = jnp.ravel(jnp.asarray(observable_fn(lam_i, v_i, p_i)))
        if jnp.iscomplexobj(obs):
            return jnp.concatenate([jnp.real(obs), jnp.imag(obs)])
        return jnp.real(obs)

    def observable_real_from_eigenpair(packed: jnp.ndarray) -> jnp.ndarray:
        lam_i = packed[0] + 1j * packed[1]
        v_real_start = 2
        v_imag_start = v_real_start + n
        v_i = packed[v_real_start:v_imag_start] + 1j * packed[v_imag_start:]
        return observable_real(lam_i, v_i, p)

    def observable_real_from_params(p_i: jnp.ndarray) -> jnp.ndarray:
        return observable_real(lam, v, p_i)

    # Split the chain rule so expensive parameter-dependent context, e.g.
    # VMEC/Boozer geometry reconstruction, is only differentiated along the
    # actual parameter directions. Differentiating one packed vector
    # [lambda, v, p] is mathematically equivalent but can replicate heavy
    # geometry tangents for every eigenvector component.
    eigenpair_base = jnp.concatenate(
        [jnp.asarray([jnp.real(lam), jnp.imag(lam)]), jnp.real(v), jnp.imag(v)]
    )
    obs_jac_eigenpair = jax.jacfwd(observable_real_from_eigenpair)(eigenpair_base)
    obs_jac_params = jax.jacfwd(observable_real_from_params)(p)
    implicit_cols = []
    eye = jnp.eye(p.size, dtype=p.dtype)
    for i in range(int(p.size)):
        eigenpair_tangent = jnp.concatenate(
            [
                jnp.asarray([jnp.real(dlam[i]), jnp.imag(dlam[i])]),
                jnp.real(dv[:, i]),
                jnp.imag(dv[:, i]),
            ]
        )
        implicit_cols.append(
            obs_jac_eigenpair @ eigenpair_tangent + obs_jac_params @ eye[i]
        )
    jac_implicit = jnp.stack(implicit_cols, axis=1)

    def branch_observable(x: jnp.ndarray) -> jnp.ndarray:
        eigvals_i, eigvecs_i = jnp.linalg.eig(jnp.asarray(matrix_fn(x)))
        branch_index = int(np.argmin(np.abs(np.asarray(eigvals_i) - np.asarray(lam))))
        obs = jnp.ravel(
            jnp.asarray(
                observable_fn(eigvals_i[branch_index], eigvecs_i[:, branch_index], x)
            )
        )
        if jnp.iscomplexobj(obs):
            return jnp.concatenate([jnp.real(obs), jnp.imag(obs)])
        return jnp.real(obs)

    jac_fd = central_finite_difference_jacobian(branch_observable, p, step=step)
    err = np.asarray(jac_implicit - jac_fd, dtype=float)
    denom = np.maximum(np.asarray(np.abs(jac_fd), dtype=float), float(atol))
    rel = np.abs(err) / denom
    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    passed = bool(
        branch_isolated and (max_abs <= float(atol) or max_rel <= float(rtol))
    )
    dlam_np = np.asarray(dlam, dtype=complex)
    return {
        "passed": passed,
        "ad_supported": True,
        "sensitivity_method": "implicit_left_right_eigenpair",
        "observable_chain_rule": "split_eigenpair_and_explicit_parameter",
        "step": float(step),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "jacobian_implicit": np.asarray(jac_implicit, dtype=float).tolist(),
        "jacobian_fd": np.asarray(jac_fd, dtype=float).tolist(),
        "eigenvalue_sensitivity_real": np.real(dlam_np).tolist(),
        "eigenvalue_sensitivity_imag": np.imag(dlam_np).tolist(),
        "selector": selector_key,
        "selected_index": index,
        "left_selected_index": left_index,
        "eigenvalue_real": float(np.real(np.asarray(lam))),
        "eigenvalue_imag": float(np.imag(np.asarray(lam))),
        "eigenvalue_gap": gap,
        "gap_floor": float(gap_floor),
        "branch_isolated": branch_isolated,
        "biorthogonal_overlap_abs": overlap_abs,
    }


__all__ = [
    "autodiff_finite_difference_report",
    "central_finite_difference_jacobian",
    "covariance_diagnostics",
    "explicit_complex_operator_matrix",
    "implicit_eigenpair_observable_sensitivity_report",
    "isolated_eigenpair_observable_sensitivity_report",
    "isolated_eigenvalue_sensitivity_report",
]
