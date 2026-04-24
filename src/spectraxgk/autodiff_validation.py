"""Autodiff validation helpers for inverse and UQ examples."""

from __future__ import annotations

import numpy as np


def covariance_diagnostics(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    regularization: float = 1.0e-9,
) -> dict[str, object]:
    """Return covariance and conditioning diagnostics for a least-squares inverse.

    The covariance uses the local Gauss-Newton approximation
    ``sigma^2 (J^T J + lambda I)^-1``. The returned dictionary is strict-JSON
    friendly and records enough metadata to decide whether an inverse demo is
    identifiable, merely locally sensitive, or ill-conditioned.
    """

    jac = np.asarray(jacobian, dtype=float)
    res = np.asarray(residual, dtype=float).reshape(-1)
    if jac.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if jac.shape[0] != res.size:
        raise ValueError("residual length must match the number of Jacobian rows")
    reg = float(regularization)
    if reg < 0.0:
        raise ValueError("regularization must be non-negative")

    sigma2 = float(np.mean(res**2) + 1.0e-12)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    if singular_values.size == 0 or float(singular_values[-1]) <= 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    rank = int(np.linalg.matrix_rank(jac))

    normal = jac.T @ jac + reg * np.eye(jac.shape[1])
    covariance = sigma2 * np.linalg.inv(normal)
    covariance = 0.5 * (covariance + covariance.T)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denom = np.outer(std, std)
    correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0.0)

    eigvals = np.linalg.eigvalsh(covariance)
    positive = eigvals[eigvals > 0.0]
    if positive.size >= 2:
        ellipse_area = float(np.pi * np.sqrt(positive[-1] * positive[-2]))
    elif positive.size == 1:
        ellipse_area = 0.0
    else:
        ellipse_area = 0.0

    return {
        "sigma2": sigma2,
        "covariance": covariance.tolist(),
        "covariance_std": std.tolist(),
        "covariance_correlation": correlation.tolist(),
        "covariance_eigenvalues": eigvals.tolist(),
        "uq_ellipse_area_1sigma": ellipse_area,
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition_number": condition_number,
        "sensitivity_map_rank": rank,
    }


__all__ = ["covariance_diagnostics"]
