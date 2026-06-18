"""Finite-difference and covariance checks for differentiable objectives."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.parallel import independent_map


def _jax_enable_x64() -> bool:
    """Return the active JAX 64-bit precision flag without relying on dynamic attrs."""

    return bool(jax.config.read("jax_enable_x64"))


def _normalize_fd_executor(executor: str) -> str:
    """Normalize finite-difference worker executor names."""

    key = str(executor).strip().lower()
    if key in {"thread", "threads"}:
        return "thread"
    if key in {"process", "processes"}:
        return "process"
    raise ValueError("parallel_executor must be 'thread' or 'process'")


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
    if jac.shape[1] == 0:
        raise ValueError("jacobian must contain at least one parameter column")
    if jac.shape[0] != res.size:
        raise ValueError("residual length must match the number of Jacobian rows")
    if not np.all(np.isfinite(jac)):
        raise ValueError("jacobian must contain only finite values")
    if not np.all(np.isfinite(res)):
        raise ValueError("residual must contain only finite values")
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
    correlation = np.divide(
        covariance, denom, out=np.zeros_like(covariance), where=denom > 0.0
    )

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


def central_finite_difference_jacobian(
    fn: Callable[[jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    step: float = 1.0e-4,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> jnp.ndarray:
    """Central finite-difference Jacobian for small differentiability gates."""

    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    executor_key = _normalize_fd_executor(parallel_executor)
    if executor_key == "process" and n_workers > 1:
        raise ValueError(
            "parallel finite differences require the thread executor because objective closures are not pickled"
        )
    eye = jnp.eye(p.size, dtype=p.dtype)

    def column(i: int) -> jnp.ndarray:
        fp = jnp.ravel(jnp.asarray(fn(p + h * eye[i])))
        fm = jnp.ravel(jnp.asarray(fn(p - h * eye[i])))
        return (fp - fm) / (2.0 * h)

    cols = independent_map(
        column, range(int(p.size)), workers=n_workers, executor=executor_key
    )
    if not cols:
        return jnp.zeros((jnp.ravel(jnp.asarray(fn(p))).size, 0), dtype=p.dtype)
    return jnp.stack(cols, axis=1)


def autodiff_finite_difference_report(
    fn: Callable[[jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    direction: jnp.ndarray | np.ndarray | None = None,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> dict[str, object]:
    """Compare JAX forward-mode derivatives against finite differences."""

    p = jnp.asarray(params, dtype=jnp.float64 if _jax_enable_x64() else jnp.float32)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    executor_key = _normalize_fd_executor(parallel_executor)

    def flat_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(jnp.asarray(fn(x)))

    jac_ad = jax.jacfwd(flat_fn)(p)
    jac_fd = central_finite_difference_jacobian(
        flat_fn,
        p,
        step=step,
        workers=n_workers,
        parallel_executor=executor_key,
    )
    err = np.asarray(jac_ad - jac_fd, dtype=float)
    denom = np.maximum(np.asarray(np.abs(jac_fd), dtype=float), float(atol))
    rel = np.abs(err) / denom

    if direction is None:
        d = jnp.ones_like(p)
        d = d / jnp.maximum(jnp.linalg.norm(d), jnp.asarray(1.0, dtype=d.dtype))
    else:
        d = jnp.asarray(direction, dtype=p.dtype)
        if d.shape != p.shape:
            raise ValueError("direction must have the same shape as params")
    tangent_ad = jac_ad @ d
    tangent_fd = (flat_fn(p + step * d) - flat_fn(p - step * d)) / (2.0 * step)
    tangent_err = np.asarray(tangent_ad - tangent_fd, dtype=float)

    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    tangent_max_abs = float(np.max(np.abs(tangent_err))) if tangent_err.size else 0.0
    passed = bool(max_abs <= float(atol) or max_rel <= float(rtol))
    return {
        "passed": passed,
        "step": float(step),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "tangent_max_abs_error": tangent_max_abs,
        "jacobian_ad": np.asarray(jac_ad, dtype=float).tolist(),
        "jacobian_fd": np.asarray(jac_fd, dtype=float).tolist(),
        "tangent_ad": np.asarray(tangent_ad, dtype=float).tolist(),
        "tangent_fd": np.asarray(tangent_fd, dtype=float).tolist(),
        "finite_difference_parallel": {
            "requested_workers": n_workers,
            "effective_workers": int(min(n_workers, max(int(p.size), 1))),
            "executor": executor_key,
            "identity_contract": "parallel finite-difference columns must match serial columns",
        },
    }


__all__ = [
    "_jax_enable_x64",
    "_normalize_fd_executor",
    "autodiff_finite_difference_report",
    "central_finite_difference_jacobian",
    "covariance_diagnostics",
]
