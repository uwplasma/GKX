"""Autodiff and finite-difference validation helpers for geometry workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.backend_discovery import _jax_float_dtype


def finite_difference_jacobian(
    fn: Any, params: jnp.ndarray, *, step: float = 1.0e-4
) -> jnp.ndarray:
    """Central finite-difference Jacobian for small validation problems."""

    p = jnp.asarray(params, dtype=_jax_float_dtype())
    h = float(step)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    if h <= 0.0:
        raise ValueError("step must be positive")
    columns = []
    for idx in range(int(p.shape[0])):
        basis = jnp.zeros_like(p).at[idx].set(h)
        columns.append(
            (jnp.asarray(fn(p + basis)) - jnp.asarray(fn(p - basis))) / (2.0 * h)
        )
    return jnp.stack(columns, axis=1)


def _json_ready(value: Any) -> Any:
    """Return a strict JSON-compatible copy, replacing nonfinite floats by null."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def _sensitivity_conditioning_metadata(
    jacobian_ad: Any,
    jacobian_fd: Any,
    params: Any,
    *,
    fd_step: float,
    observable_names: Sequence[str] | None = None,
    param_names: Sequence[str] | None = None,
    relative_floor: float = 1.0e-12,
) -> dict[str, object]:
    """Return JSON-friendly conditioning metadata for AD/FD Jacobian gates."""

    jac_ad = np.asarray(jacobian_ad, dtype=float)
    jac_fd = np.asarray(jacobian_fd, dtype=float)
    p = np.asarray(params, dtype=float).reshape(-1)
    if jac_ad.ndim != 2 or jac_fd.ndim != 2:
        raise ValueError("jacobians must be two-dimensional")
    if jac_ad.shape != jac_fd.shape:
        raise ValueError("AD and finite-difference jacobians must have matching shapes")
    if jac_ad.shape[1] != p.size:
        raise ValueError("parameter length must match jacobian columns")

    obs_names = (
        [str(name) for name in observable_names]
        if observable_names is not None
        else [f"observable_{idx}" for idx in range(jac_ad.shape[0])]
    )
    par_names = (
        [str(name) for name in param_names]
        if param_names is not None
        else [f"param_{idx}" for idx in range(jac_ad.shape[1])]
    )
    if len(obs_names) != jac_ad.shape[0]:
        raise ValueError("observable_names length must match jacobian rows")
    if len(par_names) != jac_ad.shape[1]:
        raise ValueError("param_names length must match jacobian columns")

    finite_ad = bool(np.all(np.isfinite(jac_ad)))
    finite_fd = bool(np.all(np.isfinite(jac_fd)))
    finite_params = bool(np.all(np.isfinite(p)))
    if finite_ad and jac_ad.size:
        singular_values = np.linalg.svd(jac_ad, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jac_ad))
        if singular_values.size == 0 or float(singular_values[-1]) <= 0.0:
            condition_number = float("inf")
        else:
            condition_number = float(singular_values[0] / singular_values[-1])
        column_norms = np.linalg.norm(jac_ad, axis=0)
        row_norms = np.linalg.norm(jac_ad, axis=1)
    else:
        singular_values = np.asarray([], dtype=float)
        rank = 0
        condition_number = float("inf")
        column_norms = np.full((jac_ad.shape[1],), np.nan)
        row_norms = np.full((jac_ad.shape[0],), np.nan)

    floor = float(relative_floor)
    diff = jac_ad - jac_fd
    abs_error = np.abs(diff)
    rel_error = abs_error / np.maximum(np.abs(jac_fd), floor)

    def _worst_entry(values: np.ndarray) -> dict[str, object] | None:
        if values.size == 0 or not np.any(np.isfinite(values)):
            return None
        flat_idx = int(np.nanargmax(values))
        row, col = np.unravel_index(flat_idx, values.shape)
        return {
            "observable_index": int(row),
            "observable_name": obs_names[int(row)],
            "parameter_index": int(col),
            "parameter_name": par_names[int(col)],
            "value": float(values[row, col]),
            "ad": float(jac_ad[row, col]),
            "finite_difference": float(jac_fd[row, col]),
        }

    h = abs(float(fd_step))
    fd_step_by_parameter = [
        {
            "parameter_index": int(idx),
            "parameter_name": par_names[int(idx)],
            "parameter_value": float(p[idx]) if idx < p.size else float("nan"),
            "absolute_step": h,
            "relative_step": float(h / max(abs(float(p[idx])), 1.0)),
        }
        for idx in range(p.size)
    ]

    finite_column_norms = column_norms[np.isfinite(column_norms)]
    max_column_norm = (
        float(np.max(finite_column_norms)) if finite_column_norms.size else float("nan")
    )
    min_column_norm = (
        float(np.min(finite_column_norms)) if finite_column_norms.size else float("nan")
    )

    return {
        "jacobian_shape": [int(jac_ad.shape[0]), int(jac_ad.shape[1])],
        "finite_ad_jacobian": finite_ad,
        "finite_fd_jacobian": finite_fd,
        "finite_parameters": finite_params,
        "sensitivity_map_rank": rank,
        "jacobian_condition_number": condition_number,
        "jacobian_singular_values": singular_values.tolist(),
        "ad_column_norms": column_norms.tolist(),
        "ad_row_norms": row_norms.tolist(),
        "max_ad_column_norm": max_column_norm,
        "min_ad_column_norm": min_column_norm,
        "fd_step": float(fd_step),
        "relative_error_floor": floor,
        "finite_difference_step_by_parameter": fd_step_by_parameter,
        "fd_near_zero_reference_count": int(np.sum(np.abs(jac_fd) < floor)),
        "worst_abs_error": _worst_entry(abs_error),
        "worst_rel_error": _worst_entry(rel_error),
    }


def _flat_observable_function(
    observable_fn: Callable[[jnp.ndarray], Any],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def flat_fn(x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.ravel(jnp.asarray(observable_fn(x)))
        if jnp.iscomplexobj(out):
            return jnp.concatenate([jnp.real(out), jnp.imag(out)])
        return out

    return flat_fn


def _validated_gradient_inputs(
    params: jnp.ndarray | np.ndarray,
    *,
    fd_step: float,
    rtol: float,
    atol: float,
    relative_floor: float,
    condition_number_max: float | None,
) -> tuple[jnp.ndarray, float, float, float, float, float | None]:
    p = jnp.asarray(params, dtype=_jax_float_dtype())
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    if int(p.size) == 0:
        raise ValueError("params must contain at least one parameter")
    step = float(fd_step)
    if step <= 0.0:
        raise ValueError("fd_step must be positive")
    rel_tol = float(rtol)
    abs_tol = float(atol)
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    floor = float(relative_floor)
    if floor <= 0.0:
        raise ValueError("relative_floor must be positive")
    cond_max = None if condition_number_max is None else float(condition_number_max)
    if cond_max is not None and cond_max <= 0.0:
        raise ValueError("condition_number_max must be positive or None")
    return p, step, rel_tol, abs_tol, floor, cond_max


def _resolve_report_names(
    n_obs: int,
    n_params: int,
    *,
    observable_names: Sequence[str] | None,
    param_names: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    obs_names = (
        [str(name) for name in observable_names]
        if observable_names is not None
        else [f"observable_{idx}" for idx in range(n_obs)]
    )
    par_names = (
        [str(name) for name in param_names]
        if param_names is not None
        else [f"param_{idx}" for idx in range(n_params)]
    )
    if len(obs_names) != n_obs:
        raise ValueError("observable_names length must match jacobian rows")
    if len(par_names) != n_params:
        raise ValueError("param_names length must match jacobian columns")
    return obs_names, par_names


def _autodiff_and_fd_jacobians(
    flat_fn: Callable[[jnp.ndarray], jnp.ndarray],
    p: jnp.ndarray,
    *,
    step: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    observables = flat_fn(p)
    if int(observables.size) == 0:
        raise ValueError("observable_fn must return at least one observable")
    jac_ad = jax.jacfwd(flat_fn)(p)
    jac_fd = finite_difference_jacobian(flat_fn, p, step=step)
    if jac_ad.ndim != 2 or jac_fd.ndim != 2 or jac_ad.shape != jac_fd.shape:
        raise ValueError(
            "AD and finite-difference jacobians must be two-dimensional and "
            "shape-aligned"
        )
    return observables, jac_ad, jac_fd


def _tangent_direction(
    p: jnp.ndarray,
    tangent: jnp.ndarray | np.ndarray | None,
) -> jnp.ndarray:
    if tangent is None:
        direction = jnp.ones_like(p)
        return direction / jnp.maximum(
            jnp.linalg.norm(direction), jnp.asarray(1.0, dtype=p.dtype)
        )
    direction = jnp.asarray(tangent, dtype=p.dtype)
    if direction.shape != p.shape:
        raise ValueError("tangent must have the same shape as params")
    return direction


def _jacobian_error_arrays(
    jac_ad: jnp.ndarray,
    jac_fd: jnp.ndarray,
    *,
    floor: float,
    abs_tol: float,
) -> dict[str, np.ndarray]:
    ad_np = np.asarray(jac_ad, dtype=float)
    fd_np = np.asarray(jac_fd, dtype=float)
    abs_error = np.abs(ad_np - fd_np)
    rel_error = abs_error / np.maximum(np.abs(fd_np), floor)
    # Summary relative error ignores entries that already pass the absolute
    # gate; raw per-entry relative errors remain in the report.
    rel_error_gate = np.where(abs_error <= abs_tol, 0.0, rel_error)
    return {
        "ad_np": ad_np,
        "fd_np": fd_np,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "rel_error_gate": rel_error_gate,
    }


def _tangent_error_arrays(
    flat_fn: Callable[[jnp.ndarray], jnp.ndarray],
    p: jnp.ndarray,
    jac_ad: jnp.ndarray,
    direction: jnp.ndarray,
    *,
    step: float,
    floor: float,
) -> dict[str, np.ndarray | float]:
    tangent_ad = jac_ad @ direction
    tangent_fd = (flat_fn(p + step * direction) - flat_fn(p - step * direction)) / (
        2.0 * step
    )
    tangent_ad_np = np.asarray(tangent_ad, dtype=float)
    tangent_fd_np = np.asarray(tangent_fd, dtype=float)
    tangent_abs_error = np.abs(tangent_ad_np - tangent_fd_np)
    tangent_rel_error = tangent_abs_error / np.maximum(np.abs(tangent_fd_np), floor)
    return {
        "tangent_ad_np": tangent_ad_np,
        "tangent_fd_np": tangent_fd_np,
        "tangent_abs_error": tangent_abs_error,
        "tangent_rel_error": tangent_rel_error,
        "tangent_max_abs": (
            float(np.nanmax(tangent_abs_error)) if tangent_abs_error.size else 0.0
        ),
        "tangent_max_rel": (
            float(np.nanmax(tangent_rel_error)) if tangent_rel_error.size else 0.0
        ),
    }


def _finite_report_flags(
    p: jnp.ndarray,
    observables: jnp.ndarray,
    jacobian_errors: Mapping[str, np.ndarray],
    direction: jnp.ndarray,
    tangent_errors: Mapping[str, np.ndarray | float],
) -> dict[str, bool]:
    return {
        "params": bool(np.all(np.isfinite(np.asarray(p, dtype=float)))),
        "observables": bool(np.all(np.isfinite(np.asarray(observables, dtype=float)))),
        "autodiff_jacobian": bool(np.all(np.isfinite(jacobian_errors["ad_np"]))),
        "finite_difference_jacobian": bool(
            np.all(np.isfinite(jacobian_errors["fd_np"]))
        ),
        "abs_error": bool(np.all(np.isfinite(jacobian_errors["abs_error"]))),
        "rel_error": bool(np.all(np.isfinite(jacobian_errors["rel_error"]))),
        "tangent_direction": bool(
            np.all(np.isfinite(np.asarray(direction, dtype=float)))
        ),
        "tangent_autodiff": bool(
            np.all(np.isfinite(cast(np.ndarray, tangent_errors["tangent_ad_np"])))
        ),
        "tangent_finite_difference": bool(
            np.all(np.isfinite(cast(np.ndarray, tangent_errors["tangent_fd_np"])))
        ),
    }


def _gradient_checks(
    obs_names: Sequence[str],
    par_names: Sequence[str],
    jacobian_errors: Mapping[str, np.ndarray],
    *,
    abs_tol: float,
    rel_tol: float,
) -> list[dict[str, object]]:
    ad_np = jacobian_errors["ad_np"]
    fd_np = jacobian_errors["fd_np"]
    abs_error = jacobian_errors["abs_error"]
    rel_error = jacobian_errors["rel_error"]
    checks: list[dict[str, object]] = []
    for i, observable_name in enumerate(obs_names):
        for j, parameter_name in enumerate(par_names):
            entry_abs = float(abs_error[i, j])
            entry_rel = float(rel_error[i, j])
            checks.append(
                {
                    "observable": observable_name,
                    "parameter": parameter_name,
                    "autodiff": float(ad_np[i, j]),
                    "finite_difference": float(fd_np[i, j]),
                    "abs_error": entry_abs,
                    "rel_error": entry_rel,
                    "atol": abs_tol,
                    "rtol": rel_tol,
                    "passed": bool(entry_abs <= abs_tol or entry_rel <= rel_tol),
                }
            )
    return checks


def _conditioning_gate(
    conditioning: Mapping[str, object],
    *,
    n_obs: int,
    n_params: int,
    min_rank: int | None,
    condition_number_max: float | None,
    finite_flags: Mapping[str, bool],
) -> tuple[dict[str, object], bool, bool, bool]:
    required_rank = min(n_obs, n_params) if min_rank is None else int(min_rank)
    if required_rank < 0:
        raise ValueError("min_rank must be non-negative")
    rank = int(cast(int | float, conditioning["sensitivity_map_rank"]))
    condition_number = float(cast(float, conditioning["jacobian_condition_number"]))
    condition_number_finite = bool(np.isfinite(condition_number))
    condition_number_passed = bool(
        condition_number_max is None
        or (condition_number_finite and condition_number <= float(condition_number_max))
    )
    rank_passed = bool(rank >= required_rank)
    conditioning_passed = bool(
        finite_flags["autodiff_jacobian"]
        and finite_flags["finite_difference_jacobian"]
        and rank_passed
        and condition_number_passed
    )
    return (
        {
            "passed": conditioning_passed,
            "required_rank": int(required_rank),
            "rank_passed": rank_passed,
            "condition_number_max": condition_number_max,
            "condition_number_finite": condition_number_finite,
            "condition_number_passed": condition_number_passed,
        },
        conditioning_passed,
        rank_passed,
        condition_number_passed,
    )


def _failure_reasons(
    finite_flags: Mapping[str, bool],
    *,
    finite_passed: bool,
    derivative_passed: bool,
    tangent_passed: bool,
    conditioning_passed: bool,
    rank_passed: bool,
    condition_number_passed: bool,
) -> list[str]:
    reasons: list[str] = []
    if not finite_passed:
        failed = [name for name, flag in finite_flags.items() if not flag]
        reasons.append("nonfinite:" + ",".join(failed))
    if not derivative_passed:
        reasons.append("ad_fd_tolerance")
    if not tangent_passed:
        reasons.append("tangent_ad_fd_tolerance")
    if not conditioning_passed:
        if not rank_passed:
            reasons.append("rank_below_required")
        if not condition_number_passed:
            reasons.append("ill_conditioned")
    return reasons


def _assemble_gradient_validation_report(
    *,
    report_kind: str,
    finite_passed: bool,
    derivative_passed: bool,
    tangent_passed: bool,
    conditioning_passed: bool,
    failure_reasons: Sequence[str],
    step: float,
    rel_tol: float,
    abs_tol: float,
    floor: float,
    obs_names: Sequence[str],
    par_names: Sequence[str],
    p: jnp.ndarray,
    observables: jnp.ndarray,
    jacobian_errors: Mapping[str, np.ndarray],
    gradient_checks: Sequence[Mapping[str, object]],
    finite_flags: Mapping[str, bool],
    direction: jnp.ndarray,
    tangent_errors: Mapping[str, np.ndarray | float],
    conditioning_gate: Mapping[str, object],
    conditioning: Mapping[str, object],
) -> dict[str, object]:
    ad_np = jacobian_errors["ad_np"]
    fd_np = jacobian_errors["fd_np"]
    abs_error = jacobian_errors["abs_error"]
    rel_error = jacobian_errors["rel_error"]
    rel_error_gate = jacobian_errors["rel_error_gate"]
    tangent_ad_np = cast(np.ndarray, tangent_errors["tangent_ad_np"])
    tangent_fd_np = cast(np.ndarray, tangent_errors["tangent_fd_np"])
    tangent_abs_error = cast(np.ndarray, tangent_errors["tangent_abs_error"])
    tangent_rel_error = cast(np.ndarray, tangent_errors["tangent_rel_error"])
    tangent_max_abs = float(tangent_errors["tangent_max_abs"])
    tangent_max_rel = float(tangent_errors["tangent_max_rel"])
    report = {
        "kind": str(report_kind),
        "passed": bool(
            finite_passed
            and derivative_passed
            and tangent_passed
            and conditioning_passed
        ),
        "finite_passed": finite_passed,
        "derivative_tolerance_passed": derivative_passed,
        "tangent_tolerance_passed": tangent_passed,
        "conditioning_passed": conditioning_passed,
        "failure_reasons": list(failure_reasons),
        "fd_step": step,
        "rtol": rel_tol,
        "atol": abs_tol,
        "relative_error_floor": floor,
        "observable_names": list(obs_names),
        "parameter_names": list(par_names),
        "params": np.asarray(p, dtype=float).tolist(),
        "observables": np.asarray(observables, dtype=float).tolist(),
        "jacobian_ad": ad_np.tolist(),
        "jacobian_fd": fd_np.tolist(),
        "abs_error": abs_error.tolist(),
        "rel_error": rel_error.tolist(),
        "max_abs_ad_fd_error": float(np.nanmax(abs_error)) if abs_error.size else 0.0,
        "max_rel_ad_fd_error": (
            float(np.nanmax(rel_error_gate)) if rel_error_gate.size else 0.0
        ),
        "max_rel_ad_fd_error_raw": (
            float(np.nanmax(rel_error)) if rel_error.size else 0.0
        ),
        "gradient_checks": list(gradient_checks),
        "finite_flags": dict(finite_flags),
        "tangent_direction": np.asarray(direction, dtype=float).tolist(),
        "tangent_direction_norm": float(
            np.linalg.norm(np.asarray(direction, dtype=float))
        ),
        "tangent_ad": tangent_ad_np.tolist(),
        "tangent_fd": tangent_fd_np.tolist(),
        "tangent_abs_error": tangent_abs_error.tolist(),
        "tangent_rel_error": tangent_rel_error.tolist(),
        "tangent_max_abs_error": tangent_max_abs,
        "tangent_max_rel_error": tangent_max_rel,
        "tangent_ad_norm": float(np.linalg.norm(tangent_ad_np)),
        "tangent_fd_norm": float(np.linalg.norm(tangent_fd_np)),
        "conditioning_gate": dict(conditioning_gate),
        "conditioning": dict(conditioning),
    }
    return cast(dict[str, object], _json_ready(report))


def observable_gradient_validation_report(
    observable_fn: Callable[[jnp.ndarray], Any],
    params: jnp.ndarray | np.ndarray,
    *,
    fd_step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    observable_names: Sequence[str] | None = None,
    param_names: Sequence[str] | None = None,
    tangent: jnp.ndarray | np.ndarray | None = None,
    relative_floor: float = 1.0e-12,
    min_rank: int | None = None,
    condition_number_max: float | None = 1.0e12,
    report_kind: str = "observable_gradient_validation",
) -> dict[str, object]:
    """Validate observable gradients by AD, finite differences, and conditioning.

    ``observable_fn(params)`` may return any array-like observable vector.  The
    returned report is strict JSON-compatible: nonfinite diagnostic numbers are
    represented as ``None`` while finite flags and failure reasons preserve why
    the gate failed.
    """
    p, step, rel_tol, abs_tol, floor, cond_max = _validated_gradient_inputs(
        params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        relative_floor=relative_floor,
        condition_number_max=condition_number_max,
    )
    flat_fn = _flat_observable_function(observable_fn)
    observables, jac_ad, jac_fd = _autodiff_and_fd_jacobians(
        flat_fn, p, step=step
    )
    n_obs, n_params = int(jac_ad.shape[0]), int(jac_ad.shape[1])
    obs_names, par_names = _resolve_report_names(
        n_obs,
        n_params,
        observable_names=observable_names,
        param_names=param_names,
    )
    direction = _tangent_direction(p, tangent)
    jacobian_errors = _jacobian_error_arrays(
        jac_ad, jac_fd, floor=floor, abs_tol=abs_tol
    )
    tangent_errors = _tangent_error_arrays(
        flat_fn, p, jac_ad, direction, step=step, floor=floor
    )
    finite_flags = _finite_report_flags(
        p, observables, jacobian_errors, direction, tangent_errors
    )
    finite_passed = bool(all(finite_flags.values()))
    gradient_checks = _gradient_checks(
        obs_names, par_names, jacobian_errors, abs_tol=abs_tol, rel_tol=rel_tol
    )
    derivative_passed = bool(
        gradient_checks and all(bool(row["passed"]) for row in gradient_checks)
    )
    tangent_max_abs = float(tangent_errors["tangent_max_abs"])
    tangent_max_rel = float(tangent_errors["tangent_max_rel"])
    tangent_passed = bool(tangent_max_abs <= abs_tol or tangent_max_rel <= rel_tol)

    conditioning = _sensitivity_conditioning_metadata(
        jac_ad,
        jac_fd,
        p,
        fd_step=step,
        observable_names=obs_names,
        param_names=par_names,
        relative_floor=floor,
    )
    (
        conditioning_gate,
        conditioning_passed,
        rank_passed,
        condition_number_passed,
    ) = _conditioning_gate(
        conditioning,
        n_obs=n_obs,
        n_params=n_params,
        min_rank=min_rank,
        condition_number_max=cond_max,
        finite_flags=finite_flags,
    )
    failure_reasons = _failure_reasons(
        finite_flags,
        finite_passed=finite_passed,
        derivative_passed=derivative_passed,
        tangent_passed=tangent_passed,
        conditioning_passed=conditioning_passed,
        rank_passed=rank_passed,
        condition_number_passed=condition_number_passed,
    )
    return _assemble_gradient_validation_report(
        report_kind=report_kind,
        finite_passed=finite_passed,
        derivative_passed=derivative_passed,
        tangent_passed=tangent_passed,
        conditioning_passed=conditioning_passed,
        failure_reasons=failure_reasons,
        step=step,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        floor=floor,
        obs_names=obs_names,
        par_names=par_names,
        p=p,
        observables=observables,
        jacobian_errors=jacobian_errors,
        gradient_checks=gradient_checks,
        finite_flags=finite_flags,
        direction=direction,
        tangent_errors=tangent_errors,
        conditioning_gate=conditioning_gate,
        conditioning=conditioning,
    )


__all__ = [
    "_json_ready",
    "_sensitivity_conditioning_metadata",
    "finite_difference_jacobian",
    "observable_gradient_validation_report",
]
