"""Autodiff and finite-difference validation helpers for geometry workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from solvax import chunked_jacfwd

from gkx.geometry.backend_discovery import _jax_float_dtype


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


@dataclass(frozen=True)
class _ConditioningArrays:
    jac_ad: np.ndarray
    jac_fd: np.ndarray
    params: np.ndarray
    obs_names: list[str]
    par_names: list[str]


@dataclass(frozen=True)
class _JacobianConditioningStats:
    finite_ad: bool
    finite_fd: bool
    finite_params: bool
    singular_values: np.ndarray
    rank: int
    condition_number: float
    column_norms: np.ndarray
    row_norms: np.ndarray


@dataclass(frozen=True)
class _GradientValidationInputs:
    p: jnp.ndarray
    step: float
    rel_tol: float
    abs_tol: float
    floor: float
    cond_max: float | None


@dataclass(frozen=True)
class _GradientDerivativeData:
    flat_fn: Callable[[jnp.ndarray], jnp.ndarray]
    observables: jnp.ndarray
    jac_ad: jnp.ndarray
    jac_fd: jnp.ndarray
    obs_names: list[str]
    par_names: list[str]
    direction: jnp.ndarray
    jacobian_mode: str


@dataclass(frozen=True)
class _GradientGateData:
    jacobian_errors: dict[str, np.ndarray]
    tangent_errors: dict[str, np.ndarray | float]
    finite_flags: dict[str, bool]
    finite_passed: bool
    gradient_checks: list[dict[str, object]]
    derivative_passed: bool
    tangent_passed: bool
    conditioning: dict[str, object]
    conditioning_gate: dict[str, object]
    conditioning_passed: bool
    rank_passed: bool
    condition_number_passed: bool
    failure_reasons: list[str]


def _conditioned_jacobian_arrays(
    jacobian_ad: Any,
    jacobian_fd: Any,
    params: Any,
    *,
    observable_names: Sequence[str] | None,
    param_names: Sequence[str] | None,
) -> _ConditioningArrays:
    jac_ad = np.asarray(jacobian_ad, dtype=float)
    jac_fd = np.asarray(jacobian_fd, dtype=float)
    p = np.asarray(params, dtype=float).reshape(-1)
    if jac_ad.ndim != 2 or jac_fd.ndim != 2:
        raise ValueError("jacobians must be two-dimensional")
    if jac_ad.shape != jac_fd.shape:
        raise ValueError("AD and finite-difference jacobians must have matching shapes")
    if jac_ad.shape[1] != p.size:
        raise ValueError("parameter length must match jacobian columns")

    obs_names, par_names = _resolve_report_names(
        int(jac_ad.shape[0]),
        int(jac_ad.shape[1]),
        observable_names=observable_names,
        param_names=param_names,
    )
    return _ConditioningArrays(
        jac_ad=jac_ad,
        jac_fd=jac_fd,
        params=p,
        obs_names=obs_names,
        par_names=par_names,
    )


def _jacobian_conditioning_stats(
    arrays: _ConditioningArrays,
) -> _JacobianConditioningStats:
    jac_ad = arrays.jac_ad
    finite_ad = bool(np.all(np.isfinite(jac_ad)))
    finite_fd = bool(np.all(np.isfinite(arrays.jac_fd)))
    finite_params = bool(np.all(np.isfinite(arrays.params)))
    if finite_ad and jac_ad.size:
        singular_values = np.linalg.svd(jac_ad, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jac_ad))
        condition_number = (
            float("inf")
            if singular_values.size == 0 or float(singular_values[-1]) <= 0.0
            else float(singular_values[0] / singular_values[-1])
        )
        column_norms = np.linalg.norm(jac_ad, axis=0)
        row_norms = np.linalg.norm(jac_ad, axis=1)
    else:
        singular_values = np.asarray([], dtype=float)
        rank = 0
        condition_number = float("inf")
        column_norms = np.full((jac_ad.shape[1],), np.nan)
        row_norms = np.full((jac_ad.shape[0],), np.nan)
    return _JacobianConditioningStats(
        finite_ad=finite_ad,
        finite_fd=finite_fd,
        finite_params=finite_params,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        column_norms=column_norms,
        row_norms=row_norms,
    )


def _worst_conditioning_entry(
    values: np.ndarray,
    arrays: _ConditioningArrays,
) -> dict[str, object] | None:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    flat_idx = int(np.nanargmax(values))
    row, col = np.unravel_index(flat_idx, values.shape)
    return {
        "observable_index": int(row),
        "observable_name": arrays.obs_names[int(row)],
        "parameter_index": int(col),
        "parameter_name": arrays.par_names[int(col)],
        "value": float(values[row, col]),
        "ad": float(arrays.jac_ad[row, col]),
        "finite_difference": float(arrays.jac_fd[row, col]),
    }


def _conditioning_fd_step_rows(
    arrays: _ConditioningArrays,
    *,
    fd_step: float,
) -> list[dict[str, object]]:
    h = abs(float(fd_step))
    return [
        {
            "parameter_index": int(idx),
            "parameter_name": arrays.par_names[int(idx)],
            "parameter_value": (
                float(arrays.params[idx]) if idx < arrays.params.size else float("nan")
            ),
            "absolute_step": h,
            "relative_step": float(h / max(abs(float(arrays.params[idx])), 1.0)),
        }
        for idx in range(arrays.params.size)
    ]


def _finite_norm_bounds(norms: np.ndarray) -> tuple[float, float]:
    finite_norms = norms[np.isfinite(norms)]
    if not finite_norms.size:
        return float("nan"), float("nan")
    return float(np.max(finite_norms)), float(np.min(finite_norms))


def _conditioning_error_arrays(
    arrays: _ConditioningArrays,
    *,
    relative_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    abs_error = np.abs(arrays.jac_ad - arrays.jac_fd)
    rel_error = abs_error / np.maximum(np.abs(arrays.jac_fd), float(relative_floor))
    return abs_error, rel_error


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

    floor = float(relative_floor)
    arrays = _conditioned_jacobian_arrays(
        jacobian_ad,
        jacobian_fd,
        params,
        observable_names=observable_names,
        param_names=param_names,
    )
    stats = _jacobian_conditioning_stats(arrays)
    abs_error, rel_error = _conditioning_error_arrays(arrays, relative_floor=floor)
    max_column_norm, min_column_norm = _finite_norm_bounds(stats.column_norms)

    return {
        "jacobian_shape": [int(arrays.jac_ad.shape[0]), int(arrays.jac_ad.shape[1])],
        "finite_ad_jacobian": stats.finite_ad,
        "finite_fd_jacobian": stats.finite_fd,
        "finite_parameters": stats.finite_params,
        "sensitivity_map_rank": stats.rank,
        "jacobian_condition_number": stats.condition_number,
        "jacobian_singular_values": stats.singular_values.tolist(),
        "ad_column_norms": stats.column_norms.tolist(),
        "ad_row_norms": stats.row_norms.tolist(),
        "max_ad_column_norm": max_column_norm,
        "min_ad_column_norm": min_column_norm,
        "fd_step": float(fd_step),
        "relative_error_floor": floor,
        "finite_difference_step_by_parameter": _conditioning_fd_step_rows(
            arrays,
            fd_step=fd_step,
        ),
        "fd_near_zero_reference_count": int(np.sum(np.abs(arrays.jac_fd) < floor)),
        "worst_abs_error": _worst_conditioning_entry(abs_error, arrays),
        "worst_rel_error": _worst_conditioning_entry(rel_error, arrays),
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
) -> _GradientValidationInputs:
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
    return _GradientValidationInputs(
        p=p,
        step=step,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        floor=floor,
        cond_max=cond_max,
    )


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
    jacobian_chunk_size: int | str | None,
    jacobian_mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    observables = flat_fn(p)
    if int(observables.size) == 0:
        raise ValueError("observable_fn must return at least one observable")
    requested = str(jacobian_mode).strip().lower()
    if requested not in {"auto", "forward", "reverse"}:
        raise ValueError("jacobian_mode must be 'auto', 'forward', or 'reverse'")
    resolved = requested
    if resolved == "auto":
        resolved = (
            "forward"
            if jacobian_chunk_size is not None or int(p.size) <= int(observables.size)
            else "reverse"
        )
    if resolved == "reverse" and jacobian_chunk_size is not None:
        raise ValueError("jacobian_chunk_size is only valid for forward mode")
    jac_ad = (
        chunked_jacfwd(flat_fn, chunk_size=jacobian_chunk_size)(p)
        if resolved == "forward"
        else jax.jacrev(flat_fn)(p)
    )
    jac_fd = finite_difference_jacobian(flat_fn, p, step=step)
    if jac_ad.ndim != 2 or jac_fd.ndim != 2 or jac_ad.shape != jac_fd.shape:
        raise ValueError(
            "AD and finite-difference jacobians must be two-dimensional and "
            "shape-aligned"
        )
    return observables, jac_ad, jac_fd, resolved


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


def _tangent_tolerance_passed(
    tangent_errors: Mapping[str, np.ndarray | float],
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    tangent_abs_error = cast(np.ndarray, tangent_errors["tangent_abs_error"])
    tangent_rel_error = cast(np.ndarray, tangent_errors["tangent_rel_error"])
    if not tangent_abs_error.size:
        return True
    return bool(np.all((tangent_abs_error <= abs_tol) | (tangent_rel_error <= rel_tol)))


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


def _gradient_derivative_data(
    observable_fn: Callable[[jnp.ndarray], Any],
    inputs: _GradientValidationInputs,
    *,
    observable_names: Sequence[str] | None,
    param_names: Sequence[str] | None,
    tangent: jnp.ndarray | np.ndarray | None,
    jacobian_chunk_size: int | str | None,
    jacobian_mode: str,
) -> _GradientDerivativeData:
    flat_fn = _flat_observable_function(observable_fn)
    observables, jac_ad, jac_fd, resolved_mode = _autodiff_and_fd_jacobians(
        flat_fn,
        inputs.p,
        step=inputs.step,
        jacobian_chunk_size=jacobian_chunk_size,
        jacobian_mode=jacobian_mode,
    )
    obs_names, par_names = _resolve_report_names(
        int(jac_ad.shape[0]),
        int(jac_ad.shape[1]),
        observable_names=observable_names,
        param_names=param_names,
    )
    return _GradientDerivativeData(
        flat_fn=flat_fn,
        observables=observables,
        jac_ad=jac_ad,
        jac_fd=jac_fd,
        obs_names=obs_names,
        par_names=par_names,
        direction=_tangent_direction(inputs.p, tangent),
        jacobian_mode=resolved_mode,
    )


def _gradient_gate_data(
    derivatives: _GradientDerivativeData,
    inputs: _GradientValidationInputs,
    *,
    min_rank: int | None,
) -> _GradientGateData:
    jacobian_errors = _jacobian_error_arrays(
        derivatives.jac_ad,
        derivatives.jac_fd,
        floor=inputs.floor,
        abs_tol=inputs.abs_tol,
    )
    tangent_errors = _tangent_error_arrays(
        derivatives.flat_fn,
        inputs.p,
        derivatives.jac_ad,
        derivatives.direction,
        step=inputs.step,
        floor=inputs.floor,
    )
    finite_flags = _finite_report_flags(
        inputs.p,
        derivatives.observables,
        jacobian_errors,
        derivatives.direction,
        tangent_errors,
    )
    gradient_checks = _gradient_checks(
        derivatives.obs_names,
        derivatives.par_names,
        jacobian_errors,
        abs_tol=inputs.abs_tol,
        rel_tol=inputs.rel_tol,
    )
    conditioning = _sensitivity_conditioning_metadata(
        derivatives.jac_ad,
        derivatives.jac_fd,
        inputs.p,
        fd_step=inputs.step,
        observable_names=derivatives.obs_names,
        param_names=derivatives.par_names,
        relative_floor=inputs.floor,
    )
    (
        conditioning_gate,
        conditioning_passed,
        rank_passed,
        condition_number_passed,
    ) = _conditioning_gate(
        conditioning,
        n_obs=int(derivatives.jac_ad.shape[0]),
        n_params=int(derivatives.jac_ad.shape[1]),
        min_rank=min_rank,
        condition_number_max=inputs.cond_max,
        finite_flags=finite_flags,
    )
    finite_passed = bool(all(finite_flags.values()))
    derivative_passed = bool(
        gradient_checks and all(bool(row["passed"]) for row in gradient_checks)
    )
    tangent_passed = _tangent_tolerance_passed(
        tangent_errors,
        abs_tol=inputs.abs_tol,
        rel_tol=inputs.rel_tol,
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
    return _GradientGateData(
        jacobian_errors=jacobian_errors,
        tangent_errors=tangent_errors,
        finite_flags=finite_flags,
        finite_passed=finite_passed,
        gradient_checks=gradient_checks,
        derivative_passed=derivative_passed,
        tangent_passed=tangent_passed,
        conditioning=conditioning,
        conditioning_gate=conditioning_gate,
        conditioning_passed=conditioning_passed,
        rank_passed=rank_passed,
        condition_number_passed=condition_number_passed,
        failure_reasons=failure_reasons,
    )


def _jacobian_report_fields(
    jacobian_errors: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Return JSON-ready Jacobian AD/FD comparison fields."""

    ad_np = jacobian_errors["ad_np"]
    fd_np = jacobian_errors["fd_np"]
    abs_error = jacobian_errors["abs_error"]
    rel_error = jacobian_errors["rel_error"]
    rel_error_gate = jacobian_errors["rel_error_gate"]
    return {
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
    }


def _tangent_report_fields(
    direction: jnp.ndarray,
    tangent_errors: Mapping[str, np.ndarray | float],
) -> dict[str, object]:
    """Return JSON-ready tangent AD/FD comparison fields."""

    tangent_ad_np = cast(np.ndarray, tangent_errors["tangent_ad_np"])
    tangent_fd_np = cast(np.ndarray, tangent_errors["tangent_fd_np"])
    tangent_abs_error = cast(np.ndarray, tangent_errors["tangent_abs_error"])
    tangent_rel_error = cast(np.ndarray, tangent_errors["tangent_rel_error"])
    direction_np = np.asarray(direction, dtype=float)
    return {
        "tangent_direction": direction_np.tolist(),
        "tangent_direction_norm": float(np.linalg.norm(direction_np)),
        "tangent_ad": tangent_ad_np.tolist(),
        "tangent_fd": tangent_fd_np.tolist(),
        "tangent_abs_error": tangent_abs_error.tolist(),
        "tangent_rel_error": tangent_rel_error.tolist(),
        "tangent_max_abs_error": float(tangent_errors["tangent_max_abs"]),
        "tangent_max_rel_error": float(tangent_errors["tangent_max_rel"]),
        "tangent_ad_norm": float(np.linalg.norm(tangent_ad_np)),
        "tangent_fd_norm": float(np.linalg.norm(tangent_fd_np)),
    }


def _assemble_gradient_validation_report(
    report_kind: str,
    inputs: _GradientValidationInputs,
    derivatives: _GradientDerivativeData,
    gates: _GradientGateData,
) -> dict[str, object]:
    report = {
        "kind": str(report_kind),
        "passed": bool(
            gates.finite_passed
            and gates.derivative_passed
            and gates.tangent_passed
            and gates.conditioning_passed
        ),
        "finite_passed": gates.finite_passed,
        "derivative_tolerance_passed": gates.derivative_passed,
        "tangent_tolerance_passed": gates.tangent_passed,
        "conditioning_passed": gates.conditioning_passed,
        "failure_reasons": gates.failure_reasons,
        "fd_step": inputs.step,
        "rtol": inputs.rel_tol,
        "atol": inputs.abs_tol,
        "relative_error_floor": inputs.floor,
        "observable_names": derivatives.obs_names,
        "parameter_names": derivatives.par_names,
        "params": np.asarray(inputs.p, dtype=float).tolist(),
        "observables": np.asarray(derivatives.observables, dtype=float).tolist(),
        "gradient_checks": gates.gradient_checks,
        "finite_flags": gates.finite_flags,
        "conditioning_gate": gates.conditioning_gate,
        "conditioning": gates.conditioning,
    }
    report.update(_jacobian_report_fields(gates.jacobian_errors))
    report.update(_tangent_report_fields(derivatives.direction, gates.tangent_errors))
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
    jacobian_chunk_size: int | str | None = None,
    jacobian_mode: str = "auto",
    report_kind: str = "observable_gradient_validation",
) -> dict[str, object]:
    """Validate observable gradients by AD, finite differences, and conditioning.

    ``observable_fn(params)`` may return any array-like observable vector.  The
    returned report is strict JSON-compatible: nonfinite diagnostic numbers are
    represented as ``None`` while finite flags and failure reasons preserve why
    the gate failed. ``jacobian_chunk_size`` bounds the number of simultaneous
    forward-mode directions; use ``"auto"`` for SOLVAX's device-aware policy,
    an integer for a fixed memory budget, or ``None`` for one full ``vmap``.
    ``jacobian_mode="auto"`` chooses forward mode for few parameters (or when
    chunking is requested) and reverse mode for few observables.
    """
    inputs = _validated_gradient_inputs(
        params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        relative_floor=relative_floor,
        condition_number_max=condition_number_max,
    )
    derivatives = _gradient_derivative_data(
        observable_fn,
        inputs,
        observable_names=observable_names,
        param_names=param_names,
        tangent=tangent,
        jacobian_chunk_size=jacobian_chunk_size,
        jacobian_mode=jacobian_mode,
    )
    gates = _gradient_gate_data(
        derivatives,
        inputs,
        min_rank=min_rank,
    )
    report = _assemble_gradient_validation_report(
        report_kind,
        inputs,
        derivatives,
        gates,
    )
    report["jacobian_chunk_size"] = jacobian_chunk_size
    report["jacobian_mode"] = derivatives.jacobian_mode
    return report


__all__ = [
    "_json_ready",
    "_sensitivity_conditioning_metadata",
    "finite_difference_jacobian",
    "observable_gradient_validation_report",
]
