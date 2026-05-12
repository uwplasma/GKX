"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace as dc_replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import covariance_diagnostics
from spectraxgk.geometry import FluxTubeGeometryData


_ARRAY_FIELDS = (
    "theta",
    "gradpar",
    "bmag",
    "bgrad",
    "gds2",
    "gds21",
    "gds22",
    "cvdrift",
    "gbdrift",
    "cvdrift0",
    "gbdrift0",
)
_GEOMETRY_OBSERVABLE_NAMES = (
    "mean_bmag",
    "relative_bmag_ripple",
    "metric_frobenius_rms",
    "drift_rms",
    "mean_jacobian",
    "mean_gradpar",
)
_VMEC_METRIC_OBSERVABLE_NAMES = (
    "sqrtg_rms",
    "mean_g_ss",
    "mean_g_tt",
    "mean_g_pp",
    "g_st_rms",
    "g_sp_rms",
    "g_tp_rms",
)
_VMEC_FIELD_LINE_OBSERVABLE_NAMES = (
    "mean_bmag",
    "relative_bmag_ripple",
    "sqrtg_rms",
    "mean_g_tt",
    "mean_g_pp",
    "g_tp_rms",
    "mean_g_ss",
)
_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21


def _jax_float_dtype() -> Any:
    return jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32


def _candidate_paths(env_names: Sequence[str], defaults: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for name in env_names:
        raw = os.environ.get(name)
        if raw:
            paths.append(Path(os.path.expandvars(raw)).expanduser())
    paths.extend(defaults)

    out: list[Path] = []
    seen: set[Path] = set()
    for base in paths:
        for candidate in (base, base / "src"):
            resolved = candidate.resolve(strict=False)
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            out.append(resolved)
    return out


def _find_importable_module(name: str, paths: Sequence[Path]) -> Any | None:
    import sys

    def _module_file(module_name: str) -> Path | None:
        module = sys.modules.get(module_name)
        raw = None if module is None else getattr(module, "__file__", None)
        if raw is None:
            return None
        return Path(str(raw)).resolve(strict=False)

    def _inside_candidates(path: Path | None) -> bool:
        if path is None:
            return False
        for root in paths:
            try:
                path.relative_to(root.resolve(strict=False))
                return True
            except ValueError:
                continue
        return False

    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Prefer explicitly configured/local differentiable-geometry checkouts over
    # globally installed packages.  Some editable VMEC checkouts carry example
    # data that the site package does not ship, so importing the site package
    # first can silently disable the real validation gates.
    root_name = name.split(".", maxsplit=1)[0]
    root_path = _module_file(root_name)
    if root_path is not None and not _inside_candidates(root_path):
        for module_name in list(sys.modules):
            if module_name == root_name or module_name.startswith(f"{root_name}."):
                sys.modules.pop(module_name, None)

    try:
        return importlib.import_module(name)
    except Exception:
        pass
    return None


def _is_traced(value: Any) -> bool:
    """Return true when host NumPy validation would break JAX tracing."""

    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, (tuple, list)):
        return any(_is_traced(item) for item in value)
    if isinstance(value, Mapping):
        return any(_is_traced(item) for item in value.values())
    return False


def discover_differentiable_geometry_backends() -> dict[str, object]:
    """Discover optional ``vmec_jax`` and ``booz_xform_jax`` bridge APIs."""

    repo_parent = Path(__file__).resolve().parents[3].parent
    home = Path.home()
    vmec_paths = _candidate_paths(
        ("SPECTRAX_VMEC_JAX_PATH", "VMEC_JAX_PATH"),
        (
            repo_parent / "vmec_jax",
            home / "vmec_jax",
            home / "local" / "vmec_jax",
        ),
    )
    booz_paths = _candidate_paths(
        ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"),
        (
            repo_parent / "booz_xform_jax",
            home / "booz_xform_jax",
            home / "local" / "booz_xform_jax",
        ),
    )
    vmec = _find_importable_module("vmec_jax", vmec_paths)
    booz = _find_importable_module("booz_xform_jax", booz_paths)
    booz_jax_api = (
        None
        if booz is None
        else _find_importable_module("booz_xform_jax.jax_api", booz_paths)
    )

    vmec_boundary_api = vmec is not None and all(
        hasattr(vmec, name)
        for name in (
            "BoundaryCoeffs",
            "boundary_aspect_ratio",
            "build_helical_basis",
            "make_angle_grid",
            "vmec_mode_table",
        )
    )
    booz_api = (
        booz_jax_api is not None
        and hasattr(booz_jax_api, "prepare_booz_xform_constants_from_inputs")
        and hasattr(booz_jax_api, "booz_xform_from_inputs")
        and hasattr(booz_jax_api, "booz_xform_jax_impl")
    )

    return {
        "vmec_jax_available": vmec is not None,
        "vmec_jax_boundary_api_available": vmec_boundary_api,
        "booz_xform_jax_available": booz is not None,
        "booz_xform_jax_api_available": booz_api,
        "vmec_jax_paths": [str(path) for path in vmec_paths],
        "booz_xform_jax_paths": [str(path) for path in booz_paths],
    }


def _array(
    mapping: Mapping[str, Any],
    key: str,
    ntheta: int | None = None,
    *,
    validate_finite: bool = True,
) -> jnp.ndarray:
    if key not in mapping:
        raise ValueError(f"missing differentiable geometry field {key!r}")
    arr = jnp.asarray(mapping[key])
    if arr.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    if ntheta is not None and int(arr.shape[0]) != int(ntheta):
        raise ValueError(
            f"{key} length {arr.shape[0]} does not match theta length {ntheta}"
        )
    if (
        validate_finite
        and not _is_traced(arr)
        and not bool(np.all(np.isfinite(np.asarray(arr))))
    ):
        raise ValueError(f"{key} contains non-finite values")
    return arr


def _scalar(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    *,
    validate_finite: bool = True,
) -> Any:
    value = mapping.get(key, default)
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{key} must be scalar")
    if _is_traced(arr):
        return arr
    out = float(np.asarray(arr))
    if validate_finite and not np.isfinite(out):
        raise ValueError(f"{key} contains a non-finite value")
    return out


def flux_tube_geometry_from_mapping(
    data: Mapping[str, Any],
    *,
    source_model: str = "vmec_jax",
    validate_finite: bool = True,
) -> FluxTubeGeometryData:
    """Build ``FluxTubeGeometryData`` from an in-memory differentiable backend.

    The input is intentionally the solver-ready flux-tube contract, not a fake
    equilibrium. ``vmec_jax`` / ``booz_xform_jax`` pipelines should first
    produce the sampled field-line arrays named here, then this function
    validates shapes/finite values and hands them to the existing solver.
    """

    theta = _array(data, "theta", validate_finite=validate_finite)
    ntheta = int(theta.shape[0])
    if ntheta < 1:
        raise ValueError("theta must contain at least one sample")
    arrays = {
        name: _array(data, name, ntheta, validate_finite=validate_finite)
        for name in _ARRAY_FIELDS
        if name != "theta"
    }
    jacobian = (
        _array(data, "jacobian", ntheta, validate_finite=validate_finite)
        if "jacobian" in data
        else 1.0 / arrays["gradpar"] / arrays["bmag"]
    )
    grho = (
        _array(data, "grho", ntheta, validate_finite=validate_finite)
        if "grho" in data
        else jnp.ones_like(theta)
    )

    gradpar_value: Any
    if _is_traced(arrays["gradpar"]):
        gradpar_value = jnp.mean(arrays["gradpar"])
    else:
        gradpar_values = np.asarray(arrays["gradpar"])
        gradpar_value = float(np.mean(gradpar_values))
        if validate_finite and not np.allclose(
            gradpar_values, gradpar_value, rtol=1.0e-5, atol=1.0e-7
        ):
            raise ValueError("gradpar must be constant along the sampled field line")

    nfp = int(data.get("nfp", 1))
    if nfp < 1:
        raise ValueError("nfp must be a positive integer")

    return FluxTubeGeometryData(
        theta=theta,
        gradpar_value=gradpar_value,
        bmag_profile=arrays["bmag"],
        bgrad_profile=arrays["bgrad"],
        gds2_profile=arrays["gds2"],
        gds21_profile=arrays["gds21"],
        gds22_profile=arrays["gds22"],
        cv_profile=arrays["cvdrift"],
        gb_profile=arrays["gbdrift"],
        cv0_profile=arrays["cvdrift0"],
        gb0_profile=arrays["gbdrift0"],
        jacobian_profile=jacobian,
        grho_profile=grho,
        q=_scalar(data, "q", 1.0, validate_finite=validate_finite),
        s_hat=_scalar(
            data, "s_hat", data.get("shat", 0.0), validate_finite=validate_finite
        ),
        epsilon=_scalar(data, "epsilon", 0.0, validate_finite=validate_finite),
        R0=_scalar(data, "R0", 1.0, validate_finite=validate_finite),
        B0=_scalar(data, "B0", 1.0, validate_finite=validate_finite),
        alpha=_scalar(data, "alpha", 0.0, validate_finite=validate_finite),
        drift_scale=_scalar(data, "drift_scale", 1.0, validate_finite=validate_finite),
        kxfac=_scalar(data, "kxfac", 1.0, validate_finite=validate_finite),
        theta_scale=_scalar(data, "theta_scale", 1.0, validate_finite=validate_finite),
        nfp=nfp,
        kperp2_bmag=bool(data.get("kperp2_bmag", True)),
        bessel_bmag_power=_scalar(data, "bessel_bmag_power", 0.0),
        source_model=str(source_model),
        theta_closed_interval=bool(data.get("theta_closed_interval", False)),
    )


def geometry_observable_names() -> tuple[str, ...]:
    """Return the ordered geometry observables used by bridge AD checks."""

    return _GEOMETRY_OBSERVABLE_NAMES


def vmec_metric_tensor_observable_names() -> tuple[str, ...]:
    """Return the ordered observables used by the VMEC metric-tensor gate."""

    return _VMEC_METRIC_OBSERVABLE_NAMES


def vmec_field_line_tensor_observable_names() -> tuple[str, ...]:
    """Return the ordered observables used by the VMEC field-line tensor gate."""

    return _VMEC_FIELD_LINE_OBSERVABLE_NAMES


def flux_tube_geometry_observables(geom: FluxTubeGeometryData) -> jnp.ndarray:
    """Return differentiable scalar observables from solver-ready geometry.

    The observables are intentionally geometry-level quantities: mean field
    strength, relative ripple, metric norm, drift norm, mean Jacobian, and mean
    parallel-gradient factor. They are used to validate the differentiable
    ``vmec_jax`` / ``booz_xform_jax`` bridge before any turbulence observable
    is promoted into an optimization claim.
    """

    bmag = jnp.asarray(geom.bmag_profile)
    jac = jnp.abs(jnp.asarray(geom.jacobian_profile))
    weights = jac / jnp.maximum(jnp.sum(jac), jnp.asarray(1.0e-300, dtype=jac.dtype))
    mean_b = jnp.sum(weights * bmag)
    ripple = jnp.sqrt(
        jnp.sum(weights * (bmag / jnp.maximum(jnp.abs(mean_b), 1.0e-300) - 1.0) ** 2)
    )
    metric = jnp.sqrt(
        jnp.sum(
            weights
            * (
                jnp.asarray(geom.gds2_profile) ** 2
                + 2.0 * jnp.asarray(geom.gds21_profile) ** 2
                + jnp.asarray(geom.gds22_profile) ** 2
            )
        )
    )
    drift = jnp.sqrt(
        jnp.sum(
            weights
            * (
                jnp.asarray(geom.cv_profile) ** 2
                + jnp.asarray(geom.gb_profile) ** 2
                + jnp.asarray(geom.cv0_profile) ** 2
                + jnp.asarray(geom.gb0_profile) ** 2
            )
        )
    )
    return jnp.asarray(
        [
            mean_b,
            ripple,
            metric,
            drift,
            jnp.mean(jnp.asarray(geom.jacobian_profile)),
            jnp.mean(jnp.asarray(geom.gradpar_value)),
        ]
    )


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
    if condition_number_max is not None and float(condition_number_max) <= 0.0:
        raise ValueError("condition_number_max must be positive or None")

    def flat_fn(x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.ravel(jnp.asarray(observable_fn(x)))
        if jnp.iscomplexobj(out):
            return jnp.concatenate([jnp.real(out), jnp.imag(out)])
        return out

    observables = flat_fn(p)
    if int(observables.size) == 0:
        raise ValueError("observable_fn must return at least one observable")
    jac_ad = jax.jacfwd(flat_fn)(p)
    jac_fd = finite_difference_jacobian(flat_fn, p, step=step)
    if jac_ad.ndim != 2 or jac_fd.ndim != 2 or jac_ad.shape != jac_fd.shape:
        raise ValueError("AD and finite-difference jacobians must be two-dimensional and shape-aligned")

    n_obs, n_params = int(jac_ad.shape[0]), int(jac_ad.shape[1])
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

    if tangent is None:
        direction = jnp.ones_like(p)
        direction = direction / jnp.maximum(
            jnp.linalg.norm(direction), jnp.asarray(1.0, dtype=p.dtype)
        )
    else:
        direction = jnp.asarray(tangent, dtype=p.dtype)
        if direction.shape != p.shape:
            raise ValueError("tangent must have the same shape as params")

    tangent_ad = jac_ad @ direction
    tangent_fd = (flat_fn(p + step * direction) - flat_fn(p - step * direction)) / (
        2.0 * step
    )

    ad_np = np.asarray(jac_ad, dtype=float)
    fd_np = np.asarray(jac_fd, dtype=float)
    diff = ad_np - fd_np
    abs_error = np.abs(diff)
    rel_error = abs_error / np.maximum(np.abs(fd_np), floor)
    tangent_ad_np = np.asarray(tangent_ad, dtype=float)
    tangent_fd_np = np.asarray(tangent_fd, dtype=float)
    tangent_diff = tangent_ad_np - tangent_fd_np
    tangent_abs_error = np.abs(tangent_diff)
    tangent_rel_error = tangent_abs_error / np.maximum(np.abs(tangent_fd_np), floor)

    finite_flags = {
        "params": bool(np.all(np.isfinite(np.asarray(p, dtype=float)))),
        "observables": bool(np.all(np.isfinite(np.asarray(observables, dtype=float)))),
        "autodiff_jacobian": bool(np.all(np.isfinite(ad_np))),
        "finite_difference_jacobian": bool(np.all(np.isfinite(fd_np))),
        "abs_error": bool(np.all(np.isfinite(abs_error))),
        "rel_error": bool(np.all(np.isfinite(rel_error))),
        "tangent_direction": bool(np.all(np.isfinite(np.asarray(direction, dtype=float)))),
        "tangent_autodiff": bool(np.all(np.isfinite(tangent_ad_np))),
        "tangent_finite_difference": bool(np.all(np.isfinite(tangent_fd_np))),
    }
    finite_passed = bool(all(finite_flags.values()))

    gradient_checks: list[dict[str, object]] = []
    for i, observable_name in enumerate(obs_names):
        for j, parameter_name in enumerate(par_names):
            entry_abs = float(abs_error[i, j])
            entry_rel = float(rel_error[i, j])
            gradient_checks.append(
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

    derivative_passed = bool(
        gradient_checks and all(bool(row["passed"]) for row in gradient_checks)
    )
    tangent_max_abs = (
        float(np.nanmax(tangent_abs_error)) if tangent_abs_error.size else 0.0
    )
    tangent_max_rel = (
        float(np.nanmax(tangent_rel_error)) if tangent_rel_error.size else 0.0
    )
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
    required_rank = (
        min(n_obs, n_params) if min_rank is None else int(min_rank)
    )
    if required_rank < 0:
        raise ValueError("min_rank must be non-negative")
    rank = int(cast(int | float, conditioning["sensitivity_map_rank"]))
    condition_number = float(cast(float, conditioning["jacobian_condition_number"]))
    condition_number_finite = bool(np.isfinite(condition_number))
    condition_number_passed = bool(
        condition_number_max is None
        or (
            condition_number_finite
            and condition_number <= float(condition_number_max)
        )
    )
    rank_passed = bool(rank >= required_rank)
    conditioning_passed = bool(
        finite_flags["autodiff_jacobian"]
        and finite_flags["finite_difference_jacobian"]
        and rank_passed
        and condition_number_passed
    )
    conditioning_gate = {
        "passed": conditioning_passed,
        "required_rank": int(required_rank),
        "rank_passed": rank_passed,
        "condition_number_max": (
            None if condition_number_max is None else float(condition_number_max)
        ),
        "condition_number_finite": condition_number_finite,
        "condition_number_passed": condition_number_passed,
    }

    failure_reasons: list[str] = []
    if not finite_passed:
        failed = [name for name, flag in finite_flags.items() if not flag]
        failure_reasons.append("nonfinite:" + ",".join(failed))
    if not derivative_passed:
        failure_reasons.append("ad_fd_tolerance")
    if not tangent_passed:
        failure_reasons.append("tangent_ad_fd_tolerance")
    if not conditioning_passed:
        if not rank_passed:
            failure_reasons.append("rank_below_required")
        if not condition_number_passed:
            failure_reasons.append("ill_conditioned")

    report = {
        "kind": str(report_kind),
        "passed": bool(
            finite_passed and derivative_passed and tangent_passed and conditioning_passed
        ),
        "finite_passed": finite_passed,
        "derivative_tolerance_passed": derivative_passed,
        "tangent_tolerance_passed": tangent_passed,
        "conditioning_passed": conditioning_passed,
        "failure_reasons": failure_reasons,
        "fd_step": step,
        "rtol": rel_tol,
        "atol": abs_tol,
        "relative_error_floor": floor,
        "observable_names": obs_names,
        "parameter_names": par_names,
        "params": np.asarray(p, dtype=float).tolist(),
        "observables": np.asarray(observables, dtype=float).tolist(),
        "jacobian_ad": ad_np.tolist(),
        "jacobian_fd": fd_np.tolist(),
        "abs_error": abs_error.tolist(),
        "rel_error": rel_error.tolist(),
        "max_abs_ad_fd_error": float(np.nanmax(abs_error)) if abs_error.size else 0.0,
        "max_rel_ad_fd_error": float(np.nanmax(rel_error)) if rel_error.size else 0.0,
        "gradient_checks": gradient_checks,
        "finite_flags": finite_flags,
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
        "conditioning_gate": conditioning_gate,
        "conditioning": conditioning,
    }
    return _json_ready(report)


def _array_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, object]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    metrics: dict[str, object] = {
        "candidate_shape": [int(v) for v in cand.shape],
        "reference_shape": [int(v) for v in ref.shape],
        "shape_match": bool(cand.shape == ref.shape),
    }
    if cand.shape != ref.shape:
        return metrics
    diff = cand - ref
    ref_scale = max(float(np.nanmax(np.abs(ref))) if ref.size else 0.0, float(floor))
    local_scale = np.maximum(np.abs(ref), float(floor))
    metrics.update(
        {
            "max_abs": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
            "rms_abs": float(np.sqrt(np.nanmean(diff * diff))) if diff.size else 0.0,
            "max_rel_pointwise": float(np.nanmax(np.abs(diff) / local_scale))
            if diff.size
            else 0.0,
            "rms_rel_pointwise": float(np.sqrt(np.nanmean((diff / local_scale) ** 2)))
            if diff.size
            else 0.0,
            "reference_scale": ref_scale,
            "normalized_max_abs": float(np.nanmax(np.abs(diff)) / ref_scale)
            if diff.size
            else 0.0,
            "candidate_min": float(np.nanmin(cand)) if cand.size else 0.0,
            "candidate_max": float(np.nanmax(cand)) if cand.size else 0.0,
            "reference_min": float(np.nanmin(ref)) if ref.size else 0.0,
            "reference_max": float(np.nanmax(ref)) if ref.size else 0.0,
        }
    )
    return metrics


def _scalar_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, float]:
    cand = float(np.asarray(candidate))
    ref = float(np.asarray(reference))
    diff = cand - ref
    scale = max(abs(ref), float(floor))
    return {
        "candidate": cand,
        "reference": ref,
        "abs": abs(diff),
        "rel": abs(diff) / scale,
    }


def _interp_radial(
    values: jnp.ndarray, s_grid: jnp.ndarray, s_value: float
) -> jnp.ndarray:
    arr = jnp.asarray(values)
    s = jnp.asarray(s_grid, dtype=arr.dtype)
    s_val = jnp.asarray(float(s_value), dtype=arr.dtype)
    if arr.ndim == 1:
        return jnp.interp(s_val, s, arr)
    if arr.ndim == 2:
        return jax.vmap(lambda col: jnp.interp(s_val, s, col), in_axes=1, out_axes=0)(
            arr
        )
    raise ValueError("radial interpolation expects a one- or two-dimensional array")


def _boozer_half_mesh_s_grid(
    raw_jlist: Any | None,
    *,
    ns_b: int,
    ns_b_full: int,
    dtype: Any,
) -> jnp.ndarray:
    """Return normalized Boozer half-mesh coordinates from API surface indices.

    ``booz_xform_jax`` exposes ``jlist`` using VMEC/Fortran half-mesh indexing
    where the first interior half mesh is ``j=2``.  The corresponding normalized
    radial coordinate is therefore ``(j - 1.5) / ns_b_full``, matching the
    imported VMEC/EIK half mesh ``0.5 * (s_full[:-1] + s_full[1:])``.
    """

    if raw_jlist is None:
        return (jnp.arange(int(ns_b), dtype=dtype) + 0.5) / float(
            max(int(ns_b_full), 1)
        )
    jlist = jnp.asarray(raw_jlist, dtype=dtype)
    return (jlist - 1.5) / float(max(int(ns_b_full), 1))


def _radial_derivative_profile(values: jnp.ndarray, spacing: float) -> jnp.ndarray:
    arr = jnp.asarray(values)
    if arr.ndim != 1:
        raise ValueError("radial derivative expects a one-dimensional profile")
    if int(arr.shape[0]) < 2:
        return jnp.zeros_like(arr)
    ds = jnp.asarray(float(spacing), dtype=arr.dtype)
    deriv = jnp.empty_like(arr)
    deriv = deriv.at[0].set((arr[1] - arr[0]) / ds)
    deriv = deriv.at[-1].set((arr[-1] - arr[-2]) / ds)
    if int(arr.shape[0]) > 2:
        deriv = deriv.at[1:-1].set((arr[2:] - arr[:-2]) / (2.0 * ds))
    return deriv


def _radial_derivative_array(values: jnp.ndarray, spacing: float) -> jnp.ndarray:
    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(
            "radial derivative expects a two-dimensional surface-mode array"
        )
    if int(arr.shape[0]) < 2:
        return jnp.zeros_like(arr)
    ds = jnp.asarray(float(spacing), dtype=arr.dtype)
    deriv = jnp.empty_like(arr)
    deriv = deriv.at[0].set((arr[1] - arr[0]) / ds)
    deriv = deriv.at[-1].set((arr[-1] - arr[-2]) / ds)
    if int(arr.shape[0]) > 2:
        deriv = deriv.at[1:-1].set((arr[2:] - arr[:-2]) / (2.0 * ds))
    return deriv


def _evaluate_boozer_cosine_series_on_field_line(
    theta: jnp.ndarray,
    *,
    coeffs: jnp.ndarray,
    ixm_b: jnp.ndarray,
    ixn_b: jnp.ndarray,
    iota: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta_arr = jnp.asarray(theta)
    coeff_arr = jnp.asarray(coeffs, dtype=theta_arr.dtype)
    m = jnp.asarray(ixm_b, dtype=theta_arr.dtype)
    n = jnp.asarray(ixn_b, dtype=theta_arr.dtype)
    iota_safe = jnp.where(
        jnp.abs(iota) < 1.0e-12, jnp.sign(iota + 1.0e-30) * 1.0e-12, iota
    )
    zeta = (theta_arr - jnp.asarray(float(alpha), dtype=theta_arr.dtype)) / iota_safe
    phase = m[:, None] * theta_arr[None, :] - n[:, None] * zeta[None, :]
    dphase_dtheta = m[:, None] - n[:, None] / iota_safe
    values = jnp.sum(coeff_arr[:, None] * jnp.cos(phase), axis=0)
    dvalues_dtheta = jnp.sum(
        -coeff_arr[:, None] * dphase_dtheta * jnp.sin(phase), axis=0
    )
    return values, dvalues_dtheta


def _cumulative_trapezoid(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(y)
    grid = jnp.asarray(x, dtype=values.dtype)
    if values.ndim != 1 or grid.ndim != 1 or int(values.shape[0]) != int(grid.shape[0]):
        raise ValueError("cumulative trapezoid expects matching one-dimensional arrays")
    if int(values.shape[0]) < 2:
        return jnp.zeros_like(values)
    areas = 0.5 * (values[1:] + values[:-1]) * (grid[1:] - grid[:-1])
    return jnp.concatenate([jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(areas)])


def _periodic_bilinear_sample_2d(
    values: jnp.ndarray, theta: jnp.ndarray, zeta: jnp.ndarray
) -> jnp.ndarray:
    """Sample a uniform periodic ``(theta,zeta)`` grid with fixed bilinear weights."""

    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError("values must be a two-dimensional theta-zeta array")
    theta_arr = jnp.asarray(theta)
    zeta_arr = jnp.asarray(zeta)
    if theta_arr.shape != zeta_arr.shape:
        raise ValueError("theta and zeta samples must have the same shape")

    ntheta = int(arr.shape[0])
    nzeta = int(arr.shape[1])
    if ntheta < 1 or nzeta < 1:
        raise ValueError("values must have non-empty theta and zeta dimensions")

    twopi = jnp.asarray(2.0 * np.pi, dtype=theta_arr.dtype)
    theta_index = jnp.mod(theta_arr, twopi) * (float(ntheta) / twopi)
    zeta_index = jnp.mod(zeta_arr, twopi) * (float(nzeta) / twopi)
    i0 = jnp.floor(theta_index).astype(jnp.int32)
    j0 = jnp.floor(zeta_index).astype(jnp.int32)
    wi = theta_index - i0
    wj = zeta_index - j0
    i1 = (i0 + 1) % int(ntheta)
    j1 = (j0 + 1) % int(nzeta)

    v00 = arr[i0, j0]
    v10 = arr[i1, j0]
    v01 = arr[i0, j1]
    v11 = arr[i1, j1]
    return (
        (1.0 - wi) * (1.0 - wj) * v00
        + wi * (1.0 - wj) * v10
        + (1.0 - wi) * wj * v01
        + wi * wj * v11
    )


def vmec_boundary_aspect_sensitivity_report(
    params: jnp.ndarray,
    *,
    fd_step: float = 2.0e-5,
    mpol: int = 2,
    ntor: int = 0,
    ntheta: int = 96,
    nphi: int = 1,
    nfp: int = 1,
) -> dict[str, object]:
    """Validate a real ``vmec_jax`` boundary-aspect derivative when available.

    The check intentionally stops at the boundary Fourier API. Full VMEC solves
    are too expensive and environment-sensitive for the default package tests,
    but the boundary-aspect path verifies that SPECTRAX-GK can discover a
    ``vmec_jax`` checkout and differentiate through its JAX-native boundary
    data structures before higher-cost optimization workflows are promoted.
    """

    p = jnp.asarray(params, dtype=_jax_float_dtype())
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a one-dimensional length-2 vector")
    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_boundary_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "aspect": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
        }

    import vmec_jax as vj  # type: ignore[import-untyped, import-not-found]

    modes = vj.vmec_mode_table(int(mpol), int(ntor))
    grid = vj.make_angle_grid(int(ntheta), int(nphi), int(nfp))
    basis = vj.build_helical_basis(modes, grid)

    def aspect_fn(x: jnp.ndarray) -> jnp.ndarray:
        ripple, elongation = x
        r0 = 1.0
        minor = 0.22 * (1.0 + 0.5 * ripple)
        r_cos = jnp.zeros(modes.K, dtype=p.dtype).at[0].set(r0).at[1].set(minor)
        z_sin = jnp.zeros(modes.K, dtype=p.dtype).at[1].set(minor * (1.0 + elongation))
        zeros = jnp.zeros_like(r_cos)
        boundary = vj.BoundaryCoeffs(R_cos=r_cos, R_sin=zeros, Z_cos=zeros, Z_sin=z_sin)
        return vj.boundary_aspect_ratio(boundary, basis)

    grad_ad = jax.grad(aspect_fn)(p)
    grad_fd = finite_difference_jacobian(
        lambda x: jnp.asarray([aspect_fn(x)]), p, step=fd_step
    )[0]
    diff = grad_ad - grad_fd
    conditioning = _sensitivity_conditioning_metadata(
        jnp.asarray(grad_ad)[None, :],
        jnp.asarray(grad_fd)[None, :],
        p,
        fd_step=float(fd_step),
        observable_names=("aspect_ratio",),
        param_names=("ripple", "elongation"),
    )
    return {
        "available": True,
        "backend_info": info,
        "aspect": float(aspect_fn(p)),
        "grad_ad": np.asarray(grad_ad).tolist(),
        "grad_fd": np.asarray(grad_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "conditioning": conditioning,
        "fd_step": float(fd_step),
        "mpol": int(mpol),
        "ntor": int(ntor),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "nfp": int(nfp),
    }


def booz_xform_spectral_sensitivity_report(  # pragma: no cover
    *,
    ripple: float = 0.05,
    fd_step: float = 2.0e-5,
    mboz: int = 2,
    nboz: int = 0,
) -> dict[str, object]:
    """Validate a real ``booz_xform_jax`` spectral derivative when available.

    This is a deliberately tiny Boozer-transform gate. It constructs an
    axisymmetric one-surface VMEC-to-Boozer input bundle, runs the real
    ``booz_xform_jax`` functional API, and checks the derivative of a Boozer
    magnetic-spectrum norm with respect to a magnetic-ripple coefficient against
    central finite differences.

    The gate strengthens the bridge beyond import discovery while remaining
    bounded enough for examples and optional local validation. It is not a full
    VMEC-state-to-flux-tube parity claim; that requires an equilibrium solve,
    field-line sampling, and comparison against the production imported-VMEC
    geometry path.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "objective": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
        }

    bx = importlib.import_module("booz_xform_jax.jax_api")

    xm = jnp.asarray([0, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0], dtype=jnp.int32)

    def _inputs(ripple_value: Any) -> SimpleNamespace:
        r = jnp.asarray(ripple_value)
        one = jnp.asarray(1.0, dtype=r.dtype)
        zero = jnp.asarray(0.0, dtype=r.dtype)
        minor = jnp.asarray(0.22, dtype=r.dtype)
        return SimpleNamespace(
            rmnc=jnp.asarray([[one, minor]], dtype=r.dtype),
            zmns=jnp.asarray([[zero, minor]], dtype=r.dtype),
            lmns=jnp.asarray([[zero, zero]], dtype=r.dtype),
            bmnc=jnp.asarray([[one, r]], dtype=r.dtype),
            bsubumnc=jnp.asarray([[0.1, 0.0]], dtype=r.dtype),
            bsubvmnc=jnp.asarray([[one, zero]], dtype=r.dtype),
            iota=jnp.asarray([0.41], dtype=r.dtype),
            xm=xm,
            xn=xn,
            xm_nyq=xm,
            xn_nyq=xn,
            nfp=1,
            bmns=None,
            bsubumns=None,
            bsubvmns=None,
        )

    try:
        base_inputs = _inputs(jnp.asarray(ripple, dtype=jnp.float64))
        constants, grids = bx.prepare_booz_xform_constants_from_inputs(
            inputs=base_inputs,
            mboz=int(mboz),
            nboz=int(nboz),
            asym=False,
        )

        def objective_fn(ripple_value: jnp.ndarray) -> jnp.ndarray:
            out = bx.booz_xform_from_inputs(
                inputs=_inputs(ripple_value),
                constants=constants,
                grids=grids,
                jit=False,
            )
            bmnc_b = jnp.asarray(out["bmnc_b"])
            return jnp.sum(bmnc_b * bmnc_b)

        r0 = jnp.asarray(float(ripple), dtype=jnp.float64)
        grad_ad = jax.grad(objective_fn)(r0)
        h = jnp.asarray(float(fd_step), dtype=r0.dtype)
        grad_fd = (objective_fn(r0 + h) - objective_fn(r0 - h)) / (2.0 * h)
        out = bx.booz_xform_from_inputs(
            inputs=base_inputs,
            constants=constants,
            grids=grids,
            jit=False,
        )
        diff = grad_ad - grad_fd
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "objective": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "objective": float(objective_fn(r0)),
        "grad_ad": float(grad_ad),
        "grad_fd": float(grad_fd),
        "max_abs_ad_fd_error": float(jnp.abs(diff)),
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "bmnc_b": np.asarray(out["bmnc_b"]).tolist(),
        "rmnc_b": np.asarray(out["rmnc_b"]).tolist(),
        "zmns_b": np.asarray(out["zmns_b"]).tolist(),
        "iota_b": np.asarray(out["iota_b"]).tolist(),
        "ixm_b": np.asarray(out["ixm_b"]).tolist(),
        "ixn_b": np.asarray(out["ixn_b"]).tolist(),
    }


def evaluate_boozer_bmag_on_field_line(
    theta: jnp.ndarray,
    *,
    bmnc_b: jnp.ndarray,
    ixm_b: jnp.ndarray,
    ixn_b: jnp.ndarray,
    iota: jnp.ndarray | float,
    alpha: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a Boozer ``|B|`` spectrum and theta derivative on a field line.

    The field-line label convention is :math:`\\alpha = \\theta - \\iota\\zeta`.
    This helper is intentionally small and JAX-native so that the
    ``booz_xform_jax`` spectral output can be differentiated all the way into
    the sampled SPECTRAX-GK geometry contract.
    """

    theta_arr = jnp.asarray(theta)
    modes_m = jnp.asarray(ixm_b, dtype=theta_arr.dtype)
    modes_n = jnp.asarray(ixn_b, dtype=theta_arr.dtype)
    coeffs = jnp.asarray(bmnc_b, dtype=theta_arr.dtype)
    iota_arr = jnp.asarray(iota, dtype=theta_arr.dtype)
    iota_safe = jnp.where(
        jnp.abs(iota_arr) < 1.0e-12, jnp.sign(iota_arr + 1.0e-30) * 1.0e-12, iota_arr
    )
    zeta = (theta_arr - jnp.asarray(float(alpha), dtype=theta_arr.dtype)) / iota_safe
    phase = theta_arr[:, None] * modes_m[None, :] - zeta[:, None] * modes_n[None, :]
    dphase_dtheta = modes_m[None, :] - modes_n[None, :] / iota_safe
    bmag = jnp.sum(coeffs[None, :] * jnp.cos(phase), axis=1)
    dbmag_dtheta = jnp.sum(-coeffs[None, :] * dphase_dtheta * jnp.sin(phase), axis=1)
    return bmag, dbmag_dtheta


def booz_xform_flux_tube_mapping_from_inputs(  # pragma: no cover
    inputs: Any,
    *,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 96,
    alpha: float = 0.0,
    surface_index: int = 0,
    magnetic_shear: float = 0.35,
    R0: float = 1.0,
    B0: float = 1.0,
    drift_scale: float = 1.0,
    jit: bool = False,
) -> dict[str, Any]:
    """Build a solver-ready flux-tube mapping from ``booz_xform_jax`` output.

    This is the first bounded production bridge step between JAX-native Boozer
    coordinates and SPECTRAX-GK. It uses the real Boozer magnetic-field
    spectrum for ``bmag``/``bgrad`` and supplies smooth metric/drift profiles
    with the same solver-ready names accepted by
    :func:`flux_tube_geometry_from_mapping`.

    Full VMEC/Boozer metric parity remains a separate promotion gate: a
    high-fidelity backend must replace the smooth metric/drift closure here
    with sampled VMEC/Boozer metric tensors before nonlinear optimization
    claims are made.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        raise RuntimeError("booz_xform_jax functional API is not available")

    bx = importlib.import_module("booz_xform_jax.jax_api")
    constants, grids = bx.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(inputs, "bmns", None) is not None),
    )
    out = bx.booz_xform_from_inputs(
        inputs=inputs, constants=constants, grids=grids, jit=bool(jit)
    )
    idx = int(surface_index)
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False)
    bmnc_b = jnp.asarray(out["bmnc_b"])[idx]
    iota = jnp.asarray(out["iota_b"])[idx]
    bmag, dbmag_dtheta = evaluate_boozer_bmag_on_field_line(
        theta,
        bmnc_b=bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=iota,
        alpha=float(alpha),
    )
    q = 1.0 / jnp.maximum(jnp.abs(iota), jnp.asarray(1.0e-12, dtype=theta.dtype))
    gradpar_value = 1.0 / (q * float(R0))
    gradpar = gradpar_value * jnp.ones_like(theta)
    shear = jnp.asarray(float(magnetic_shear), dtype=theta.dtype)
    field_line_shift = shear * theta
    gds2 = 1.0 + field_line_shift * field_line_shift
    gds21 = -shear * field_line_shift
    gds22 = (1.0 + shear * shear) * jnp.ones_like(theta)
    cv = (
        float(drift_scale)
        * (jnp.cos(theta) + field_line_shift * jnp.sin(theta))
        / float(R0)
    )
    cv0 = -float(drift_scale) * shear * jnp.sin(theta) / float(R0)
    bmag_safe = jnp.maximum(jnp.abs(bmag), jnp.asarray(1.0e-12, dtype=theta.dtype))
    return {
        "theta": theta,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": gradpar_value * dbmag_dtheta / bmag_safe,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cv,
        "gbdrift": cv,
        "cvdrift0": cv0,
        "gbdrift0": cv0,
        "jacobian": 1.0 / (gradpar * bmag_safe),
        "grho": jnp.ones_like(theta),
        "q": q,
        "s_hat": shear,
        "epsilon": jnp.sqrt(jnp.mean((bmag / jnp.mean(bmag) - 1.0) ** 2)),
        "R0": float(R0),
        "B0": float(B0),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(jnp.asarray(getattr(inputs, "nfp", 1))),
        "booz_xform": {
            "bmnc_b": bmnc_b,
            "ixm_b": jnp.asarray(out["ixm_b"]),
            "ixn_b": jnp.asarray(out["ixn_b"]),
            "iota_b": iota,
        },
    }


def booz_xform_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    fd_step: float = 2.0e-5,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 64,
) -> dict[str, object]:
    """AD/FD-check a Boozer-spectrum-to-flux-tube geometry bridge.

    ``params = [axisymmetric_B_ripple, helical_B_ripple]`` perturbs a tiny
    one-surface VMEC-to-Boozer input bundle. The real ``booz_xform_jax``
    transform is run for each parameter vector; its Boozer ``|B|`` spectrum is
    sampled on a field line and converted into ``FluxTubeGeometryData``.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
        }

    p = jnp.asarray([0.05, 0.02] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    xm = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0, 2], dtype=jnp.int32)

    def _inputs(x: jnp.ndarray) -> SimpleNamespace:
        axisym_ripple, helical_ripple = x
        one = jnp.asarray(1.0, dtype=x.dtype)
        zero = jnp.asarray(0.0, dtype=x.dtype)
        minor = jnp.asarray(0.22, dtype=x.dtype)
        helical_shape = jnp.asarray(0.02, dtype=x.dtype)
        return SimpleNamespace(
            rmnc=jnp.asarray([[one, minor, helical_shape]], dtype=x.dtype),
            zmns=jnp.asarray([[zero, minor, helical_shape]], dtype=x.dtype),
            lmns=jnp.asarray([[zero, zero, zero]], dtype=x.dtype),
            bmnc=jnp.asarray([[one, axisym_ripple, helical_ripple]], dtype=x.dtype),
            bsubumnc=jnp.asarray([[0.1, 0.0, 0.0]], dtype=x.dtype),
            bsubvmnc=jnp.asarray([[one, zero, zero]], dtype=x.dtype),
            iota=jnp.asarray([0.41], dtype=x.dtype),
            xm=xm,
            xn=xn,
            xm_nyq=xm,
            xn_nyq=xn,
            nfp=2,
            bmns=None,
            bsubumns=None,
            bsubvmns=None,
        )

    try:
        sensitivity = geometry_sensitivity_report(
            lambda x: booz_xform_flux_tube_mapping_from_inputs(
                _inputs(x),
                mboz=int(mboz),
                nboz=int(nboz),
                ntheta=int(ntheta),
                magnetic_shear=0.35,
                jit=False,
            ),
            p,
            fd_step=float(fd_step),
            source_model="booz_xform_jax:field-line-bmag",
        )
        mapping = booz_xform_flux_tube_mapping_from_inputs(
            _inputs(p),
            mboz=int(mboz),
            nboz=int(nboz),
            ntheta=int(ntheta),
            magnetic_shear=0.35,
            jit=False,
        )
        booz_meta = mapping["booz_xform"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "params": np.asarray(p).tolist(),
        "sensitivity": sensitivity,
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "ntheta": int(ntheta),
        "bmnc_b": np.asarray(booz_meta["bmnc_b"]).tolist(),
        "ixm_b": np.asarray(booz_meta["ixm_b"]).tolist(),
        "ixn_b": np.asarray(booz_meta["ixn_b"]).tolist(),
        "iota_b": float(np.asarray(booz_meta["iota_b"])),
    }


def vmec_jax_boozer_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "circular_tokamak",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-5,
    mboz: int = 2,
    nboz: int = 0,
    ntheta: int = 32,
) -> dict[str, object]:
    """AD/FD-check ``vmec_jax`` state coefficients through the Boozer bridge.

    This is the first end-to-end optional-backend gate that starts from a real
    ``vmec_jax`` ``VMECState`` instead of a hand-built Boozer input bundle. It
    loads a small bundled VMEC example, perturbs two VMEC Fourier coefficients
    ``[Rcos(radial_index, mode_index), Zsin(radial_index, mode_index)]``,
    converts the perturbed state to ``booz_xform_jax`` inputs, samples the
    resulting Boozer ``|B|`` spectrum on a field line, and checks
    SPECTRAX-GK geometry-observable derivatives against central finite
    differences.

    The current metric/drift closure is still intentionally smooth and local to
    SPECTRAX-GK. Full production promotion requires replacing it with sampled
    VMEC/Boozer metric tensors and parity-checking those arrays against the
    imported VMEC/EIK path.
    """

    p = jnp.asarray([1.0e-3, 1.0e-3] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    info = discover_differentiable_geometry_backends()
    if not (
        info.get("vmec_jax_available", False)
        and info.get("booz_xform_jax_api_available", False)
    ):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax or booz_xform_jax functional API is not available",
        }

    try:
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        booz_input_mod = importlib.import_module("vmec_jax.booz_input")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 2))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0]) - 1):
            raise ValueError(
                "surface_index is outside the VMEC half-mesh Boozer surface grid"
            )

        def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            inputs = booz_input_mod.booz_xform_inputs_from_state(
                state=traced_state,
                static=static,
                indata=indata,
                signgs=wout.signgs,
            )
            return booz_xform_flux_tube_mapping_from_inputs(
                inputs,
                mboz=int(mboz),
                nboz=int(nboz),
                ntheta=int(ntheta),
                surface_index=int(sidx),
                magnetic_shear=0.35,
                jit=False,
            )

        sensitivity = geometry_sensitivity_report(
            mapping_fn,
            p,
            fd_step=float(fd_step),
            source_model="vmec_jax:state->booz_xform_jax:field-line-bmag",
        )
        mapping = mapping_fn(p)
        booz_meta = mapping["booz_xform"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(p).tolist(),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "sensitivity": sensitivity,
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "ntheta": int(ntheta),
        "bmnc_b": np.asarray(booz_meta["bmnc_b"]).tolist(),
        "ixm_b": np.asarray(booz_meta["ixm_b"]).tolist(),
        "ixn_b": np.asarray(booz_meta["ixn_b"]).tolist(),
        "iota_b": float(np.asarray(booz_meta["iota_b"])),
    }


def vmec_jax_metric_tensor_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "circular_tokamak",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-5,
    rms_epsilon: float = 1.0e-24,
) -> dict[str, object]:
    """AD/FD-check real ``vmec_jax`` metric tensors from a ``VMECState``.

    The Boozer bridge validates the straight-field-line ``|B|`` spectrum, but
    SPECTRAX-GK's production geometry contract also needs sampled metric and
    drift tensors. This gate stays upstream of any reduced closure: it loads a
    real ``vmec_jax`` example state, perturbs two VMEC Fourier coefficients,
    evaluates ``vmec_jax.geom.eval_geom``, and checks metric-tensor observable
    derivatives against central finite differences.

    This is a prerequisite for replacing the smooth metric/drift closure in
    :func:`booz_xform_flux_tube_mapping_from_inputs`; it is not by itself the
    final Boozer-field-line metric parity gate.
    """

    p = jnp.asarray([1.0e-3, 1.0e-3] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        geom_mod = importlib.import_module("vmec_jax.geom")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 1))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0])):
            raise ValueError("surface_index is outside the VMEC metric radial grid")

        eps = jnp.asarray(float(rms_epsilon), dtype=p.dtype)

        def _rms(arr: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(arr)
            return jnp.sqrt(jnp.mean(arr * arr) + eps)

        def metric_observables(x: jnp.ndarray) -> jnp.ndarray:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            geom = geom_mod.eval_geom(traced_state, static)
            sqrtg = jnp.asarray(geom.sqrtg)[sidx]
            g_ss = jnp.asarray(geom.g_ss)[sidx]
            g_st = jnp.asarray(geom.g_st)[sidx]
            g_sp = jnp.asarray(geom.g_sp)[sidx]
            g_tt = jnp.asarray(geom.g_tt)[sidx]
            g_tp = jnp.asarray(geom.g_tp)[sidx]
            g_pp = jnp.asarray(geom.g_pp)[sidx]
            return jnp.asarray(
                [
                    _rms(sqrtg),
                    jnp.mean(g_ss),
                    jnp.mean(g_tt),
                    jnp.mean(g_pp),
                    _rms(g_st),
                    _rms(g_sp),
                    _rms(g_tp),
                ]
            )

        observables = metric_observables(p)
        jac_ad = jax.jacfwd(metric_observables)(p)
        jac_fd = finite_difference_jacobian(metric_observables, p, step=float(fd_step))
        diff = jac_ad - jac_fd
        max_abs = jnp.max(jnp.abs(diff))
        max_rel = jnp.max(jnp.abs(diff) / (jnp.abs(jac_fd) + 1.0e-12))
        geom0 = geom_mod.eval_geom(state, static)
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->metric-tensors",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "observable_names": list(_VMEC_METRIC_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(observables).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.asarray(max_abs)),
        "max_rel_ad_fd_error": float(np.asarray(max_rel)),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            p,
            fd_step=float(fd_step),
            observable_names=_VMEC_METRIC_OBSERVABLE_NAMES,
            param_names=("delta_Rcos", "delta_Zsin"),
            relative_floor=1.0e-12,
        ),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "fd_step": float(fd_step),
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_field_line_tensor_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    fd_step: float = 1.0e-6,
    b2_floor: float = 1.0e-24,
    rms_epsilon: float = 1.0e-24,
) -> dict[str, object]:
    """AD/FD-check VMEC field-line metric and ``|B|`` tensors from ``vmec_jax``.

    This optional-backend gate is deliberately upstream of the production
    SPECTRAX-GK metric/drift closure. It loads a real stellarator
    ``vmec_jax`` example state, perturbs two VMEC Fourier coefficients,
    evaluates ``vmec_jax.geom.eval_geom`` and ``vmec_jax.vmec_bcovar``, samples
    raw metric/``|B|`` tensors on a fixed VMEC field-line convention, and checks
    those observable derivatives against central finite differences.

    The gate proves differentiability from ``VMECState`` through real VMEC
    metric and magnetic-field tensors. The later production gate must still
    convert those tensors into the exact SPECTRAX-GK ``gds*``/drift contract and
    compare against the imported VMEC/EIK path.
    """

    p = jnp.asarray([1.0e-4, 1.0e-4] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        geom_mod = importlib.import_module("vmec_jax.geom")
        bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
        field_mod = importlib.import_module("vmec_jax.field")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(1, min(ridx, int(base_Rcos.shape[0]) - 2))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0])):
            raise ValueError("surface_index is outside the VMEC metric radial grid")

        iota_profile = jnp.asarray(getattr(wout, "iotas"))
        if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx:
            raise RuntimeError(
                "vmec_jax wout iotas profile is missing or incompatible with the state grid"
            )
        iota_line = iota_profile[sidx]
        iota_safe = jnp.where(
            jnp.abs(iota_line) < 1.0e-12,
            jnp.sign(iota_line + 1.0e-30) * 1.0e-12,
            iota_line,
        )
        theta_line = jnp.linspace(
            -jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=p.dtype
        )
        theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
        zeta_line = jnp.mod(
            (theta_vmec - jnp.asarray(float(alpha), dtype=p.dtype)) / iota_safe,
            2.0 * jnp.pi,
        )
        b2_floor_arr = jnp.asarray(float(b2_floor), dtype=p.dtype)
        eps = jnp.asarray(float(rms_epsilon), dtype=p.dtype)

        def _rms(arr: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(arr)
            return jnp.sqrt(jnp.mean(arr * arr) + eps)

        def field_line_observables(x: jnp.ndarray) -> jnp.ndarray:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            geom = geom_mod.eval_geom(traced_state, static)
            bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
                state=traced_state,
                static=static,
                wout=wout,
                pres=getattr(wout, "pres", None),
            )
            b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
            bmag = jnp.sqrt(
                jnp.maximum(
                    _periodic_bilinear_sample_2d(b2[sidx], theta_vmec, zeta_line),
                    b2_floor_arr,
                )
            )
            sqrtg = _periodic_bilinear_sample_2d(
                geom.sqrtg[sidx], theta_vmec, zeta_line
            )
            g_tt = _periodic_bilinear_sample_2d(geom.g_tt[sidx], theta_vmec, zeta_line)
            g_tp = _periodic_bilinear_sample_2d(geom.g_tp[sidx], theta_vmec, zeta_line)
            g_pp = _periodic_bilinear_sample_2d(geom.g_pp[sidx], theta_vmec, zeta_line)
            g_ss = _periodic_bilinear_sample_2d(geom.g_ss[sidx], theta_vmec, zeta_line)
            mean_b = jnp.mean(bmag)
            ripple = jnp.std(bmag) / jnp.maximum(
                jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=bmag.dtype)
            )
            return jnp.asarray(
                [
                    mean_b,
                    ripple,
                    _rms(sqrtg),
                    jnp.mean(g_tt),
                    jnp.mean(g_pp),
                    _rms(g_tp),
                    jnp.mean(g_ss),
                ]
            )

        observables = field_line_observables(p)
        jac_ad = jax.jacfwd(field_line_observables)(p)
        jac_fd = finite_difference_jacobian(
            field_line_observables, p, step=float(fd_step)
        )
        diff = jac_ad - jac_fd
        max_abs = jnp.max(jnp.abs(diff))
        max_rel = jnp.max(jnp.abs(diff) / (jnp.abs(jac_fd) + 1.0e-10))
        geom0 = geom_mod.eval_geom(state, static)
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->field-line-metric-and-b",
        "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "observable_names": list(_VMEC_FIELD_LINE_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(observables).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.asarray(max_abs)),
        "max_rel_ad_fd_error": float(np.asarray(max_rel)),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            p,
            fd_step=float(fd_step),
            observable_names=_VMEC_FIELD_LINE_OBSERVABLE_NAMES,
            param_names=("delta_Rcos", "delta_Zsin"),
            relative_floor=1.0e-10,
        ),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "iota": float(np.asarray(iota_line)),
        "alpha": float(alpha),
        "ntheta": int(ntheta_int),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "fd_step": float(fd_step),
        "b2_floor": float(b2_floor),
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_flux_tube_mapping_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    b2_floor: float = 1.0e-24,
    metric_floor: float = 1.0e-24,
    reference_length: float | None = None,
    reference_b: float | None = None,
    drift_scale: float = 1.0,
) -> dict[str, Any]:
    """Build a solver-ready flux-tube mapping directly from ``vmec_jax`` tensors.

    This is the VMEC-native bridge step: it evaluates ``vmec_jax.geom`` and
    ``vmec_jax.vmec_bcovar``, samples the covariant metric and ``|B|`` on a
    fixed field line, inverts the sampled metric to construct perpendicular
    flux-tube metric coefficients, and emits the
    :func:`flux_tube_geometry_from_mapping` contract.

    The metric and magnetic-field derivatives are differentiable with respect
    to the VMEC state.  The drift coefficients use a local grad-:math:`B`
    projection closure so that downstream solver contracts remain populated;
    the full Hegna-Nakajima/imported-VMEC drift parity gate remains a separate
    production promotion step.
    """

    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")

    geom_mod = importlib.import_module("vmec_jax.geom")
    bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
    field_mod = importlib.import_module("vmec_jax.field")

    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos array must be two-dimensional")
    ns = int(base_Rcos.shape[0])
    if ns < 3:
        raise RuntimeError("vmec_jax state needs at least three radial surfaces")
    sidx = max(1, min(ns // 2, ns - 2)) if surface_index is None else int(surface_index)
    if not (0 < sidx < ns - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")

    iota_profile = jnp.asarray(getattr(wout, "iotas"))
    if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx + 1:
        raise RuntimeError(
            "vmec_jax wout iotas profile is missing or incompatible with the state grid"
        )
    iota_line = iota_profile[sidx]
    iota_safe = jnp.where(
        jnp.abs(iota_line) < 1.0e-12, jnp.sign(iota_line + 1.0e-30) * 1.0e-12, iota_line
    )

    s_grid = jnp.linspace(0.0, 1.0, ns, dtype=base_Rcos.dtype)
    s_val = jnp.maximum(s_grid[sidx], jnp.asarray(1.0e-12, dtype=base_Rcos.dtype))
    ds = s_grid[1] - s_grid[0]
    d_iota_ds = (iota_profile[sidx + 1] - iota_profile[sidx - 1]) / (2.0 * ds)
    s_hat = -2.0 * s_val * d_iota_ds / iota_safe

    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    L_reference = (
        raw_length if np.isfinite(raw_length) and abs(raw_length) > 0.0 else 1.0
    )
    if reference_b is None:
        phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
        edge_toroidal_flux_over_2pi = abs(float(phi_profile[-1]) / (2.0 * np.pi))
        raw_b = 2.0 * edge_toroidal_flux_over_2pi / (L_reference * L_reference)
        B_reference = raw_b if np.isfinite(raw_b) and abs(raw_b) > 0.0 else 1.0
    else:
        B_reference = float(reference_b)
    B_reference = (
        B_reference if np.isfinite(B_reference) and abs(B_reference) > 0.0 else 1.0
    )

    theta_line = jnp.linspace(
        -jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=base_Rcos.dtype
    )
    theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
    zeta_line = jnp.mod(
        (theta_vmec - jnp.asarray(float(alpha), dtype=base_Rcos.dtype)) / iota_safe,
        2.0 * jnp.pi,
    )

    geom = geom_mod.eval_geom(state, static)
    bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        pres=getattr(wout, "pres", None),
    )
    b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
    b2_floor_arr = jnp.asarray(float(b2_floor), dtype=base_Rcos.dtype)
    metric_floor_arr = jnp.asarray(float(metric_floor), dtype=base_Rcos.dtype)

    bmag_all = jnp.sqrt(jnp.maximum(jnp.asarray(b2), b2_floor_arr)) / float(B_reference)
    bmag_grid = bmag_all[sidx]
    dtheta_grid = 2.0 * jnp.pi / float(bmag_grid.shape[0])
    dzeta_grid = 2.0 * jnp.pi / float(bmag_grid.shape[1])
    db_dtheta_grid = (
        jnp.roll(bmag_grid, -1, axis=0) - jnp.roll(bmag_grid, 1, axis=0)
    ) / (2.0 * dtheta_grid)
    db_dzeta_grid = (
        jnp.roll(bmag_grid, -1, axis=1) - jnp.roll(bmag_grid, 1, axis=1)
    ) / (2.0 * dzeta_grid)
    db_ds_grid = (bmag_all[sidx + 1] - bmag_all[sidx - 1]) / (2.0 * ds)

    def _line(values: jnp.ndarray) -> jnp.ndarray:
        return _periodic_bilinear_sample_2d(values, theta_vmec, zeta_line)

    bmag = _line(bmag_grid)
    sqrtg = _line(jnp.asarray(geom.sqrtg)[sidx])
    g_ss = _line(jnp.asarray(geom.g_ss)[sidx])
    g_st = _line(jnp.asarray(geom.g_st)[sidx])
    g_sp = _line(jnp.asarray(geom.g_sp)[sidx])
    g_tt = _line(jnp.asarray(geom.g_tt)[sidx])
    g_tp = _line(jnp.asarray(geom.g_tp)[sidx])
    g_pp = _line(jnp.asarray(geom.g_pp)[sidx])
    db_ds = _line(db_ds_grid)
    db_dt = _line(db_dtheta_grid)
    db_dp = _line(db_dzeta_grid)

    cov_metric = jnp.stack(
        (
            jnp.stack((g_ss, g_st, g_sp), axis=-1),
            jnp.stack((g_st, g_tt, g_tp), axis=-1),
            jnp.stack((g_sp, g_tp, g_pp), axis=-1),
        ),
        axis=-2,
    )
    contra = jnp.linalg.inv(cov_metric)
    alpha_coeff = (
        jnp.asarray([0.0, 1.0, -1.0], dtype=base_Rcos.dtype).at[2].set(-iota_safe)
    )
    grad_alpha_metric = jnp.einsum("i,zij,j->z", alpha_coeff, contra, alpha_coeff)
    grad_s_metric = contra[:, 0, 0]
    grad_s_dot_alpha = contra[:, 0, 1] - iota_safe * contra[:, 0, 2]

    grad_alpha_metric = jnp.maximum(grad_alpha_metric, metric_floor_arr)
    grad_s_metric = jnp.maximum(grad_s_metric, metric_floor_arr)
    sqrt_s = jnp.sqrt(s_val)
    L = jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
    Bref = jnp.asarray(float(B_reference), dtype=base_Rcos.dtype)
    shat_safe = jnp.where(
        jnp.abs(s_hat) < 1.0e-12, jnp.asarray(0.0, dtype=base_Rcos.dtype), s_hat
    )
    gds2 = L * L * s_val * grad_alpha_metric
    gds21 = (
        shat_safe
        * grad_s_dot_alpha
        / jnp.maximum(Bref, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    )
    gds22 = (
        shat_safe
        * shat_safe
        * grad_s_metric
        / jnp.maximum(
            L * L * Bref * Bref * s_val, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )
    grho = jnp.sqrt(grad_s_metric) / jnp.maximum(
        L * Bref * sqrt_s, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )

    gradpar_profile = jnp.abs(
        L
        * iota_safe
        / jnp.maximum(
            jnp.abs(bmag * sqrtg), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )
    gradpar_value = jnp.mean(gradpar_profile)
    gradpar = gradpar_value * jnp.ones_like(theta_line)
    dlogb_dtheta = (
        jnp.roll(jnp.log(jnp.maximum(bmag, 1.0e-30)), -1)
        - jnp.roll(jnp.log(jnp.maximum(bmag, 1.0e-30)), 1)
    ) / (2.0 * (2.0 * jnp.pi / float(ntheta_int)))
    bgrad = gradpar_value * dlogb_dtheta

    grad_b_cov = jnp.stack((db_ds, db_dt, db_dp), axis=-1)
    grad_b_dot_alpha = jnp.einsum("zi,zij,j->z", grad_b_cov, contra, alpha_coeff)
    grad_b_dot_s = jnp.einsum("zi,zi->z", grad_b_cov, contra[:, :, 0])
    bmag_sq = jnp.maximum(bmag * bmag, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    gbdrift = -2.0 * float(drift_scale) * L * L * sqrt_s * grad_b_dot_alpha / bmag_sq
    gbdrift0 = (
        -2.0
        * float(drift_scale)
        * shat_safe
        * grad_b_dot_s
        / jnp.maximum(
            Bref * bmag_sq * sqrt_s, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )

    return {
        "theta": theta_line,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": bgrad,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": gbdrift,
        "gbdrift": gbdrift,
        "cvdrift0": gbdrift0,
        "gbdrift0": gbdrift0,
        "jacobian": jnp.abs(sqrtg)
        / jnp.maximum(
            jnp.mean(jnp.abs(sqrtg)), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        ),
        "grho": grho,
        "q": 1.0
        / jnp.maximum(jnp.abs(iota_safe), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "s_hat": shat_safe,
        "epsilon": jnp.std(bmag)
        / jnp.maximum(jnp.mean(bmag), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "R0": float(L_reference),
        "B0": float(B_reference),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(getattr(wout, "nfp", 1)),
        "vmec_jax": {
            "surface_index": int(sidx),
            "iota": iota_line,
            "reference_length": float(L_reference),
            "reference_b": float(B_reference),
            "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        },
    }


def vmec_jax_boozer_equal_arc_core_profiles_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    torflux: float | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    jit: bool = False,
    surface_stencil_width: int | None = None,
    reference_length: float | None = None,
    reference_b: float | None = None,
) -> dict[str, Any]:
    """Return Boozer equal-arc core profiles from a real ``vmec_jax`` state.

    This bridge follows the same high-level convention as the imported VMEC/EIK
    runtime path for scalar/core field-line quantities and the zero-beta Boozer
    metric/drift terms that can be reconstructed directly from
    ``booz_xform_jax`` output: Boozer ``|B|``, equal-arc constant ``gradpar``,
    ``q``, magnetic shear, solver Jacobian normalization, ``gds*``/``grho``,
    and loaded-convention ``cvdrift``/``gbdrift`` coefficients.  General
    finite-beta pressure corrections and broader-equilibrium drift gates remain
    separate promotion steps.
    """

    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")
    mboz_int = int(mboz)
    nboz_int = int(nboz)
    if (
        mboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
        or nboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
    ):
        raise ValueError(
            "mboz and nboz must both be >= "
            f"{_VMEC_BOOZER_PARITY_MIN_MODE_COUNT} for VMEC/Boozer parity gates"
        )

    info = discover_differentiable_geometry_backends()
    if not (
        info.get("vmec_jax_available", False)
        and info.get("booz_xform_jax_api_available", False)
    ):
        raise RuntimeError("vmec_jax and booz_xform_jax functional APIs are required")

    booz_input_mod = importlib.import_module("vmec_jax.booz_input")
    bx = importlib.import_module("booz_xform_jax.jax_api")

    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos array must be two-dimensional")
    ns_full = int(base_Rcos.shape[0])
    if ns_full < 3:
        raise RuntimeError("vmec_jax state needs at least three radial surfaces")

    sidx = (
        max(1, min(ns_full // 2, ns_full - 2))
        if surface_index is None
        else int(surface_index)
    )
    if not (0 < sidx < ns_full - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")
    s_value = (
        float(sidx) / float(max(ns_full - 1, 1)) if torflux is None else float(torflux)
    )
    if not (0.0 < s_value < 1.0):
        raise ValueError("torflux must lie inside (0, 1)")
    if surface_stencil_width is not None and int(surface_stencil_width) < 3:
        raise ValueError("surface_stencil_width must be >= 3 when provided")

    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    L_reference = (
        raw_length if np.isfinite(raw_length) and abs(raw_length) > 0.0 else 1.0
    )
    if reference_b is None:
        phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
        edge_toroidal_flux_over_2pi = -float(phi_profile[-1]) / (2.0 * np.pi)
        raw_b = 2.0 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
        B_reference = raw_b if np.isfinite(raw_b) and abs(raw_b) > 0.0 else 1.0
    else:
        edge_toroidal_flux_over_2pi = -float(
            np.asarray(getattr(wout, "phi", [0.0, np.pi]))[-1]
        ) / (2.0 * np.pi)
        B_reference = float(reference_b)
    B_reference = (
        B_reference if np.isfinite(B_reference) and abs(B_reference) > 0.0 else 1.0
    )

    inputs = booz_input_mod.booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=getattr(wout, "signgs", 1),
    )
    constants, grids = bx.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=mboz_int,
        nboz=nboz_int,
        asym=bool(getattr(inputs, "bmns", None) is not None),
    )
    surface_indices = None
    if surface_stencil_width is not None:
        ns_b_est = max(1, ns_full - 1)
        width = min(int(surface_stencil_width), ns_b_est)
        center = int(round(s_value * float(ns_b_est) - 0.5))
        half_width = width // 2
        start = max(0, min(center - half_width, ns_b_est - width))
        surface_indices = jnp.arange(start, start + width, dtype=jnp.int32)
    out = bx.booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=surface_indices,
        jit=bool(jit),
    )

    bmnc_b_all = jnp.asarray(out["bmnc_b"], dtype=base_Rcos.dtype)
    if bmnc_b_all.ndim != 2:
        raise RuntimeError(
            "booz_xform_jax bmnc_b output must have shape (surface, mode)"
        )
    ns_b = int(bmnc_b_all.shape[0])
    if ns_b < 2:
        raise RuntimeError("booz_xform_jax output needs at least two radial surfaces")
    ns_b_full = int(np.asarray(out.get("ns_b", ns_b)))
    s_half = _boozer_half_mesh_s_grid(
        out.get("jlist"),
        ns_b=ns_b,
        ns_b_full=ns_b_full,
        dtype=base_Rcos.dtype,
    )

    radial_spacing = float(s_half[1] - s_half[0])
    bmnc_b = _interp_radial(bmnc_b_all, s_half, s_value)
    rmnc_b = _interp_radial(
        jnp.asarray(out["rmnc_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    zmns_b = _interp_radial(
        jnp.asarray(out["zmns_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    numns_b = -_interp_radial(
        jnp.asarray(out["pmns_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    d_bmnc_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["bmnc_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_rmnc_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["rmnc_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_zmns_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["zmns_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_numns_b_d_s = -_interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["pmns_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    iota_profile = jnp.asarray(out["iota_b"], dtype=base_Rcos.dtype)
    iota = _interp_radial(iota_profile, s_half, s_value)
    d_iota_ds = _interp_radial(
        _radial_derivative_profile(iota_profile, radial_spacing), s_half, s_value
    )
    iota_safe = jnp.where(
        jnp.abs(iota) < 1.0e-12, jnp.sign(iota + 1.0e-30) * 1.0e-12, iota
    )
    s_hat = -2.0 * jnp.asarray(s_value, dtype=base_Rcos.dtype) * d_iota_ds / iota_safe

    boozer_i = _interp_radial(
        jnp.asarray(out["buco_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    boozer_g = _interp_radial(
        jnp.asarray(out["bvco_b"], dtype=base_Rcos.dtype), s_half, s_value
    )

    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, ntheta_int + 1, dtype=base_Rcos.dtype)
    mod_b, _dmod_b_dtheta = _evaluate_boozer_cosine_series_on_field_line(
        theta_closed,
        coeffs=bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=iota_safe,
        alpha=float(alpha),
    )
    mod_b_safe = jnp.maximum(
        jnp.abs(mod_b), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )
    sqrt_g_booz = (boozer_g + iota_safe * boozer_i) / (mod_b_safe * mod_b_safe)
    gradpar_raw = jnp.abs(
        jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
        * iota_safe
        / jnp.maximum(
            jnp.abs(mod_b_safe * sqrt_g_booz),
            jnp.asarray(1.0e-30, dtype=base_Rcos.dtype),
        )
    )
    inv_gradpar_int = _cumulative_trapezoid(1.0 / gradpar_raw, theta_closed)
    gradpar_eqarc = (
        2.0
        * jnp.pi
        / jnp.maximum(inv_gradpar_int[-1], jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    )
    theta_eqarc = gradpar_eqarc * inv_gradpar_int - jnp.pi
    theta_uniform_closed = jnp.linspace(
        -jnp.pi, jnp.pi, ntheta_int + 1, dtype=base_Rcos.dtype
    )
    bmag_closed = jnp.asarray(
        jnp.interp(theta_uniform_closed, theta_eqarc, mod_b_safe / float(B_reference))
    )
    theta = theta_uniform_closed[:-1]
    bmag = bmag_closed[:-1]
    gradpar = gradpar_eqarc * jnp.ones_like(theta)
    dtheta = 2.0 * jnp.pi / float(ntheta_int)
    bmag_safe = jnp.maximum(jnp.abs(bmag), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    wave_number = 2.0 * jnp.pi * jnp.fft.fftfreq(ntheta_int, d=float(dtheta))
    dbmag_dtheta = jnp.real(jnp.fft.ifft(1j * wave_number * jnp.fft.fft(bmag)))
    bgrad = gradpar_eqarc * dbmag_dtheta / bmag_safe

    dpsidrho = (
        2.0
        * jnp.sqrt(jnp.asarray(s_value, dtype=base_Rcos.dtype))
        * jnp.asarray(
            edge_toroidal_flux_over_2pi,
            dtype=base_Rcos.dtype,
        )
    )
    drhodpsi = 1.0 / jnp.maximum(
        jnp.abs(dpsidrho), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )
    jacobian = 1.0 / jnp.maximum(
        jnp.abs(drhodpsi * gradpar_eqarc * bmag_safe),
        jnp.asarray(1.0e-30, dtype=base_Rcos.dtype),
    )

    m = jnp.asarray(out["ixm_b"], dtype=base_Rcos.dtype)
    n = jnp.asarray(out["ixn_b"], dtype=base_Rcos.dtype)
    phi_b = (
        theta_closed - jnp.asarray(float(alpha), dtype=base_Rcos.dtype)
    ) / iota_safe
    phase = m[:, None] * theta_closed[None, :] - n[:, None] * phi_b[None, :]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    m_cos = m[:, None] * cos_phase
    m_sin = m[:, None] * sin_phase
    n_cos = n[:, None] * cos_phase
    n_sin = n[:, None] * sin_phase

    r_b = jnp.sum(rmnc_b[:, None] * cos_phase, axis=0)
    d_mod_b_d_s = jnp.sum(d_bmnc_b_d_s[:, None] * cos_phase, axis=0)
    d_mod_b_d_theta = -jnp.sum(bmnc_b[:, None] * m_sin, axis=0)
    d_mod_b_d_phi = jnp.sum(bmnc_b[:, None] * n_sin, axis=0)
    d_r_b_d_s = jnp.sum(d_rmnc_b_d_s[:, None] * cos_phase, axis=0)
    d_r_b_d_theta = -jnp.sum(rmnc_b[:, None] * m_sin, axis=0)
    d_r_b_d_phi = jnp.sum(rmnc_b[:, None] * n_sin, axis=0)
    d_z_b_d_s = jnp.sum(d_zmns_b_d_s[:, None] * sin_phase, axis=0)
    d_z_b_d_theta = jnp.sum(zmns_b[:, None] * m_cos, axis=0)
    d_z_b_d_phi = -jnp.sum(zmns_b[:, None] * n_cos, axis=0)
    nu_b = jnp.sum(numns_b[:, None] * sin_phase, axis=0)
    d_nu_b_d_s = jnp.sum(d_numns_b_d_s[:, None] * sin_phase, axis=0)
    d_nu_b_d_theta = jnp.sum(numns_b[:, None] * m_cos, axis=0)
    d_nu_b_d_phi = -jnp.sum(numns_b[:, None] * n_cos, axis=0)

    phi_cyl = phi_b - nu_b
    sin_phi = jnp.sin(phi_cyl)
    cos_phi = jnp.cos(phi_cyl)
    d_x_d_theta = d_r_b_d_theta * cos_phi - r_b * sin_phi * (-d_nu_b_d_theta)
    d_x_d_phi = d_r_b_d_phi * cos_phi - r_b * sin_phi * (1.0 - d_nu_b_d_phi)
    d_x_d_s = d_r_b_d_s * cos_phi - r_b * sin_phi * (-d_nu_b_d_s)
    d_y_d_theta = d_r_b_d_theta * sin_phi + r_b * cos_phi * (-d_nu_b_d_theta)
    d_y_d_phi = d_r_b_d_phi * sin_phi + r_b * cos_phi * (1.0 - d_nu_b_d_phi)
    d_y_d_s = d_r_b_d_s * sin_phi + r_b * cos_phi * (-d_nu_b_d_s)

    grad_psi_x = (d_y_d_theta * d_z_b_d_phi - d_z_b_d_theta * d_y_d_phi) / sqrt_g_booz
    grad_psi_y = (d_z_b_d_theta * d_x_d_phi - d_x_d_theta * d_z_b_d_phi) / sqrt_g_booz
    grad_psi_z = (d_x_d_theta * d_y_d_phi - d_y_d_theta * d_x_d_phi) / sqrt_g_booz
    g_sup_psi_psi = grad_psi_x**2 + grad_psi_y**2 + grad_psi_z**2
    g_sup_psi_psi_safe = jnp.maximum(
        g_sup_psi_psi, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )

    etf = jnp.asarray(edge_toroidal_flux_over_2pi, dtype=base_Rcos.dtype)
    grad_theta_x = (d_y_d_phi * d_z_b_d_s - d_z_b_d_phi * d_y_d_s) / (sqrt_g_booz * etf)
    grad_theta_y = (d_z_b_d_phi * d_x_d_s - d_x_d_phi * d_z_b_d_s) / (sqrt_g_booz * etf)
    grad_theta_z = (d_x_d_phi * d_y_d_s - d_y_d_phi * d_x_d_s) / (sqrt_g_booz * etf)
    grad_phi_x = (d_y_d_s * d_z_b_d_theta - d_z_b_d_s * d_y_d_theta) / (
        sqrt_g_booz * etf
    )
    grad_phi_y = (d_z_b_d_s * d_x_d_theta - d_x_d_s * d_z_b_d_theta) / (
        sqrt_g_booz * etf
    )
    grad_phi_z = (d_x_d_s * d_y_d_theta - d_y_d_s * d_x_d_theta) / (sqrt_g_booz * etf)
    zeta_center = -jnp.asarray(float(alpha), dtype=base_Rcos.dtype) / iota_safe
    shear_phase = phi_b - zeta_center
    grad_alpha_x = (
        -shear_phase * d_iota_ds * grad_psi_x / etf
        + grad_theta_x
        - iota_safe * grad_phi_x
    )
    grad_alpha_y = (
        -shear_phase * d_iota_ds * grad_psi_y / etf
        + grad_theta_y
        - iota_safe * grad_phi_y
    )
    grad_alpha_z = (
        -shear_phase * d_iota_ds * grad_psi_z / etf
        + grad_theta_z
        - iota_safe * grad_phi_z
    )
    grad_alpha_dot_grad_psi = (
        grad_alpha_x * grad_psi_x
        + grad_alpha_y * grad_psi_y
        + grad_alpha_z * grad_psi_z
    )
    local_shear_l1 = grad_alpha_dot_grad_psi / g_sup_psi_psi_safe
    s_arr = jnp.asarray(s_value, dtype=base_Rcos.dtype)
    L = jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
    Bref = jnp.asarray(float(B_reference), dtype=base_Rcos.dtype)
    shat_metric = s_hat
    metric_bmag_sq = mod_b_safe * mod_b_safe
    gds2_raw = (
        (metric_bmag_sq / g_sup_psi_psi_safe + g_sup_psi_psi_safe * local_shear_l1**2)
        * L
        * L
        * s_arr
    )
    gds21_raw = g_sup_psi_psi_safe * local_shear_l1 * shat_metric / Bref
    gds22_raw = (
        g_sup_psi_psi_safe * shat_metric * shat_metric / (L * L * Bref * Bref * s_arr)
    )
    grho_raw = jnp.sqrt(g_sup_psi_psi_safe / (L * L * Bref * Bref * s_arr))

    boozer_current_sum = boozer_g + iota_safe * boozer_i
    d_sqrt_g_booz_d_theta = (
        -2.0 * boozer_current_sum * d_mod_b_d_theta / (mod_b_safe**3)
    )
    d_sqrt_g_booz_d_phi = -2.0 * boozer_current_sum * d_mod_b_d_phi / (mod_b_safe**3)
    curvature_numerator = (
        boozer_g * d_sqrt_g_booz_d_theta - boozer_i * d_sqrt_g_booz_d_phi
    )
    curvature_denom = 2.0 * sqrt_g_booz * boozer_current_sum
    eps = jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    curvature_denom_safe = jnp.where(
        jnp.abs(curvature_denom) < eps,
        jnp.sign(curvature_denom + eps) * eps,
        curvature_denom,
    )
    etf_safe = jnp.where(jnp.abs(etf) < eps, jnp.sign(etf + eps) * eps, etf)
    kappa_g = curvature_numerator / curvature_denom_safe
    local_shear_l0 = -(local_shear_l1 + d_iota_ds / etf_safe * shear_phase)
    kappa_n = d_mod_b_d_s / (mod_b_safe * etf_safe) + local_shear_l0 * kappa_g
    b_cross_kappa_dot_grad_alpha = (kappa_n + kappa_g * local_shear_l1) * metric_bmag_sq
    b_cross_kappa_dot_grad_psi = kappa_g * metric_bmag_sq
    toroidal_flux_sign = jnp.sign(etf)
    sqrt_s = jnp.sqrt(s_arr)
    drift_cvdrift0_raw = (
        -b_cross_kappa_dot_grad_psi
        * 2.0
        * shat_metric
        / jnp.maximum(metric_bmag_sq * sqrt_s, eps)
        * toroidal_flux_sign
    )
    drift_cvdrift_raw = (
        -2.0
        * Bref
        * L
        * L
        * sqrt_s
        * b_cross_kappa_dot_grad_alpha
        / metric_bmag_sq
        * toroidal_flux_sign
    )
    # Root-level VMEC/EIK drift coefficients are stored at the pre-loader (2x)
    # level; SPECTRAX-GK compares against the loaded solver convention.
    drift_loader_factor = jnp.asarray(0.5, dtype=base_Rcos.dtype)
    gds2 = jnp.interp(theta_uniform_closed, theta_eqarc, gds2_raw)[:-1]
    gds21 = jnp.interp(theta_uniform_closed, theta_eqarc, gds21_raw)[:-1]
    gds22 = jnp.interp(theta_uniform_closed, theta_eqarc, gds22_raw)[:-1]
    grho = jnp.interp(theta_uniform_closed, theta_eqarc, grho_raw)[:-1]
    cvdrift = (
        drift_loader_factor
        * jnp.interp(theta_uniform_closed, theta_eqarc, drift_cvdrift_raw)[:-1]
    )
    gbdrift = cvdrift
    cvdrift0 = (
        drift_loader_factor
        * jnp.interp(theta_uniform_closed, theta_eqarc, drift_cvdrift0_raw)[:-1]
    )
    gbdrift0 = cvdrift0

    return {
        "theta": theta,
        "theta_equal_arc_closed": theta_eqarc,
        "theta_uniform_closed": theta_uniform_closed,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": bgrad,
        "jacobian": jacobian,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cvdrift,
        "gbdrift": gbdrift,
        "cvdrift0": cvdrift0,
        "gbdrift0": gbdrift0,
        "grho": grho,
        "q": 1.0
        / jnp.maximum(jnp.abs(iota_safe), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "s_hat": s_hat,
        "iota": iota,
        "torflux": float(s_value),
        "surface_index": int(sidx),
        "reference_length": float(L_reference),
        "reference_b": float(B_reference),
        "mboz": mboz_int,
        "nboz": nboz_int,
        "surface_stencil_width": None
        if surface_stencil_width is None
        else int(surface_stencil_width),
        "boozer_surface_indices": None
        if surface_indices is None
        else [int(x) for x in np.asarray(surface_indices)],
        "field_line_convention": "Boozer theta, alpha=theta-iota*zeta, equal-arc remap",
        "scope": (
            "Boozer equal-arc bmag/gradpar/Jacobian plus zero-beta metric/drift parity; "
            "finite-beta pressure corrections and broad-equilibrium drift gates remain open"
        ),
    }


def flux_tube_geometry_from_vmec_boozer_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    torflux: float | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    jit: bool = False,
    surface_stencil_width: int | None = None,
    reference_length: float | None = None,
    reference_b: float | None = None,
    source_model: str = "mode21_vmec_boozer_state",
    validate_finite: bool = True,
) -> FluxTubeGeometryData:
    """Build solver-ready geometry directly from a solved ``vmec_jax`` state.

    This is the production-facing in-memory bridge for differentiable
    optimization workflows. It keeps the path inside JAX-compatible objects:

    ``VMECState -> BoozXformInputs -> booz_xform_jax -> FluxTubeGeometryData``.

    Runtime VMEC file generation can still use the NetCDF/EIK route, but
    differentiable stellarator optimization should call this function or a
    higher-level objective wrapper around it so gradients never pass through
    filesystem artifacts.
    """

    mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
        state,
        static,
        indata,
        wout,
        surface_index=surface_index,
        torflux=torflux,
        alpha=alpha,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        jit=jit,
        surface_stencil_width=surface_stencil_width,
        reference_length=reference_length,
        reference_b=reference_b,
    )
    return flux_tube_geometry_from_mapping(
        mapping,
        source_model=source_model,
        validate_finite=validate_finite,
    )


def vmec_jax_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 24,
    fd_step: float = 2.0e-6,
) -> dict[str, object]:
    """AD/FD-check VMEC-state derivatives through a solver-ready flux tube.

    Unlike the Boozer-only bridge, this report starts from a real
    ``vmec_jax`` state, evaluates VMEC metric and magnetic-field tensors, emits
    the SPECTRAX-GK ``FluxTubeGeometryData`` mapping, and differentiates
    geometry observables through that path.
    """

    p = jnp.asarray([1.0e-4, 1.0e-4] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")

        def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            return vmec_jax_flux_tube_mapping_from_state(
                traced_state,
                static,
                wout,
                surface_index=surface_index,
                alpha=float(alpha),
                ntheta=int(ntheta),
            )

        sensitivity = geometry_sensitivity_report(
            mapping_fn,
            p,
            fd_step=float(fd_step),
            source_model="vmec_jax:state->tensor-flux-tube",
        )
        mapping = mapping_fn(p)
        vmec_meta = mapping["vmec_jax"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(p).tolist(),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(vmec_meta["surface_index"]),
        "iota": float(np.asarray(vmec_meta["iota"])),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "sensitivity": sensitivity,
        "fd_step": float(fd_step),
        "reference_length": float(vmec_meta["reference_length"]),
        "reference_b": float(vmec_meta["reference_b"]),
        "field_line_convention": str(vmec_meta["field_line_convention"]),
        "scope": (
            "VMEC-native metric and grad-B flux-tube mapping; full imported-VMEC/EIK "
            "drift parity and production transport-gradient gates remain separate."
        ),
    }


def vmec_jax_flux_tube_array_parity_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 16,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    boundary: str = "none",
    include_shear_variation: bool = True,
    include_pressure_variation: bool = True,
    core_tolerance: float = 5.0e-2,
    scalar_tolerance: float = 5.0e-3,
    equal_arc_core_tolerance: float = 1.0e-2,
    equal_arc_derivative_tolerance: float = 3.0e-2,
    equal_arc_metric_tolerance: float = 8.0e-2,
    equal_arc_drift_tolerance: float = 8.0e-2,
) -> dict[str, object]:
    """Compare the direct ``vmec_jax`` flux-tube arrays to imported VMEC/EIK.

    This is a diagnostic promotion gate, not a differentiability check.  It
    starts from the same real ``vmec_jax`` example state used by
    :func:`vmec_jax_flux_tube_sensitivity_report`, builds the direct
    VMEC-tensor-derived flux-tube mapping, then generates the existing imported
    VMEC/EIK geometry on the same surface and compares solver-facing arrays.

    The expected current result is that ``q`` and magnetic shear are close
    while metric/drift arrays remain open because the direct path still uses a
    VMEC-coordinate/equal-theta convention and a local grad-:math:`B` closure
    instead of the production Boozer equal-arc/Hegna-Nakajima convention.
    """

    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")
    mboz_int = int(mboz)
    nboz_int = int(nboz)
    if (
        mboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
        or nboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
    ):
        raise ValueError(
            "mboz and nboz must both be >= "
            f"{_VMEC_BOOZER_PARITY_MIN_MODE_COUNT} for VMEC/Boozer parity gates"
        )

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return {
            "available": False,
            "backend_info": info,
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        from spectraxgk.from_gx.vmec import (
            generate_vmec_eik_internal,
            internal_vmec_backend_available,
        )
        from spectraxgk.geometry import load_gx_geometry_netcdf

        if not internal_vmec_backend_available():
            return {
                "available": False,
                "backend_info": info,
                "case_name": str(case_name),
                "reason": "internal VMEC/EIK backend is not available",
            }

        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        ns = int(jnp.asarray(state.Rcos).shape[0])
        sidx = (
            max(1, min(ns // 2, ns - 2))
            if surface_index is None
            else int(surface_index)
        )
        torflux = float(sidx) / float(max(ns - 1, 1))
        direct_mapping = vmec_jax_flux_tube_mapping_from_state(
            state,
            static,
            wout,
            surface_index=sidx,
            alpha=float(alpha),
            ntheta=ntheta_int,
        )
        direct = flux_tube_geometry_from_mapping(
            direct_mapping,
            source_model="vmec_jax:state->tensor-flux-tube",
            validate_finite=False,
        )

        request = SimpleNamespace(
            vmec_file=str(wout_path),
            ntheta=ntheta_int,
            boundary=str(boundary),
            y0=10.0,
            x0=None,
            jtwist=None,
            beta=0.0,
            alpha=float(alpha),
            torflux=torflux,
            npol=1.0,
            npol_min=None,
            isaxisym=False,
            which_crossing=None,
            include_shear_variation=bool(include_shear_variation),
            include_pressure_variation=bool(include_pressure_variation),
            betaprim=None,
            z=(1.0, -1.0),
            mass=(1.0, 2.7e-4),
            dens=(1.0, 1.0),
            temp=(1.0, 1.0),
            tprim=(3.0, 0.0),
            fprim=(1.0, 0.0),
            vnewk=(0.0, 0.0),
            species_type=("ion", "electron"),
        )
        with tempfile.TemporaryDirectory(prefix="spectrax_vmec_eik_parity_") as tmp:
            eik_path = Path(tmp) / f"{case_name}.eik.nc"
            generate_vmec_eik_internal(output_path=eik_path, request=request)
            imported = load_gx_geometry_netcdf(eik_path)
            if imported.theta.shape[0] == direct.theta.shape[0] + 1:
                imported = imported.trim_terminal_theta_point()

        array_pairs = {
            "theta": (direct.theta, imported.theta),
            "bmag": (direct.bmag_profile, imported.bmag_profile),
            "bgrad": (direct.bgrad_profile, imported.bgrad_profile),
            "gds2": (direct.gds2_profile, imported.gds2_profile),
            "gds21": (direct.gds21_profile, imported.gds21_profile),
            "gds22": (direct.gds22_profile, imported.gds22_profile),
            "cvdrift": (direct.cv_profile, imported.cv_profile),
            "gbdrift": (direct.gb_profile, imported.gb_profile),
            "cvdrift0": (direct.cv0_profile, imported.cv0_profile),
            "gbdrift0": (direct.gb0_profile, imported.gb0_profile),
            "jacobian": (direct.jacobian_profile, imported.jacobian_profile),
            "grho": (direct.grho_profile, imported.grho_profile),
        }
        array_metrics = {
            name: _array_parity_metrics(candidate, reference)
            for name, (candidate, reference) in array_pairs.items()
        }
        scalar_metrics = {
            "gradpar": _scalar_parity_metrics(
                direct.gradpar_value, imported.gradpar_value
            ),
            "q": _scalar_parity_metrics(direct.q, imported.q),
            "s_hat": _scalar_parity_metrics(direct.s_hat, imported.s_hat),
        }
        core_names = (
            "bmag",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "jacobian",
            "grho",
        )
        core_values: list[float] = []
        for name in core_names:
            metrics = array_metrics[name]
            if not bool(metrics.get("shape_match", False)):
                continue
            raw_value = metrics.get("normalized_max_abs")
            core_values.append(
                float(raw_value)
                if isinstance(raw_value, int | float | np.floating)
                else np.inf
            )
        worst_core = max(core_values) if core_values else np.inf
        worst_scalar = max(float(values["rel"]) for values in scalar_metrics.values())
        production_parity_passed = bool(
            worst_core <= float(core_tolerance)
            and worst_scalar <= float(scalar_tolerance)
        )

        equal_arc_core_error = None
        equal_arc_core_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_metric_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_drift_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_core_scalar_metrics: dict[str, dict[str, float]] = {}
        equal_arc_core_worst = np.inf
        equal_arc_core_worst_scalar = np.inf
        equal_arc_derivative_worst = np.inf
        equal_arc_metric_worst = np.inf
        equal_arc_drift_worst = np.inf
        equal_arc_core_passed = False
        equal_arc_derivative_passed = False
        equal_arc_metric_passed = False
        equal_arc_drift_passed = False
        if bool(info.get("booz_xform_jax_api_available", False)):
            try:
                equal_arc_core = vmec_jax_boozer_equal_arc_core_profiles_from_state(
                    state,
                    static,
                    indata,
                    wout,
                    surface_index=sidx,
                    torflux=torflux,
                    alpha=float(alpha),
                    ntheta=ntheta_int,
                    mboz=mboz_int,
                    nboz=nboz_int,
                    jit=False,
                )
                equal_arc_core_pairs = {
                    "theta": (equal_arc_core["theta"], imported.theta),
                    "bmag": (equal_arc_core["bmag"], imported.bmag_profile),
                    "bgrad": (equal_arc_core["bgrad"], imported.bgrad_profile),
                    "jacobian": (equal_arc_core["jacobian"], imported.jacobian_profile),
                }
                equal_arc_metric_pairs = {
                    "gds2": (equal_arc_core["gds2"], imported.gds2_profile),
                    "gds21": (equal_arc_core["gds21"], imported.gds21_profile),
                    "gds22": (equal_arc_core["gds22"], imported.gds22_profile),
                    "grho": (equal_arc_core["grho"], imported.grho_profile),
                }
                equal_arc_drift_pairs = {
                    "cvdrift": (equal_arc_core["cvdrift"], imported.cv_profile),
                    "gbdrift": (equal_arc_core["gbdrift"], imported.gb_profile),
                    "cvdrift0": (equal_arc_core["cvdrift0"], imported.cv0_profile),
                    "gbdrift0": (equal_arc_core["gbdrift0"], imported.gb0_profile),
                }
                equal_arc_core_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_core_pairs.items()
                }
                equal_arc_metric_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_metric_pairs.items()
                }
                equal_arc_drift_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_drift_pairs.items()
                }
                equal_arc_core_scalar_metrics = {
                    "gradpar": _scalar_parity_metrics(
                        jnp.asarray(equal_arc_core["gradpar"])[0],
                        imported.gradpar_value,
                    ),
                    "q": _scalar_parity_metrics(equal_arc_core["q"], imported.q),
                    "s_hat": _scalar_parity_metrics(
                        equal_arc_core["s_hat"], imported.s_hat
                    ),
                }
                equal_arc_core_names = ("theta", "bmag", "jacobian")
                equal_arc_values: list[float] = []
                for name in equal_arc_core_names:
                    metrics = equal_arc_core_array_metrics[name]
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    equal_arc_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_core_worst = (
                    max(equal_arc_values) if equal_arc_values else np.inf
                )
                equal_arc_core_worst_scalar = max(
                    float(values["rel"])
                    for values in equal_arc_core_scalar_metrics.values()
                )
                derivative_metrics = equal_arc_core_array_metrics["bgrad"]
                raw_derivative = derivative_metrics.get("normalized_max_abs")
                equal_arc_derivative_worst = (
                    float(raw_derivative)
                    if isinstance(raw_derivative, int | float | np.floating)
                    else np.inf
                )
                equal_arc_core_passed = bool(
                    equal_arc_core_worst <= float(equal_arc_core_tolerance)
                    and equal_arc_core_worst_scalar <= float(equal_arc_core_tolerance)
                )
                equal_arc_derivative_passed = bool(
                    equal_arc_derivative_worst <= float(equal_arc_derivative_tolerance)
                )
                metric_values: list[float] = []
                for metrics in equal_arc_metric_array_metrics.values():
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    metric_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_metric_worst = max(metric_values) if metric_values else np.inf
                equal_arc_metric_passed = bool(
                    equal_arc_metric_worst <= float(equal_arc_metric_tolerance)
                )
                drift_values: list[float] = []
                for metrics in equal_arc_drift_array_metrics.values():
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    drift_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_drift_worst = max(drift_values) if drift_values else np.inf
                equal_arc_drift_passed = bool(
                    equal_arc_drift_worst <= float(equal_arc_drift_tolerance)
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - optional-backend diagnostic detail
                equal_arc_core_error = f"{type(exc).__name__}: {exc}"
        else:
            equal_arc_core_error = "booz_xform_jax functional API is not available"
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": int(sidx),
        "torflux": float(torflux),
        "alpha": float(alpha),
        "ntheta": int(ntheta_int),
        "mboz": mboz_int,
        "nboz": nboz_int,
        "boundary": str(boundary),
        "include_shear_variation": bool(include_shear_variation),
        "include_pressure_variation": bool(include_pressure_variation),
        "array_metrics": array_metrics,
        "scalar_metrics": scalar_metrics,
        "equal_arc_core_array_metrics": equal_arc_core_array_metrics,
        "equal_arc_metric_array_metrics": equal_arc_metric_array_metrics,
        "equal_arc_drift_array_metrics": equal_arc_drift_array_metrics,
        "equal_arc_core_scalar_metrics": equal_arc_core_scalar_metrics,
        "equal_arc_core_worst_normalized_max_abs": float(equal_arc_core_worst),
        "equal_arc_core_worst_scalar_rel": float(equal_arc_core_worst_scalar),
        "equal_arc_derivative_worst_normalized_max_abs": float(
            equal_arc_derivative_worst
        ),
        "equal_arc_metric_worst_normalized_max_abs": float(equal_arc_metric_worst),
        "equal_arc_drift_worst_normalized_max_abs": float(equal_arc_drift_worst),
        "equal_arc_core_tolerance": float(equal_arc_core_tolerance),
        "equal_arc_derivative_tolerance": float(equal_arc_derivative_tolerance),
        "equal_arc_metric_tolerance": float(equal_arc_metric_tolerance),
        "equal_arc_drift_tolerance": float(equal_arc_drift_tolerance),
        "equal_arc_core_passed": bool(equal_arc_core_passed),
        "equal_arc_derivative_passed": bool(equal_arc_derivative_passed),
        "equal_arc_metric_passed": bool(equal_arc_metric_passed),
        "equal_arc_drift_passed": bool(equal_arc_drift_passed),
        "equal_arc_core_error": equal_arc_core_error,
        "worst_core_normalized_max_abs": float(worst_core),
        "worst_scalar_rel": float(worst_scalar),
        "core_tolerance": float(core_tolerance),
        "scalar_tolerance": float(scalar_tolerance),
        "production_parity_passed": production_parity_passed,
        "status": "passed" if production_parity_passed else "diagnostic_open",
        "interpretation": (
            "The direct VMEC tensor path proves state-level differentiability. "
            "The Boozer equal-arc core audit narrows the production gap to the "
            "full imported VMEC/EIK metric and drift convention when it passes."
        ),
    }


def geometry_sensitivity_report(
    mapping_fn: Any,
    params: jnp.ndarray,
    *,
    fd_step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    source_model: str = "vmec_jax:in-memory",
) -> dict[str, object]:
    """Validate geometry-observable sensitivities by AD and finite differences.

    ``mapping_fn(params)`` must return the solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The report is strict
    JSON friendly so examples and CI gates can preserve the derivative
    contract without depending on large VMEC solves.
    """

    p = jnp.asarray(params, dtype=jnp.float64)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)

    param_names = tuple(f"param_{idx}" for idx in range(int(p.shape[0])))

    report = observable_gradient_validation_report(
        observable_fn,
        p,
        fd_step=float(fd_step),
        rtol=float(rtol),
        atol=float(atol),
        observable_names=_GEOMETRY_OBSERVABLE_NAMES,
        param_names=param_names,
        relative_floor=1.0e-12,
        report_kind="geometry_sensitivity_ad_fd_gate",
    )
    report["source_model"] = str(source_model)
    return report


def geometry_inverse_design_report(
    mapping_fn: Any,
    initial_params: jnp.ndarray,
    target_observables: jnp.ndarray,
    *,
    observable_indices: Sequence[int] | None = None,
    max_steps: int = 8,
    damping: float = 1.0e-8,
    fd_step: float = 1.0e-4,
    regularization: float = 1.0e-8,
    source_model: str = "vmec_jax:in-memory",
) -> dict[str, object]:
    """Run a small Gauss-Newton geometry inverse-design validation.

    ``mapping_fn(params)`` must be the same solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The routine is meant
    for differentiable ``vmec_jax`` / ``booz_xform_jax`` workflows: it keeps
    the optimization, sensitivity check, and local UQ covariance in one
    JSON-friendly report so examples can validate the full AD contract without
    depending on a long equilibrium solve in CI.
    """

    params = jnp.asarray(initial_params, dtype=_jax_float_dtype())
    if params.ndim != 1:
        raise ValueError("initial_params must be one-dimensional")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")
    if float(damping) < 0.0:
        raise ValueError("damping must be non-negative")

    if observable_indices is None:
        indices_np = np.arange(len(_GEOMETRY_OBSERVABLE_NAMES), dtype=int)
    else:
        indices_np = np.asarray(list(observable_indices), dtype=int)
    if indices_np.ndim != 1 or indices_np.size == 0:
        raise ValueError(
            "observable_indices must be a non-empty one-dimensional sequence"
        )
    if np.any(indices_np < 0) or np.any(indices_np >= len(_GEOMETRY_OBSERVABLE_NAMES)):
        raise ValueError("observable_indices contains an out-of-range observable index")

    target = jnp.asarray(target_observables, dtype=params.dtype)
    if target.ndim != 1 or int(target.shape[0]) != int(indices_np.size):
        raise ValueError("target_observables length must match observable_indices")
    indices = jnp.asarray(indices_np, dtype=jnp.int32)

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)[indices]

    history: list[dict[str, object]] = []
    p = params
    residual = observable_fn(p) - target
    for step in range(int(max_steps) + 1):
        obs = observable_fn(p)
        residual = obs - target
        objective = 0.5 * jnp.dot(residual, residual)
        history.append(
            {
                "step": int(step),
                "params": np.asarray(p).tolist(),
                "observables": np.asarray(obs).tolist(),
                "objective": float(objective),
                "residual_norm": float(jnp.linalg.norm(residual)),
            }
        )
        if step == int(max_steps):
            break
        jac = jax.jacfwd(observable_fn)(p)
        normal = jac.T @ jac + float(damping) * jnp.eye(int(p.shape[0]), dtype=p.dtype)
        delta = jnp.linalg.solve(normal, jac.T @ residual)
        p = p - delta

    jac_ad = jax.jacfwd(observable_fn)(p)
    jac_fd = finite_difference_jacobian(observable_fn, p, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)
    uq = covariance_diagnostics(
        np.asarray(jac_ad), np.asarray(residual), regularization=regularization
    )
    selected_observable_names = [
        str(_GEOMETRY_OBSERVABLE_NAMES[int(i)]) for i in indices_np
    ]
    param_names = tuple(f"param_{idx}" for idx in range(int(params.shape[0])))

    return {
        "observable_names": selected_observable_names,
        "initial_params": np.asarray(params).tolist(),
        "final_params": np.asarray(p).tolist(),
        "target_observables": np.asarray(target).tolist(),
        "final_observables": np.asarray(observable_fn(p)).tolist(),
        "final_residual": np.asarray(residual).tolist(),
        "final_residual_norm": float(jnp.linalg.norm(residual)),
        "history": history,
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(
            np.max(np.abs(np.asarray(diff) / np.asarray(scale)))
        ),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            p,
            fd_step=float(fd_step),
            observable_names=selected_observable_names,
            param_names=param_names,
            relative_floor=1.0e-12,
        ),
        "uq": uq,
        "fd_step": float(fd_step),
        "damping": float(damping),
        "regularization": float(regularization),
        "source_model": str(source_model),
        "backend_info": discover_differentiable_geometry_backends(),
    }


__all__ = [
    "booz_xform_flux_tube_mapping_from_inputs",
    "booz_xform_flux_tube_sensitivity_report",
    "booz_xform_spectral_sensitivity_report",
    "discover_differentiable_geometry_backends",
    "evaluate_boozer_bmag_on_field_line",
    "finite_difference_jacobian",
    "flux_tube_geometry_from_mapping",
    "flux_tube_geometry_from_vmec_boozer_state",
    "flux_tube_geometry_observables",
    "geometry_inverse_design_report",
    "geometry_observable_names",
    "geometry_sensitivity_report",
    "observable_gradient_validation_report",
    "vmec_jax_boozer_flux_tube_sensitivity_report",
    "vmec_jax_boozer_equal_arc_core_profiles_from_state",
    "vmec_jax_field_line_tensor_sensitivity_report",
    "vmec_jax_flux_tube_array_parity_report",
    "vmec_jax_flux_tube_mapping_from_state",
    "vmec_jax_flux_tube_sensitivity_report",
    "vmec_jax_metric_tensor_sensitivity_report",
    "vmec_boundary_aspect_sensitivity_report",
    "vmec_field_line_tensor_observable_names",
    "vmec_metric_tensor_observable_names",
]
