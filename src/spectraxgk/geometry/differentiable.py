"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
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
    vmec_paths = _candidate_paths(
        ("SPECTRAX_VMEC_JAX_PATH", "VMEC_JAX_PATH"),
        (repo_parent / "vmec_jax", Path("/Users/rogeriojorge/local/vmec_jax")),
    )
    booz_paths = _candidate_paths(
        ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"),
        (repo_parent / "booz_xform_jax", Path("/Users/rogeriojorge/local/booz_xform_jax")),
    )
    vmec = _find_importable_module("vmec_jax", vmec_paths)
    booz = _find_importable_module("booz_xform_jax", booz_paths)
    booz_jax_api = None if booz is None else _find_importable_module("booz_xform_jax.jax_api", booz_paths)

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
        raise ValueError(f"{key} length {arr.shape[0]} does not match theta length {ntheta}")
    if validate_finite and not _is_traced(arr) and not bool(np.all(np.isfinite(np.asarray(arr)))):
        raise ValueError(f"{key} contains non-finite values")
    return arr


def _scalar(mapping: Mapping[str, Any], key: str, default: float) -> Any:
    value = mapping.get(key, default)
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{key} must be scalar")
    if _is_traced(arr):
        return arr
    return float(np.asarray(arr))


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
    grho = _array(data, "grho", ntheta, validate_finite=validate_finite) if "grho" in data else jnp.ones_like(theta)

    gradpar_value: Any
    if _is_traced(arrays["gradpar"]):
        gradpar_value = jnp.mean(arrays["gradpar"])
    else:
        gradpar_values = np.asarray(arrays["gradpar"])
        gradpar_value = float(np.mean(gradpar_values))
        if validate_finite and not np.allclose(gradpar_values, gradpar_value, rtol=1.0e-5, atol=1.0e-7):
            raise ValueError("gradpar must be constant along the sampled field line")

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
        q=_scalar(data, "q", 1.0),
        s_hat=_scalar(data, "s_hat", data.get("shat", 0.0)),
        epsilon=_scalar(data, "epsilon", 0.0),
        R0=_scalar(data, "R0", 1.0),
        B0=_scalar(data, "B0", 1.0),
        alpha=_scalar(data, "alpha", 0.0),
        drift_scale=_scalar(data, "drift_scale", 1.0),
        kxfac=_scalar(data, "kxfac", 1.0),
        theta_scale=_scalar(data, "theta_scale", 1.0),
        nfp=int(data.get("nfp", 1)),
        kperp2_bmag=bool(data.get("kperp2_bmag", True)),
        bessel_bmag_power=_scalar(data, "bessel_bmag_power", 0.0),
        source_model=str(source_model),
        theta_closed_interval=bool(data.get("theta_closed_interval", False)),
    )


def geometry_observable_names() -> tuple[str, ...]:
    """Return the ordered geometry observables used by bridge AD checks."""

    return _GEOMETRY_OBSERVABLE_NAMES


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
    ripple = jnp.sqrt(jnp.sum(weights * (bmag / jnp.maximum(jnp.abs(mean_b), 1.0e-300) - 1.0) ** 2))
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


def finite_difference_jacobian(fn: Any, params: jnp.ndarray, *, step: float = 1.0e-4) -> jnp.ndarray:
    """Central finite-difference Jacobian for small validation problems."""

    p = jnp.asarray(params, dtype=jnp.float64)
    h = float(step)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    columns = []
    for idx in range(int(p.shape[0])):
        basis = jnp.zeros_like(p).at[idx].set(h)
        columns.append((jnp.asarray(fn(p + basis)) - jnp.asarray(fn(p - basis))) / (2.0 * h))
    return jnp.stack(columns, axis=1)


def geometry_sensitivity_report(
    mapping_fn: Any,
    params: jnp.ndarray,
    *,
    fd_step: float = 1.0e-4,
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

    obs = observable_fn(p)
    jac_ad = jax.jacfwd(observable_fn)(p)
    jac_fd = finite_difference_jacobian(observable_fn, p, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)

    return {
        "observable_names": list(_GEOMETRY_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(obs).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(np.max(np.abs(np.asarray(diff) / np.asarray(scale)))),
        "fd_step": float(fd_step),
        "source_model": str(source_model),
    }


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

    params = jnp.asarray(initial_params, dtype=jnp.float64)
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
        raise ValueError("observable_indices must be a non-empty one-dimensional sequence")
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
    uq = covariance_diagnostics(np.asarray(jac_ad), np.asarray(residual), regularization=regularization)

    return {
        "observable_names": [str(_GEOMETRY_OBSERVABLE_NAMES[int(i)]) for i in indices_np],
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
        "max_rel_ad_fd_error": float(np.max(np.abs(np.asarray(diff) / np.asarray(scale)))),
        "uq": uq,
        "fd_step": float(fd_step),
        "damping": float(damping),
        "regularization": float(regularization),
        "source_model": str(source_model),
        "backend_info": discover_differentiable_geometry_backends(),
    }


__all__ = [
    "discover_differentiable_geometry_backends",
    "finite_difference_jacobian",
    "flux_tube_geometry_from_mapping",
    "flux_tube_geometry_observables",
    "geometry_inverse_design_report",
    "geometry_observable_names",
    "geometry_sensitivity_report",
]
