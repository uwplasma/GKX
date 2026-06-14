"""Solver-ready flux-tube geometry contract for differentiable backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryData
from spectraxgk.geometry.backend_discovery import _is_traced

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
        bessel_bmag_power=float(data["bessel_bmag_power"])
        if "bessel_bmag_power" in data
        else 0.0,
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


__all__ = [
    "_ARRAY_FIELDS",
    "_GEOMETRY_OBSERVABLE_NAMES",
    "_VMEC_FIELD_LINE_OBSERVABLE_NAMES",
    "_VMEC_METRIC_OBSERVABLE_NAMES",
    "_array",
    "_scalar",
    "flux_tube_geometry_from_mapping",
    "flux_tube_geometry_observables",
    "geometry_observable_names",
    "vmec_field_line_tensor_observable_names",
    "vmec_metric_tensor_observable_names",
]
