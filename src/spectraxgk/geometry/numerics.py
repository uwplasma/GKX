"""Pure numerical helpers for differentiable geometry bridge routines."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


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


def _interp_equal_arc_profile(
    theta_uniform_closed: jnp.ndarray,
    theta_equal_arc_closed: jnp.ndarray,
    values_closed: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate onto the equal-arc grid through the coordinate map.

    The Boozer transform feeding this remap must use safe divisions in inactive
    Fourier branches; otherwise nonfinite upstream cotangents contaminate the
    moving equal-arc coordinate sensitivity. Sparse boundary finite-difference
    gates remain the promotion criterion for VMEC-JAX transport gradients.
    """

    values = jnp.asarray(values_closed)
    theta_uniform = jnp.asarray(theta_uniform_closed, dtype=values.dtype)
    theta_equal_arc = jnp.asarray(theta_equal_arc_closed, dtype=values.dtype)
    return jnp.interp(theta_uniform, theta_equal_arc, values)


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


__all__ = [
    "_array_parity_metrics",
    "_boozer_half_mesh_s_grid",
    "_cumulative_trapezoid",
    "_evaluate_boozer_cosine_series_on_field_line",
    "_interp_equal_arc_profile",
    "_interp_radial",
    "_periodic_bilinear_sample_2d",
    "_radial_derivative_array",
    "_radial_derivative_profile",
    "_scalar_parity_metrics",
]
