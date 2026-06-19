"""Direct VMEC tensor to flux-tube mapping bridge."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.numerics import _periodic_bilinear_sample_2d
from spectraxgk.geometry.vmec_field_line_sampling import (
    _vmec_field_line_sampling_coordinates,
)


@dataclass(frozen=True)
class _Surface:
    base_Rcos: Any
    index: int
    s_value: Any
    ds: Any
    iota_line: Any
    iota_safe: Any
    shear: Any


@dataclass(frozen=True)
class _Scales:
    length: float
    b_ref: float


@dataclass(frozen=True)
class _Line:
    theta: Any
    theta_vmec: Any
    zeta: Any
    ntheta: int


def _safe_iota(iota: Any, dtype: Any) -> Any:
    return jnp.where(
        jnp.abs(iota) < 1.0e-12,
        jnp.sign(iota + 1.0e-30) * jnp.asarray(1.0e-12, dtype=dtype),
        iota,
    )


def _surface(state: Any, wout: Any, surface_index: int | None) -> _Surface:
    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos array must be two-dimensional")
    ns = int(base_Rcos.shape[0])
    if ns < 3:
        raise RuntimeError("vmec_jax state needs at least three radial surfaces")

    index = max(1, min(ns // 2, ns - 2)) if surface_index is None else int(surface_index)
    if not (0 < index < ns - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")

    iota_profile = jnp.asarray(getattr(wout, "iotas"))
    if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= index + 1:
        raise RuntimeError(
            "vmec_jax wout iotas profile is missing or incompatible with the state grid"
        )

    s_grid = jnp.linspace(0.0, 1.0, ns, dtype=base_Rcos.dtype)
    s_value = jnp.maximum(s_grid[index], jnp.asarray(1.0e-12, dtype=base_Rcos.dtype))
    ds = s_grid[1] - s_grid[0]
    iota_line = iota_profile[index]
    iota_safe = _safe_iota(iota_line, base_Rcos.dtype)
    d_iota_ds = (iota_profile[index + 1] - iota_profile[index - 1]) / (2.0 * ds)
    shear = -2.0 * s_value * d_iota_ds / iota_safe
    return _Surface(base_Rcos, index, s_value, ds, iota_line, iota_safe, shear)


def _finite_nonzero(value: float) -> float:
    return value if np.isfinite(value) and abs(value) > 0.0 else 1.0


def _reference_scales(
    wout: Any, reference_length: float | None, reference_b: float | None
) -> _Scales:
    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    length = _finite_nonzero(raw_length)
    if reference_b is not None:
        return _Scales(length, _finite_nonzero(float(reference_b)))

    phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
    edge_flux = abs(float(phi_profile[-1]) / (2.0 * np.pi))
    return _Scales(length, _finite_nonzero(2.0 * edge_flux / (length * length)))


def _field_line(wout: Any, surface: _Surface, *, alpha: float, ntheta: int) -> _Line:
    _iota, _iota_safe, theta, theta_vmec, zeta = _vmec_field_line_sampling_coordinates(
        wout,
        surface_index=surface.index,
        alpha=alpha,
        ntheta=ntheta,
        dtype=surface.base_Rcos.dtype,
    )
    return _Line(theta, theta_vmec, zeta, int(ntheta))


def _load_raw_vmec_tensors(
    state: Any,
    static: Any,
    wout: Any,
    surface: _Surface,
    scales: _Scales,
    *,
    b2_floor: float,
) -> tuple[Any, Any]:
    geom_mod = importlib.import_module("vmec_jax.geom")
    bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
    field_mod = importlib.import_module("vmec_jax.field")
    geom = geom_mod.eval_geom(state, static)
    bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        pres=getattr(wout, "pres", None),
    )
    b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
    floor = jnp.asarray(float(b2_floor), dtype=surface.base_Rcos.dtype)
    bmag_all = jnp.sqrt(jnp.maximum(jnp.asarray(b2), floor)) / scales.b_ref
    return geom, bmag_all


def _line_sample(values: Any, line: _Line) -> Any:
    return _periodic_bilinear_sample_2d(values, line.theta_vmec, line.zeta)


def _sample_line_tensors(
    geom: Any, bmag_all: Any, surface: _Surface, line: _Line
) -> dict[str, Any]:
    bmag_grid = bmag_all[surface.index]
    dtheta = 2.0 * jnp.pi / float(bmag_grid.shape[0])
    dzeta = 2.0 * jnp.pi / float(bmag_grid.shape[1])
    db_dtheta = (jnp.roll(bmag_grid, -1, 0) - jnp.roll(bmag_grid, 1, 0)) / (
        2.0 * dtheta
    )
    db_dzeta = (jnp.roll(bmag_grid, -1, 1) - jnp.roll(bmag_grid, 1, 1)) / (
        2.0 * dzeta
    )
    db_ds = (bmag_all[surface.index + 1] - bmag_all[surface.index - 1]) / (
        2.0 * surface.ds
    )
    g = {name: _line_sample(jnp.asarray(getattr(geom, name))[surface.index], line) for name in (
        "sqrtg",
        "g_ss",
        "g_st",
        "g_sp",
        "g_tt",
        "g_tp",
        "g_pp",
    )}
    cov_metric = jnp.stack(
        (
            jnp.stack((g["g_ss"], g["g_st"], g["g_sp"]), axis=-1),
            jnp.stack((g["g_st"], g["g_tt"], g["g_tp"]), axis=-1),
            jnp.stack((g["g_sp"], g["g_tp"], g["g_pp"]), axis=-1),
        ),
        axis=-2,
    )
    return {
        "bmag": _line_sample(bmag_grid, line),
        "sqrtg": g["sqrtg"],
        "contra": jnp.linalg.inv(cov_metric),
        "db_ds": _line_sample(db_ds, line),
        "db_dtheta": _line_sample(db_dtheta, line),
        "db_dzeta": _line_sample(db_dzeta, line),
    }


def _safe_shear(surface: _Surface) -> Any:
    return jnp.where(
        jnp.abs(surface.shear) < 1.0e-12,
        jnp.asarray(0.0, dtype=surface.base_Rcos.dtype),
        surface.shear,
    )


def _metric_profiles(
    sampled: dict[str, Any],
    surface: _Surface,
    scales: _Scales,
    *,
    metric_floor: float,
) -> dict[str, Any]:
    dtype = surface.base_Rcos.dtype
    floor = jnp.asarray(float(metric_floor), dtype=dtype)
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    alpha_cov = jnp.stack(
        (
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
            -surface.iota_safe,
        )
    )
    contra = sampled["contra"]
    grad_alpha_metric = jnp.maximum(
        jnp.einsum("i,zij,j->z", alpha_cov, contra, alpha_cov), floor
    )
    grad_s_metric = jnp.maximum(contra[:, 0, 0], floor)
    grad_s_dot_alpha = contra[:, 0, 1] - surface.iota_safe * contra[:, 0, 2]
    length = jnp.asarray(scales.length, dtype=dtype)
    b_ref = jnp.asarray(scales.b_ref, dtype=dtype)
    shear = _safe_shear(surface)
    sqrt_s = jnp.sqrt(surface.s_value)
    gradpar_profile = jnp.abs(
        length
        * surface.iota_safe
        / jnp.maximum(jnp.abs(sampled["bmag"] * sampled["sqrtg"]), tiny)
    )
    gradpar_value = jnp.mean(gradpar_profile)
    return {
        "alpha_cov": alpha_cov,
        "gradpar": gradpar_value * jnp.ones_like(sampled["bmag"]),
        "gradpar_value": gradpar_value,
        "gds2": length * length * surface.s_value * grad_alpha_metric,
        "gds21": shear * grad_s_dot_alpha / jnp.maximum(b_ref, tiny),
        "gds22": shear
        * shear
        * grad_s_metric
        / jnp.maximum(length * length * b_ref * b_ref * surface.s_value, tiny),
        "grho": jnp.sqrt(grad_s_metric) / jnp.maximum(length * b_ref * sqrt_s, tiny),
    }


def _drift_profiles(
    sampled: dict[str, Any],
    metric: dict[str, Any],
    surface: _Surface,
    line: _Line,
    scales: _Scales,
    *,
    drift_scale: float,
) -> dict[str, Any]:
    dtype = surface.base_Rcos.dtype
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    length = jnp.asarray(scales.length, dtype=dtype)
    b_ref = jnp.asarray(scales.b_ref, dtype=dtype)
    sqrt_s = jnp.sqrt(surface.s_value)
    shear = _safe_shear(surface)
    log_b = jnp.log(jnp.maximum(sampled["bmag"], tiny))
    dlogb_dtheta = (jnp.roll(log_b, -1) - jnp.roll(log_b, 1)) / (
        2.0 * (2.0 * jnp.pi / float(line.ntheta))
    )
    grad_b_cov = jnp.stack(
        (sampled["db_ds"], sampled["db_dtheta"], sampled["db_dzeta"]), axis=-1
    )
    contra = sampled["contra"]
    grad_b_dot_alpha = jnp.einsum(
        "zi,zij,j->z", grad_b_cov, contra, metric["alpha_cov"]
    )
    grad_b_dot_s = jnp.einsum("zi,zi->z", grad_b_cov, contra[:, :, 0])
    bmag_sq = jnp.maximum(sampled["bmag"] * sampled["bmag"], tiny)
    gbdrift = (
        -2.0 * float(drift_scale) * length * length * sqrt_s * grad_b_dot_alpha / bmag_sq
    )
    gbdrift0 = (
        -2.0
        * float(drift_scale)
        * shear
        * grad_b_dot_s
        / jnp.maximum(b_ref * bmag_sq * sqrt_s, tiny)
    )
    return {
        "bgrad": metric["gradpar_value"] * dlogb_dtheta,
        "gbdrift": gbdrift,
        "gbdrift0": gbdrift0,
    }


def _assemble_mapping(
    wout: Any,
    surface: _Surface,
    scales: _Scales,
    line: _Line,
    sampled: dict[str, Any],
    metric: dict[str, Any],
    drift: dict[str, Any],
    *,
    alpha: float,
    drift_scale: float,
) -> dict[str, Any]:
    tiny = jnp.asarray(1.0e-30, dtype=surface.base_Rcos.dtype)
    jacobian = jnp.abs(sampled["sqrtg"]) / jnp.maximum(
        jnp.mean(jnp.abs(sampled["sqrtg"])), tiny
    )
    return {
        "theta": line.theta,
        "gradpar": metric["gradpar"],
        "bmag": sampled["bmag"],
        "bgrad": drift["bgrad"],
        "gds2": metric["gds2"],
        "gds21": metric["gds21"],
        "gds22": metric["gds22"],
        "cvdrift": drift["gbdrift"],
        "gbdrift": drift["gbdrift"],
        "cvdrift0": drift["gbdrift0"],
        "gbdrift0": drift["gbdrift0"],
        "jacobian": jacobian,
        "grho": metric["grho"],
        "q": 1.0 / jnp.maximum(jnp.abs(surface.iota_safe), tiny),
        "s_hat": _safe_shear(surface),
        "epsilon": jnp.std(sampled["bmag"]) / jnp.maximum(jnp.mean(sampled["bmag"]), tiny),
        "R0": float(scales.length),
        "B0": float(scales.b_ref),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(getattr(wout, "nfp", 1)),
        "vmec_jax": {
            "surface_index": int(surface.index),
            "iota": surface.iota_line,
            "reference_length": float(scales.length),
            "reference_b": float(scales.b_ref),
            "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        },
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

    surface = _surface(state, wout, surface_index)
    scales = _reference_scales(wout, reference_length, reference_b)
    line = _field_line(wout, surface, alpha=alpha, ntheta=int(ntheta))
    geom, bmag_all = _load_raw_vmec_tensors(
        state, static, wout, surface, scales, b2_floor=b2_floor
    )
    sampled = _sample_line_tensors(geom, bmag_all, surface, line)
    metric = _metric_profiles(sampled, surface, scales, metric_floor=metric_floor)
    drift = _drift_profiles(
        sampled, metric, surface, line, scales, drift_scale=drift_scale
    )
    return _assemble_mapping(
        wout,
        surface,
        scales,
        line,
        sampled,
        metric,
        drift,
        alpha=alpha,
        drift_scale=drift_scale,
    )


__all__ = ["vmec_jax_flux_tube_mapping_from_state"]
