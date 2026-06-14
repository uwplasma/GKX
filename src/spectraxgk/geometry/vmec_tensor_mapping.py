"""Direct VMEC tensor to flux-tube mapping bridge."""

from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.numerics import _periodic_bilinear_sample_2d


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


__all__ = ["vmec_jax_flux_tube_mapping_from_state"]
