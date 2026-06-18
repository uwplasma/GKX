"""VMEC-JAX to Boozer equal-arc core-profile bridge."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.core import FluxTubeGeometryData
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.geometry.numerics import (
    _boozer_half_mesh_s_grid,
    _cumulative_trapezoid,
    _evaluate_boozer_cosine_series_on_field_line,
    _interp_equal_arc_profile,
    _interp_radial,
    _radial_derivative_array,
    _radial_derivative_profile,
)
from spectraxgk.geometry.vmec_boozer_constants import (
    _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    _cached_booz_xform_constants,
    prewarm_vmec_boozer_equal_arc_cache,
)


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
    asym = bool(getattr(inputs, "bmns", None) is not None)
    cfg = getattr(static, "cfg", SimpleNamespace())
    nfp_raw = getattr(wout, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(cfg, "nfp", 1)
    nfp_int = 1 if nfp_raw is None else int(nfp_raw)
    try:
        constants, grids = _cached_booz_xform_constants(
            nfp=nfp_int,
            mpol=int(getattr(cfg, "mpol", max(2, base_Rcos.shape[1]))),
            ntor=int(getattr(cfg, "ntor", max(1, base_Rcos.shape[1] - 1))),
            ntheta=int(getattr(cfg, "ntheta", max(16, ntheta_int))),
            nzeta=int(getattr(cfg, "nzeta", max(16, 2 * ntheta_int))),
            mboz=mboz_int,
            nboz=nboz_int,
            asym=asym,
        )
    except (AttributeError, ModuleNotFoundError):
        constants, grids = bx.prepare_booz_xform_constants_from_inputs(
            inputs=inputs,
            mboz=mboz_int,
            nboz=nboz_int,
            asym=asym,
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
    ns_b_full = max(int(ns_full) - 1, int(ns_b))
    s_half = _boozer_half_mesh_s_grid(
        out.get("jlist"),
        ns_b=ns_b,
        ns_b_full=ns_b_full,
        dtype=base_Rcos.dtype,
    )

    radial_spacing = 1.0 / float(max(ns_b_full, 1))
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
        _interp_equal_arc_profile(
            theta_uniform_closed,
            theta_eqarc,
            mod_b_safe / float(B_reference),
        )
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
    gds2 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds2_raw)[:-1]
    gds21 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds21_raw)[:-1]
    gds22 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds22_raw)[:-1]
    grho = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, grho_raw)[:-1]
    cvdrift = (
        drift_loader_factor
        * _interp_equal_arc_profile(
            theta_uniform_closed, theta_eqarc, drift_cvdrift_raw
        )[:-1]
    )
    gbdrift = cvdrift
    cvdrift0 = (
        drift_loader_factor
        * _interp_equal_arc_profile(
            theta_uniform_closed, theta_eqarc, drift_cvdrift0_raw
        )[:-1]
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
    """Build solver-ready geometry directly from an in-memory VMEC/Boozer state."""

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


__all__ = [
    "flux_tube_geometry_from_vmec_boozer_state",
    "prewarm_vmec_boozer_equal_arc_cache",
    "vmec_jax_boozer_equal_arc_core_profiles_from_state",
]
