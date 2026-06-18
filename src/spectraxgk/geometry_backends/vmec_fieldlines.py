"""VMEC/Boozer field-line geometry assembly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.integrate import simpson as _simps
from scipy.interpolate import InterpolatedUnivariateSpline

from spectraxgk.geometry_backends.vmec_backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _import_booz_backend,
    _new_booz_object,
)
from spectraxgk.geometry_backends.vmec_types import _Struct

_MU_0 = 4.0 * np.pi * 1.0e-7

def _vmec_splines(nc_obj: Any, booz_obj: Any) -> _Struct:
    """Build radial splines from a VMEC netCDF object and a booz_xform object.

    Build VMEC spline data used by the imported-geometry pipeline.
    """

    r = _Struct()

    ns = int(nc_obj.variables["ns"][:].data)
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])

    mnmax_b = int(booz_obj.mnboz)

    r.rmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.rmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.zmns_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.zmns_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.numns_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.numns_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.d_rmnc_b_d_s = [f.derivative() for f in r.rmnc_b]
    r.d_zmns_b_d_s = [f.derivative() for f in r.zmns_b]
    r.d_numns_b_d_s = [f.derivative() for f in r.numns_b]

    r.gmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.gmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.bmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.bmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.d_bmnc_b_d_s = [f.derivative() for f in r.bmnc_b]

    r.Gfun = InterpolatedUnivariateSpline(s_half, booz_obj.Boozer_G)
    r.Ifun = InterpolatedUnivariateSpline(s_half, booz_obj.Boozer_I)
    r.pressure = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["pres"][1:])
    )
    r.d_pressure_d_s = r.pressure.derivative()
    r.psi = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["phi"][1:]) / (2.0 * np.pi)
    )
    r.d_psi_d_s = r.psi.derivative()
    r.iota = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["iotas"][1:])
    )
    r.d_iota_d_s = r.iota.derivative()

    r.phiedge = float(nc_obj.variables["phi"][-1])
    r.Aminor_p = float(nc_obj.variables["Aminor_p"][:])
    r.nfp = int(nc_obj.variables["nfp"][:])
    r.raxis_cc = np.asarray(nc_obj.variables["raxis_cc"][:])

    r.xm_b = booz_obj.xm_b
    r.xn_b = booz_obj.xn_b
    r.mnbooz = mnmax_b
    r.mboz = booz_obj.mboz
    r.nboz = booz_obj.nboz

    return r


def _vmec_fieldlines(
    vmec_fname: str | Path,
    s_val: float,
    betaprim: float,
    alpha: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    theta1d: np.ndarray,
    isaxisym: bool,
    iota_input: float | None = None,
    s_hat_input: float | None = None,
    res_theta: int = 201,
    res_phi: int = 201,
) -> _Struct:
    """Compute VMEC flux-tube geometry coefficients from a VMEC equilibrium.

    Evaluate field-line geometry from VMEC and Boozer spline data.

    Parameters
    ----------
    vmec_fname:
        Path to a VMEC ``wout_*.nc`` file.
    s_val:
        Normalised toroidal flux (normalized toroidal-flux input).
    betaprim:
        Effective beta prime for pressure gradient variation.
    alpha:
        Field-line label (Boozer alpha = theta_b - iota * phi_b).
    include_shear_variation / include_pressure_variation:
        Whether to apply Hegna-Nakajima local-equilibrium corrections.
    theta1d:
        1-D Boozer-theta array defining the field line.
    isaxisym:
        Whether the equilibrium is axisymmetric (enables flipit logic).
    iota_input / s_hat_input:
        Override values; *None* means use VMEC values.
    res_theta / res_phi:
        Resolution of the 2-D (theta, phi) grid used for flux-surface
        integrals D1 and D2.
    """

    bxform = _import_booz_backend()
    from netCDF4 import Dataset as _NC

    nc_obj = _NC(str(vmec_fname), "r")
    mpol = int(nc_obj.variables["mpol"][:])
    ntor = int(nc_obj.variables["ntor"][:])

    try:
        booz_obj = _new_booz_object(bxform, str(vmec_fname))
    except Exception as exc:
        if os.environ.get("SPECTRAX_BOOZ_BACKEND", "auto").strip().lower() in {
            "",
            "auto",
        } and _booz_read_wout_square_layout_failure(exc):
            # Some VMEC-JAX wouts can have ns == mnmax with explicit
            # (radius, mn_mode) dimensions. Older booz_xform_jax releases
            # reject that square shape as ambiguous; the classic booz_xform
            # reader handles it, so use it as a runtime EIK fallback.
            try:
                fallback = _import_booz_backend("booz_xform")
                booz_obj = _new_booz_object(fallback, str(vmec_fname))
            except Exception:
                nc_obj.close()
                raise
        else:
            nc_obj.close()
            raise
    booz_obj.mboz = int(2 * mpol)
    booz_obj.nboz = int(2 * ntor)
    booz_obj.run()

    vs = _vmec_splines(nc_obj, booz_obj)

    s = np.array([s_val])
    ns = 1
    alpha_arr = np.array([alpha])
    nalpha = 1
    nl = len(theta1d)

    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2.0 * s / iota) * d_iota_d_s
    sqrt_s = np.sqrt(s)

    nfp = vs.nfp

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2.0 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    L_reference = vs.Aminor_p
    if not np.isfinite(float(L_reference)) or abs(float(L_reference)) <= 0.0:
        nc_obj.close()
        raise ValueError(
            "VMEC geometry has an invalid reference length Aminor_p="
            f"{float(L_reference)!r}. External VMEC equilibria used for runtime "
            "EIK generation must provide a positive finite minor radius."
        )
    B_reference = 2.0 * abs(edge_toroidal_flux_over_2pi) / (L_reference**2)
    R_mag_ax = float(vs.raxis_cc[0])

    zeta_center = -alpha / float(iota[0])

    iota_input_val = float(iota[0]) if iota_input is None else float(iota_input)
    s_hat_input_val = float(shat[0]) if s_hat_input is None else float(s_hat_input)
    if abs(s_hat_input_val) < 1.0e-30:
        s_hat_input_val = 1.0e-8

    G = vs.Gfun(s)
    boozer_i = vs.Ifun(s)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    mnmax_b = vs.mnbooz

    rmnc_b = np.zeros((ns, mnmax_b))
    zmns_b = np.zeros((ns, mnmax_b))
    numns_b = np.zeros((ns, mnmax_b))
    d_rmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_zmns_b_d_s = np.zeros((ns, mnmax_b))
    d_numns_b_d_s = np.zeros((ns, mnmax_b))
    gmnc_b = np.zeros((ns, mnmax_b))
    bmnc_b = np.zeros((ns, mnmax_b))
    d_bmnc_b_d_s = np.zeros((ns, mnmax_b))

    for jmn in range(mnmax_b):
        rmnc_b[:, jmn] = vs.rmnc_b[jmn](s)
        zmns_b[:, jmn] = vs.zmns_b[jmn](s)
        numns_b[:, jmn] = vs.numns_b[jmn](s)
        d_rmnc_b_d_s[:, jmn] = vs.d_rmnc_b_d_s[jmn](s)
        d_zmns_b_d_s[:, jmn] = vs.d_zmns_b_d_s[jmn](s)
        d_numns_b_d_s[:, jmn] = vs.d_numns_b_d_s[jmn](s)
        gmnc_b[:, jmn] = vs.gmnc_b[jmn](s)
        bmnc_b[:, jmn] = vs.bmnc_b[jmn](s)
        d_bmnc_b_d_s[:, jmn] = vs.d_bmnc_b_d_s[jmn](s)

    # Field line (theta_b, phi_b) along alpha = theta_b - iota * phi_b
    theta_b = np.zeros((ns, nalpha, nl))
    phi_b = np.zeros((ns, nalpha, nl))
    for js in range(ns):
        theta_b[js, :, :] = theta1d[None, :]
        phi_b[js, :, :] = (theta1d[None, :] - alpha_arr[:, None]) / iota[js]

    # Flipit check for axisymmetric equilibria
    angle_b_chk = (
        xm_b[:, None, None, None] * theta_b[None, :, :, :]
        - xn_b[:, None, None, None] * phi_b[None, :, :, :]
    )
    R_b_chk = np.einsum("ij,jikl->ikl", rmnc_b, np.cos(angle_b_chk))
    Z_b_chk = np.einsum("ij,jikl->ikl", zmns_b, np.sin(angle_b_chk))
    flipit = 0
    if isaxisym:
        if R_b_chk[0, 0, 0] > R_b_chk[0, 0, 1] or Z_b_chk[0, 0, 1] > Z_b_chk[0, 0, 0]:
            flipit = 1

    if flipit:
        angle_b = (
            xm_b[:, None, None, None] * (theta_b[None, :, :, :] + np.pi)
            - xn_b[:, None, None, None] * phi_b[None, :, :, :]
        )
    else:
        angle_b = (
            xm_b[:, None, None, None] * theta_b[None, :, :, :]
            - xn_b[:, None, None, None] * phi_b[None, :, :, :]
        )

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b

    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    d_R_b_d_s = np.einsum("ij,jikl->ikl", d_rmnc_b_d_s, cosangle_b)
    d_R_b_d_theta_b = -np.einsum("ij,jikl->ikl", rmnc_b, msinangle_b)
    d_R_b_d_phi_b = np.einsum("ij,jikl->ikl", rmnc_b, nsinangle_b)

    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)
    d_Z_b_d_s = np.einsum("ij,jikl->ikl", d_zmns_b_d_s, sinangle_b)
    d_Z_b_d_theta_b = np.einsum("ij,jikl->ikl", zmns_b, mcosangle_b)
    d_Z_b_d_phi_b = -np.einsum("ij,jikl->ikl", zmns_b, ncosangle_b)

    nu_b = np.einsum("ij,jikl->ikl", numns_b, sinangle_b)
    d_nu_b_d_s = np.einsum("ij,jikl->ikl", d_numns_b_d_s, sinangle_b)
    d_nu_b_d_theta_b = np.einsum("ij,jikl->ikl", numns_b, mcosangle_b)
    d_nu_b_d_phi_b = -np.einsum("ij,jikl->ikl", numns_b, ncosangle_b)

    sqrt_g_booz = np.einsum("ij,jikl->ikl", gmnc_b, cosangle_b)
    d_sqrt_g_booz_d_theta_b = -np.einsum("ij,jikl->ikl", gmnc_b, msinangle_b)
    d_sqrt_g_booz_d_phi_b = np.einsum("ij,jikl->ikl", gmnc_b, nsinangle_b)
    modB_b = np.einsum("ij,jikl->ikl", bmnc_b, cosangle_b)
    d_B_b_d_s = np.einsum("ij,jikl->ikl", d_bmnc_b_d_s, cosangle_b)

    Vprime = gmnc_b[:, 0]  # flux-surface volume element (m=0, n=0 Boozer mode)

    # Lambda / beta corrections (Hegna-Nakajima)
    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    denom_mn = xm_b[1:] * iota[:, None] - xn_b[1:]
    safe_denom_mn = np.where(
        np.abs(denom_mn) < 1.0e-30,
        np.sign(denom_mn + 1.0e-300) * 1.0e-30,
        denom_mn,
    )

    delmnc_b[:, 1:] = gmnc_b[:, 1:] / Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        / edge_toroidal_flux_over_2pi
        * _MU_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / safe_denom_mn
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * boozer_i[:, None])
        / (safe_denom_mn * (G[:, None] + iota[:, None] * boozer_i[:, None]))
    )

    beta_b = np.einsum("ij,jikl->ikl", betamns_b, sinangle_b)
    lambda_b = np.einsum("ij,jikl->ikl", lambmnc_b, cosangle_b)

    # Cartesian coordinate derivatives for basis vectors
    phi_cyl = phi_b - nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)

    d_X_d_theta_b = d_R_b_d_theta_b * cosphi - R_b * sinphi * (-d_nu_b_d_theta_b)
    d_X_d_phi_b = d_R_b_d_phi_b * cosphi - R_b * sinphi * (1.0 - d_nu_b_d_phi_b)
    d_X_d_s = d_R_b_d_s * cosphi - R_b * sinphi * (-d_nu_b_d_s)

    d_Y_d_theta_b = d_R_b_d_theta_b * sinphi + R_b * cosphi * (-d_nu_b_d_theta_b)
    d_Y_d_phi_b = d_R_b_d_phi_b * sinphi + R_b * cosphi * (1.0 - d_nu_b_d_phi_b)
    d_Y_d_s = d_R_b_d_s * sinphi + R_b * cosphi * (-d_nu_b_d_s)

    # Dual (contravariant) gradient vectors via cross products
    grad_psi_X = (
        d_Y_d_theta_b * d_Z_b_d_phi_b - d_Z_b_d_theta_b * d_Y_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Y = (
        d_Z_b_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_b_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Z = (
        d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b
    ) / sqrt_g_booz

    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2

    _etf = edge_toroidal_flux_over_2pi
    grad_theta_b_X = (d_Y_d_phi_b * d_Z_b_d_s - d_Z_b_d_phi_b * d_Y_d_s) / (
        sqrt_g_booz * _etf
    )
    grad_theta_b_Y = (d_Z_b_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_b_d_s) / (
        sqrt_g_booz * _etf
    )
    grad_theta_b_Z = (d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s) / (
        sqrt_g_booz * _etf
    )

    grad_phi_b_X = (d_Y_d_s * d_Z_b_d_theta_b - d_Z_b_d_s * d_Y_d_theta_b) / (
        sqrt_g_booz * _etf
    )
    grad_phi_b_Y = (d_Z_b_d_s * d_X_d_theta_b - d_X_d_s * d_Z_b_d_theta_b) / (
        sqrt_g_booz * _etf
    )
    grad_phi_b_Z = (d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b) / (
        sqrt_g_booz * _etf
    )

    grad_alpha_X = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_X / _etf
        + grad_theta_b_X
        - iota[:, None, None] * grad_phi_b_X
    )
    grad_alpha_Y = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Y / _etf
        + grad_theta_b_Y
        - iota[:, None, None] * grad_phi_b_Y
    )
    grad_alpha_Z = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Z / _etf
        + grad_theta_b_Z
        - iota[:, None, None] * grad_phi_b_Z
    )

    # ------------------------------------------------------------------
    # Flux-surface integrals D1, D2 (Hegna-Nakajima correction)
    # ------------------------------------------------------------------
    theta_b_grid = np.linspace(-np.pi, np.pi, res_theta)
    phi_b_grid = np.linspace(-np.pi, np.pi, res_phi)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)  # (res_phi, res_theta)

    if flipit:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :] + np.pi)
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )
    else:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * th_b_2D[None, None, None, :, :]
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)
    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = np.einsum("ij,jiklm->iklm", lambmnc_b, cosangle_b_2D)
    R_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, cosangle_b_2D)
    d_R_b_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, msinangle_b_2D)
    d_R_b_d_phi_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, ncosangle_b_2D)
    d_Z_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, mcosangle_b_2D)
    d_Z_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", zmns_b, ncosangle_b_2D)
    nu_b_2D = np.einsum("ij,jiklm->iklm", numns_b, sinangle_b_2D)
    d_nu_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", numns_b, mcosangle_b_2D)
    d_nu_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", numns_b, nsinangle_b_2D)
    sqrt_g_booz_2D = np.einsum("ij,jiklm->iklm", gmnc_b, cosangle_b_2D)
    Z_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, sinangle_b_2D)  # noqa: F841

    ph_nat_2D = ph_b_2D - nu_b_2D
    sinphi_2D = np.sin(ph_nat_2D)
    cosphi_2D = np.cos(ph_nat_2D)

    d_X_d_th_b_2D = d_R_b_d_theta_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        -d_nu_b_d_theta_b_2D
    )
    d_X_d_phi_2D = d_R_b_d_phi_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        1.0 - d_nu_b_d_phi_b_2D
    )
    d_Y_d_th_b_2D = d_R_b_d_theta_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        -d_nu_b_d_theta_b_2D
    )
    d_Y_d_phi_2D = d_R_b_d_phi_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        1.0 - d_nu_b_d_phi_b_2D
    )

    grad_psi_X_2D = (
        d_Y_d_th_b_2D * d_Z_b_d_phi_b_2D - d_Z_b_d_theta_b_2D * d_Y_d_phi_2D
    ) / sqrt_g_booz_2D
    grad_psi_Y_2D = (
        d_Z_b_d_theta_b_2D * d_X_d_phi_2D - d_X_d_th_b_2D * d_Z_b_d_phi_b_2D
    ) / sqrt_g_booz_2D
    grad_psi_Z_2D = (
        d_X_d_th_b_2D * d_Y_d_phi_2D - d_Y_d_th_b_2D * d_X_d_phi_2D
    ) / sqrt_g_booz_2D

    g_sup_psi_psi_2D = grad_psi_X_2D**2 + grad_psi_Y_2D**2 + grad_psi_Z_2D**2
    g_sup_psi_psi_2D_inv = 1.0 / g_sup_psi_psi_2D
    lam_over_g_2D = lambda_b_2D * g_sup_psi_psi_2D_inv

    # Flux-surface averages over the (res_phi, res_theta) 2-D grid
    D1 = (
        _simps(
            [_simps(row, x=theta_b_grid) for row in g_sup_psi_psi_2D_inv[0, 0]],
            x=phi_b_grid,
        )
        / (2.0 * np.pi) ** 2
    )

    D2 = (
        _simps(
            [_simps(row, x=theta_b_grid) for row in lam_over_g_2D[0, 0]],
            x=phi_b_grid,
        )
        / (2.0 * np.pi) ** 2
    )

    # Cumulative integrals along the field line
    intinv_g = _ctrap(1.0 / g_sup_psi_psi, phi_b, initial=0)
    int_lam_div_g = _ctrap(lambda_b / g_sup_psi_psi, phi_b, initial=0)

    # Subtract the value at theta = 0 (field-line midpoint)
    theta_1d = theta_b[0, 0]
    spl0 = InterpolatedUnivariateSpline(theta_1d, intinv_g[0, 0])
    intinv_g = intinv_g - spl0(0.0)
    spl1 = InterpolatedUnivariateSpline(theta_1d, int_lam_div_g[0, 0])
    int_lam_div_g = int_lam_div_g - spl1(0.0)

    # HNGC correction factors
    d_iota_d_s_1 = (
        -(iota_input_val / (2.0 * s_val)) * s_hat_input_val
        + (float(iota[0]) / (2.0 * s_val)) * float(shat[0])
    ) * np.ones((ns,))
    sfac = float(shat[0]) / s_hat_input_val

    if not include_shear_variation:
        d_iota_d_s_1 = np.zeros((ns,))
        sfac = 1.0

    d_pressure_d_s_1 = betaprim / (4.0 * np.sqrt(s_val)) * B_reference**2 * np.ones(
        (ns,)
    ) - _MU_0 * d_pressure_d_s * np.ones((ns,))

    dp_ds_safe = np.where(np.abs(d_pressure_d_s) < 1.0e-30, 1.0e-8, d_pressure_d_s)
    pfac = (
        betaprim
        * B_reference**2
        / (4.0 * np.sqrt(s_val))
        / (_MU_0 * float(dp_ds_safe[0]))
    )

    if not include_pressure_variation:
        pfac = 1.0
        d_pressure_d_s_1 = np.zeros((ns,))

    D_HNGC = (
        1.0
        / _etf
        * (
            d_iota_d_s_1[:, None, None] * (intinv_g / D1 - phi_b + zeta_center)
            - d_pressure_d_s_1[:, None, None]
            * Vprime[:, None, None]
            * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
            * (int_lam_div_g - D2 * intinv_g / D1)
        )
    )

    # Integrated local shear L0, L1
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    L0 = -1.0 * (
        grad_alpha_dot_grad_psi / g_sup_psi_psi
        + 1.0 / _etf * d_iota_d_s[:, None, None] * (phi_b - zeta_center)
    )
    L1 = (
        -1.0 / _etf * d_iota_d_s_1[:, None, None] * (phi_b - zeta_center)
        + grad_alpha_dot_grad_psi / g_sup_psi_psi
        - D_HNGC
    )

    # Curvature components (normal and geodesic)
    kappa_n = (
        1.0
        / modB_b**2
        * (modB_b * d_B_b_d_s + _MU_0 * d_pressure_d_s[:, None, None])
        / _etf
        - beta_b
        / (
            2.0
            * sqrt_g_booz
            * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
        )
        * d_sqrt_g_booz_d_phi_b
        + L0
        * (
            G[:, None] * d_sqrt_g_booz_d_theta_b
            - boozer_i[:, None] * d_sqrt_g_booz_d_phi_b
        )
        / (2.0 * sqrt_g_booz * (G[:, None] + iota[:, None] * boozer_i[:, None]))
    )
    kappa_g = (
        G[:, None] * d_sqrt_g_booz_d_theta_b - boozer_i[:, None] * d_sqrt_g_booz_d_phi_b
    ) / (2.0 * sqrt_g_booz * (G[:, None] + iota[:, None] * boozer_i[:, None]))

    B_cross_kappa_dot_grad_alpha = (kappa_n + kappa_g * L1) * modB_b**2
    B_cross_kappa_dot_grad_psi = kappa_g * modB_b**2

    # Geometry coefficients -------------------------------------------
    bmag = modB_b / B_reference
    gradpar_theta_b = -L_reference / modB_b / sqrt_g_booz * iota[:, None, None]

    grad_alpha_dot_grad_alpha_b = modB_b**2 / g_sup_psi_psi + g_sup_psi_psi * L1**2
    grad_alpha_dot_grad_psi_b = g_sup_psi_psi * L1

    gds2 = grad_alpha_dot_grad_alpha_b * L_reference**2 * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi_b * sfac * shat[:, None, None] / B_reference
    gds22 = (
        g_sup_psi_psi
        * (sfac * shat[:, None, None]) ** 2
        / (L_reference**2 * B_reference**2 * s[:, None, None])
    )
    grho = np.sqrt(g_sup_psi_psi / (L_reference**2 * B_reference**2 * s[:, None, None]))

    gbdrift0 = (
        -B_cross_kappa_dot_grad_psi
        * 2.0
        * sfac
        * shat[:, None, None]
        / (modB_b**2 * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift0 = gbdrift0

    cvdrift = (
        -2.0
        * B_reference
        * L_reference**2
        * sqrt_s[:, None, None]
        * B_cross_kappa_dot_grad_alpha
        / modB_b**2
        * toroidal_flux_sign
    )
    gbdrift = cvdrift + (
        2.0
        * B_reference
        * L_reference**2
        * sqrt_s[:, None, None]
        * _MU_0
        * pfac
        * d_pressure_d_s[:, None, None]
        * toroidal_flux_sign
        / (_etf * modB_b**2)
    )

    theta_PEST = theta_b - iota[:, None, None] * nu_b
    theta_geo = np.arctan2(Z_b, R_b - R_mag_ax)

    grad_y = (
        L_reference
        * np.sqrt(s[:, None, None])
        * np.array([grad_alpha_X, grad_alpha_Y, grad_alpha_Z])
    )
    grad_x = (
        sfac
        * shat[:, None, None]
        * np.array([grad_psi_X, grad_psi_Y, grad_psi_Z])
        / (L_reference * B_reference * np.sqrt(s[:, None, None]))
    )

    nc_obj.close()

    # Pack results
    r = _Struct()
    r.iota_input = iota_input_val
    r.d_iota_d_s = d_iota_d_s
    r.d_pressure_d_s = d_pressure_d_s
    r.s_hat_input = s_hat_input_val
    r.alpha = alpha
    r.theta_b = theta_b
    r.phi_b = phi_b
    r.theta_PEST = theta_PEST
    r.theta_geo = theta_geo
    r.edge_toroidal_flux_over_2pi = edge_toroidal_flux_over_2pi
    r.R_b = R_b
    r.Z_b = Z_b
    r.betaprim = betaprim
    r.bmag = bmag
    r.gradpar_theta_b = gradpar_theta_b
    r.gradpar_phi = L_reference / modB_b / sqrt_g_booz
    r.gds2 = gds2
    r.gds21 = gds21
    r.gds22 = gds22
    r.gbdrift = gbdrift
    r.gbdrift0 = gbdrift0
    r.cvdrift = cvdrift
    r.cvdrift0 = cvdrift0
    r.grho = grho
    r.grad_y = grad_y
    r.grad_x = grad_x
    r.zeta_center = zeta_center
    r.nfp = nfp
    r.L_reference = L_reference
    r.B_reference = B_reference
    r.dpsidrho = 2.0 * np.sqrt(s_val) * edge_toroidal_flux_over_2pi
    return r
