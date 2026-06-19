"""VMEC/Boozer field-line geometry assembly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np

from spectraxgk.geometry_backends.vmec_backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _import_booz_backend,
    _new_booz_object,
)
from spectraxgk.geometry_backends.vmec_splines import _vmec_splines
from spectraxgk.geometry_backends.vmec_types import _Struct

from spectraxgk.geometry_backends.vmec_fieldline_numerics import (
    _MU_0,
    _axisym_flip_required,
    _boozer_mode_angle,
    _boozer_mode_sum,
    _boozer_trig_basis,
    _centered_fieldline_integral,
    _fieldline_boozer_coordinates,
    _fieldline_boozer_tensors,
    _fieldline_cartesian_derivatives,
    _fieldline_coordinate_gradients,
    _flux_surface_hngc_averages,
    _hngc_pressure_correction,
    _hngc_shear_correction,
    _input_iota_shear,
    _sample_boozer_mode_table,
    _safe_mode_denominator,
    _surface_average_2d,  # noqa: F401 - re-exported for helper-level tests.
    _validated_reference_scales,
)


def _new_boozer_object_with_auto_fallback(
    primary_backend: Any, vmec_fname: str | Path, nc_obj: Any
) -> Any:
    """Create a Boozer transform object, using the classic reader if needed.

    Some VMEC-JAX WOUT files expose a square ``(radius, mode)`` layout that old
    ``booz_xform_jax`` readers reject as ambiguous. In automatic backend mode,
    the imported-geometry path can safely fall back to the classic
    ``booz_xform`` reader; explicit backend selections remain fail-fast.
    """

    try:
        return _new_booz_object(primary_backend, str(vmec_fname))
    except Exception as exc:
        auto_backend = os.environ.get("SPECTRAX_BOOZ_BACKEND", "auto").strip().lower()
        if auto_backend in {"", "auto"} and _booz_read_wout_square_layout_failure(exc):
            try:
                fallback = _import_booz_backend("booz_xform")
                return _new_booz_object(fallback, str(vmec_fname))
            except Exception:
                nc_obj.close()
                raise
        nc_obj.close()
        raise


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

    booz_obj = _new_boozer_object_with_auto_fallback(bxform, vmec_fname, nc_obj)
    booz_obj.mboz = int(2 * mpol)
    booz_obj.nboz = int(2 * ntor)
    booz_obj.run()

    vs = _vmec_splines(nc_obj, booz_obj)

    s = np.array([s_val])
    ns = 1
    alpha_arr = np.array([alpha])

    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2.0 * s / iota) * d_iota_d_s
    sqrt_s = np.sqrt(s)

    nfp = vs.nfp

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2.0 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    try:
        L_reference, B_reference, R_mag_ax = _validated_reference_scales(
            vs, edge_toroidal_flux_over_2pi
        )
    except ValueError:
        nc_obj.close()
        raise

    zeta_center = -alpha / float(iota[0])

    iota_input_val, s_hat_input_val = _input_iota_shear(
        iota, shat, iota_input, s_hat_input
    )

    G = vs.Gfun(s)
    boozer_i = vs.Ifun(s)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    (
        rmnc_b,
        zmns_b,
        numns_b,
        d_rmnc_b_d_s,
        d_zmns_b_d_s,
        d_numns_b_d_s,
        gmnc_b,
        bmnc_b,
        d_bmnc_b_d_s,
    ) = _sample_boozer_mode_table(vs, s, ns)
    mnmax_b = rmnc_b.shape[1]

    theta_b, phi_b = _fieldline_boozer_coordinates(theta1d, alpha_arr, iota)
    flipit = _axisym_flip_required(
        isaxisym=isaxisym,
        xm_b=xm_b,
        xn_b=xn_b,
        theta_b=theta_b,
        phi_b=phi_b,
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )

    angle_b = _boozer_mode_angle(xm_b, xn_b, theta_b, phi_b, flipit=flipit)

    (
        cosangle_b,
        sinangle_b,
        mcosangle_b,
        msinangle_b,
        ncosangle_b,
        nsinangle_b,
    ) = _boozer_trig_basis(xm_b, xn_b, angle_b)

    tensors = _fieldline_boozer_tensors(
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        d_rmnc_b_d_s=d_rmnc_b_d_s,
        d_zmns_b_d_s=d_zmns_b_d_s,
        d_numns_b_d_s=d_numns_b_d_s,
        gmnc_b=gmnc_b,
        bmnc_b=bmnc_b,
        d_bmnc_b_d_s=d_bmnc_b_d_s,
        cosangle_b=cosangle_b,
        sinangle_b=sinangle_b,
        mcosangle_b=mcosangle_b,
        msinangle_b=msinangle_b,
        ncosangle_b=ncosangle_b,
        nsinangle_b=nsinangle_b,
    )
    R_b = tensors.R_b
    Z_b = tensors.Z_b
    nu_b = tensors.nu_b
    sqrt_g_booz = tensors.sqrt_g_booz
    modB_b = tensors.modB_b

    Vprime = gmnc_b[:, 0]  # flux-surface volume element (m=0, n=0 Boozer mode)

    # Lambda / beta corrections (Hegna-Nakajima)
    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    safe_denom_mn = _safe_mode_denominator(xm_b, xn_b, iota)

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

    beta_b = _boozer_mode_sum(betamns_b, sinangle_b)
    lambda_b = _boozer_mode_sum(lambmnc_b, cosangle_b)

    _etf = edge_toroidal_flux_over_2pi
    cartesian = _fieldline_cartesian_derivatives(tensors=tensors, phi_b=phi_b)
    gradients = _fieldline_coordinate_gradients(
        tensors=tensors,
        cartesian=cartesian,
        edge_toroidal_flux_over_2pi=_etf,
    )
    grad_psi_X = gradients.grad_psi_X
    grad_psi_Y = gradients.grad_psi_Y
    grad_psi_Z = gradients.grad_psi_Z

    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2

    grad_alpha_X = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_X / _etf
        + gradients.grad_theta_b_X
        - iota[:, None, None] * gradients.grad_phi_b_X
    )
    grad_alpha_Y = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Y / _etf
        + gradients.grad_theta_b_Y
        - iota[:, None, None] * gradients.grad_phi_b_Y
    )
    grad_alpha_Z = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Z / _etf
        + gradients.grad_theta_b_Z
        - iota[:, None, None] * gradients.grad_phi_b_Z
    )

    D1, D2 = _flux_surface_hngc_averages(
        xm_b=xm_b,
        xn_b=xn_b,
        flipit=bool(flipit),
        lambmnc_b=lambmnc_b,
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        gmnc_b=gmnc_b,
        res_theta=res_theta,
        res_phi=res_phi,
    )

    # Cumulative integrals along the field line
    theta_1d = theta_b[0, 0]
    intinv_g = _centered_fieldline_integral(1.0 / g_sup_psi_psi, phi_b, theta_1d)
    int_lam_div_g = _centered_fieldline_integral(
        lambda_b / g_sup_psi_psi, phi_b, theta_1d
    )

    # HNGC correction factors
    d_iota_d_s_1, sfac = _hngc_shear_correction(
        s_val=s_val,
        iota=iota,
        shat=shat,
        iota_input_val=iota_input_val,
        s_hat_input_val=s_hat_input_val,
        include_shear_variation=include_shear_variation,
    )
    d_pressure_d_s_1, pfac = _hngc_pressure_correction(
        s_val=s_val,
        betaprim=betaprim,
        B_reference=B_reference,
        d_pressure_d_s=d_pressure_d_s,
        include_pressure_variation=include_pressure_variation,
    )

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
        * (modB_b * tensors.d_B_b_d_s + _MU_0 * d_pressure_d_s[:, None, None])
        / _etf
        - beta_b
        / (
            2.0
            * sqrt_g_booz
            * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
        )
        * tensors.d_sqrt_g_booz_d_phi_b
        + L0
        * (
            G[:, None] * tensors.d_sqrt_g_booz_d_theta_b
            - boozer_i[:, None] * tensors.d_sqrt_g_booz_d_phi_b
        )
        / (2.0 * sqrt_g_booz * (G[:, None] + iota[:, None] * boozer_i[:, None]))
    )
    kappa_g = (
        G[:, None] * tensors.d_sqrt_g_booz_d_theta_b
        - boozer_i[:, None] * tensors.d_sqrt_g_booz_d_phi_b
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

    return _Struct(
        iota_input=iota_input_val,
        d_iota_d_s=d_iota_d_s,
        d_pressure_d_s=d_pressure_d_s,
        s_hat_input=s_hat_input_val,
        alpha=alpha,
        theta_b=theta_b,
        phi_b=phi_b,
        theta_PEST=theta_PEST,
        theta_geo=theta_geo,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
        R_b=R_b,
        Z_b=Z_b,
        betaprim=betaprim,
        bmag=bmag,
        gradpar_theta_b=gradpar_theta_b,
        gradpar_phi=L_reference / modB_b / sqrt_g_booz,
        gds2=gds2,
        gds21=gds21,
        gds22=gds22,
        gbdrift=gbdrift,
        gbdrift0=gbdrift0,
        cvdrift=cvdrift,
        cvdrift0=cvdrift0,
        grho=grho,
        grad_y=grad_y,
        grad_x=grad_x,
        zeta_center=zeta_center,
        nfp=nfp,
        L_reference=L_reference,
        B_reference=B_reference,
        dpsidrho=2.0 * np.sqrt(s_val) * edge_toroidal_flux_over_2pi,
    )
