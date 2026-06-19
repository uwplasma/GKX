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
from spectraxgk.geometry_backends.vmec_splines import _vmec_splines
from spectraxgk.geometry_backends.vmec_types import _Struct

_MU_0 = 4.0 * np.pi * 1.0e-7


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


def _sample_boozer_mode_table(vs: Any, s: np.ndarray, ns: int) -> tuple[np.ndarray, ...]:
    """Sample Boozer Fourier amplitudes and radial derivatives at one surface."""

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

    return (
        rmnc_b,
        zmns_b,
        numns_b,
        d_rmnc_b_d_s,
        d_zmns_b_d_s,
        d_numns_b_d_s,
        gmnc_b,
        bmnc_b,
        d_bmnc_b_d_s,
    )


def _boozer_mode_angle(
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    theta_b: np.ndarray,
    phi_b: np.ndarray,
    *,
    flipit: bool,
) -> np.ndarray:
    """Return ``m theta - n phi`` with the axisymmetric flip convention."""

    mode_index = (slice(None),) + (None,) * theta_b.ndim
    theta_eval = theta_b + np.pi if flipit else theta_b
    return xm_b[mode_index] * theta_eval[None, ...] - xn_b[mode_index] * phi_b[None, ...]


def _boozer_mode_sum(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Contract Boozer mode coefficients with a basis carrying ``(mode, surface, ...)``."""

    return np.einsum("ij,ji...->i...", coefficients, basis)


def _fieldline_boozer_coordinates(
    theta1d: np.ndarray,
    alpha_arr: np.ndarray,
    iota: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``theta_b`` and ``phi_b`` on alpha = theta_b - iota * phi_b."""

    theta1d = np.asarray(theta1d)
    alpha_arr = np.asarray(alpha_arr)
    iota = np.asarray(iota)
    theta_b = np.broadcast_to(
        theta1d[None, None, :], (iota.size, alpha_arr.size, theta1d.size)
    ).copy()
    phi_b = (theta_b - alpha_arr[None, :, None]) / iota[:, None, None]
    return theta_b, phi_b


def _axisym_flip_required(
    *,
    isaxisym: bool,
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    theta_b: np.ndarray,
    phi_b: np.ndarray,
    rmnc_b: np.ndarray,
    zmns_b: np.ndarray,
) -> bool:
    """Detect the axisymmetric theta flip convention from the first two points."""

    if not isaxisym:
        return False
    angle_b_chk = _boozer_mode_angle(xm_b, xn_b, theta_b, phi_b, flipit=False)
    r_check = _boozer_mode_sum(rmnc_b, np.cos(angle_b_chk))
    z_check = _boozer_mode_sum(zmns_b, np.sin(angle_b_chk))
    return bool(
        r_check[0, 0, 0] > r_check[0, 0, 1]
        or z_check[0, 0, 1] > z_check[0, 0, 0]
    )


def _safe_mode_denominator(
    xm_b: np.ndarray, xn_b: np.ndarray, iota: np.ndarray
) -> np.ndarray:
    """Guard resonant Boozer denominators without changing resolved modes."""

    denom_mn = xm_b[1:] * iota[:, None] - xn_b[1:]
    return np.where(
        np.abs(denom_mn) < 1.0e-30,
        np.sign(denom_mn + 1.0e-300) * 1.0e-30,
        denom_mn,
    )


def _surface_average_2d(
    values: np.ndarray, theta_b_grid: np.ndarray, phi_b_grid: np.ndarray
) -> float:
    """Average a 2-D Boozer-surface quantity over ``theta`` and ``phi``."""

    integral = _simps(
        [_simps(row, x=theta_b_grid) for row in values],
        x=phi_b_grid,
    )
    return float(integral / (2.0 * np.pi) ** 2)


def _centered_fieldline_integral(
    integrand: np.ndarray, coordinate: np.ndarray, theta_1d: np.ndarray
) -> np.ndarray:
    """Cumulatively integrate along the field line and set theta=0 to zero."""

    integrated = _ctrap(integrand, coordinate, initial=0)
    midpoint_value = InterpolatedUnivariateSpline(theta_1d, integrated[0, 0])(0.0)
    return integrated - midpoint_value


def _validated_reference_scales(
    vs: Any,
    edge_toroidal_flux_over_2pi: float,
) -> tuple[float, float, float]:
    """Return positive finite reference scales used for VMEC normalization."""

    L_reference = float(vs.Aminor_p)
    if not np.isfinite(L_reference) or abs(L_reference) <= 0.0:
        raise ValueError(
            "VMEC geometry has an invalid reference length Aminor_p="
            f"{L_reference!r}. External VMEC equilibria used for runtime "
            "EIK generation must provide a positive finite minor radius."
        )
    B_reference = float(2.0 * abs(edge_toroidal_flux_over_2pi) / (L_reference**2))
    R_mag_ax = float(vs.raxis_cc[0])
    return L_reference, B_reference, R_mag_ax


def _input_iota_shear(
    iota: np.ndarray,
    shat: np.ndarray,
    iota_input: float | None,
    s_hat_input: float | None,
) -> tuple[float, float]:
    """Resolve user iota/shear overrides and protect the zero-shear limit."""

    iota_input_val = float(iota[0]) if iota_input is None else float(iota_input)
    s_hat_input_val = float(shat[0]) if s_hat_input is None else float(s_hat_input)
    if abs(s_hat_input_val) < 1.0e-30:
        s_hat_input_val = 1.0e-8
    return iota_input_val, s_hat_input_val


def _hngc_shear_correction(
    *,
    s_val: float,
    iota: np.ndarray,
    shat: np.ndarray,
    iota_input_val: float,
    s_hat_input_val: float,
    include_shear_variation: bool,
) -> tuple[np.ndarray, float]:
    """Return the Hegna-Nakajima shear correction and scale factor."""

    if not include_shear_variation:
        return np.zeros_like(np.asarray(iota, dtype=float)), 1.0
    d_iota_d_s_1 = (
        -(iota_input_val / (2.0 * s_val)) * s_hat_input_val
        + (float(iota[0]) / (2.0 * s_val)) * float(shat[0])
    ) * np.ones_like(np.asarray(iota, dtype=float))
    return d_iota_d_s_1, float(shat[0]) / s_hat_input_val


def _hngc_pressure_correction(
    *,
    s_val: float,
    betaprim: float,
    B_reference: float,
    d_pressure_d_s: np.ndarray,
    include_pressure_variation: bool,
) -> tuple[np.ndarray, float]:
    """Return the Hegna-Nakajima pressure correction and scale factor."""

    pressure_profile = np.asarray(d_pressure_d_s, dtype=float)
    if not include_pressure_variation:
        return np.zeros_like(pressure_profile), 1.0

    drive = betaprim * B_reference**2 / (4.0 * np.sqrt(s_val))
    d_pressure_d_s_1 = drive * np.ones_like(pressure_profile) - _MU_0 * pressure_profile
    dp_ds_safe = np.where(np.abs(pressure_profile) < 1.0e-30, 1.0e-8, pressure_profile)
    pfac = drive / (_MU_0 * float(dp_ds_safe[0]))
    return d_pressure_d_s_1, float(pfac)


def _flux_surface_hngc_averages(
    *,
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    flipit: bool,
    lambmnc_b: np.ndarray,
    rmnc_b: np.ndarray,
    zmns_b: np.ndarray,
    numns_b: np.ndarray,
    gmnc_b: np.ndarray,
    res_theta: int,
    res_phi: int,
) -> tuple[float, float]:
    """Return ``D1`` and ``D2`` Hegna-Nakajima flux-surface averages."""

    theta_b_grid = np.linspace(-np.pi, np.pi, res_theta)
    phi_b_grid = np.linspace(-np.pi, np.pi, res_phi)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)

    angle_b_2D = _boozer_mode_angle(
        xm_b,
        xn_b,
        th_b_2D[None, None, :, :],
        ph_b_2D[None, None, :, :],
        flipit=bool(flipit),
    )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)
    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = _boozer_mode_sum(lambmnc_b, cosangle_b_2D)
    R_b_2D = _boozer_mode_sum(rmnc_b, cosangle_b_2D)
    d_R_b_d_theta_b_2D = -_boozer_mode_sum(rmnc_b, msinangle_b_2D)
    d_R_b_d_phi_b_2D = _boozer_mode_sum(rmnc_b, ncosangle_b_2D)
    d_Z_b_d_theta_b_2D = _boozer_mode_sum(zmns_b, mcosangle_b_2D)
    d_Z_b_d_phi_b_2D = -_boozer_mode_sum(zmns_b, ncosangle_b_2D)
    nu_b_2D = _boozer_mode_sum(numns_b, sinangle_b_2D)
    d_nu_b_d_theta_b_2D = _boozer_mode_sum(numns_b, mcosangle_b_2D)
    d_nu_b_d_phi_b_2D = -_boozer_mode_sum(numns_b, nsinangle_b_2D)
    sqrt_g_booz_2D = _boozer_mode_sum(gmnc_b, cosangle_b_2D)

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

    return (
        _surface_average_2d(g_sup_psi_psi_2D_inv[0, 0], theta_b_grid, phi_b_grid),
        _surface_average_2d(lam_over_g_2D[0, 0], theta_b_grid, phi_b_grid),
    )


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

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b

    R_b = _boozer_mode_sum(rmnc_b, cosangle_b)
    d_R_b_d_s = _boozer_mode_sum(d_rmnc_b_d_s, cosangle_b)
    d_R_b_d_theta_b = -_boozer_mode_sum(rmnc_b, msinangle_b)
    d_R_b_d_phi_b = _boozer_mode_sum(rmnc_b, nsinangle_b)

    Z_b = _boozer_mode_sum(zmns_b, sinangle_b)
    d_Z_b_d_s = _boozer_mode_sum(d_zmns_b_d_s, sinangle_b)
    d_Z_b_d_theta_b = _boozer_mode_sum(zmns_b, mcosangle_b)
    d_Z_b_d_phi_b = -_boozer_mode_sum(zmns_b, ncosangle_b)

    nu_b = _boozer_mode_sum(numns_b, sinangle_b)
    d_nu_b_d_s = _boozer_mode_sum(d_numns_b_d_s, sinangle_b)
    d_nu_b_d_theta_b = _boozer_mode_sum(numns_b, mcosangle_b)
    d_nu_b_d_phi_b = -_boozer_mode_sum(numns_b, ncosangle_b)

    sqrt_g_booz = _boozer_mode_sum(gmnc_b, cosangle_b)
    d_sqrt_g_booz_d_theta_b = -_boozer_mode_sum(gmnc_b, msinangle_b)
    d_sqrt_g_booz_d_phi_b = _boozer_mode_sum(gmnc_b, nsinangle_b)
    modB_b = _boozer_mode_sum(bmnc_b, cosangle_b)
    d_B_b_d_s = _boozer_mode_sum(d_bmnc_b_d_s, cosangle_b)

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
