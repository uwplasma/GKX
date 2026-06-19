"""Boozer field-line numerical helpers for imported VMEC geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.integrate import simpson as _simps
from scipy.interpolate import InterpolatedUnivariateSpline


_MU_0 = 4.0 * np.pi * 1.0e-7


@dataclass(frozen=True)
class _FieldlineBoozerTensors:
    R_b: np.ndarray
    d_R_b_d_s: np.ndarray
    d_R_b_d_theta_b: np.ndarray
    d_R_b_d_phi_b: np.ndarray
    Z_b: np.ndarray
    d_Z_b_d_s: np.ndarray
    d_Z_b_d_theta_b: np.ndarray
    d_Z_b_d_phi_b: np.ndarray
    nu_b: np.ndarray
    d_nu_b_d_s: np.ndarray
    d_nu_b_d_theta_b: np.ndarray
    d_nu_b_d_phi_b: np.ndarray
    sqrt_g_booz: np.ndarray
    d_sqrt_g_booz_d_theta_b: np.ndarray
    d_sqrt_g_booz_d_phi_b: np.ndarray
    modB_b: np.ndarray
    d_B_b_d_s: np.ndarray


@dataclass(frozen=True)
class _FieldlineCartesianDerivatives:
    d_X_d_theta_b: np.ndarray
    d_X_d_phi_b: np.ndarray
    d_X_d_s: np.ndarray
    d_Y_d_theta_b: np.ndarray
    d_Y_d_phi_b: np.ndarray
    d_Y_d_s: np.ndarray


@dataclass(frozen=True)
class _FieldlineCoordinateGradients:
    grad_psi_X: np.ndarray
    grad_psi_Y: np.ndarray
    grad_psi_Z: np.ndarray
    grad_theta_b_X: np.ndarray
    grad_theta_b_Y: np.ndarray
    grad_theta_b_Z: np.ndarray
    grad_phi_b_X: np.ndarray
    grad_phi_b_Y: np.ndarray
    grad_phi_b_Z: np.ndarray


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


def _boozer_trig_basis(
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Boozer trigonometric basis arrays and mode-weighted derivatives."""

    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mode_index = (slice(None),) + (None,) * (angle.ndim - 1)
    m = xm_b[mode_index]
    n = xn_b[mode_index]
    return (
        cosangle,
        sinangle,
        m * cosangle,
        m * sinangle,
        n * cosangle,
        n * sinangle,
    )


def _fieldline_boozer_tensors(
    *,
    rmnc_b: np.ndarray,
    zmns_b: np.ndarray,
    numns_b: np.ndarray,
    d_rmnc_b_d_s: np.ndarray,
    d_zmns_b_d_s: np.ndarray,
    d_numns_b_d_s: np.ndarray,
    gmnc_b: np.ndarray,
    bmnc_b: np.ndarray,
    d_bmnc_b_d_s: np.ndarray,
    cosangle_b: np.ndarray,
    sinangle_b: np.ndarray,
    mcosangle_b: np.ndarray,
    msinangle_b: np.ndarray,
    ncosangle_b: np.ndarray,
    nsinangle_b: np.ndarray,
) -> _FieldlineBoozerTensors:
    """Evaluate Boozer spectral tensors and first derivatives on a field line."""

    return _FieldlineBoozerTensors(
        R_b=_boozer_mode_sum(rmnc_b, cosangle_b),
        d_R_b_d_s=_boozer_mode_sum(d_rmnc_b_d_s, cosangle_b),
        d_R_b_d_theta_b=-_boozer_mode_sum(rmnc_b, msinangle_b),
        d_R_b_d_phi_b=_boozer_mode_sum(rmnc_b, nsinangle_b),
        Z_b=_boozer_mode_sum(zmns_b, sinangle_b),
        d_Z_b_d_s=_boozer_mode_sum(d_zmns_b_d_s, sinangle_b),
        d_Z_b_d_theta_b=_boozer_mode_sum(zmns_b, mcosangle_b),
        d_Z_b_d_phi_b=-_boozer_mode_sum(zmns_b, ncosangle_b),
        nu_b=_boozer_mode_sum(numns_b, sinangle_b),
        d_nu_b_d_s=_boozer_mode_sum(d_numns_b_d_s, sinangle_b),
        d_nu_b_d_theta_b=_boozer_mode_sum(numns_b, mcosangle_b),
        d_nu_b_d_phi_b=-_boozer_mode_sum(numns_b, ncosangle_b),
        sqrt_g_booz=_boozer_mode_sum(gmnc_b, cosangle_b),
        d_sqrt_g_booz_d_theta_b=-_boozer_mode_sum(gmnc_b, msinangle_b),
        d_sqrt_g_booz_d_phi_b=_boozer_mode_sum(gmnc_b, nsinangle_b),
        modB_b=_boozer_mode_sum(bmnc_b, cosangle_b),
        d_B_b_d_s=_boozer_mode_sum(d_bmnc_b_d_s, cosangle_b),
    )


def _fieldline_cartesian_derivatives(
    *,
    tensors: _FieldlineBoozerTensors,
    phi_b: np.ndarray,
) -> _FieldlineCartesianDerivatives:
    """Convert Boozer R/nu derivatives into cylindrical X/Y derivatives."""

    phi_cyl = phi_b - tensors.nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)
    return _FieldlineCartesianDerivatives(
        d_X_d_theta_b=(
            tensors.d_R_b_d_theta_b * cosphi
            - tensors.R_b * sinphi * (-tensors.d_nu_b_d_theta_b)
        ),
        d_X_d_phi_b=(
            tensors.d_R_b_d_phi_b * cosphi
            - tensors.R_b * sinphi * (1.0 - tensors.d_nu_b_d_phi_b)
        ),
        d_X_d_s=(
            tensors.d_R_b_d_s * cosphi
            - tensors.R_b * sinphi * (-tensors.d_nu_b_d_s)
        ),
        d_Y_d_theta_b=(
            tensors.d_R_b_d_theta_b * sinphi
            + tensors.R_b * cosphi * (-tensors.d_nu_b_d_theta_b)
        ),
        d_Y_d_phi_b=(
            tensors.d_R_b_d_phi_b * sinphi
            + tensors.R_b * cosphi * (1.0 - tensors.d_nu_b_d_phi_b)
        ),
        d_Y_d_s=(
            tensors.d_R_b_d_s * sinphi
            + tensors.R_b * cosphi * (-tensors.d_nu_b_d_s)
        ),
    )


def _fieldline_coordinate_gradients(
    *,
    tensors: _FieldlineBoozerTensors,
    cartesian: _FieldlineCartesianDerivatives,
    edge_toroidal_flux_over_2pi: float,
) -> _FieldlineCoordinateGradients:
    """Return coordinate gradients from field-line basis-vector cross products."""

    grad_psi_X = (
        cartesian.d_Y_d_theta_b * tensors.d_Z_b_d_phi_b
        - tensors.d_Z_b_d_theta_b * cartesian.d_Y_d_phi_b
    ) / tensors.sqrt_g_booz
    grad_psi_Y = (
        tensors.d_Z_b_d_theta_b * cartesian.d_X_d_phi_b
        - cartesian.d_X_d_theta_b * tensors.d_Z_b_d_phi_b
    ) / tensors.sqrt_g_booz
    grad_psi_Z = (
        cartesian.d_X_d_theta_b * cartesian.d_Y_d_phi_b
        - cartesian.d_Y_d_theta_b * cartesian.d_X_d_phi_b
    ) / tensors.sqrt_g_booz
    denominator = tensors.sqrt_g_booz * edge_toroidal_flux_over_2pi
    return _FieldlineCoordinateGradients(
        grad_psi_X=grad_psi_X,
        grad_psi_Y=grad_psi_Y,
        grad_psi_Z=grad_psi_Z,
        grad_theta_b_X=(
            cartesian.d_Y_d_phi_b * tensors.d_Z_b_d_s
            - tensors.d_Z_b_d_phi_b * cartesian.d_Y_d_s
        )
        / denominator,
        grad_theta_b_Y=(
            tensors.d_Z_b_d_phi_b * cartesian.d_X_d_s
            - cartesian.d_X_d_phi_b * tensors.d_Z_b_d_s
        )
        / denominator,
        grad_theta_b_Z=(
            cartesian.d_X_d_phi_b * cartesian.d_Y_d_s
            - cartesian.d_Y_d_phi_b * cartesian.d_X_d_s
        )
        / denominator,
        grad_phi_b_X=(
            cartesian.d_Y_d_s * tensors.d_Z_b_d_theta_b
            - tensors.d_Z_b_d_s * cartesian.d_Y_d_theta_b
        )
        / denominator,
        grad_phi_b_Y=(
            tensors.d_Z_b_d_s * cartesian.d_X_d_theta_b
            - cartesian.d_X_d_s * tensors.d_Z_b_d_theta_b
        )
        / denominator,
        grad_phi_b_Z=(
            cartesian.d_X_d_s * cartesian.d_Y_d_theta_b
            - cartesian.d_Y_d_s * cartesian.d_X_d_theta_b
        )
        / denominator,
    )


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

    (
        cosangle_b_2D,
        sinangle_b_2D,
        mcosangle_b_2D,
        msinangle_b_2D,
        ncosangle_b_2D,
        nsinangle_b_2D,
    ) = _boozer_trig_basis(xm_b, xn_b, angle_b_2D)

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


__all__ = [
    "_MU_0",
    "_axisym_flip_required",
    "_boozer_mode_angle",
    "_boozer_mode_sum",
    "_boozer_trig_basis",
    "_centered_fieldline_integral",
    "_fieldline_boozer_coordinates",
    "_fieldline_boozer_tensors",
    "_fieldline_cartesian_derivatives",
    "_fieldline_coordinate_gradients",
    "_flux_surface_hngc_averages",
    "_hngc_pressure_correction",
    "_hngc_shear_correction",
    "_input_iota_shear",
    "_safe_mode_denominator",
    "_surface_average_2d",
    "_validated_reference_scales",
]
