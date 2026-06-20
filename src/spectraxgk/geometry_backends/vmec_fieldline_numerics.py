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


@dataclass(frozen=True)
class _FieldlineAlphaGradients:
    grad_alpha_X: np.ndarray
    grad_alpha_Y: np.ndarray
    grad_alpha_Z: np.ndarray
    g_sup_psi_psi: np.ndarray
    grad_alpha_dot_grad_psi: np.ndarray


@dataclass(frozen=True)
class _FieldlineLocalShear:
    D_HNGC: np.ndarray
    L0: np.ndarray
    L1: np.ndarray


@dataclass(frozen=True)
class _FieldlineMetricDrifts:
    bmag: np.ndarray
    gradpar_theta_b: np.ndarray
    gradpar_phi: np.ndarray
    gds2: np.ndarray
    gds21: np.ndarray
    gds22: np.ndarray
    gbdrift: np.ndarray
    gbdrift0: np.ndarray
    cvdrift: np.ndarray
    cvdrift0: np.ndarray
    grho: np.ndarray
    grad_y: np.ndarray
    grad_x: np.ndarray


@dataclass(frozen=True)
class _FieldlineCurvatureComponents:
    b_cross_kappa_dot_grad_alpha: np.ndarray
    b_cross_kappa_dot_grad_psi: np.ndarray


@dataclass(frozen=True)
class _FieldlineMetricProfiles:
    bmag: np.ndarray
    gradpar_theta_b: np.ndarray
    gradpar_phi: np.ndarray
    gds2: np.ndarray
    gds21: np.ndarray
    gds22: np.ndarray
    grho: np.ndarray


@dataclass(frozen=True)
class _FieldlineDriftProfiles:
    gbdrift: np.ndarray
    gbdrift0: np.ndarray
    cvdrift: np.ndarray
    cvdrift0: np.ndarray


@dataclass(frozen=True)
class _FieldlineGradientVectors:
    grad_y: np.ndarray
    grad_x: np.ndarray


@dataclass(frozen=True)
class _FluxSurfaceGrid:
    theta_b_grid: np.ndarray
    phi_b_grid: np.ndarray
    theta_b_2d: np.ndarray
    phi_b_2d: np.ndarray


@dataclass(frozen=True)
class _FluxSurfaceBoozerGeometry:
    lambda_b: np.ndarray
    R_b: np.ndarray
    d_R_b_d_theta_b: np.ndarray
    d_R_b_d_phi_b: np.ndarray
    d_Z_b_d_theta_b: np.ndarray
    d_Z_b_d_phi_b: np.ndarray
    nu_b: np.ndarray
    d_nu_b_d_theta_b: np.ndarray
    d_nu_b_d_phi_b: np.ndarray
    sqrt_g_booz: np.ndarray


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


def _fieldline_alpha_gradients(
    *,
    gradients: _FieldlineCoordinateGradients,
    phi_b: np.ndarray,
    zeta_center: float,
    d_iota_d_s: np.ndarray,
    iota: np.ndarray,
    edge_toroidal_flux_over_2pi: float,
) -> _FieldlineAlphaGradients:
    """Return ``grad(alpha)`` and its dot product with ``grad(psi)``."""

    grad_psi_X = gradients.grad_psi_X
    grad_psi_Y = gradients.grad_psi_Y
    grad_psi_Z = gradients.grad_psi_Z
    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2
    radial_shear = (
        -(phi_b - zeta_center)
        * d_iota_d_s[:, None, None]
        / edge_toroidal_flux_over_2pi
    )
    grad_alpha_X = (
        radial_shear * grad_psi_X
        + gradients.grad_theta_b_X
        - iota[:, None, None] * gradients.grad_phi_b_X
    )
    grad_alpha_Y = (
        radial_shear * grad_psi_Y
        + gradients.grad_theta_b_Y
        - iota[:, None, None] * gradients.grad_phi_b_Y
    )
    grad_alpha_Z = (
        radial_shear * grad_psi_Z
        + gradients.grad_theta_b_Z
        - iota[:, None, None] * gradients.grad_phi_b_Z
    )
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )
    return _FieldlineAlphaGradients(
        grad_alpha_X=grad_alpha_X,
        grad_alpha_Y=grad_alpha_Y,
        grad_alpha_Z=grad_alpha_Z,
        g_sup_psi_psi=g_sup_psi_psi,
        grad_alpha_dot_grad_psi=grad_alpha_dot_grad_psi,
    )


def _fieldline_local_shear(
    *,
    edge_toroidal_flux_over_2pi: float,
    d_iota_d_s: np.ndarray,
    d_iota_d_s_1: np.ndarray,
    d_pressure_d_s_1: np.ndarray,
    Vprime: np.ndarray,
    G: np.ndarray,
    iota: np.ndarray,
    boozer_i: np.ndarray,
    phi_b: np.ndarray,
    zeta_center: float,
    intinv_g: np.ndarray,
    int_lam_div_g: np.ndarray,
    D1: float,
    D2: float,
    g_sup_psi_psi: np.ndarray,
    grad_alpha_dot_grad_psi: np.ndarray,
) -> _FieldlineLocalShear:
    """Return Hegna-Nakajima local-shear corrections ``D_HNGC``, ``L0``, ``L1``."""

    D_HNGC = (
        1.0
        / edge_toroidal_flux_over_2pi
        * (
            d_iota_d_s_1[:, None, None] * (intinv_g / D1 - phi_b + zeta_center)
            - d_pressure_d_s_1[:, None, None]
            * Vprime[:, None, None]
            * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
            * (int_lam_div_g - D2 * intinv_g / D1)
        )
    )
    L0 = -1.0 * (
        grad_alpha_dot_grad_psi / g_sup_psi_psi
        + 1.0
        / edge_toroidal_flux_over_2pi
        * d_iota_d_s[:, None, None]
        * (phi_b - zeta_center)
    )
    L1 = (
        -1.0
        / edge_toroidal_flux_over_2pi
        * d_iota_d_s_1[:, None, None]
        * (phi_b - zeta_center)
        + grad_alpha_dot_grad_psi / g_sup_psi_psi
        - D_HNGC
    )
    return _FieldlineLocalShear(D_HNGC=D_HNGC, L0=L0, L1=L1)


def _fieldline_curvature_components(
    *,
    tensors: _FieldlineBoozerTensors,
    shear: _FieldlineLocalShear,
    d_pressure_d_s: np.ndarray,
    beta_b: np.ndarray,
    G: np.ndarray,
    iota: np.ndarray,
    boozer_i: np.ndarray,
    edge_toroidal_flux_over_2pi: float,
) -> _FieldlineCurvatureComponents:
    """Return curvature terms entering VMEC magnetic-drift coefficients."""

    modB_b = tensors.modB_b
    sqrt_g_booz = tensors.sqrt_g_booz
    boozer_current = G[:, None] + iota[:, None] * boozer_i[:, None]
    boozer_current_3d = G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None]
    kappa_g = (
        G[:, None] * tensors.d_sqrt_g_booz_d_theta_b
        - boozer_i[:, None] * tensors.d_sqrt_g_booz_d_phi_b
    ) / (2.0 * sqrt_g_booz * boozer_current)
    kappa_n = (
        (modB_b * tensors.d_B_b_d_s + _MU_0 * d_pressure_d_s[:, None, None])
        / (modB_b**2 * edge_toroidal_flux_over_2pi)
        - beta_b
        * tensors.d_sqrt_g_booz_d_phi_b
        / (2.0 * sqrt_g_booz * boozer_current_3d)
        + shear.L0
        * (
            G[:, None] * tensors.d_sqrt_g_booz_d_theta_b
            - boozer_i[:, None] * tensors.d_sqrt_g_booz_d_phi_b
        )
        / (2.0 * sqrt_g_booz * boozer_current)
    )
    return _FieldlineCurvatureComponents(
        b_cross_kappa_dot_grad_alpha=(kappa_n + kappa_g * shear.L1) * modB_b**2,
        b_cross_kappa_dot_grad_psi=kappa_g * modB_b**2,
    )


def _fieldline_metric_profiles(
    *,
    tensors: _FieldlineBoozerTensors,
    alpha_gradients: _FieldlineAlphaGradients,
    shear: _FieldlineLocalShear,
    s: np.ndarray,
    shat: np.ndarray,
    sfac: float,
    L_reference: float,
    B_reference: float,
    iota: np.ndarray,
) -> _FieldlineMetricProfiles:
    """Return normalized metric profiles before magnetic-drift assembly."""

    modB_b = tensors.modB_b
    grad_alpha_dot_grad_alpha_b = (
        modB_b**2 / alpha_gradients.g_sup_psi_psi
        + alpha_gradients.g_sup_psi_psi * shear.L1**2
    )
    grad_alpha_dot_grad_psi_b = alpha_gradients.g_sup_psi_psi * shear.L1
    return _FieldlineMetricProfiles(
        bmag=modB_b / B_reference,
        gradpar_theta_b=-L_reference / modB_b / tensors.sqrt_g_booz * iota[:, None, None],
        gradpar_phi=L_reference / modB_b / tensors.sqrt_g_booz,
        gds2=grad_alpha_dot_grad_alpha_b * L_reference**2 * s[:, None, None],
        gds21=grad_alpha_dot_grad_psi_b * sfac * shat[:, None, None] / B_reference,
        gds22=(
            alpha_gradients.g_sup_psi_psi
            * (sfac * shat[:, None, None]) ** 2
            / (L_reference**2 * B_reference**2 * s[:, None, None])
        ),
        grho=np.sqrt(
            alpha_gradients.g_sup_psi_psi
            / (L_reference**2 * B_reference**2 * s[:, None, None])
        ),
    )


def _fieldline_drift_profiles(
    *,
    curvature: _FieldlineCurvatureComponents,
    tensors: _FieldlineBoozerTensors,
    s: np.ndarray,
    shat: np.ndarray,
    sfac: float,
    pfac: float,
    L_reference: float,
    B_reference: float,
    toroidal_flux_sign: float,
    edge_toroidal_flux_over_2pi: float,
    d_pressure_d_s: np.ndarray,
) -> _FieldlineDriftProfiles:
    """Return normalized grad-B and curvature drift profiles."""

    modB_b = tensors.modB_b
    sqrt_s = np.sqrt(s)
    gbdrift0 = (
        -curvature.b_cross_kappa_dot_grad_psi
        * 2.0
        * sfac
        * shat[:, None, None]
        / (modB_b**2 * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift = (
        -2.0
        * B_reference
        * L_reference**2
        * sqrt_s[:, None, None]
        * curvature.b_cross_kappa_dot_grad_alpha
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
        / (edge_toroidal_flux_over_2pi * modB_b**2)
    )
    return _FieldlineDriftProfiles(
        gbdrift=gbdrift,
        gbdrift0=gbdrift0,
        cvdrift=cvdrift,
        cvdrift0=gbdrift0,
    )


def _fieldline_gradient_vectors(
    *,
    gradients: _FieldlineCoordinateGradients,
    alpha_gradients: _FieldlineAlphaGradients,
    s: np.ndarray,
    shat: np.ndarray,
    sfac: float,
    L_reference: float,
    B_reference: float,
) -> _FieldlineGradientVectors:
    """Return solver-normalized ``x`` and ``y`` gradient-vector components."""

    sqrt_s = np.sqrt(s)
    grad_y = (
        L_reference
        * sqrt_s[:, None, None]
        * np.array(
            [
                alpha_gradients.grad_alpha_X,
                alpha_gradients.grad_alpha_Y,
                alpha_gradients.grad_alpha_Z,
            ]
        )
    )
    grad_x = (
        sfac
        * shat[:, None, None]
        * np.array(
            [
                gradients.grad_psi_X,
                gradients.grad_psi_Y,
                gradients.grad_psi_Z,
            ]
        )
        / (L_reference * B_reference * sqrt_s[:, None, None])
    )
    return _FieldlineGradientVectors(grad_y=grad_y, grad_x=grad_x)


def _pack_fieldline_metric_drifts(
    metrics: _FieldlineMetricProfiles,
    drifts: _FieldlineDriftProfiles,
    vectors: _FieldlineGradientVectors,
) -> _FieldlineMetricDrifts:
    return _FieldlineMetricDrifts(
        bmag=metrics.bmag,
        gradpar_theta_b=metrics.gradpar_theta_b,
        gradpar_phi=metrics.gradpar_phi,
        gds2=metrics.gds2,
        gds21=metrics.gds21,
        gds22=metrics.gds22,
        gbdrift=drifts.gbdrift,
        gbdrift0=drifts.gbdrift0,
        cvdrift=drifts.cvdrift,
        cvdrift0=drifts.cvdrift0,
        grho=metrics.grho,
        grad_y=vectors.grad_y,
        grad_x=vectors.grad_x,
    )


def _fieldline_metric_drifts(
    *,
    tensors: _FieldlineBoozerTensors,
    gradients: _FieldlineCoordinateGradients,
    alpha_gradients: _FieldlineAlphaGradients,
    shear: _FieldlineLocalShear,
    s: np.ndarray,
    shat: np.ndarray,
    sfac: float,
    pfac: float,
    L_reference: float,
    B_reference: float,
    toroidal_flux_sign: float,
    edge_toroidal_flux_over_2pi: float,
    d_pressure_d_s: np.ndarray,
    beta_b: np.ndarray,
    G: np.ndarray,
    iota: np.ndarray,
    boozer_i: np.ndarray,
) -> _FieldlineMetricDrifts:
    """Return normalized VMEC metric and magnetic-drift coefficients."""

    curvature = _fieldline_curvature_components(
        tensors=tensors,
        shear=shear,
        d_pressure_d_s=d_pressure_d_s,
        beta_b=beta_b,
        G=G,
        iota=iota,
        boozer_i=boozer_i,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
    )
    metrics = _fieldline_metric_profiles(
        tensors=tensors,
        alpha_gradients=alpha_gradients,
        shear=shear,
        s=s,
        shat=shat,
        sfac=sfac,
        L_reference=L_reference,
        B_reference=B_reference,
        iota=iota,
    )
    drifts = _fieldline_drift_profiles(
        curvature=curvature,
        tensors=tensors,
        s=s,
        shat=shat,
        sfac=sfac,
        pfac=pfac,
        L_reference=L_reference,
        B_reference=B_reference,
        toroidal_flux_sign=toroidal_flux_sign,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
        d_pressure_d_s=d_pressure_d_s,
    )
    vectors = _fieldline_gradient_vectors(
        gradients=gradients,
        alpha_gradients=alpha_gradients,
        s=s,
        shat=shat,
        sfac=sfac,
        L_reference=L_reference,
        B_reference=B_reference,
    )
    return _pack_fieldline_metric_drifts(metrics, drifts, vectors)


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


def _flux_surface_grid(res_theta: int, res_phi: int) -> _FluxSurfaceGrid:
    theta_b_grid = np.linspace(-np.pi, np.pi, res_theta)
    phi_b_grid = np.linspace(-np.pi, np.pi, res_phi)
    theta_b_2d, phi_b_2d = np.meshgrid(theta_b_grid, phi_b_grid)
    return _FluxSurfaceGrid(
        theta_b_grid=theta_b_grid,
        phi_b_grid=phi_b_grid,
        theta_b_2d=theta_b_2d,
        phi_b_2d=phi_b_2d,
    )


def _flux_surface_boozer_geometry(
    *,
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    flipit: bool,
    lambmnc_b: np.ndarray,
    rmnc_b: np.ndarray,
    zmns_b: np.ndarray,
    numns_b: np.ndarray,
    gmnc_b: np.ndarray,
    grid: _FluxSurfaceGrid,
) -> _FluxSurfaceBoozerGeometry:
    angle_b_2d = _boozer_mode_angle(
        xm_b,
        xn_b,
        grid.theta_b_2d[None, None, :, :],
        grid.phi_b_2d[None, None, :, :],
        flipit=bool(flipit),
    )
    cosangle, sinangle, mcosangle, msinangle, ncosangle, nsinangle = (
        _boozer_trig_basis(xm_b, xn_b, angle_b_2d)
    )
    return _FluxSurfaceBoozerGeometry(
        lambda_b=_boozer_mode_sum(lambmnc_b, cosangle),
        R_b=_boozer_mode_sum(rmnc_b, cosangle),
        d_R_b_d_theta_b=-_boozer_mode_sum(rmnc_b, msinangle),
        d_R_b_d_phi_b=_boozer_mode_sum(rmnc_b, ncosangle),
        d_Z_b_d_theta_b=_boozer_mode_sum(zmns_b, mcosangle),
        d_Z_b_d_phi_b=-_boozer_mode_sum(zmns_b, ncosangle),
        nu_b=_boozer_mode_sum(numns_b, sinangle),
        d_nu_b_d_theta_b=_boozer_mode_sum(numns_b, mcosangle),
        d_nu_b_d_phi_b=-_boozer_mode_sum(numns_b, nsinangle),
        sqrt_g_booz=_boozer_mode_sum(gmnc_b, cosangle),
    )


def _flux_surface_grad_psi_norm_inv(
    geometry: _FluxSurfaceBoozerGeometry,
    grid: _FluxSurfaceGrid,
) -> np.ndarray:
    ph_nat_2d = grid.phi_b_2d - geometry.nu_b
    sinphi_2d = np.sin(ph_nat_2d)
    cosphi_2d = np.cos(ph_nat_2d)
    d_x_d_theta_b = (
        geometry.d_R_b_d_theta_b * cosphi_2d
        - geometry.R_b * sinphi_2d * (-geometry.d_nu_b_d_theta_b)
    )
    d_x_d_phi = (
        geometry.d_R_b_d_phi_b * cosphi_2d
        - geometry.R_b * sinphi_2d * (1.0 - geometry.d_nu_b_d_phi_b)
    )
    d_y_d_theta_b = (
        geometry.d_R_b_d_theta_b * sinphi_2d
        + geometry.R_b * cosphi_2d * (-geometry.d_nu_b_d_theta_b)
    )
    d_y_d_phi = (
        geometry.d_R_b_d_phi_b * sinphi_2d
        + geometry.R_b * cosphi_2d * (1.0 - geometry.d_nu_b_d_phi_b)
    )
    grad_psi_x = (
        d_y_d_theta_b * geometry.d_Z_b_d_phi_b
        - geometry.d_Z_b_d_theta_b * d_y_d_phi
    ) / geometry.sqrt_g_booz
    grad_psi_y = (
        geometry.d_Z_b_d_theta_b * d_x_d_phi
        - d_x_d_theta_b * geometry.d_Z_b_d_phi_b
    ) / geometry.sqrt_g_booz
    grad_psi_z = (
        d_x_d_theta_b * d_y_d_phi - d_y_d_theta_b * d_x_d_phi
    ) / geometry.sqrt_g_booz
    g_sup_psi_psi = grad_psi_x**2 + grad_psi_y**2 + grad_psi_z**2
    return 1.0 / g_sup_psi_psi


def _flux_surface_hngc_average_pair(
    geometry: _FluxSurfaceBoozerGeometry,
    grid: _FluxSurfaceGrid,
) -> tuple[float, float]:
    g_sup_psi_psi_inv = _flux_surface_grad_psi_norm_inv(geometry, grid)
    lam_over_g = geometry.lambda_b * g_sup_psi_psi_inv
    return (
        _surface_average_2d(
            g_sup_psi_psi_inv[0, 0],
            grid.theta_b_grid,
            grid.phi_b_grid,
        ),
        _surface_average_2d(
            lam_over_g[0, 0],
            grid.theta_b_grid,
            grid.phi_b_grid,
        ),
    )


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

    grid = _flux_surface_grid(res_theta, res_phi)
    geometry = _flux_surface_boozer_geometry(
        xm_b=xm_b,
        xn_b=xn_b,
        flipit=flipit,
        lambmnc_b=lambmnc_b,
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        gmnc_b=gmnc_b,
        grid=grid,
    )
    return _flux_surface_hngc_average_pair(geometry, grid)


__all__ = [
    "_MU_0",
    "_axisym_flip_required",
    "_boozer_mode_angle",
    "_boozer_mode_sum",
    "_boozer_trig_basis",
    "_centered_fieldline_integral",
    "_fieldline_boozer_coordinates",
    "_fieldline_boozer_tensors",
    "_fieldline_alpha_gradients",
    "_fieldline_cartesian_derivatives",
    "_fieldline_coordinate_gradients",
    "_fieldline_local_shear",
    "_fieldline_metric_drifts",
    "_flux_surface_hngc_averages",
    "_hngc_pressure_correction",
    "_hngc_shear_correction",
    "_input_iota_shear",
    "_safe_mode_denominator",
    "_surface_average_2d",
    "_validated_reference_scales",
]
