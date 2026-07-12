"""Boozer field-line derivative algebra for differentiable geometry bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.integrate import simpson as _simps
from scipy.interpolate import InterpolatedUnivariateSpline

from spectraxgk.geometry.vmec_field_line_sampling import (
    _FluxSurfaceBoozerGeometry,
    _FluxSurfaceGrid,
    _MU_0,
    _boozer_mode_angle,
    _boozer_mode_sum,
    _boozer_trig_basis,
    _fieldline_alpha_gradients,
    _fieldline_cartesian_derivatives,
    _fieldline_coordinate_gradients,
    _fieldline_local_shear,
    _fieldline_metric_drifts,
    _Struct,
)

if TYPE_CHECKING:
    from spectraxgk.geometry.vmec_state_controls import (
        _BoozerFieldlineSamples,
        _VMECFieldlineScalars,
    )


@dataclass(frozen=True)
class BoozerFieldLineDerivatives:
    """Spectral sums and first derivatives on one Boozer field line."""

    phi_b: jnp.ndarray
    r_b: jnp.ndarray
    d_mod_b_d_s: jnp.ndarray
    d_mod_b_d_theta: jnp.ndarray
    d_mod_b_d_phi: jnp.ndarray
    d_r_b_d_s: jnp.ndarray
    d_r_b_d_theta: jnp.ndarray
    d_r_b_d_phi: jnp.ndarray
    d_z_b_d_s: jnp.ndarray
    d_z_b_d_theta: jnp.ndarray
    d_z_b_d_phi: jnp.ndarray
    nu_b: jnp.ndarray
    d_nu_b_d_s: jnp.ndarray
    d_nu_b_d_theta: jnp.ndarray
    d_nu_b_d_phi: jnp.ndarray


@dataclass(frozen=True)
class BoozerCartesianDerivatives:
    """Cylindrical x/y derivatives derived from Boozer R and nu harmonics."""

    d_x_d_theta: jnp.ndarray
    d_x_d_phi: jnp.ndarray
    d_x_d_s: jnp.ndarray
    d_y_d_theta: jnp.ndarray
    d_y_d_phi: jnp.ndarray
    d_y_d_s: jnp.ndarray


@dataclass(frozen=True)
class BoozerCoordinateGradients:
    """Contravariant gradients for psi, theta, and phi coordinates."""

    grad_psi_x: jnp.ndarray
    grad_psi_y: jnp.ndarray
    grad_psi_z: jnp.ndarray
    grad_theta_x: jnp.ndarray
    grad_theta_y: jnp.ndarray
    grad_theta_z: jnp.ndarray
    grad_phi_x: jnp.ndarray
    grad_phi_y: jnp.ndarray
    grad_phi_z: jnp.ndarray


def evaluate_boozer_field_line_derivatives(
    out: dict[str, Any],
    *,
    theta_closed: jnp.ndarray,
    alpha: float,
    iota_safe: jnp.ndarray,
    base_dtype: Any,
    bmnc_b: jnp.ndarray,
    d_bmnc_b_d_s: jnp.ndarray,
    rmnc_b: jnp.ndarray,
    d_rmnc_b_d_s: jnp.ndarray,
    zmns_b: jnp.ndarray,
    d_zmns_b_d_s: jnp.ndarray,
    numns_b: jnp.ndarray,
    d_numns_b_d_s: jnp.ndarray,
) -> BoozerFieldLineDerivatives:
    """Evaluate Boozer spectral sums and first derivatives on one field line."""

    m = jnp.asarray(out["ixm_b"], dtype=base_dtype)
    n = jnp.asarray(out["ixn_b"], dtype=base_dtype)
    phi_b = (theta_closed - jnp.asarray(float(alpha), dtype=base_dtype)) / iota_safe
    phase = m[:, None] * theta_closed[None, :] - n[:, None] * phi_b[None, :]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    m_cos = m[:, None] * cos_phase
    m_sin = m[:, None] * sin_phase
    n_cos = n[:, None] * cos_phase
    n_sin = n[:, None] * sin_phase

    return BoozerFieldLineDerivatives(
        phi_b=phi_b,
        r_b=jnp.sum(rmnc_b[:, None] * cos_phase, axis=0),
        d_mod_b_d_s=jnp.sum(d_bmnc_b_d_s[:, None] * cos_phase, axis=0),
        d_mod_b_d_theta=-jnp.sum(bmnc_b[:, None] * m_sin, axis=0),
        d_mod_b_d_phi=jnp.sum(bmnc_b[:, None] * n_sin, axis=0),
        d_r_b_d_s=jnp.sum(d_rmnc_b_d_s[:, None] * cos_phase, axis=0),
        d_r_b_d_theta=-jnp.sum(rmnc_b[:, None] * m_sin, axis=0),
        d_r_b_d_phi=jnp.sum(rmnc_b[:, None] * n_sin, axis=0),
        d_z_b_d_s=jnp.sum(d_zmns_b_d_s[:, None] * sin_phase, axis=0),
        d_z_b_d_theta=jnp.sum(zmns_b[:, None] * m_cos, axis=0),
        d_z_b_d_phi=-jnp.sum(zmns_b[:, None] * n_cos, axis=0),
        nu_b=jnp.sum(numns_b[:, None] * sin_phase, axis=0),
        d_nu_b_d_s=jnp.sum(d_numns_b_d_s[:, None] * sin_phase, axis=0),
        d_nu_b_d_theta=jnp.sum(numns_b[:, None] * m_cos, axis=0),
        d_nu_b_d_phi=-jnp.sum(numns_b[:, None] * n_cos, axis=0),
    )


def boozer_cartesian_derivatives(
    spectral: BoozerFieldLineDerivatives,
) -> BoozerCartesianDerivatives:
    """Convert Boozer spectral R/nu derivatives to cylindrical x/y derivatives."""

    phi_cyl = spectral.phi_b - spectral.nu_b
    sin_phi = jnp.sin(phi_cyl)
    cos_phi = jnp.cos(phi_cyl)
    return BoozerCartesianDerivatives(
        d_x_d_theta=(
            spectral.d_r_b_d_theta * cos_phi
            - spectral.r_b * sin_phi * (-spectral.d_nu_b_d_theta)
        ),
        d_x_d_phi=(
            spectral.d_r_b_d_phi * cos_phi
            - spectral.r_b * sin_phi * (1.0 - spectral.d_nu_b_d_phi)
        ),
        d_x_d_s=(
            spectral.d_r_b_d_s * cos_phi
            - spectral.r_b * sin_phi * (-spectral.d_nu_b_d_s)
        ),
        d_y_d_theta=(
            spectral.d_r_b_d_theta * sin_phi
            + spectral.r_b * cos_phi * (-spectral.d_nu_b_d_theta)
        ),
        d_y_d_phi=(
            spectral.d_r_b_d_phi * sin_phi
            + spectral.r_b * cos_phi * (1.0 - spectral.d_nu_b_d_phi)
        ),
        d_y_d_s=(
            spectral.d_r_b_d_s * sin_phi
            + spectral.r_b * cos_phi * (-spectral.d_nu_b_d_s)
        ),
    )


def boozer_coordinate_gradients(
    *,
    spectral: BoozerFieldLineDerivatives,
    cartesian: BoozerCartesianDerivatives,
    sqrt_g_booz: jnp.ndarray,
    etf_safe: jnp.ndarray,
) -> BoozerCoordinateGradients:
    """Return contravariant gradients for psi, theta, and phi coordinates."""

    grad_psi_x = (
        cartesian.d_y_d_theta * spectral.d_z_b_d_phi
        - spectral.d_z_b_d_theta * cartesian.d_y_d_phi
    ) / sqrt_g_booz
    grad_psi_y = (
        spectral.d_z_b_d_theta * cartesian.d_x_d_phi
        - cartesian.d_x_d_theta * spectral.d_z_b_d_phi
    ) / sqrt_g_booz
    grad_psi_z = (
        cartesian.d_x_d_theta * cartesian.d_y_d_phi
        - cartesian.d_y_d_theta * cartesian.d_x_d_phi
    ) / sqrt_g_booz
    denominator = sqrt_g_booz * etf_safe
    grad_theta_x = (
        cartesian.d_y_d_phi * spectral.d_z_b_d_s
        - spectral.d_z_b_d_phi * cartesian.d_y_d_s
    ) / denominator
    grad_theta_y = (
        spectral.d_z_b_d_phi * cartesian.d_x_d_s
        - cartesian.d_x_d_phi * spectral.d_z_b_d_s
    ) / denominator
    grad_theta_z = (
        cartesian.d_x_d_phi * cartesian.d_y_d_s
        - cartesian.d_y_d_phi * cartesian.d_x_d_s
    ) / denominator
    grad_phi_x = (
        cartesian.d_y_d_s * spectral.d_z_b_d_theta
        - spectral.d_z_b_d_s * cartesian.d_y_d_theta
    ) / denominator
    grad_phi_y = (
        spectral.d_z_b_d_s * cartesian.d_x_d_theta
        - cartesian.d_x_d_s * spectral.d_z_b_d_theta
    ) / denominator
    grad_phi_z = (
        cartesian.d_x_d_s * cartesian.d_y_d_theta
        - cartesian.d_y_d_s * cartesian.d_x_d_theta
    ) / denominator
    return BoozerCoordinateGradients(
        grad_psi_x=grad_psi_x,
        grad_psi_y=grad_psi_y,
        grad_psi_z=grad_psi_z,
        grad_theta_x=grad_theta_x,
        grad_theta_y=grad_theta_y,
        grad_theta_z=grad_theta_z,
        grad_phi_x=grad_phi_x,
        grad_phi_y=grad_phi_y,
        grad_phi_z=grad_phi_z,
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
        r_check[0, 0, 0] > r_check[0, 0, 1] or z_check[0, 0, 1] > z_check[0, 0, 0]
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
    cosangle, sinangle, mcosangle, msinangle, ncosangle, nsinangle = _boozer_trig_basis(
        xm_b, xn_b, angle_b_2d
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
    d_x_d_theta_b = geometry.d_R_b_d_theta_b * cosphi_2d - geometry.R_b * sinphi_2d * (
        -geometry.d_nu_b_d_theta_b
    )
    d_x_d_phi = geometry.d_R_b_d_phi_b * cosphi_2d - geometry.R_b * sinphi_2d * (
        1.0 - geometry.d_nu_b_d_phi_b
    )
    d_y_d_theta_b = geometry.d_R_b_d_theta_b * sinphi_2d + geometry.R_b * cosphi_2d * (
        -geometry.d_nu_b_d_theta_b
    )
    d_y_d_phi = geometry.d_R_b_d_phi_b * sinphi_2d + geometry.R_b * cosphi_2d * (
        1.0 - geometry.d_nu_b_d_phi_b
    )
    grad_psi_x = (
        d_y_d_theta_b * geometry.d_Z_b_d_phi_b - geometry.d_Z_b_d_theta_b * d_y_d_phi
    ) / geometry.sqrt_g_booz
    grad_psi_y = (
        geometry.d_Z_b_d_theta_b * d_x_d_phi - d_x_d_theta_b * geometry.d_Z_b_d_phi_b
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


@dataclass(frozen=True)
class _HNGCModeCorrections:
    beta_b: np.ndarray
    lambda_b: np.ndarray
    lambmnc_b: np.ndarray


@dataclass(frozen=True)
class _FieldlineMetricGeometry:
    gradients: Any
    alpha_gradients: Any


@dataclass(frozen=True)
class _FieldlineHNGCIntegrals:
    D1: float
    D2: float
    intinv_g: np.ndarray
    int_lam_div_g: np.ndarray


@dataclass(frozen=True)
class _FieldlineShearFactors:
    d_iota_d_s_1: np.ndarray
    d_pressure_d_s_1: np.ndarray
    sfac: float
    pfac: float


def _hngc_mode_corrections(
    scalars: _VMECFieldlineScalars, samples: _BoozerFieldlineSamples
) -> _HNGCModeCorrections:
    """Compute Lambda/beta Boozer-mode corrections for local equilibrium."""

    delmnc_b = np.zeros((scalars.ns, samples.mnmax_b))
    lambmnc_b = np.zeros((scalars.ns, samples.mnmax_b))
    betamns_b = np.zeros((scalars.ns, samples.mnmax_b))
    safe_denom_mn = _safe_mode_denominator(samples.xm_b, samples.xn_b, scalars.iota)
    delmnc_b[:, 1:] = samples.gmnc_b[:, 1:] / samples.Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        / scalars.edge_toroidal_flux_over_2pi
        * _MU_0
        * scalars.d_pressure_d_s[:, None]
        * samples.Vprime[:, None]
        / safe_denom_mn
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (
            samples.xm_b[1:] * scalars.G[:, None]
            + samples.xn_b[1:] * scalars.boozer_i[:, None]
        )
        / (
            safe_denom_mn
            * (scalars.G[:, None] + scalars.iota[:, None] * scalars.boozer_i[:, None])
        )
    )
    angle_b = _boozer_mode_angle(
        samples.xm_b,
        samples.xn_b,
        samples.theta_b,
        samples.phi_b,
        flipit=samples.flipit,
    )
    cosangle_b, sinangle_b, *_ = _boozer_trig_basis(samples.xm_b, samples.xn_b, angle_b)
    return _HNGCModeCorrections(
        beta_b=_boozer_mode_sum(betamns_b, sinangle_b),
        lambda_b=_boozer_mode_sum(lambmnc_b, cosangle_b),
        lambmnc_b=lambmnc_b,
    )


def _fieldline_metric_geometry(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
) -> _FieldlineMetricGeometry:
    """Return coordinate gradients needed for field-line metric coefficients."""

    cartesian = _fieldline_cartesian_derivatives(
        tensors=samples.tensors, phi_b=samples.phi_b
    )
    gradients = _fieldline_coordinate_gradients(
        tensors=samples.tensors,
        cartesian=cartesian,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
    )
    alpha_gradients = _fieldline_alpha_gradients(
        gradients=gradients,
        phi_b=samples.phi_b,
        zeta_center=scalars.zeta_center,
        d_iota_d_s=scalars.d_iota_d_s,
        iota=scalars.iota,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
    )
    return _FieldlineMetricGeometry(
        gradients=gradients, alpha_gradients=alpha_gradients
    )


def _fieldline_hngc_integrals(
    samples: _BoozerFieldlineSamples,
    hngc: _HNGCModeCorrections,
    alpha_gradients: Any,
    *,
    res_theta: int,
    res_phi: int,
) -> _FieldlineHNGCIntegrals:
    """Return flux-surface and field-line HNGC integrals."""

    D1, D2 = _flux_surface_hngc_averages(
        xm_b=samples.xm_b,
        xn_b=samples.xn_b,
        flipit=samples.flipit,
        lambmnc_b=hngc.lambmnc_b,
        rmnc_b=samples.rmnc_b,
        zmns_b=samples.zmns_b,
        numns_b=samples.numns_b,
        gmnc_b=samples.gmnc_b,
        res_theta=res_theta,
        res_phi=res_phi,
    )
    theta_1d = samples.theta_b[0, 0]
    intinv_g = _centered_fieldline_integral(
        1.0 / alpha_gradients.g_sup_psi_psi, samples.phi_b, theta_1d
    )
    int_lam_div_g = _centered_fieldline_integral(
        hngc.lambda_b / alpha_gradients.g_sup_psi_psi,
        samples.phi_b,
        theta_1d,
    )
    return _FieldlineHNGCIntegrals(
        D1=D1,
        D2=D2,
        intinv_g=intinv_g,
        int_lam_div_g=int_lam_div_g,
    )


def _fieldline_shear_factors(
    scalars: _VMECFieldlineScalars,
    *,
    s_val: float,
    betaprim: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
) -> _FieldlineShearFactors:
    """Return HNGC shear and pressure correction factors."""

    d_iota_d_s_1, sfac = _hngc_shear_correction(
        s_val=s_val,
        iota=scalars.iota,
        shat=scalars.shat,
        iota_input_val=scalars.iota_input_val,
        s_hat_input_val=scalars.s_hat_input_val,
        include_shear_variation=include_shear_variation,
    )
    d_pressure_d_s_1, pfac = _hngc_pressure_correction(
        s_val=s_val,
        betaprim=betaprim,
        B_reference=scalars.B_reference,
        d_pressure_d_s=scalars.d_pressure_d_s,
        include_pressure_variation=include_pressure_variation,
    )
    return _FieldlineShearFactors(
        d_iota_d_s_1=d_iota_d_s_1,
        d_pressure_d_s_1=d_pressure_d_s_1,
        sfac=sfac,
        pfac=pfac,
    )


def _fieldline_shear(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    geometry: _FieldlineMetricGeometry,
    integrals: _FieldlineHNGCIntegrals,
    factors: _FieldlineShearFactors,
) -> Any:
    """Return the local shear object used by metric/drift assembly."""

    return _fieldline_local_shear(
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        d_iota_d_s=scalars.d_iota_d_s,
        d_iota_d_s_1=factors.d_iota_d_s_1,
        d_pressure_d_s_1=factors.d_pressure_d_s_1,
        Vprime=samples.Vprime,
        G=scalars.G,
        iota=scalars.iota,
        boozer_i=scalars.boozer_i,
        phi_b=samples.phi_b,
        zeta_center=scalars.zeta_center,
        intinv_g=integrals.intinv_g,
        int_lam_div_g=integrals.int_lam_div_g,
        D1=integrals.D1,
        D2=integrals.D2,
        g_sup_psi_psi=geometry.alpha_gradients.g_sup_psi_psi,
        grad_alpha_dot_grad_psi=geometry.alpha_gradients.grad_alpha_dot_grad_psi,
    )


def _fieldline_metric_coefficients(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    hngc: _HNGCModeCorrections,
    *,
    s_val: float,
    betaprim: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    res_theta: int,
    res_phi: int,
) -> Any:
    """Assemble local shear, HNGC corrections, and metric/drift coefficients."""

    geometry = _fieldline_metric_geometry(scalars, samples)
    integrals = _fieldline_hngc_integrals(
        samples,
        hngc,
        geometry.alpha_gradients,
        res_theta=res_theta,
        res_phi=res_phi,
    )
    factors = _fieldline_shear_factors(
        scalars,
        s_val=s_val,
        betaprim=betaprim,
        include_shear_variation=include_shear_variation,
        include_pressure_variation=include_pressure_variation,
    )
    shear = _fieldline_shear(scalars, samples, geometry, integrals, factors)
    return _fieldline_metric_drifts(
        tensors=samples.tensors,
        gradients=geometry.gradients,
        alpha_gradients=geometry.alpha_gradients,
        shear=shear,
        s=scalars.s,
        shat=scalars.shat,
        sfac=factors.sfac,
        pfac=factors.pfac,
        L_reference=scalars.L_reference,
        B_reference=scalars.B_reference,
        toroidal_flux_sign=scalars.toroidal_flux_sign,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        d_pressure_d_s=scalars.d_pressure_d_s,
        beta_b=hngc.beta_b,
        G=scalars.G,
        iota=scalars.iota,
        boozer_i=scalars.boozer_i,
    )


def _assemble_fieldline_struct(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    coeffs: Any,
    *,
    s_val: float,
    alpha: float,
    betaprim: float,
) -> _Struct:
    theta_pest = samples.theta_b - scalars.iota[:, None, None] * samples.nu_b
    theta_geo = np.arctan2(samples.Z_b, samples.R_b - scalars.R_mag_ax)
    return _Struct(
        iota_input=scalars.iota_input_val,
        d_iota_d_s=scalars.d_iota_d_s,
        d_pressure_d_s=scalars.d_pressure_d_s,
        s_hat_input=scalars.s_hat_input_val,
        alpha=alpha,
        theta_b=samples.theta_b,
        phi_b=samples.phi_b,
        theta_PEST=theta_pest,
        theta_geo=theta_geo,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        R_b=samples.R_b,
        Z_b=samples.Z_b,
        betaprim=betaprim,
        bmag=coeffs.bmag,
        gradpar_theta_b=coeffs.gradpar_theta_b,
        gradpar_phi=coeffs.gradpar_phi,
        gds2=coeffs.gds2,
        gds21=coeffs.gds21,
        gds22=coeffs.gds22,
        gbdrift=coeffs.gbdrift,
        gbdrift0=coeffs.gbdrift0,
        cvdrift=coeffs.cvdrift,
        cvdrift0=coeffs.cvdrift0,
        grho=coeffs.grho,
        grad_y=coeffs.grad_y,
        grad_x=coeffs.grad_x,
        zeta_center=scalars.zeta_center,
        nfp=scalars.nfp,
        L_reference=scalars.L_reference,
        B_reference=scalars.B_reference,
        dpsidrho=2.0 * np.sqrt(s_val) * scalars.edge_toroidal_flux_over_2pi,
    )


__all__ = [
    "BoozerCartesianDerivatives",
    "BoozerCoordinateGradients",
    "BoozerFieldLineDerivatives",
    "boozer_cartesian_derivatives",
    "boozer_coordinate_gradients",
    "evaluate_boozer_field_line_derivatives",
]
