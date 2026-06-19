"""Boozer field-line derivative algebra for differentiable geometry bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


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


__all__ = [
    "BoozerCartesianDerivatives",
    "BoozerCoordinateGradients",
    "BoozerFieldLineDerivatives",
    "boozer_cartesian_derivatives",
    "boozer_coordinate_gradients",
    "evaluate_boozer_field_line_derivatives",
]
