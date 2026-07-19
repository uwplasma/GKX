"""Analytic flux-tube geometry models.

This module contains closed-form solver-ready geometry models. Sampled VMEC,
Boozer, and imported field-line data live in :mod:`gkx.geometry.flux_tube`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gkx.config import GeometryConfig

ZERO_SHAT_THRESHOLD = 1.0e-5


def zero_shear_enabled(
    s_hat: float,
    *,
    zero_shat: bool = False,
    threshold: float = ZERO_SHAT_THRESHOLD,
) -> bool:
    """Return the effective zero-shear state."""

    return bool(zero_shat) or abs(float(s_hat)) < float(threshold)


def effective_boundary(
    boundary: str,
    *,
    s_hat: float,
    zero_shat: bool = False,
    threshold: float = ZERO_SHAT_THRESHOLD,
) -> str:
    """Return the effective boundary after zero-shear promotion."""

    if zero_shear_enabled(s_hat, zero_shat=zero_shat, threshold=threshold):
        return "periodic"
    return str(boundary)


@dataclass(frozen=True)
class MillerCoreParams:
    """Core Miller parameters needed by the low-level Miller geometry formulas."""

    ntgrid: int
    nperiod: int
    rhoc: float
    qinp: float
    shat: float
    rmaj: float
    r_geo: float
    shift: float
    akappa: float
    tri: float
    akappri: float
    tripri: float
    betaprim: float
    delrho: float = 1.0e-3


def build_collocation_surfaces(
    params: MillerCoreParams,
) -> dict[str, np.ndarray | float]:
    """Construct the Miller surface on the collocation grid."""

    no_of_surfs = 3
    theta = np.linspace(0.0, np.pi, int(params.ntgrid), dtype=float)
    r0 = np.array(
        [
            params.rmaj - params.shift * params.delrho,
            params.rmaj,
            params.rmaj + params.shift * params.delrho,
        ],
        dtype=float,
    )
    rho = np.array(
        [params.rhoc - params.delrho, params.rhoc, params.rhoc + params.delrho],
        dtype=float,
    )
    qfac = np.array(
        [
            params.qinp - params.shat * (params.qinp / params.rhoc) * params.delrho,
            params.qinp,
            params.qinp + params.shat * (params.qinp / params.rhoc) * params.delrho,
        ],
        dtype=float,
    )
    kappa = np.array(
        [
            params.akappa - params.akappri * params.delrho,
            params.akappa,
            params.akappa + params.akappri * params.delrho,
        ],
        dtype=float,
    )
    delta = np.array(
        [
            params.tri - params.tripri * params.delrho,
            params.tri,
            params.tri + params.tripri * params.delrho,
        ],
        dtype=float,
    )

    r = np.array(
        [
            r0[i] + rho[i] * np.cos(theta + np.arcsin(delta[i]) * np.sin(theta))
            for i in range(no_of_surfs)
        ],
        dtype=float,
    )
    z = np.array(
        [kappa[i] * rho[i] * np.sin(theta) for i in range(no_of_surfs)], dtype=float
    )
    theta_common_mag_axis = np.arctan2(z, r - params.rmaj)

    return {
        "theta": theta,
        "rho": rho,
        "qfac": qfac,
        "r": r,
        "z": z,
        "theta_common_mag_axis": theta_common_mag_axis,
        "dpdrho": 0.5 * float(params.betaprim),
        "no_of_surfs": float(no_of_surfs),
    }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SAlphaGeometry:
    """Simple s-alpha geometry with circular concentric flux surfaces."""

    q: float
    s_hat: float
    epsilon: float
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 1.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0

    def tree_flatten(self):
        children = (
            self.q,
            self.s_hat,
            self.epsilon,
            self.R0,
            self.B0,
            self.alpha,
            self.drift_scale,
            self.kperp2_bmag,
            self.bessel_bmag_power,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def from_config(cfg: GeometryConfig) -> "SAlphaGeometry":
        zero_shat = zero_shear_enabled(cfg.s_hat, zero_shat=cfg.zero_shat)
        return SAlphaGeometry(
            q=cfg.q,
            s_hat=0.0 if zero_shat else cfg.s_hat,
            epsilon=cfg.epsilon,
            R0=cfg.R0,
            B0=cfg.B0,
            alpha=cfg.alpha,
            drift_scale=cfg.drift_scale,
            kperp2_bmag=cfg.kperp2_bmag,
            bessel_bmag_power=cfg.bessel_bmag_power,
        )

    def kx_effective(
        self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        """Field-aligned kx(theta) with s-alpha shear shift."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        return kx0 - shear * ky

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic field strength for circular s-alpha geometry."""

        return 1.0 / (1.0 + self.epsilon * jnp.cos(theta))

    def gradpar(self) -> float:
        """Parallel gradient factor for s-alpha geometry (constant for equal-arc)."""

        return float(abs(1.0 / (self.q * self.R0)))

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic field gradient term used in mirror force."""

        bmag = self.bmag(theta)
        return self.gradpar() * self.epsilon * jnp.sin(theta) * bmag

    def metric_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Metric coefficients (gds2, gds21, gds22) for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        gds2 = 1.0 + shear * shear
        gds21 = -self.s_hat * shear
        gds22 = jnp.asarray(self.s_hat) * jnp.asarray(self.s_hat)
        return gds2, gds21, gds22

    def k_perp2(
        self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        """Perpendicular wave-number squared for s-alpha geometry."""

        gds2, gds21, gds22 = self.metric_coeffs(theta)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        kperp2 = ky * (ky * gds2 + 2.0 * kx_hat * gds21) + (kx_hat * kx_hat) * gds22
        if self.kperp2_bmag:
            bmag_inv = 1.0 / self.bmag(theta)
            return kperp2 * (bmag_inv * bmag_inv)
        return kperp2

    def drift_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Curvature and grad-B drift coefficients for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        base = jnp.cos(theta) + shear * jnp.sin(theta)
        scale = jnp.asarray(self.drift_scale)
        cv = scale * base / self.R0
        gb = cv
        cv0 = scale * (-self.s_hat * jnp.sin(theta)) / self.R0
        gb0 = cv0
        return cv, gb, cv0, gb0

    def drift_components(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return cv_d and gb_d drift components in (ky, kx, theta)."""

        kx0 = kx[None, :, None]
        ky0 = ky[:, None, None]
        theta0 = theta[None, None, :]
        cv, gb, cv0, gb0 = self.drift_coeffs(theta0)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        cv_d = ky0 * cv + kx_hat * cv0
        gb_d = ky0 * gb + kx_hat * gb0
        return cv_d, gb_d

    def omega_d(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        """Magnetic drift frequency for s-alpha geometry."""

        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SlabGeometry:
    """Reference slab geometry contract."""

    s_hat: float = 0.0
    z0: float | None = None
    q: float = 1.0
    epsilon: float = 0.0
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 0.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0
    zero_shat: bool = False

    def tree_flatten(self):
        children = (
            self.s_hat,
            self.q,
            self.epsilon,
            self.R0,
            self.B0,
            self.alpha,
            self.drift_scale,
            self.kperp2_bmag,
            self.bessel_bmag_power,
        )
        return children, {"z0": self.z0, "zero_shat": self.zero_shat}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            s_hat,
            q,
            epsilon,
            R0,
            B0,
            alpha,
            drift_scale,
            kperp2_bmag,
            bessel_bmag_power,
        ) = children
        return cls(
            s_hat=s_hat,
            z0=aux_data["z0"],
            q=q,
            epsilon=epsilon,
            R0=R0,
            B0=B0,
            alpha=alpha,
            drift_scale=drift_scale,
            kperp2_bmag=kperp2_bmag,
            bessel_bmag_power=bessel_bmag_power,
            zero_shat=aux_data["zero_shat"],
        )

    @staticmethod
    def from_config(cfg: GeometryConfig) -> "SlabGeometry":
        zero_shat = zero_shear_enabled(cfg.s_hat, zero_shat=cfg.zero_shat)
        shat = 0.0 if zero_shat else float(cfg.s_hat)
        return SlabGeometry(
            s_hat=shat,
            z0=cfg.z0,
            q=1.0,
            epsilon=0.0,
            R0=cfg.R0,
            B0=cfg.B0,
            alpha=0.0,
            drift_scale=0.0,
            kperp2_bmag=cfg.kperp2_bmag,
            bessel_bmag_power=cfg.bessel_bmag_power,
            zero_shat=zero_shat,
        )

    def kx_effective(
        self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        shear = jnp.asarray(self.s_hat) * theta
        return kx0 - shear * ky

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(jnp.asarray(theta))

    def gradpar(self) -> float:
        if self.z0 is not None and float(self.z0) > 0.0:
            return float(1.0 / float(self.z0))
        return 1.0

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(jnp.asarray(theta))

    def metric_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta_arr = jnp.asarray(theta)
        shear = jnp.asarray(self.s_hat) * theta_arr
        gds2 = 1.0 + shear * shear
        gds21 = -jnp.asarray(self.s_hat) * shear
        if float(self.s_hat) == 0.0:
            gds22 = jnp.ones_like(theta_arr)
        else:
            gds22 = jnp.full_like(theta_arr, float(self.s_hat) * float(self.s_hat))
        return gds2, gds21, gds22

    def k_perp2(
        self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        gds2, gds21, gds22 = self.metric_coeffs(theta)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        kperp2 = ky * (ky * gds2 + 2.0 * kx_hat * gds21) + (kx_hat * kx_hat) * gds22
        return kperp2

    def drift_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta_arr = jnp.asarray(theta)
        zeros = jnp.zeros_like(theta_arr)
        return zeros, zeros, zeros, zeros

    def drift_components(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        kx0 = jnp.asarray(kx)
        ky0 = jnp.asarray(ky)
        theta0 = jnp.asarray(theta)
        zeros = jnp.zeros(
            (ky0.shape[0], kx0.shape[0], theta0.shape[0]), dtype=theta0.dtype
        )
        return zeros, zeros

    def omega_d(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        kx0 = jnp.asarray(kx)
        ky0 = jnp.asarray(ky)
        theta0 = jnp.asarray(theta)
        return jnp.zeros(
            (ky0.shape[0], kx0.shape[0], theta0.shape[0]), dtype=theta0.dtype
        )
