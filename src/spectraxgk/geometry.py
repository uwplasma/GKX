"""Analytic flux-tube geometry for the Cyclone base case."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import GeometryConfig, GridConfig


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
        return SAlphaGeometry(
            q=cfg.q,
            s_hat=cfg.s_hat,
            epsilon=cfg.epsilon,
            R0=cfg.R0,
            B0=cfg.B0,
            alpha=cfg.alpha,
            drift_scale=cfg.drift_scale,
            kperp2_bmag=cfg.kperp2_bmag,
            bessel_bmag_power=cfg.bessel_bmag_power,
        )

    def kx_effective(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    def metric_coeffs(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Metric coefficients (gds2, gds21, gds22) for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        gds2 = 1.0 + shear * shear
        gds21 = -self.s_hat * shear
        gds22 = jnp.asarray(self.s_hat) * jnp.asarray(self.s_hat)
        return gds2, gds21, gds22

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    def omega_d(self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic drift frequency for s-alpha geometry."""

        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


@dataclass(frozen=True)
class FluxTubeGeometryData:
    """Sampled flux-tube geometry contract for solver-ready metric profiles."""

    theta: jnp.ndarray
    gradpar_value: float
    bmag_profile: jnp.ndarray
    bgrad_profile: jnp.ndarray
    gds2_profile: jnp.ndarray
    gds21_profile: jnp.ndarray
    gds22_profile: jnp.ndarray
    cv_profile: jnp.ndarray
    gb_profile: jnp.ndarray
    cv0_profile: jnp.ndarray
    gb0_profile: jnp.ndarray
    q: float
    s_hat: float
    epsilon: float
    R0: float
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 1.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0
    source_model: str = "sampled"

    def _theta_matches(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta_arr = jnp.asarray(theta)
        if theta_arr.shape[-1] != self.theta.shape[0]:
            raise ValueError("theta must have the same last dimension as the sampled geometry grid")
        if isinstance(theta_arr, jax.core.Tracer):
            return theta_arr
        theta_line = theta_arr if theta_arr.ndim == 1 else theta_arr.reshape(-1, theta_arr.shape[-1])[0]
        if isinstance(theta_line, jax.core.Tracer):
            return theta_arr
        if not np.allclose(np.asarray(theta_line), np.asarray(self.theta)):
            raise ValueError("theta does not match the sampled geometry grid")
        return theta_arr

    def _broadcast_profile(self, theta: jnp.ndarray, profile: jnp.ndarray) -> jnp.ndarray:
        theta_arr = self._theta_matches(theta)
        if theta_arr.ndim == 1:
            return profile
        shape = (1,) * (theta_arr.ndim - 1) + (profile.shape[0],)
        return jnp.broadcast_to(profile.reshape(shape), theta_arr.shape)

    def gradpar(self) -> float:
        return float(self.gradpar_value)

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.bmag_profile)

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.bgrad_profile)

    def metric_coeffs(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self._broadcast_profile(theta, self.gds2_profile),
            self._broadcast_profile(theta, self.gds21_profile),
            self._broadcast_profile(theta, self.gds22_profile),
        )

    def drift_coeffs(
        self,
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self._broadcast_profile(theta, self.cv_profile),
            self._broadcast_profile(theta, self.gb_profile),
            self._broadcast_profile(theta, self.cv0_profile),
            self._broadcast_profile(theta, self.gb0_profile),
        )

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    def drift_components(
        self,
        kx: jnp.ndarray,
        ky: jnp.ndarray,
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def omega_d(self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


def sample_flux_tube_geometry(geom: SAlphaGeometry, theta: jnp.ndarray) -> FluxTubeGeometryData:
    """Sample an analytic geometry model onto a flux-tube theta grid."""

    theta_arr = jnp.asarray(theta)
    gds2, gds21, gds22 = geom.metric_coeffs(theta_arr)
    gds22_arr = gds22 if gds22.ndim else jnp.full_like(theta_arr, gds22)
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta_arr)
    return FluxTubeGeometryData(
        theta=theta_arr,
        gradpar_value=float(geom.gradpar()),
        bmag_profile=geom.bmag(theta_arr),
        bgrad_profile=geom.bgrad(theta_arr),
        gds2_profile=gds2,
        gds21_profile=gds21,
        gds22_profile=gds22_arr,
        cv_profile=cv,
        gb_profile=gb,
        cv0_profile=cv0,
        gb0_profile=gb0,
        q=float(geom.q),
        s_hat=float(geom.s_hat),
        epsilon=float(geom.epsilon),
        R0=float(geom.R0),
        B0=float(geom.B0),
        alpha=float(geom.alpha),
        drift_scale=float(geom.drift_scale),
        kperp2_bmag=bool(geom.kperp2_bmag),
        bessel_bmag_power=float(geom.bessel_bmag_power),
        source_model="s-alpha",
    )


def gx_twist_shift_params(
    geom: SAlphaGeometry,
    grid: GridConfig,
) -> tuple[int, float]:
    """Return (jtwist, x0) following GX twist-and-shift defaults."""

    y0 = float(grid.y0) if grid.y0 is not None else float(grid.Ly) / (2.0 * jnp.pi)
    if grid.ntheta is not None:
        if grid.zp is not None:
            zp = int(grid.zp)
        elif grid.nperiod is not None:
            zp = 2 * int(grid.nperiod) - 1
        else:
            zp = 1
        theta_min = -jnp.pi * float(zp)
    else:
        theta_min = float(grid.z_min)
    _gds2, gds21, gds22 = geom.metric_coeffs(jnp.asarray([theta_min]))
    gds21_val = float(gds21[0])
    gds22_val = float(gds22)
    shat = float(geom.s_hat)
    twist_shift_geo_fac = 2.0 * shat * gds21_val / gds22_val if gds22_val != 0.0 else 0.0
    if grid.jtwist is None:
        jtwist = int(round(twist_shift_geo_fac))
        if jtwist == 0:
            jtwist = 1
    else:
        jtwist = int(grid.jtwist)
        if jtwist == 0:
            jtwist = 1
    if twist_shift_geo_fac == 0.0:
        x0 = y0
    else:
        x0 = y0 * abs(jtwist) / abs(twist_shift_geo_fac)
    return jtwist, x0
