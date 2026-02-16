"""Analytic flux-tube geometry for the Cyclone base case."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from spectraxgk.config import GeometryConfig


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

    def tree_flatten(self):
        children = (self.q, self.s_hat, self.epsilon, self.R0, self.B0, self.alpha)
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
        )

    def kx_effective(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Field-aligned kx(theta) with s-alpha shear shift."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        return kx0 - shear * ky

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic field strength for circular s-alpha geometry."""

        return 1.0 / (1.0 + self.epsilon * jnp.cos(theta))

    def gradpar(self) -> jnp.ndarray:
        """Parallel gradient factor for s-alpha geometry (constant for equal-arc)."""

        return jnp.abs(1.0 / (self.q * self.R0))

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
        bmag_inv = 1.0 / self.bmag(theta)
        return kperp2 * (bmag_inv * bmag_inv)

    def drift_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Curvature and grad-B drift coefficients for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        base = jnp.cos(theta) + shear * jnp.sin(theta)
        cv = base / self.R0
        gb = cv
        cv0 = (-self.s_hat * jnp.sin(theta)) / self.R0
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
