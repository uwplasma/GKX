"""Analytic flux-tube geometry for the Cyclone base case."""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp

from spectraxgk.config import GeometryConfig


@dataclass(frozen=True)
class SAlphaGeometry:
    """Simple s-alpha geometry with circular concentric flux surfaces."""

    q: float
    s_hat: float
    epsilon: float
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0

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
        """Field-aligned kx(theta) = kx0 + s_hat * theta * ky."""

        return kx0 + self.s_hat * theta * ky

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Perpendicular wave-number squared for s-alpha geometry."""

        kx_t = self.kx_effective(kx0, ky, theta)
        return kx_t * kx_t + ky * ky
