from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .basis import hermite_coupling_factors, lb_eigenvalues


class StreamingOperator(eqx.Module):
    """Hermite streaming along straight B; Fourier in z with kpar.

    dC_{n,m}/dt += -i * (kpar*vth/sqrt(2)) * [ sqrt(n+1) C_{n+1,m} + sqrt(n) C_{n-1,m} ]
    """
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    vth: float
    sqrt_n: jnp.ndarray
    sqrt_np1: jnp.ndarray

    def __init__(self, Nn: int, Nm: int, kpar: float, vth: float):
        object.__setattr__(self, "Nn", Nn)
        object.__setattr__(self, "Nm", Nm)
        object.__setattr__(self, "kpar", kpar)
        object.__setattr__(self, "vth", vth)
        sn, snp1 = hermite_coupling_factors(Nn)
        object.__setattr__(self, "sqrt_n", sn)
        object.__setattr__(self, "sqrt_np1", snp1)

    @property
    def a(self) -> jnp.ndarray:
        return self.kpar * self.vth / jnp.sqrt(2.0)

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        """Complex-space streaming RHS."""
        # nearest-neighbor Hermite coupling (same Laguerre m)
        upper = jnp.pad(C[1:, :], ((0, 1), (0, 0))) * self.sqrt_np1[:, None]
        lower = jnp.pad(C[:-1, :], ((1, 0), (0, 0))) * self.sqrt_n[:, None]
        return -1j * self.a * (upper + lower)


class LenardBernstein(eqx.Module):
    """Diagonal Lenard–Bernstein model in Hermite–Laguerre space: dC/dt += -nu*(α n + β m) C."""
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    nu: float
    alpha: float = eqx.field(static=True, default=1.0)
    beta:  float = eqx.field(static=True, default=2.0)

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        lam = lb_eigenvalues(self.Nn, self.Nm, self.alpha, self.beta)
        return -self.nu * lam * C


class ElectrostaticDrive(eqx.Module):
    """Very simple electrostatic drive:
       E_∥ ~ C_{0,0} / kpar; dC_{1,0}/dt += - coef * sqrt(2) * i * E_∥
    """
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    coef: float

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        k = self.kpar
        E = jnp.where(k != 0.0, 1j * C[0, 0] / k, 0.0 + 0.0j)
        dC = jnp.zeros_like(C)
        dC = dC.at[1, 0].add(-1 * self.coef * jnp.sqrt(2.0) * E)
        return dC
