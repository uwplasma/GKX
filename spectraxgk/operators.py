from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import equinox as eqx
import jax.numpy as jnp

from .basis import hermite_coupling_factors, lb_eigenvalues


class StreamingOperator(eqx.Module):
    """Hermite streaming along a straight-B direction; Fourier in z with kpar.

    dC_{n,m}/dt += -i kpar vth / sqrt(2) * [ sqrt(n+1) C_{n+1,m} + sqrt(n) C_{n-1,m} ]

    We **precompute** sqrt(n) and sqrt(n+1) once and store them as fields to avoid
    rebuilding small arrays in the hot path.
    """

    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    vth: float
    # precomputed Hermite coupling factors (dynamic PyTree fields)
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
        # scalar factor kpar * vth / sqrt(2)
        return self.kpar * self.vth / jnp.sqrt(2.0)

    def couple_real(self, X: jnp.ndarray) -> jnp.ndarray:
        """Apply the Hermite nearest-neighbour coupling to a REAL (Nn,Nm) array.
        Returns (upper + lower). No -i factor here.
        """
        upper = jnp.pad(X[1:, :], ((0, 1), (0, 0))) * self.sqrt_np1[:, None]
        lower = jnp.pad(X[:-1, :], ((1, 0), (0, 0))) * self.sqrt_n[:, None]
        return upper + lower


class LenardBernstein(eqx.Module):
    """Diagonal Lenard–Bernstein model in Hermite–Laguerre space.

    C_dot += -nu * (alpha*n + beta*m) * C_{n,m}
    """

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
    E_parallel ~ C_{0,0} / kpar
    dC_{1,0}/dt += coef * E_parallel
    """

    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    coef: float

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        # Guard kpar=0 to avoid division by zero
        E = jnp.where(self.kpar != 0.0, C[0, 0] / self.kpar, 0.0 + 0.0j)
        dC = jnp.zeros_like(C)
        # Only (n=1,m=0) gets the drive
        dC = dC.at[1, 0].add(-1j * self.coef * E)
        return dC

class StreamingRHS(eqx.Module):
    """dCr = -a*S(Ci), dCi = +a*S(Cr) using StreamingOperator.couple_real and factor a."""
    stream: "StreamingOperator"

    def __call__(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        S_Cr = self.stream.couple_real(Cr)
        S_Ci = self.stream.couple_real(Ci)
        a = self.stream.a
        dCr = -a * S_Ci
        dCi =  a * S_Cr
        return dCr, dCi


class CollisionsRHS(eqx.Module):
    """dCr -= nu*λ*Cr, dCi -= nu*λ*Ci in HL space (diagonal)."""
    collide: "LenardBernstein"

    def __call__(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        Nn, Nm = self.collide.Nn, self.collide.Nm
        lam = lb_eigenvalues(Nn, Nm, self.collide.alpha, self.collide.beta)
        dCr = -self.collide.nu * lam * Cr
        dCi = -self.collide.nu * lam * Ci
        return dCr, dCi


class ElectrostaticDriveRHS(eqx.Module):
    """E_∥ ~ C_{0,0}/kpar; inject only into (n=1, m=0)."""
    drive: "ElectrostaticDrive"

    def __call__(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        Nn, Nm = self.drive.Nn, self.drive.Nm
        dCr = jnp.zeros((Nn, Nm), dtype=Cr.dtype)
        dCi = jnp.zeros((Nn, Nm), dtype=Ci.dtype)
        k = self.drive.kpar
        Er = jnp.where(k != 0.0, Cr[0, 0] / k, 0.0)
        Ei = jnp.where(k != 0.0, Ci[0, 0] / k, 0.0)
        dCr = dCr.at[1, 0].add(self.drive.coef * Er)
        dCi = dCi.at[1, 0].add(self.drive.coef * Ei)
        return dCr, dCi