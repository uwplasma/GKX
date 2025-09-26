from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .basis import hermite_coupling_factors, lb_eigenvalues


class StreamingOperator(eqx.Module):
    """Hermite streaming along a straight-B direction; Fourier in z with kpar.

    dC_{n,m}/dt += -i kpar vth / sqrt(2) * [ sqrt(n+1) C_{n+1,m} + sqrt(n) C_{n-1,m} ]

    We **precompute** sqrt(n) and sqrt(n+1) once and store them as fields to avoid
    rebuilding small arrays in the hot path.
    """

    Nn: int = eqx.static_field()
    Nm: int = eqx.static_field()
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

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        # C has shape (Nn, Nm). Build the two nearest-neighbor couplings in n.
        # For the C_{n+1} term: shift C down by one (pad last row with zeros) and
        # multiply by sqrt(n+1) evaluated at each n.
        upper = jnp.pad(C[1:, :], ((0, 1), (0, 0))) * self.sqrt_np1[:, None]
        # For the C_{n-1} term: shift C up by one (pad first row with zeros) and
        # multiply by sqrt(n) evaluated at each n.
        lower = jnp.pad(C[:-1, :], ((1, 0), (0, 0))) * self.sqrt_n[:, None]
        rhs = -1j * self.kpar * self.vth / jnp.sqrt(2.0) * (upper + lower)
        return rhs


class LenardBernstein(eqx.Module):
    """Diagonal Lenard–Bernstein model in Hermite–Laguerre space.

    C_dot += -nu * (alpha*n + beta*m) * C_{n,m}
    """

    Nn: int = eqx.static_field()
    Nm: int = eqx.static_field()
    nu: float
    alpha: float = eqx.static_field(default=1.0)
    beta: float = eqx.static_field(default=2.0)

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        lam = lb_eigenvalues(self.Nn, self.Nm, self.alpha, self.beta)
        return -self.nu * lam * C


class ElectrostaticDrive(eqx.Module):
    """Very simple electrostatic drive:
    E_parallel ~ C_{0,0} / kpar
    dC_{1,0}/dt += coef * E_parallel
    """

    Nn: int = eqx.static_field()
    Nm: int = eqx.static_field()
    kpar: float
    coef: float

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        # Guard kpar=0 to avoid division by zero
        E = jnp.where(self.kpar != 0.0, C[0, 0] / self.kpar, 0.0 + 0.0j)
        dC = jnp.zeros_like(C)
        # Only (n=1,m=0) gets the drive
        dC = dC.at[1, 0].add(-1j * self.coef * E)
        return dC
