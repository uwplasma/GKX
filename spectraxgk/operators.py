from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jax import jit
from .basis import hermite_coupling_factors, lb_eigenvalues

class StreamingOperator(eqx.Module):
    """Hermite streaming along a straight-B direction; Fourier in z with kpar.
    dC_{n,m}/dt += -i kpar vth / sqrt(2) * [ sqrt(n+1) C_{n+1,m} + sqrt(n) C_{n-1,m} ]
    """
    Nn: int = eqx.static_field()
    Nm: int = eqx.static_field()
    kpar: float
    vth: float

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        Nn, Nm = self.Nn, self.Nm
        sqrt_n, _ = hermite_coupling_factors(Nn)
        # term with C_{n+1}: pad bottom row with zeros
        upper = jnp.pad(C[1:, :] * jnp.sqrt(jnp.arange(1, Nn))[:, None], ((0, 1), (0, 0)))
        # term with C_{n-1}: pad top row with zeros
        lower = jnp.pad(C[:-1, :] * jnp.sqrt(jnp.arange(1, Nn))[:, None], ((1, 0), (0, 0)))
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