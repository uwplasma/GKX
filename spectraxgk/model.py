from __future__ import annotations

from typing import Optional, Tuple, Protocol
import jax.numpy as jnp
import equinox as eqx
from jax import random
from .operators import (
    StreamingOperator, LenardBernstein, ElectrostaticDrive,
    StreamingRHS, CollisionsRHS, ElectrostaticDriveRHS, 
)
from .basis import lb_eigenvalues


class RealRHSTerm(Protocol):
    def __call__(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]: ...


class LinearGK(eqx.Module):
    """Linear slab GK (streaming + LB collisions) in Hermite–Laguerre space.

    Future: add electrostatic closure via quasi-neutrality with Laguerre algebra.
    """

    stream: StreamingOperator # streaming term
    collide: LenardBernstein # Lenard-Bernstein collisions
    drive: Optional[ElectrostaticDrive] = None # optional ES drive term
    terms: Tuple[RealRHSTerm, ...] = () # sum of contributions to RHS

    def init_state(self, ic_kind: str, amp: float, phase: float) -> jnp.ndarray:
        """Return a COMPLEX, flattened initial state C0.reshape(-1)."""
        Nn, Nm = self.stream.Nn, self.stream.Nm
        if ic_kind == "n0_mode":
            C0 = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
            C0 = C0.at[0, 0].set(amp * jnp.exp(1j * phase))
        elif ic_kind == "random":
            key = random.PRNGKey(0)
            real = 1e-3 * random.normal(key, (Nn, Nm))
            imag = 1e-3 * random.normal(key, (Nn, Nm))
            C0 = (real + 1j * imag).astype(jnp.complex64)
        else:
            raise ValueError(f"Unknown ic.kind: {ic_kind}")
        return C0.reshape(-1)

    # --- helpers for real/imag packing ---
    def _shape(self):
        return self.stream.Nn, self.stream.Nm

    def pack_real(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([Cr.reshape(-1), Ci.reshape(-1)], axis=0)

    def unpack_real(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        Nn, Nm = self._shape()
        M = Nn * Nm
        Cr = y[:M].reshape(Nn, Nm)
        Ci = y[M:].reshape(Nn, Nm)
        return Cr, Ci

    def init_state_real(self, ic_kind: str, amp: float, phase: float) -> jnp.ndarray:
        y0c = self.init_state(ic_kind, amp, phase)           # complex flat
        Nn, Nm = self._shape()
        C0 = y0c.reshape(Nn, Nm)
        Cr0 = jnp.real(C0)
        Ci0 = jnp.imag(C0)
        return self.pack_real(Cr0, Ci0)

    def rhs_real(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """Real-valued vector field: state y = [Re(C); Im(C)], sum contributions from terms."""
        Cr, Ci = self.unpack_real(y)
        dCr = jnp.zeros_like(Cr)
        dCi = jnp.zeros_like(Ci)
        for term in self.terms:
            tCr, tCi = term(Cr, Ci)
            dCr = dCr + tCr
            dCi = dCi + tCi
        return self.pack_real(dCr, dCi)