from __future__ import annotations

from typing import Optional, Tuple
import equinox as eqx
import jax.numpy as jnp
from jax import random, vmap

from .operators import StreamingOperator, LenardBernstein, ElectrostaticDrive
from .types import ComplexTerm

class LinearGK(eqx.Module):
    stream: StreamingOperator
    collide: LenardBernstein
    drive: Optional[ElectrostaticDrive] = None
    terms: Tuple[ComplexTerm, ...] = ()  # unified complex-space terms
    Nk: int = eqx.field(static=True, default=1)

    def _shape_tuple(self) -> Tuple[int, int, int]:
        return (self.Nk, self.stream.Nn, self.stream.Nm)

    def pack_real(self, Cr: jnp.ndarray, Ci: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([Cr.reshape(-1), Ci.reshape(-1)], axis=0)

    def unpack_real(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        Nk, Nn, Nm = self._shape_tuple()
        M = Nk * Nn * Nm
        Cr = y[:M].reshape(Nk, Nn, Nm)
        Ci = y[M:].reshape(Nk, Nn, Nm)
        return Cr, Ci

    def rhs_real(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        Cr, Ci = self.unpack_real(y)
        C = Cr + 1j * Ci   # (Nk,Nn,Nm)
        dC = jnp.zeros_like(C)

        def apply_linear(term, C3):
            # term expects (Nn,Nm): apply per-k
            return vmap(term, in_axes=0, out_axes=0)(C3)

        for term in self.terms:
            if getattr(term, "accepts_3d", False):
                # term expects (Nk,Nn,Nm)
                dC = dC + term(C)
            else:
                # term expects (Nn,Nm); vmap over k
                dC = dC + apply_linear(term, C)

        return self.pack_real(jnp.real(dC), jnp.imag(dC))

    # Initial conditions (now produce Nk×Nn×Nm)
    def init_state(self, ic_kind: str, amp: float, phase: float) -> jnp.ndarray:
        Nk, Nn, Nm = self._shape_tuple()
        C0 = jnp.zeros((Nk, Nn, Nm), dtype=jnp.complex64)
        if ic_kind == "n0_mode":
            # excite n=0, m=0 at the central mode (k=0) if present; else k index 0
            k0 = Nk // 2 if Nk > 1 else 0
            C0 = C0.at[k0, 0, 0].set(amp * jnp.exp(1j * phase))
        elif ic_kind == "random":
            keyr = random.PRNGKey(0)
            real = 1e-3 * random.normal(keyr, (Nk, Nn, Nm))
            imag = 1e-3 * random.normal(random.PRNGKey(1), (Nk, Nn, Nm))
            C0 = (real + 1j * imag).astype(jnp.complex64)
        else:
            raise ValueError(f"Unknown ic.kind: {ic_kind}")
        return C0.reshape(-1)

    def init_state_real(self, ic_kind: str, amp: float, phase: float) -> jnp.ndarray:
        C0_flat = self.init_state(ic_kind, amp, phase)
        Nk, Nn, Nm = self._shape_tuple()
        C0 = C0_flat.reshape(Nk, Nn, Nm)
        return self.pack_real(jnp.real(C0), jnp.imag(C0))