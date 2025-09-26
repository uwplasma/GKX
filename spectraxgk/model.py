from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .operators import ElectrostaticDrive, LenardBernstein, StreamingOperator


class LinearGK(eqx.Module):
    """Linear slab GK (streaming + LB collisions) in Hermite–Laguerre space.

    Future: add electrostatic closure via quasi-neutrality with Laguerre algebra.
    """

    stream: StreamingOperator
    collide: LenardBernstein
    drive: ElectrostaticDrive | None = None

    def rhs(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """Diffrax vector field signature (t, y, args). `args` is unused."""
        Nn, Nm = self.stream.Nn, self.stream.Nm
        C = y.reshape(Nn, Nm)
        rhs = self.stream(C) + self.collide(C)
        if self.drive is not None:
            rhs = rhs + self.drive(C)
        return rhs.reshape(-1)

    def init_state(self, ic_kind: str, amp: float, phase: float) -> jnp.ndarray:
        Nn, Nm = self.stream.Nn, self.stream.Nm
        if ic_kind == "n0_mode":
            C0 = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
            C0 = C0.at[0, 0].set(amp * jnp.exp(1j * phase))
        elif ic_kind == "random":
            key = jax.random.PRNGKey(0)
            real = 1e-3 * jax.random.normal(key, (Nn, Nm))
            imag = 1e-3 * jax.random.normal(key, (Nn, Nm))
            C0 = (real + 1j * imag).astype(jnp.complex64)
        else:
            raise ValueError(f"Unknown ic.kind: {ic_kind}")
        return C0.reshape(-1)
