"""Exponential of the end absorber for an explicit step (prototype; no run path uses it).

The absorber damps the non-adiabatic ``H = G + B(fields)``, with the fields a linear
function ``F`` of ``G``, so it is not diagonal in ``G``: ``A = -e (I + B F)``, where the rate
``e(ky, z)`` is one number per grid point and ``B F`` has rank ``nf`` (``phi``, ``apar``,
``bpar``) at each point. Then ``exp(-s (I + B F)) = exp(-s) (I + B g(s K) F)`` with
``K = F B`` an ``nf x nf`` matrix per point and ``g(s K) = (exp(-s K) - I) K^-1``, so
``exp(h A)`` costs one field solve and one ``B`` per application.  A diagonal
integrating factor ``exp(-h e)`` alone leaves the field-amplified part of ``A`` explicit.

``lawson_rk4`` is the integrating-factor RK4 step in forward form (no ``exp(+hA)``
factors): ``y' = A y + N(y)`` with ``N`` the right-hand side without the absorber.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.dissipation import end_damping_contribution
from gkx.operators.linear.moments import build_H
from gkx.terms.assembly import _species_arrays, compute_fields_cached


@dataclass(frozen=True)
class AbsorberExponential:
    """``exp(h A)`` as ``exp(-h e) (G + B(psi))``, ``psi = V g(h e, lam) V^-1 F(G)``."""

    rate: jnp.ndarray  # (ky, 1, z)  e(ky, z) >= 0, broadcast over kx
    lam: jnp.ndarray  # (nf, ky, kx, z)  eigenvalues of K
    v: jnp.ndarray  # (nf, nf, ky, kx, z)
    vinv: jnp.ndarray  # (nf, nf, ky, kx, z)
    fields: Callable[
        [jnp.ndarray], tuple[jnp.ndarray, ...]
    ]  # G -> (phi, apar, bpar)[:nf]
    inject: Callable[[tuple[jnp.ndarray, ...]], jnp.ndarray]  # fields -> B(fields)

    def __call__(self, G: jnp.ndarray, h: Any) -> jnp.ndarray:
        s = h * self.rate  # (ky, kx|1, z)
        lam = self.lam
        x = -s * lam
        small = jnp.abs(x) < 1e-3
        xs = jnp.where(small, 1.0, x)
        # g = (exp(-s lam) - 1) / lam = -s phi1(-s lam)
        phi1 = jnp.where(small, 1 + x / 2 + x**2 / 6, jnp.expm1(xs) / xs)
        g = -s * phi1
        f = jnp.stack(self.fields(G))
        c = jnp.einsum("ij...,j...->i...", self.vinv, f)
        psi = jnp.einsum("ij...,j...->i...", self.v, g * c)
        return jnp.exp(-s) * (G + self.inject(tuple(psi)))


def build_absorber_exponential(
    cache: Any, params: Any, terms: Any, shape: tuple[int, ...], dtype: Any
) -> AbsorberExponential:
    """Probe ``F`` and ``B`` once and eigendecompose ``K = F B`` at every point."""
    real = jnp.real(jnp.zeros((), dtype)).dtype
    ns = shape[0]
    sp = _species_arrays(params, ns, real)
    use_apar = float(terms.apar) != 0.0
    use_bpar = float(terms.bpar) != 0.0
    names = ("phi",) + (("apar",) if use_apar else ()) + (("bpar",) if use_bpar else ())
    zero = jnp.zeros(shape, dtype)
    fshape = shape[3:]

    def fields(G):
        f = compute_fields_cached(G, cache, params, terms=terms)
        return tuple(getattr(f, n) for n in names)

    def inject(parts):
        d = dict(zip(names, parts))
        return build_H(
            zero,
            cache.Jl,
            d["phi"],
            sp.tz,
            apar=d.get("apar"),
            vth=sp.vth,
            bpar=d.get("bpar"),
            JlB=cache.JlB,
        )

    nf = len(names)
    k = np.zeros((nf, nf, *fshape), dtype=complex)
    for j in range(nf):
        unit = tuple(
            jnp.ones(fshape, dtype) if i == j else jnp.zeros(fshape, dtype)
            for i in range(nf)
        )
        out = fields(inject(unit))
        for i in range(nf):
            k[i, j] = np.asarray(out[i])
    # eig of a stack of nf x nf matrices, point by point
    flat = np.moveaxis(k.reshape(nf, nf, -1), -1, 0)
    lam, v = np.linalg.eig(flat)
    vinv = np.linalg.inv(v)
    restore = lambda a, lead: np.moveaxis(a, 0, -1).reshape(*lead, *fshape)  # noqa: E731
    one = jnp.ones(shape, dtype)
    damp = end_damping_contribution(
        one,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        linked_damp_profile=cache.linked_damp_profile,
        damp_amp=params.end_damping_rate(cache.dz * cache.damp_profile.shape[-1], real),
        weight=jnp.asarray(terms.end_damping, real),
    )
    rate = -jnp.real(
        damp[0, 0, 0]
    )  # (ky, kx, z): the same for every species and moment
    cdt = dtype
    return AbsorberExponential(
        rate=rate,
        lam=jnp.asarray(restore(lam, (nf,)), cdt),
        v=jnp.asarray(restore(v.reshape(-1, nf * nf), (nf, nf)), cdt),
        vinv=jnp.asarray(restore(vinv.reshape(-1, nf * nf), (nf, nf)), cdt),
        fields=fields,
        inject=inject,
    )


def lawson_rk4(
    G: jnp.ndarray,
    h: Any,
    rhs: Callable[[jnp.ndarray], jnp.ndarray],
    expo: AbsorberExponential,
) -> jnp.ndarray:
    """One integrating-factor RK4 step of ``y' = A y + N(y)``; ``rhs`` is ``N`` (no absorber)."""
    e_half = lambda x: expo(x, h / 2)  # noqa: E731
    k1 = rhs(G)
    e2y, e2k1 = e_half(G), e_half(k1)
    u2 = e2y + 0.5 * h * e2k1
    k2 = rhs(u2)
    u3 = e2y + 0.5 * h * k2
    k3 = rhs(u3)
    e2k3 = e_half(k3)
    u4 = expo(G, h) + h * e2k3
    k4 = rhs(u4)
    ek1 = e_half(e2k1)
    return expo(G, h) + (h / 6.0) * (ek1 + 2.0 * e_half(k2 + k3) + k4)
