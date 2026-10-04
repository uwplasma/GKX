"""Implicit linear operator per linked chain, for the ``imex-ars*`` methods.

Kinetic-electron runs are step-limited by electron parallel streaming and the
electromagnetic electron mode (|lambda| ~ 500-700 against drift rates ~ 1 on
the Cyclone tutorial box), not by accuracy. The linear operator couples modes
only along one ``(ky, kx)`` twist-shift chain, so its restriction to a chain is
a dense matrix of size ``ns Nl Nm Nz L`` (``L`` chain links). This module
materializes those matrices once, by probing the linear part of the run's own
RHS (its odd part, so every linear term -- streaming, fields, mirror,
drifts, dissipation, end damping -- enters exactly as the explicit route
applies it), and inverts ``I - gamma dt L`` per chain.

The time step is an Ascher-Ruuth-Spiteri IMEX Runge-Kutta scheme: L-stable
SDIRK with one diagonal coefficient for the linear part (one factorization per
dt), explicit for the nonlinear bracket. It is the plan's section 5.4 route with dense per-chain
factors in place of the banded response-matrix solve: memory is
``sum over chains of (ns Nl Nm Nz L)^2`` complex entries, which fits the
tutorial and moderate decks and not production resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ARS_TABLEAUX",
    "IMEX_CHAIN_METHODS",
    "ChainImplicitLinear",
    "build_chain_implicit_linear",
]

_G2 = 1.0 - 1.0 / 2.0**0.5
_D2 = 1.0 - 1.0 / (2.0 * _G2)
_G3 = 0.4358665215
_B1 = -1.5 * _G3**2 + 4.0 * _G3 - 0.25
_B2 = 1.5 * _G3**2 - 5.0 * _G3 + 1.25
# (explicit A, implicit A, explicit b, implicit b), Ascher-Ruuth-Spiteri 1997.
# ARS(2,2,2): two-stage explicit part, whose RK2 has no imaginary-axis interval
# (bracket advection is weakly unstable at any dt). ARS(3,4,3): third order,
# four explicit stages with an imaginary-axis interval, three RHS per step.
ARS_TABLEAUX = {
    "imex-ars2": (
        ((0, 0, 0), (_G2, 0, 0), (_D2, 1 - _D2, 0)),
        ((0, 0, 0), (0, _G2, 0), (0, 1 - _G2, _G2)),
        (_D2, 1 - _D2, 0),
        (0, 1 - _G2, _G2),
    ),
    "imex-ars3": (
        (
            (0, 0, 0, 0),
            (_G3, 0, 0, 0),
            (0.3212788860, 0.3966543747, 0, 0),
            (-0.105858296, 0.5529291479, 0.5529291479, 0),
        ),
        (
            (0, 0, 0, 0),
            (0, _G3, 0, 0),
            (0, (1 - _G3) / 2, _G3, 0),
            (0, _B1, _B2, _G3),
        ),
        (0, _B1, _B2, _G3),
        (0, _B1, _B2, _G3),
    ),
}
IMEX_CHAIN_METHODS = frozenset(ARS_TABLEAUX)


@dataclass(frozen=True)
class _ChainGroup:
    ky: np.ndarray  # (chains, L) row indices
    kx: np.ndarray  # (chains, L) column indices
    lmat: jnp.ndarray  # (chains, n, n) linear operator
    inv: jnp.ndarray  # (chains, n, n) inverse of I - gamma dt L


@dataclass(frozen=True)
class ChainImplicitLinear:
    """Per-chain linear operator and the inverse of ``I - gamma dt L``."""

    groups: tuple[_ChainGroup, ...]
    shape: tuple[int, ...]
    dt: float
    scheme: str

    @property
    def nbytes(self) -> int:
        return sum(int(g.lmat.nbytes + g.inv.nbytes) for g in self.groups)

    def _gather(self, g: _ChainGroup, G: jnp.ndarray) -> jnp.ndarray:
        x = G[:, :, :, g.ky, g.kx, :]  # (s, l, m, chains, L, z)
        return jnp.moveaxis(x, (3, 4), (0, 1)).reshape(g.ky.shape[0], -1)

    def _scatter(self, g: _ChainGroup, X: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
        x = X.reshape(g.ky.shape[0], g.ky.shape[1], *self.shape[:3], self.shape[-1])
        return out.at[:, :, :, g.ky, g.kx, :].set(jnp.moveaxis(x, (0, 1), (3, 4)))

    def _apply(self, G: jnp.ndarray, which: str) -> jnp.ndarray:
        out = G if which == "inv" else jnp.zeros_like(G)
        for g in self.groups:
            mat = g.inv if which == "inv" else g.lmat
            y = jnp.einsum("cij,cj->ci", mat, self._gather(g, G).astype(mat.dtype))
            out = self._scatter(g, y.astype(G.dtype), out)
        return out

    def matvec(self, G: jnp.ndarray) -> jnp.ndarray:
        """``L G`` on the chains (zero on rows no chain owns)."""
        return self._apply(G, "lmat")

    def solve(self, R: jnp.ndarray) -> jnp.ndarray:
        """``(I - gamma dt L)^{-1} R`` on the chains (identity elsewhere)."""
        return self._apply(R, "inv")

    def ars_step(
        self,
        G: jnp.ndarray,
        dG: jnp.ndarray,
        rhs: Callable[[jnp.ndarray], jnp.ndarray],
        project: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """One ARS step; ``dG = rhs(G)`` is the full (linear + bracket) RHS."""
        a_exp, a_imp, b_exp, b_imp = ARS_TABLEAUX[self.scheme]
        dt, gam = self.dt, a_imp[1][1]
        lin = [self.matvec(G)]
        non = [dG - lin[0]]
        for i in range(1, len(b_exp)):
            r = G + dt * sum(
                a_exp[i][j] * non[j] + a_imp[i][j] * lin[j] for j in range(i)
            )
            y = project(self.solve(r))
            lin.append((y - r) / (gam * dt))  # y = r + gamma dt L y
            non.append(rhs(y) - lin[i] if b_exp[i] or i + 1 < len(b_exp) else 0.0)
        return G + dt * sum(
            b_exp[j] * non[j] + b_imp[j] * lin[j] for j in range(len(b_exp))
        )


def _chains(lin: Callable, shape: tuple[int, ...], modes: np.ndarray, dtype) -> list:
    """Connected sets of the linear operator among the ``(ky, kx)`` ``modes``."""
    rng = np.random.default_rng(0)
    parent = {(int(r), int(c)): (int(r), int(c)) for r, c in np.argwhere(modes)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Modes at different ky never couple, so one probe per kx column finds the
    # kx links of every row at once.
    for j in np.nonzero(modes.any(axis=0))[0]:
        rows = np.nonzero(modes[:, j])[0]
        v = np.zeros(shape, dtype=complex)
        v[:, :, :, rows, j, :] = rng.normal(size=v[:, :, :, rows, j, :].shape)
        out = np.abs(np.asarray(lin(jnp.asarray(v, dtype=dtype)))).max(
            axis=(0, 1, 2, 5)
        )
        tol = 1e-9 * max(float(out.max()), 1e-300)
        for r in rows:
            for q in np.nonzero(out[r] > tol)[0]:
                if (int(r), int(q)) in parent:
                    parent[find((int(r), int(q)))] = find((int(r), int(j)))
    sets: dict = {}
    for mode in parent:
        sets.setdefault(find(mode), []).append(mode)
    return [sorted(c) for c in sets.values()]


def build_chain_implicit_linear(
    rhs: Callable[[jnp.ndarray], jnp.ndarray],
    shape: tuple[int, ...],
    dt: float,
    *,
    modes: np.ndarray,
    scheme: str = "imex-ars3",
    dtype=jnp.complex64,
) -> ChainImplicitLinear:
    """Probe the linear part of ``rhs`` per chain and factor ``I - gamma dt L``.

    ``rhs`` maps a state of ``shape = (ns, Nl, Nm, Nky, Nkx, Nz)`` to its full
    RHS; its odd part ``(rhs(v) - rhs(-v)) / 2`` is the linear operator.
    ``modes`` (``Nky x Nkx`` bool) selects what is solved: every ``ky >= 0``
    mode, dealiased or not -- the explicit route evolves the dealiased-out
    modes linearly too, and leaving them out of the solve makes their stiff
    linear terms explicit (non-finite at the first sample). The ``ky < 0`` rows
    of a two-sided layout pass through unchanged; the projector rebuilds them.
    """
    zero = jnp.zeros(shape, dtype)
    # The bracket is quadratic in G (fields are linear in G) and any source is
    # constant, so the odd part (rhs(v) - rhs(-v)) / 2 is the linear operator
    # exactly. Not a JVP: the field solve is a custom_vjp, which has no JVP rule.
    lin = jax.jit(lambda v: 0.5 * (rhs(v) - rhs(-v)))
    by_len: dict[int, list] = {}
    for chain in _chains(lin, shape, np.asarray(modes, dtype=bool), dtype):
        by_len.setdefault(len(chain), []).append(chain)
    blk = int(np.prod(shape[:3])) * shape[-1]
    idx = [
        (np.array(c)[..., 0], np.array(c)[..., 1]) for _, c in sorted(by_len.items())
    ]
    probe = ChainImplicitLinear(
        tuple(_ChainGroup(k, x, None, None) for k, x in idx), tuple(shape), dt, scheme
    )
    n_max = max(k.shape[1] for k, _ in idx) * blk

    def column(j):
        # Unit impulse at local index j of every chain of every group at once:
        # chains are disjoint and do not couple, so one RHS gives all columns.
        v = zero
        for g in probe.groups:
            n = g.ky.shape[1] * blk
            e = (
                jnp.zeros((g.ky.shape[0], n), dtype)
                .at[:, jnp.minimum(j, n - 1)]
                .set(jnp.where(j < n, 1.0, 0.0).astype(dtype))
            )
            v = probe._scatter(g, e, v)
        out = lin(v)
        return tuple(probe._gather(g, out) for g in probe.groups)

    cols = jax.lax.map(column, jnp.arange(n_max))  # per group: (n_max, chains, n)
    groups = []
    for g, c in zip(probe.groups, cols):
        n = g.ky.shape[1] * blk
        lmat = jnp.moveaxis(c[:n], 0, 2)  # (chains, out, in)
        inv = jnp.linalg.inv(
            jnp.eye(n, dtype=dtype) - ARS_TABLEAUX[scheme][1][1][1] * dt * lmat
        )
        groups.append(_ChainGroup(g.ky, g.kx, lmat, inv))
    return ChainImplicitLinear(tuple(groups), tuple(shape), float(dt), scheme)
