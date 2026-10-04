"""Implicit linear operator per linked chain, for the ``imex-ars2`` method.

Kinetic-electron runs are step-limited by electron parallel streaming and the
electromagnetic electron mode (|lambda| ~ 500-700 against drift rates ~ 1 on
the Cyclone tutorial box), not by accuracy. The linear operator couples modes
only along one ``(ky, kx)`` twist-shift chain, so its restriction to a chain is
a dense matrix of size ``ns Nl Nm Nz L`` (``L`` chain links). This module
materializes those matrices once, by probing the linear part of the run's own
RHS (its JVP at zero, so every linear term -- streaming, fields, mirror,
drifts, dissipation, end damping -- enters exactly as the explicit route
applies it), and inverts ``I - gamma dt L`` per chain.

The time step is ARS(2,2,2) (Ascher-Ruuth-Spiteri 1997): L-stable SDIRK for
the linear part, explicit for the nonlinear bracket, second order, one
factorization per dt. It is the plan's section 5.4 route with dense per-chain
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

__all__ = ["ARS2_GAMMA", "ChainImplicitLinear", "build_chain_implicit_linear"]

ARS2_GAMMA = 1.0 - 1.0 / 2.0**0.5
ARS2_DELTA = 1.0 - 1.0 / (2.0 * ARS2_GAMMA)


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

    def ars2_step(
        self,
        G: jnp.ndarray,
        dG: jnp.ndarray,
        rhs: Callable[[jnp.ndarray], jnp.ndarray],
        project: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """One ARS(2,2,2) step; ``dG = rhs(G)`` is the full (linear+bracket) RHS."""
        dt, gam, dlt = self.dt, ARS2_GAMMA, ARS2_DELTA
        n1 = dG - self.matvec(G)
        y2 = project(self.solve(G + gam * dt * n1))
        ly2 = self.matvec(y2)
        n2 = rhs(y2) - ly2
        return self.solve(
            G + dt * (dlt * n1 + (1.0 - dlt) * n2) + dt * (1.0 - gam) * ly2
        )


def _chains(lin: Callable, shape: tuple[int, ...], rows: np.ndarray, dtype) -> list:
    """Connected (ky, kx) sets of the linear operator on the stored ``rows``."""
    ny, nx = shape[3], shape[4]
    rng = np.random.default_rng(0)
    parent = {(int(r), j): (int(r), j) for r in rows for j in range(nx)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Modes at different ky never couple, so one probe per kx column finds the
    # kx links of every row at once.
    for j in range(nx):
        v = np.zeros(shape, dtype=complex)
        v[:, :, :, rows, j, :] = rng.normal(size=v[:, :, :, rows, j, :].shape)
        out = np.abs(np.asarray(lin(jnp.asarray(v, dtype=dtype)))).max(
            axis=(0, 1, 2, 5)
        )
        tol = 1e-9 * max(float(out.max()), 1e-300)
        for r in rows:
            for q in np.nonzero(out[r] > tol)[0]:
                parent[find((int(r), int(q)))] = find((int(r), j))
    del ny
    sets: dict = {}
    for mode in parent:
        sets.setdefault(find(mode), []).append(mode)
    return [sorted(c) for c in sets.values()]


def build_chain_implicit_linear(
    rhs: Callable[[jnp.ndarray], jnp.ndarray],
    shape: tuple[int, ...],
    dt: float,
    *,
    ky: np.ndarray,
    dtype=jnp.complex64,
) -> ChainImplicitLinear:
    """Probe the linear part of ``rhs`` per chain and factor ``I - gamma dt L``.

    ``rhs`` maps a state of ``shape = (ns, Nl, Nm, Nky, Nkx, Nz)`` to its full
    RHS; its JVP at zero is the linear operator. Only rows with ``ky >= 0`` are
    solved: on a two-sided layout the projector rebuilds the others.
    """
    zero = jnp.zeros(shape, dtype)
    lin = jax.jit(lambda v: jax.jvp(rhs, (zero,), (v,))[1])
    rows = np.nonzero(np.asarray(ky) >= 0.0)[0]
    by_len: dict[int, list] = {}
    for chain in _chains(lin, shape, rows, dtype):
        by_len.setdefault(len(chain), []).append(chain)
    blk = int(np.prod(shape[:3])) * shape[-1]
    idx = [
        (np.array(c)[..., 0], np.array(c)[..., 1]) for _, c in sorted(by_len.items())
    ]
    probe = ChainImplicitLinear(
        tuple(_ChainGroup(k, x, None, None) for k, x in idx), tuple(shape), dt
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
        inv = jnp.linalg.inv(jnp.eye(n, dtype=dtype) - ARS2_GAMMA * dt * lmat)
        groups.append(_ChainGroup(g.ky, g.kx, lmat, inv))
    return ChainImplicitLinear(tuple(groups), tuple(shape), float(dt))
