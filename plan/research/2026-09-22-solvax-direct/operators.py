# ruff: noqa: E402
"""Assemble the linear systems GKX solves as exact sparse matrices (CSR).

SOLVAX-DIRECT (plan G.2), step 2. Measurement only: no GKX or SOLVAX source is
changed. Every matrix is recovered from GKX's own matrix-free operator, so it is
the operator the solvers see, not a re-derivation.

Operators (one linked Cyclone ky at a time, complex128):

``full``        ``A``: every term of the deck (streaming, mirror, drifts, drive,
                end damping, hypercollisions, collisions, the phi response).
                Shift-invert solves ``A - sigma I``; the implicit linear step and
                the IMEX step solve ``I - dt A``: the same pattern.
``streaming``   ``A_S``: parallel streaming and its phi coupling only. ``I - dt A_S``
                is the implicit-streaming system of plan section 5.4.
``collisions``  ``A_C``: the collision operator only, at ``nu = 0.01``. ``I - dt A_C``
                is an implicit collision step.

Assembly is Curtis-Powell-Reid compression (``solvax.compression``): the pattern
is found by probing the operator once per ``(l, m)`` moment at two ``z`` points,
which tells, for every pair of moments, whether they couple and whether that
coupling is z-local or spans the chain (spectral streaming does); the pattern is
the Kronecker expansion of that moment graph, and ``verify_products`` checks the
recovered matrix against the operator on random vectors.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

REPO = Path(__file__).resolve().parents[3]
DECK = "examples/linear/axisymmetric/cyclone.toml"

# name: (Nz, ntheta, nperiod, Nl, Nm). "r48", "r96" and "prod" are the rungs of
# the 2026-09-13 exact ladder and of Q7/Q21/Q28, so the rows compare directly.
CASES = {
    "c24": (24, 24, 1, 8, 16),
    "r48": (48, 16, 2, 8, 16),
    "c48": (48, 16, 2, 16, 48),
    "r96": (96, 32, 2, 8, 24),
    "prod": (96, 32, 2, 16, 48),
}
# Shifts near the ITG branch; ky=0.3 is the ladder's reference shift.
SHIFTS = {
    0.1: complex(0.03, -0.09),
    0.3: complex(0.09302951, -0.28199404),
    0.5: complex(0.10, -0.45),
}
TERM_FIELDS = (
    "streaming mirror curvature gradb diamagnetic collisions hypercollisions "
    "hyperdiffusion end_damping apar bpar nonlinear"
).split()


def _terms(base, **over):
    from gkx.terms.config import TermConfig

    return TermConfig(**{f: float(over.get(f, getattr(base, f))) for f in TERM_FIELDS})


class Case:
    """One linked Cyclone ky: cache, params, terms and the flat operators."""

    def __init__(self, name: str, ky: float = 0.3, nu: float = 0.0):
        from gkx.operators.linear.params import linear_terms_to_term_config
        from gkx.runtime import _runtime_linear_dispatch_deps
        from gkx.workflows.linear import _prepare_linear_runtime_context
        from gkx.workflows.runtime.toml import load_runtime_from_toml

        Nz, ntheta, nperiod, Nl, Nm = CASES[name]
        cfg0, _ = load_runtime_from_toml(REPO / DECK)
        species = tuple(replace(s, nu=nu) for s in cfg0.species) if nu else cfg0.species
        cfg = replace(
            cfg0,
            species=species,
            grid=replace(
                cfg0.grid, Nx=1, Ny=24, Nz=Nz, ntheta=ntheta, nperiod=nperiod, jtwist=1
            ),
            time=replace(cfg0.time, damp_ends_rate=0.1),
        )
        deps = _runtime_linear_dispatch_deps().full_deps
        ctx = _prepare_linear_runtime_context(
            cfg,
            deps=deps,
            ky_target=ky,
            n_laguerre=Nl,
            n_hermite=Nm,
            solver="krylov",
            fit_signal="auto",
            return_state=False,
            initial_state=None,
            status_callback=None,
        )
        self.name, self.ky_target = name, ky
        self.ky = float(np.asarray(ctx.grid.ky).reshape(-1)[0])
        self.dims = dict(Nz=Nz, ntheta=ntheta, nperiod=nperiod, Nl=Nl, Nm=Nm)
        self.params = ctx.params
        self.seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
        self.cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, Nl, Nm)
        self.terms = linear_terms_to_term_config(ctx.terms)
        self.shape = tuple(int(s) for s in self.seed.shape)
        ns, nl, nm, ny, nx, nz = self.shape
        if ns * ny * nx != 1:
            raise ValueError("one species, one ky and one kx per case")
        self.n = int(self.seed.size)
        self.sigma = SHIFTS.get(round(ky, 3), SHIFTS[0.3])

    def term_config(self, which: str):
        T = self.terms
        if which == "full":
            return T
        zero = {f: 0.0 for f in TERM_FIELDS}
        if which == "streaming":
            return _terms(T, **(zero | {"streaming": 1.0}))
        if which == "collisions":
            return _terms(T, **(zero | {"collisions": 1.0}))
        raise ValueError(which)

    def apply_fn(self, which: str):
        from gkx.solvers_linear_krylov_algorithms import _apply_operator

        terms, cache, params, shape = self.term_config(which), self.cache, self.params, self.shape

        def apply(x):
            return _apply_operator(x.reshape(shape), cache, params, terms).reshape(-1)

        return apply


def moment_graph(apply, shape, *, probes=(1 / 3, 2 / 3), rtol=0.0):
    """Couplings between (l, m) moments, split into z-local and chain-wide.

    Returns two boolean (Nl*Nm, Nl*Nm) arrays ``local[row, col]`` and
    ``chain[row, col]``: column moment ``col`` reaches row moment ``row`` only at
    the probed z (local), or at other z too (chain-wide).
    """

    _, nl, nm, _, _, nz = shape
    nb = nl * nm
    local = np.zeros((nb, nb), bool)
    chain = np.zeros((nb, nb), bool)
    batched = jax.jit(jax.vmap(apply))
    for frac in probes:
        z0 = min(max(int(round(frac * (nz - 1))), 0), nz - 1)
        seeds = np.zeros((nb, nb, nz), np.complex128)
        seeds[np.arange(nb), np.arange(nb), z0] = 1.0 + 0.5j
        out = np.asarray(batched(jnp.asarray(seeds.reshape(nb, -1)))).reshape(nb, nb, nz)
        scale = np.abs(out).max()
        hit = np.abs(out) > rtol * scale
        at = hit[:, :, z0]
        off = hit.copy()
        off[:, :, z0] = False
        away = off.any(axis=2)
        chain |= away.T
        local |= (at & ~away).T
    local &= ~chain
    return local, chain


def kron_groups(local, chain, nz):
    """Column groups for the Kronecker pattern, from a colouring of the moments.

    Columns ``(c, z)`` and ``(c', z')`` share a row when some row moment is
    reached by both and their z-supports meet: always if either link is
    chain-wide, only at ``z = z'`` if both are local. Moments are coloured
    greedily on that conflict graph; a colour class with no chain-wide column
    link takes all z in one group, any other class one group per z. This is the
    pattern's structure used directly, so no per-column colouring is needed.
    """

    link = local | chain
    li, ch = link.astype(np.int64), chain.astype(np.int64)
    conflict = (ch.T @ li + li.T @ ch + li.T @ li) > 0
    nb = link.shape[0]
    chain_col = chain.any(axis=0)
    order = np.argsort(-(conflict.sum(axis=1) + nb * chain_col), kind="stable")
    color = np.full(nb, -1)
    for c in order:
        used = set(color[conflict[c] & (color >= 0)].tolist())
        k = 0
        while k in used:
            k += 1
        color[c] = k
    groups = []
    for k in range(color.max() + 1):
        members = np.flatnonzero(color == k)
        if chain_col[members].any():
            groups += [members * nz + z for z in range(nz)]
        else:
            groups.append((members[:, None] * nz + np.arange(nz)[None, :]).ravel())
    return groups


def pattern_from_graph(local, chain, nz):
    """Kronecker expansion of the moment graph: z-local -> I, chain-wide -> ones."""

    return (
        sp.kron(sp.csr_matrix(local), sp.identity(nz, format="csr"))
        + sp.kron(sp.csr_matrix(chain), sp.csr_matrix(np.ones((nz, nz))))
    ).tocsr().astype(bool).astype(np.float64)


def assemble(case: Case, which: str, *, batch: int = 64):
    """Exact CSR matrix of one operator, with assembly statistics."""

    from solvax.compression import matrix_from_products

    apply = case.apply_fn(which)
    stats: dict = {}
    t = time.perf_counter()
    local, chain = moment_graph(apply, case.shape)
    pattern = pattern_from_graph(local, chain, case.shape[-1])
    stats["probe_s"] = time.perf_counter() - t
    t = time.perf_counter()
    groups = kron_groups(local, chain, case.shape[-1])
    stats["coloring_s"] = time.perf_counter() - t
    batched = jax.jit(jax.vmap(apply))
    n = case.n
    seeds = []
    for g in groups:
        s = np.zeros(n, np.complex128)
        s[g] = 1.0
        seeds.append(s)
    t = time.perf_counter()
    images = []
    for start in range(0, len(seeds), batch):
        images.append(np.asarray(batched(jnp.asarray(np.stack(seeds[start : start + batch])))))
    images = np.concatenate(images)
    stats["products_s"] = time.perf_counter() - t
    def cached(v):  # matrix_from_products calls groups in order; replay the batch
        cached.k += 1
        return images[cached.k - 1]

    cached.k = 0
    t = time.perf_counter()
    matrix = matrix_from_products(cached, pattern, groups=groups, dtype=np.complex128)
    matrix.eliminate_zeros()
    stats["recover_s"] = time.perf_counter() - t
    stats["verify_rel"] = verify(matrix, apply)
    stats["products"] = len(groups)
    stats["pattern_nnz"] = int(pattern.nnz)
    stats["moment_links_local"] = int(local.sum())
    stats["moment_links_chain"] = int(chain.sum())
    return matrix.tocsr(), stats


def verify(matrix, apply, samples: int = 3, seed: int = 0) -> float:
    """Largest relative 2-norm difference of ``matrix @ v`` and ``apply(v)``.

    Complex probes: ``solvax.compression.verify_products`` casts to float64 and
    would compare real parts only.
    """

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        v = rng.standard_normal(matrix.shape[1]) + 1j * rng.standard_normal(matrix.shape[1])
        ref = np.asarray(apply(jnp.asarray(v)))
        worst = max(worst, float(np.linalg.norm(matrix @ v - ref) / np.linalg.norm(ref)))
    return worst


def structure(matrix) -> dict:
    """Sparsity, bandwidth (natural and RCM) and row statistics."""

    A = matrix.tocsr()
    n = A.shape[0]
    coo = A.tocoo()
    row_nnz = np.diff(A.indptr)
    bw = int(np.abs(coo.row - coo.col).max()) if A.nnz else 0
    perm = reverse_cuthill_mckee((abs(A) + abs(A.T)).tocsr(), symmetric_mode=True)
    P = A[perm][:, perm].tocoo()
    bw_rcm = int(np.abs(P.row - P.col).max()) if A.nnz else 0
    sym = abs(abs(A) - abs(A).T).sum() / max(abs(A).sum(), 1e-300)
    return {
        "n": n,
        "nnz": int(A.nnz),
        "density": A.nnz / n**2,
        "nnz_row_mean": float(row_nnz.mean()),
        "nnz_row_max": int(row_nnz.max()),
        "bandwidth": bw,
        "bandwidth_rcm": bw_rcm,
        "pattern_asymmetry": float(sym),
    }


if __name__ == "__main__":  # pragma: no cover - smoke run
    import argparse
    import json

    jax.config.update("jax_enable_x64", True)
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="c24")
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--which", default="full")
    a = p.parse_args()
    c = Case(a.case, a.ky, nu=0.01 if a.which == "collisions" else 0.0)
    M, st = assemble(c, a.which)
    print(json.dumps({"case": a.case, "ky": c.ky, "which": a.which} | st | structure(M)))
