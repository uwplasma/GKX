"""Sparse-direct shift-invert replacement for gkx.runtime.dominant_eigenpair (research only).

The KBM shift-invert Krylov route fails its inner FGMRES solves on the EM decks
(residual ~0.99 at every probe, section 2 of REPORT.md), so this assembles the exact
sparse operator by column-group probing (the pattern logic of
gkx.objectives.core._sparse_direct_eigenvalue), factors A - sigma I with SuperLU and
runs ARPACK shift-invert. The returned pair is gated at a 1e-8 relative residual.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers_linear_krylov import EigenSolveStatus

SHIFT = complex(os.environ.get("KBM_SHIFT", "0.3-1.0j").replace(" ", ""))
NCAND = int(os.environ.get("KBM_NCAND", "8"))
INFO = {}


def assemble(v0, cache, params, terms):
    shape = tuple(v0.shape)
    nz = int(shape[-1])
    n = int(np.prod(shape))
    blocks = n // nz
    dtype = jnp.result_type(v0.dtype, jnp.complex64)

    def op(x):
        return linear_rhs_cached(
            x.reshape(shape),
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )[0].reshape(-1)

    probe = jax.jit(jax.vmap(op))
    local = np.zeros((blocks, blocks), bool)
    chain = np.zeros((blocks, blocks), bool)
    chunk = 256
    for z0 in {nz // 3, (2 * nz) // 3}:
        for b0 in range(0, blocks, chunk):
            bs = np.arange(b0, min(blocks, b0 + chunk))
            seeds = np.zeros((len(bs), blocks, nz))
            seeds[np.arange(len(bs)), bs, z0] = 1.0
            hit = (
                np.abs(
                    np.asarray(probe(jnp.asarray(seeds.reshape(len(bs), n), dtype)))
                ).reshape(len(bs), blocks, nz)
                > 0
            )
            away = np.delete(hit, z0, axis=2).any(axis=2)
            chain[:, bs] |= away.T
            local[:, bs] |= (hit[:, :, z0] & ~away).T
    local = (local | np.eye(blocks, dtype=bool)) & ~chain
    mask = (
        sp.kron(sp.csr_matrix(local), sp.identity(nz))
        + sp.kron(sp.csr_matrix(chain), sp.csr_matrix(np.ones((nz, nz))))
    ).tocsc()
    mask.sort_indices()
    # block-level distance-2 colouring: columns (b, z) with equal (colour(b), z) share no row
    B = sp.csr_matrix(local | chain)
    reach = (
        (B.T @ B).tolil().rows
    )  # column blocks sharing a row block (B[i, b]: column b hits row i)
    colour = -np.ones(blocks, int)
    for b in range(blocks):
        taken = {colour[c] for c in reach[b] if colour[c] >= 0}
        colour[b] = next(k for k in range(blocks) if k not in taken)
    group_of = (colour[:, None] * nz + np.arange(nz)[None, :]).reshape(-1)
    ng = int(group_of.max()) + 1
    rows_of = [mask.indices[mask.indptr[j] : mask.indptr[j + 1]] for j in range(n)]
    members = [[] for _ in range(ng)]
    for j, g in enumerate(group_of):
        members[g].append(j)
    rows, cols, vals = [], [], []
    for g0 in range(0, ng, 64):
        gs = np.arange(g0, min(ng, g0 + 64))
        seeds = np.zeros((len(gs), n))
        for k, g in enumerate(gs):
            seeds[k, members[g]] = 1.0
        prod = np.asarray(probe(jnp.asarray(seeds, dtype)))
        for k, g in enumerate(gs):
            for j in members[g]:
                r = rows_of[j]
                rows.append(r)
                cols.append(np.full(len(r), j))
                vals.append(prod[k, r])
    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    A.eliminate_zeros()
    # spot check against the matrix-free operator
    x = np.random.default_rng(1).standard_normal(n) + 0j
    ref = np.asarray(op(jnp.asarray(x, dtype)))
    INFO.update(
        n=n,
        nnz=int(A.nnz),
        groups=ng,
        assembly_err=float(np.linalg.norm(A @ x - ref) / np.linalg.norm(ref)),
    )
    return A, op, dtype


def dominant_eigenpair(v0, cache, params, terms=None, **kw):
    t0 = time.perf_counter()
    A, op, dtype = assemble(v0, cache, params, terms)
    t1 = time.perf_counter()
    if os.environ.get(
        "KBM_SAVE_A"
    ):  # keep the exact operator for the Galerkin basis study, then stop
        sp.save_npz(os.environ["KBM_SAVE_A"], A.tocsr())
        np.save(os.environ["KBM_SAVE_A"] + ".shape.npy", np.asarray(v0.shape))
        raise SystemExit(0)
    if os.environ.get(
        "KBM_DENSE"
    ):  # whole spectrum (small n only): the five largest growth rates
        import scipy.linalg as sla

        ev = sla.eigvals(A.toarray())
        top = ev[np.argsort(-ev.real)[:5]]
        INFO.update(dense_top=[(float(z.real), float(z.imag)) for z in top])
    lu = spla.splu(
        (A - SHIFT * sp.identity(A.shape[0], format="csc")).tocsc(), permc_spec="COLAMD"
    )
    inv = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=complex)
    mu, V = spla.eigs(inv, k=NCAND, which="LM", tol=1e-12)
    lam = SHIFT + 1.0 / mu
    order = np.argsort(-lam.real)
    best = []
    for i in order:
        v = V[:, i]
        Av = A @ v
        res = np.linalg.norm(Av - lam[i] * v) / max(
            np.linalg.norm(Av), abs(lam[i]) * np.linalg.norm(v)
        )
        best.append((lam[i], res))
    INFO.update(
        candidates=[
            (complex(lam_).real, complex(lam_).imag, float(r)) for lam_, r in best
        ],
        assemble_s=round(t1 - t0, 1),
        factor_eigs_s=round(time.perf_counter() - t1, 1),
    )
    lam0, res0 = best[0]
    v = V[:, order[0]]
    assert res0 < 1e-8, ("residual gate", res0, INFO)
    status = EigenSolveStatus(
        method="sparse-direct",
        route="scipy-splu-arpack",
        residual=float(res0),
        tolerance=1e-8,
        certified=True,
    )
    vec = jnp.asarray(v.reshape(v0.shape), dtype)
    return jnp.asarray(lam0), vec, status
