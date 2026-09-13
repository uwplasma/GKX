# ruff: noqa: E402
"""D1b: where do the Re=0 eigenvalues next to sigma live? (read-only review)

Same case as D1. Splits the state by kx rows covered / not covered by the linked
chains, checks block coupling of the exact operator, and reports eigenvector and
seed support on the uncovered rows.
"""

import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path("/Users/rogeriojorge/local/GKX-worktrees/perf-review")
sys.path.insert(0, str(REPO))
from gkx.runtime import _runtime_linear_dispatch_deps  # noqa: E402
from gkx.workflows.linear import _prepare_linear_runtime_context  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402
from gkx.operators.linear.params import linear_terms_to_term_config  # noqa: E402
from gkx.solvers_linear_krylov_algorithms import _apply_operator, _normalize  # noqa: E402
from solvax import sparse_operator_matrix  # noqa: E402

cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
cfg = replace(
    cfg,
    grid=replace(cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(cfg.time, damp_ends_rate=0.1),
)
deps = _runtime_linear_dispatch_deps().full_deps
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=deps,
    ky_target=0.3,
    n_laguerre=4,
    n_hermite=8,
    solver="krylov",
    fit_signal="auto",
    return_state=False,
    initial_state=None,
    status_callback=None,
)
seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
shape = seed.shape  # (s, l, m, ky, kx, z)
n = seed.size
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 4, 8)
terms = linear_terms_to_term_config(ctx.terms)
sigma = complex(0.09302951 - 0.28199404j)
Ny, Nx = shape[-3], shape[-2]
covered_flat = np.unique(
    np.concatenate([np.asarray(m).reshape(-1) for m in cache.linked_indices])
)
# linked flat index = kx * Ny + ky (swapaxes(ky, kx) then reshape)
covered_kx = sorted({int(i) // Ny for i in covered_flat})
print(
    "shape",
    shape,
    "n",
    n,
    "kx grid",
    np.round(np.asarray(ctx.grid.kx).reshape(-1), 4).tolist(),
    "covered kx indices",
    covered_kx,
    flush=True,
)
print(
    "dealias kx row (ky sel)",
    np.asarray(cache.dealias_mask).reshape(-1, Nx)[0].astype(int).tolist()
    if np.asarray(cache.dealias_mask).size % Nx == 0
    else "n/a",
    flush=True,
)

kx_of = np.broadcast_to(np.arange(Nx)[None, None, None, None, :, None], shape).reshape(
    -1
)
cov = np.isin(kx_of, covered_kx)
A = (
    sparse_operator_matrix(
        lambda s: _apply_operator(s, cache, ctx.params, terms), seed, batch_size=64
    )
    .astype(np.complex128)
    .tocsr()
)
Acc, Acu = A[cov][:, cov], A[cov][:, ~cov]
Auc, Auu = A[~cov][:, cov], A[~cov][:, ~cov]
print(
    f"rows covered={cov.sum()} uncovered={(~cov).sum()} | nnz Acc={Acc.nnz} Acu={Acu.nnz} "
    f"Auc={Auc.nnz} Auu={Auu.nnz} | ||Acu||_max={abs(Acu).max() if Acu.nnz else 0:.2e} "
    f"||Auc||_max={abs(Auc).max() if Auc.nnz else 0:.2e}",
    flush=True,
)
off = Auu - sp.diags(Auu.diagonal())
print(
    f"Auu diagonal only? off-diag nnz={off.nnz}; diag real range "
    f"[{Auu.diagonal().real.min():.2e},{Auu.diagonal().real.max():.2e}] imag range "
    f"[{Auu.diagonal().imag.min():.3f},{Auu.diagonal().imag.max():.3f}]",
    flush=True,
)
b = np.asarray(_normalize(seed).reshape(-1))
print(
    f"seed norm fraction on uncovered rows = {np.linalg.norm(b[~cov]) ** 2:.3e}",
    flush=True,
)
lu = spla.splu((A - sigma * sp.eye(n, format="csc", dtype=A.dtype)).tocsc())
vals, vecs = spla.eigs(
    A,
    k=10,
    sigma=sigma,
    OPinv=spla.LinearOperator((n, n), matvec=lu.solve, dtype=np.complex128),
    tol=1e-12,
    maxiter=20000,
)
for i in np.argsort(np.abs(vals - sigma)):
    v = vecs[:, i] / np.linalg.norm(vecs[:, i])
    print(
        f"  eig {vals[i].real:+.6f}{vals[i].imag:+.6f}j  |eig-sigma|={abs(vals[i] - sigma):.4f} "
        f"uncovered-row weight={np.linalg.norm(v[~cov]) ** 2:.3e}",
        flush=True,
    )
# Spectrum of the covered block alone near sigma
Sc = (Acc - sigma * sp.eye(Acc.shape[0], format="csc", dtype=A.dtype)).tocsc()
luc = spla.splu(Sc)
vc = spla.eigs(
    Acc,
    k=6,
    sigma=sigma,
    return_eigenvectors=False,
    OPinv=spla.LinearOperator(Acc.shape, matvec=luc.solve, dtype=np.complex128),
    tol=1e-12,
)
vc = vc[np.argsort(np.abs(vc - sigma))]
print(
    "covered block nearest eigs:",
    "; ".join(f"{z.real:+.6f}{z.imag:+.6f}j" for z in vc),
    flush=True,
)
