# ruff: noqa: E402
"""D5: preconditioner ceiling and recycling bake-off on the exact operator (read-only review).

Same case as D1 (Ny16, ky=+.3, Nl4/Nm8, rate .1, seed shift). Host-side SciPy
preconditioners are review instruments, not proposed production code.
Part A: iterations to 1e-5 (unrestarted and GMRES(20)) per preconditioner.
Part B: 12 outer shift-invert Arnoldi RHSs (generated with exact LU) solved by
        cold SOLVAX gmres(restart 20) vs gcrot(m=20, k=10) carrying the recycle space.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
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
from gkx.solvers_linear_krylov_algorithms import (  # noqa: E402
    _apply_operator,
    _normalize,
    build_shift_invert_preconditioner,
)
import solvax  # noqa: E402
from solvax import sparse_operator_matrix  # noqa: E402

t0 = time.perf_counter()
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
shape, n = seed.shape, seed.size
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 4, 8)
terms = linear_terms_to_term_config(ctx.terms)
sigma = complex(0.09302951 - 0.28199404j)
A = sparse_operator_matrix(
    lambda s: _apply_operator(s, cache, ctx.params, terms), seed, batch_size=64
).astype(np.complex128)
S = (A - sigma * sp.eye(n, format="csr", dtype=A.dtype)).tocsr()
Sc = S.tocsc()
b = np.asarray(_normalize(seed).reshape(-1))
_, hl_op = build_shift_invert_preconditioner(
    seed, cache, ctx.params, terms, jnp.asarray(sigma, seed.dtype), "hermite-line"
)
hl_jit = jax.jit(hl_op)
print(f"setup {time.perf_counter() - t0:.1f}s n={n} nnz(S)={S.nnz}", flush=True)

idx = np.arange(n).reshape(shape)


def block_jacobi(axis):
    blocks = []
    for k in range(shape[axis]):
        rows = np.take(idx, k, axis=axis).reshape(-1)
        blocks.append((rows, spla.splu(Sc[rows][:, rows].tocsc())))

    def apply(v):
        out = np.zeros_like(v)
        for rows, lu in blocks:
            out[rows] = lu.solve(v[rows])
        return out

    fill = sum(lu.L.nnz + lu.U.nnz for _, lu in blocks)
    return apply, fill


pre = {
    "none": (lambda v: v, 0),
    "hermite-line": (lambda v: np.asarray(hl_jit(jnp.asarray(v))), 0),
}
pre["block-jacobi-l"] = block_jacobi(1)
pre["block-jacobi-m"] = block_jacobi(2)
for drop, ff in ((1e-2, 3), (1e-4, 10)):
    ilu = spla.spilu(Sc, drop_tol=drop, fill_factor=ff)
    pre[f"ilu(drop={drop:g},fill={ff})"] = (ilu.solve, ilu.L.nnz + ilu.U.nnz)
exact = spla.splu(Sc)
print(f"exact LU fill={exact.L.nnz + exact.U.nnz}", flush=True)


def iters_to(minv, restart, cap=300, tol=1e-5):
    hist = []
    B = spla.LinearOperator((n, n), matvec=lambda v: S @ minv(v), dtype=np.complex128)
    y, _ = spla.gmres(
        B,
        b,
        rtol=1e-12,
        atol=0.0,
        restart=restart,
        maxiter=max(1, cap // restart),
        callback=lambda r: hist.append(float(r)),
        callback_type="pr_norm",
    )
    hist = np.asarray(hist)
    hit = np.flatnonzero(hist <= tol)
    true = np.linalg.norm(b - S @ minv(y)) / np.linalg.norm(b)
    return (int(hit[0]) + 1 if hit.size else None), true


print("\nPart A: iterations to 1e-5 (cap 300); fill = factor nnz", flush=True)
for name, (minv, fill) in pre.items():
    t = time.perf_counter()
    full_its, full_true = iters_to(minv, 300)
    r20_its, r20_true = iters_to(minv, 20)
    print(
        f"  {name:26s} unrestarted={full_its} (true {full_true:.1e})  GMRES(20)={r20_its} "
        f"(true {r20_true:.1e})  fill={fill}  t={time.perf_counter() - t:.1f}s",
        flush=True,
    )

print(
    "\nPart B: 12 outer shift-invert RHSs, hermite-line right preconditioner, rtol 1e-5",
    flush=True,
)
V = [b / np.linalg.norm(b)]
for _ in range(11):
    w = exact.solve(V[-1])
    for v in V:
        w = w - np.vdot(v, w) * v
    w = w - sum(np.vdot(v, w) * v for v in V)
    V.append(w / np.linalg.norm(w))


def mv(x):
    return (
        _apply_operator(x.reshape(shape), cache, ctx.params, terms)
        - sigma * x.reshape(shape)
    ).reshape(-1)


gm = jax.jit(
    lambda r: solvax.gmres(
        mv, r, x0=None, precond=hl_op, rtol=1e-5, restart=20, max_restarts=20
    )
)
cold = []
for v in V:
    s = gm(jnp.asarray(v))
    cold.append((int(s.iterations), bool(s.converged)))
print(
    "  cold gmres(20):  its =",
    [c[0] for c in cold],
    "converged =",
    sum(c[1] for c in cold),
    "/12",
    "total =",
    sum(c[0] for c in cold),
    flush=True,
)
for strategy in ("fifo", "harmonic"):
    try:
        recycle = None
        its = []
        conv = 0
        for j, v in enumerate(V):
            s = solvax.gcrot(
                mv,
                jnp.asarray(v),
                precond=hl_op,
                m=20,
                k=10,
                rtol=1e-5,
                max_restarts=20,
                recycle=recycle,
                recycle_strategy=strategy,
            )
            recycle = s.recycle
            its.append(int(s.iterations))
            conv += bool(s.converged)
        print(
            f"  gcrot(20,10) {strategy:8s}: its = {its} converged = {conv}/12 total = {sum(its)}",
            flush=True,
        )
    except Exception as exc:
        print(f"  gcrot {strategy} failed: {type(exc).__name__}: {exc}", flush=True)
print(f"\ntotal {time.perf_counter() - t0:.1f}s", flush=True)
