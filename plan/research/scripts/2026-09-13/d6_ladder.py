# ruff: noqa: E402
"""D6: exact-solver size ladder and drift-dominance recheck (read-only review).

For each (Nx, Nz, Nl, Nm): assemble the exact sparse operator by column probing,
SuperLU-factor A - sigma I, report nnz/fill/times, the certified eigenvalue nearest
sigma (sigma = the exact target of the smallest rung, re-targeted per rung by
taking the eigenvalue nearest the previous rung's), and unrestarted GMRES
iterations to 1e-5 with the Hermite-line preconditioner on the full operator and
with drifts off.  Iteration counts are load-independent; times are indicative.
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
from gkx.terms.config import TermConfig  # noqa: E402
from gkx.solvers_linear_krylov_algorithms import (  # noqa: E402
    _apply_operator,
    _normalize,
    build_shift_invert_preconditioner,
)
from solvax import sparse_operator_matrix  # noqa: E402

FIELDS = (
    "streaming mirror curvature gradb diamagnetic collisions hypercollisions "
    "hyperdiffusion end_damping apar bpar nonlinear"
).split()
LADDER = [  # (Nx, Ny, Nz, ntheta, nperiod, Nl, Nm)
    (1, 4, 16, 16, 1, 4, 8),
    (1, 4, 32, 32, 1, 4, 8),
    (1, 4, 32, 32, 1, 8, 16),
    (1, 4, 48, 16, 2, 8, 16),
    (1, 4, 64, 64, 1, 8, 32),
    (4, 4, 32, 32, 1, 8, 16),
]
deps = _runtime_linear_dispatch_deps().full_deps
sigma = complex(0.09302951 - 0.28199404j)
cfg0, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")


def its_to(S, minv, b, tol=1e-5, cap=400):
    hist = []
    B = spla.LinearOperator(S.shape, matvec=lambda v: S @ minv(v), dtype=np.complex128)
    spla.gmres(
        B,
        b,
        rtol=1e-12,
        atol=0.0,
        restart=cap,
        maxiter=1,
        callback=lambda r: hist.append(float(r)),
        callback_type="pr_norm",
    )
    hist = np.asarray(hist)
    hit = np.flatnonzero(hist <= tol)
    return (int(hit[0]) + 1 if hit.size else None), (
        float(hist[-1]) if hist.size else None
    )


for Nx, Ny, Nz, ntheta, nperiod, Nl, Nm in LADDER:
    t0 = time.perf_counter()
    cfg = replace(
        cfg0,
        grid=replace(
            cfg0.grid, Nx=Nx, Ny=Ny, Nz=Nz, ntheta=ntheta, nperiod=nperiod, jtwist=1
        ),
        time=replace(cfg0.time, damp_ends_rate=0.1),
    )
    ctx = _prepare_linear_runtime_context(
        cfg,
        deps=deps,
        ky_target=0.3,
        n_laguerre=Nl,
        n_hermite=Nm,
        solver="krylov",
        fit_signal="auto",
        return_state=False,
        initial_state=None,
        status_callback=None,
    )
    seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
    shape, n = seed.shape, int(seed.size)
    cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, Nl, Nm)
    terms = linear_terms_to_term_config(ctx.terms)
    chains = [tuple(x.shape) for x in cache.linked_indices]
    t1 = time.perf_counter()
    A = sparse_operator_matrix(
        lambda s: _apply_operator(s, cache, ctx.params, terms), seed, batch_size=64
    ).astype(np.complex128)
    t_asm = time.perf_counter() - t1
    S = (A - sigma * sp.eye(n, format="csr", dtype=A.dtype)).tocsc()
    t2 = time.perf_counter()
    lu = spla.splu(S)
    t_lu = time.perf_counter() - t2
    b = np.asarray(_normalize(seed).reshape(-1))
    t3 = time.perf_counter()
    lu.solve(b)
    t_solve = time.perf_counter() - t3
    try:
        vals = spla.eigs(
            A,
            k=4,
            sigma=sigma,
            return_eigenvectors=False,
            tol=1e-10,
            maxiter=20000,
            OPinv=spla.LinearOperator((n, n), matvec=lu.solve, dtype=np.complex128),
        )
        vals = vals[np.argsort(np.abs(vals - sigma))]
        eig_txt = "; ".join(f"{v.real:+.5f}{v.imag:+.5f}j" for v in vals[:3])
    except Exception as exc:
        eig_txt = f"eigs failed {type(exc).__name__}"
    # dead rows: columns with no coupling to the seed's support and zero real diagonal
    kx_of = np.broadcast_to(
        np.arange(shape[-2])[None, None, None, None, :, None], shape
    ).reshape(-1)
    covered = (
        np.unique(
            np.concatenate([np.asarray(m).reshape(-1) for m in cache.linked_indices])
        )
        if chains
        else np.array([])
    )
    cov_kx = (
        sorted({int(i) // shape[-3] for i in covered})
        if chains
        else list(range(shape[-2]))
    )
    dead = float(np.mean(~np.isin(kx_of, cov_kx)))
    _, hl = build_shift_invert_preconditioner(
        seed, cache, ctx.params, terms, jnp.asarray(sigma, seed.dtype), "hermite-line"
    )
    hl = jax.jit(hl)

    def mv(v):
        return np.asarray(hl(jnp.asarray(v, jnp.complex128)))

    its_full, res_full = its_to(S, mv, b)
    # drifts off
    terms_nd = TermConfig(
        **{
            f: (0.0 if f in ("curvature", "gradb") else float(getattr(terms, f)))
            for f in FIELDS
        }
    )
    A_nd = sparse_operator_matrix(
        lambda s: _apply_operator(s, cache, ctx.params, terms_nd), seed, batch_size=64
    ).astype(np.complex128)
    S_nd = (A_nd - sigma * sp.eye(n, format="csr", dtype=A.dtype)).tocsr()
    _, hl_nd = build_shift_invert_preconditioner(
        seed,
        cache,
        ctx.params,
        terms_nd,
        jnp.asarray(sigma, seed.dtype),
        "hermite-line",
    )
    hl_nd = jax.jit(hl_nd)
    its_nd, res_nd = its_to(
        S_nd, lambda v: np.asarray(hl_nd(jnp.asarray(v, jnp.complex128))), b
    )
    print(
        f"Nx{Nx} Nz{Nz} (ntheta{ntheta} nperiod{nperiod}) Nl{Nl} Nm{Nm}: n={n} chains={chains} dead_rows={dead:.2f} "
        f"nnz={A.nnz} nnz/row={A.nnz / n:.1f} LU_fill={lu.L.nnz + lu.U.nnz} fill_ratio={(lu.L.nnz + lu.U.nnz) / A.nnz:.1f} "
        f"asm={t_asm:.1f}s lu={t_lu:.2f}s solve={t_solve * 1e3:.1f}ms | nearest eigs: {eig_txt} | "
        f"hermite-line its_to_1e-5: full={its_full} (res@400 {res_full:.1e}) drifts_off={its_nd} | rung {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
