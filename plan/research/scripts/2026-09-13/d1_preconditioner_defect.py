# ruff: noqa: E402
"""D1: exact-operator discriminator for the rejected shift-invert pilot (read-only review).

Same case as /tmp/gkx-gmres-restart-control.beZ0sR/run.py (Ny16, ky=+.3, Nl4/Nm8,
rate .1, seed shift). No GKX source is modified. For each term ablation:
  * assemble the exact sparse operator A (CSR, no dense matrix),
  * exact eigenvalues nearest sigma via SuperLU shift-invert (ARPACK),
  * full-history GMRES on B = (A - sigma I) M^{-1} (true residual), M = preconditioner,
  * a few extreme eigenvalues of B (largest |z| directly, smallest |z| via M (A-sigma I)^{-1}).
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
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
import solvax  # noqa: E402
from solvax import sparse_operator_matrix  # noqa: E402

FIELDS = (
    "streaming mirror curvature gradb diamagnetic collisions hypercollisions "
    "hyperdiffusion end_damping apar bpar nonlinear"
).split()

t0 = time.perf_counter()
print(
    "env",
    sys.version.split()[0],
    jax.__version__,
    np.__version__,
    scipy.__version__,
    solvax.__version__,
    jax.devices(),
    flush=True,
)
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
actual_ky = float(np.asarray(ctx.grid.ky).reshape(-1)[0])
seed = jnp.asarray(np.asarray(ctx.initial_state), dtype=jnp.complex128)
b = np.asarray(_normalize(seed).reshape(-1))
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 4, 8)
base_terms = linear_terms_to_term_config(ctx.terms)
sigma = complex(0.09302951 - 0.28199404j)
n = seed.size
print(
    "case shape",
    seed.shape,
    "n",
    n,
    "ky",
    actual_ky,
    "linked",
    [tuple(x.shape) for x in cache.linked_indices],
    "base_terms",
    {f: float(getattr(base_terms, f)) for f in FIELDS},
    flush=True,
)
print("setup_s", round(time.perf_counter() - t0, 2), flush=True)

ABLATIONS = {
    "full": {},
    "no_end_damping": {"end_damping": 0.0},
    "no_mirror": {"mirror": 0.0},
    "no_diamagnetic": {"diamagnetic": 0.0},
    "covered_only": {
        "mirror": 0.0,
        "diamagnetic": 0.0,
        "end_damping": 0.0,
        "apar": 0.0,
        "bpar": 0.0,
    },
    "covered_no_drifts": {
        "mirror": 0.0,
        "diamagnetic": 0.0,
        "end_damping": 0.0,
        "apar": 0.0,
        "bpar": 0.0,
        "curvature": 0.0,
        "gradb": 0.0,
    },
}
selected = sys.argv[1:] or list(ABLATIONS)
MODES = ("none", "damping", "hermite-line", "field-corrected")
KMAX = 200


def gmres_history(op, rhs, kmax):
    hist = []
    y, info = spla.gmres(
        op,
        rhs,
        rtol=1e-12,
        atol=0.0,
        restart=kmax,
        maxiter=1,
        callback=lambda r: hist.append(float(r)),
        callback_type="pr_norm",
    )
    return y, np.asarray(hist)


def first_below(hist, tol):
    idx = np.flatnonzero(hist <= tol)
    return int(idx[0]) + 1 if idx.size else None


for name in selected:
    over = ABLATIONS[name]
    terms = TermConfig(
        **{f: float(over.get(f, getattr(base_terms, f))) for f in FIELDS}
    )
    t1 = time.perf_counter()

    def apply(state, _terms=terms):
        return _apply_operator(state, cache, ctx.params, _terms)

    A = sparse_operator_matrix(apply, seed, batch_size=64, drop_tolerance=0.0).astype(
        np.complex128
    )
    S = (A - sigma * sp.eye(n, format="csr", dtype=A.dtype)).tocsc()
    t_asm = time.perf_counter() - t1
    t2 = time.perf_counter()
    lu = spla.splu(S)
    t_lu = time.perf_counter() - t2
    x_exact = lu.solve(b)
    exact_res = np.linalg.norm(b - S @ x_exact) / np.linalg.norm(b)
    opinv = spla.LinearOperator((n, n), matvec=lu.solve, dtype=np.complex128)
    vals, vecs = spla.eigs(A, k=6, sigma=sigma, OPinv=opinv, tol=1e-12, maxiter=20000)
    order = np.argsort(np.abs(vals - sigma))
    vals, vecs = vals[order], vecs[:, order]
    eres = [
        np.linalg.norm(A @ vecs[:, i] - vals[i] * vecs[:, i])
        / np.linalg.norm(vecs[:, i])
        for i in range(len(vals))
    ]
    # Spectral scale for context: largest |eig(A)|.
    big = spla.eigs(
        A, k=3, which="LM", return_eigenvectors=False, tol=1e-6, maxiter=20000
    )
    smin = (
        1.0
        / spla.svds(
            spla.LinearOperator(
                (n, n),
                matvec=lu.solve,
                rmatvec=lambda v: lu.solve(v, trans="H"),
                dtype=np.complex128,
            ),
            k=1,
            which="LM",
            return_singular_vectors=False,
            tol=1e-6,
        )[0]
    )
    print(
        f"\n== {name} overrides={over} nnz={A.nnz} asm_s={t_asm:.2f} lu_s={t_lu:.3f} "
        f"lu_nnz={lu.L.nnz + lu.U.nnz} exact_solve_rel_res={exact_res:.2e}",
        flush=True,
    )
    print(
        "   nearest eigs:",
        "; ".join(
            f"{v.real:+.6f}{v.imag:+.6f}j (res {r:.1e})" for v, r in zip(vals, eres)
        ),
        flush=True,
    )
    print(
        f"   |lambda0-sigma|={abs(vals[0] - sigma):.4e} |lambda1-sigma|={abs(vals[1] - sigma):.4e} "
        f"max|eig A|~{np.max(np.abs(big)):.3e} sigma_min(A-sI)={smin:.3e}",
        flush=True,
    )
    for mode in MODES:
        t3 = time.perf_counter()
        if mode == "none":
            minv = None
        else:
            _, pop = build_shift_invert_preconditioner(
                seed, cache, ctx.params, terms, jnp.asarray(sigma, seed.dtype), mode
            )
            minv = jax.jit(pop)
        mv = (
            (lambda v: np.asarray(v))
            if minv is None
            else (lambda v: np.asarray(minv(jnp.asarray(v, jnp.complex128))))
        )
        B = spla.LinearOperator((n, n), matvec=lambda v: S @ mv(v), dtype=np.complex128)
        y, hist = gmres_history(B, b, KMAX)
        x = mv(y)
        true_res = np.linalg.norm(b - S @ x) / np.linalg.norm(b)
        at = {
            k: (float(hist[k - 1]) if len(hist) >= k else None)
            for k in (20, 60, 120, 200)
        }
        line = (
            f"   [{mode:15s}] GMRES({KMAX}) rel_res@20/60/120/200="
            + "/".join(
                "—" if at[k] is None else f"{at[k]:.2e}" for k in (20, 60, 120, 200)
            )
            + f" its_to_1e-5={first_below(hist, 1e-5)} its_to_1e-8={first_below(hist, 1e-8)}"
            + f" final_true={true_res:.2e}"
        )
        if minv is not None:
            # Extreme eigenvalues of B: largest |z| directly; smallest |z| as 1/largest of M S^{-1}.
            try:
                zl = spla.eigs(
                    B,
                    k=4,
                    which="LM",
                    return_eigenvectors=False,
                    tol=1e-6,
                    maxiter=5000,
                )
                C = spla.LinearOperator(
                    (n, n), matvec=lambda v: mv(lu.solve(v)), dtype=np.complex128
                )
                zs = 1.0 / spla.eigs(
                    C,
                    k=4,
                    which="LM",
                    return_eigenvectors=False,
                    tol=1e-6,
                    maxiter=5000,
                )
                line += (
                    f" | max|z|={np.max(np.abs(zl)):.3e} min|z|={np.min(np.abs(zs)):.3e}"
                    f" smallest z={', '.join(f'{z.real:+.2e}{z.imag:+.2e}j' for z in zs[np.argsort(np.abs(zs))][:2])}"
                )
            except Exception as exc:  # ARPACK non-convergence is itself informative
                line += f" | eigs failed: {type(exc).__name__}"
        print(line + f" t={time.perf_counter() - t3:.1f}s", flush=True)

print("\ntotal_s", round(time.perf_counter() - t0, 1), flush=True)
