# ruff: noqa: E402
"""SOLVAX-DIRECT step 3: sparse direct against ``pr3-cm`` on GKX's shifted operator.

One fresh single-threaded process per case (``--case``), x64, CPU. For the shift
GKX's own shift-invert route would use (``_shift_seed`` with the Q28 options),
it records

``assembly``  compressed assembly of ``A`` (products, compile, recovery) and its
              check against the matrix-free operator;
``direct``    per backend/ordering: analysis, factorization, 1-RHS solves with
              ``trans`` N/T/H, a 16-RHS solve, factor entries and memory, and
              the relative residual and normwise backward error of each solve;
``pr3``       ``pr3-cm`` setup, then right-preconditioned FGMRES solves of the
              same system at rtol 1e-6 and 1e-10 (iterations, warm wall time,
              true relative residual against the assembled matrix);
``eigen``     the certified eigenpair by shift-invert Arnoldi (ARPACK) on the
              MUMPS factor, the left eigenvector from conjugate-transpose solves
              on the same factor, and both residuals; the right pair is
              certified with GKX's own ``_eigenpair_relative_residual`` gate.

Matvec-equivalents use Q21/Q28's definition: wall seconds over the median
single-operator time ``t_mv`` measured in the same process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

T0_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import operators as op

MUMPS_ORDERINGS = {"amd": 0, "scotch": 3, "pord": 4, "metis": 5}


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(
        f"[{time.perf_counter() - T0_PROCESS:8.1f}s rss<={rss:.2f}G] {msg}",
        file=sys.stderr,
        flush=True,
    )


def median_seconds(fn, *args, reps: int = 7) -> float:
    jax.block_until_ready(fn(*args))
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append(time.perf_counter() - t)
    return float(np.median(out))


def environment(argv) -> dict:
    root = HERE.parents[2]
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    import scipy
    import solvax

    try:
        import mumps  # noqa: F401

        mumps_ok = True
    except ImportError:
        mumps_ok = False
    return {
        "argv": argv,
        "host": platform.node(),
        "machine": platform.machine(),
        "loadavg": list(os.getloadavg()),
        "sha": sha,
        "dirty_src": bool(dirty),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "solvax": solvax.__version__,
        "mumps_binding": mumps_ok,
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
        },
    }


class Mumps:
    """Thin timed PyMUMPS wrapper: explicit ordering, multi-RHS, N/T/H solves."""

    def __init__(self, matrix, ordering: str, memory_limit_mb: int):
        import mumps
        from mpi4py import MPI

        self.n = matrix.shape[0]
        self.ctx = mumps.ZMumpsContext(par=1, sym=0, comm=MPI.COMM_SELF)
        self.ctx.set_silent()
        coo = matrix.tocoo()
        self._keep = coo
        self.ctx.set_centralized_sparse(coo)
        self.ctx.set_icntl(7, MUMPS_ORDERINGS[ordering])
        t = time.perf_counter()
        self.ctx.run(job=1)
        self.analysis_s = time.perf_counter() - t
        self.estimated_mb = int(self.ctx.get_infog(16))
        self.ordering_used = int(self.ctx.get_infog(7))
        self.factored = False
        if self.estimated_mb * 1.2 > memory_limit_mb:
            return
        self.ctx.set_icntl(23, memory_limit_mb)
        t = time.perf_counter()
        self.ctx.run(job=2)
        self.factor_s = time.perf_counter() - t
        self.factored = True
        self.factor_entries = int(self.ctx.get_infog(29))
        self.effective_mb = int(self.ctx.get_infog(21))

    def solve(self, b: np.ndarray, trans: str = "N") -> np.ndarray:
        x = np.asfortranarray(
            np.conj(b) if trans == "H" else b, dtype=np.complex128
        ).copy(order="F")
        k = 1 if x.ndim == 1 else x.shape[1]
        self.ctx.set_icntl(9, 1 if trans == "N" else 0)
        self.ctx._refs.update(rhs=x)
        self.ctx.id.nrhs = k
        self.ctx.id.lrhs = self.n
        self.ctx.id.rhs = self.ctx.cast_array(x)
        self.ctx.run(job=3)
        return np.conj(x) if trans == "H" else x

    def close(self):
        self.ctx.destroy()


class SuperLU:
    def __init__(self, matrix, ordering: str):
        csc = matrix.tocsc()
        t = time.perf_counter()
        self.lu = spla.splu(csc, permc_spec=ordering)
        self.factor_s = time.perf_counter() - t
        self.analysis_s = 0.0
        self.factored = True
        self.factor_entries = int(self.lu.L.nnz + self.lu.U.nnz)
        self.effective_mb = int(self.factor_entries * 20 / 1e6)
        self.estimated_mb = None
        self.ordering_used = ordering

    def solve(self, b, trans="N"):
        return self.lu.solve(np.asarray(b, dtype=np.complex128), trans=trans)

    def close(self):
        self.lu = None


def residuals(B, x, b, trans) -> dict:
    M = {"N": B, "T": B.T, "H": B.conj().T}[trans]
    r = b - M @ x
    norm_inf = abs(B).sum(axis=1).max() if trans == "N" else abs(B).sum(axis=0).max()
    rel = float(np.linalg.norm(r) / np.linalg.norm(b))
    bwd = float(np.abs(r).max() / (float(norm_inf) * np.abs(x).max() + np.abs(b).max()))
    return {"relres": rel, "backward": bwd}


def time_solves(fac, B, rng) -> dict:
    n = B.shape[0]
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    out = {}
    for trans in ("N", "T", "H"):
        fac.solve(b, trans)
        t = time.perf_counter()
        x = fac.solve(b, trans)
        out[f"solve_{trans}_s"] = time.perf_counter() - t
        out[f"res_{trans}"] = residuals(B, np.asarray(x).reshape(-1), b, trans)
    b16 = rng.standard_normal((n, 16)) + 1j * rng.standard_normal((n, 16))
    t = time.perf_counter()
    x16 = np.asarray(fac.solve(b16, "N"))
    out["solve_16rhs_s"] = time.perf_counter() - t
    out["res_16rhs_max"] = float(
        max(
            np.linalg.norm(b16[:, j] - B @ x16[:, j]) / np.linalg.norm(b16[:, j])
            for j in range(16)
        )
    )
    return out


def shift_and_seed(case):
    import gkx.solvers_linear_krylov as klv
    from gkx.solvers_linear_krylov import _shift_seed

    options = dict(
        method="shift_invert",
        shift=None,
        shift_preconditioner="pr3-cm",
        shift_precond_block_solve="block-thomas",
        shift_maxiter=600,
        shift_restart=600,
        shift_tol=1e-6,
        krylov_dim=48,
        restarts=2,
    )
    cfg = klv._normalized_config(dict(klv.KrylovConfig(**options).__dict__))
    seed = case.seed
    sigma, v_init = _shift_seed(
        seed, seed, case.cache, case.params, case.terms, cfg, None, select_overlap=False
    )
    return complex(np.asarray(sigma)), v_init


def run_pr3(case, sigma, v_init, B, rec, args):
    from gkx.solvers_linear_krylov_algorithms import _shift_invert_apply_factory
    from gkx.solvers_linear_precond_pr3 import build_pr3_factors

    t = time.perf_counter()
    factors, meta = build_pr3_factors(
        v_init, case.cache, case.params, case.terms, sigma, block_solve="block-thomas"
    )
    jax.block_until_ready(factors)
    rec["pr3_setup_s"] = time.perf_counter() - t
    rec["pr3_meta"] = {
        k: v
        for k, v in meta.items()
        if k in ("block_solve", "block_solve_reason", "factor_bytes", "alpha", "s1")
    }
    rng = np.random.default_rng(1)
    b = rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n)
    rows = {}
    for rtol in args.pr3_rtols:
        fn = _shift_invert_apply_factory(
            v_init,
            case.cache,
            case.params,
            case.terms,
            sigma_val=jnp.asarray(sigma, dtype=jnp.complex128),
            gmres_tol=rtol,
            gmres_maxiter=args.pr3_maxiter,
            gmres_restart=args.pr3_restart,
            shift_preconditioner="pr3-cm",
            precond_factors=factors,
        )
        solve = jax.jit(lambda x: fn(x, None, None, None))
        bj = jnp.asarray(b.reshape(case.shape))
        t = time.perf_counter()
        jax.block_until_ready(solve(bj))
        first = time.perf_counter() - t
        t = time.perf_counter()
        x, stats = jax.block_until_ready(solve(bj))
        warm = time.perf_counter() - t
        x = np.asarray(x).reshape(-1)
        rows[f"{rtol:g}"] = {
            "first_call_s": first,
            "warm_s": warm,
            "iterations": int(stats.total_iterations),
            "converged": int(stats.unconverged_solves) == 0,
            "reported_relres": float(stats.max_relative_residual),
            "true_relres": float(np.linalg.norm(b - B @ x) / np.linalg.norm(b)),
        }
        log(f"pr3 rtol {rtol:g}: {rows[f'{rtol:g}']}")
    rec["pr3"] = rows


def run_eigen(case, sigma, fac, A, rec, args):
    from gkx.solvers_linear_krylov import (
        _eigenpair_relative_residual,
        certifiable_residual_tolerance,
    )

    n = case.n
    v0 = np.asarray(case.seed).reshape(-1)
    counts = {"N": 0, "H": 0}

    def opinv(trans):
        def mv(v):
            counts[trans] += 1
            return np.asarray(fac.solve(np.asarray(v).reshape(-1), trans)).reshape(-1)

        return spla.LinearOperator((n, n), matvec=mv, dtype=np.complex128)

    t = time.perf_counter()
    ncv = min(n - 1, 2 * args.nev + 8)
    vals, vecs = spla.eigs(
        A,
        k=args.nev,
        ncv=ncv,
        sigma=sigma,
        OPinv=opinv("N"),
        v0=v0,
        tol=args.eig_tol,
        which="LM",
    )
    right_s = time.perf_counter() - t
    # GKX's shift-invert route keeps the largest growth rate among its Ritz
    # values; this does the same over the nev eigenvalues nearest sigma.
    j = int(np.argmax(vals.real))
    lam = complex(vals[j])
    x = vecs[:, j]
    gate = certifiable_residual_tolerance(1.0e-9, jnp.complex128)
    t = time.perf_counter()
    res = float(
        _eigenpair_relative_residual(
            jnp.asarray(lam),
            jnp.asarray(x.reshape(case.shape)),
            case.cache,
            case.params,
            case.terms,
        )
    )
    cert_s = time.perf_counter() - t
    # Left eigenvector from the same factor: A^H y = conj(lam) y, i.e. an
    # eigenvector of (A - sigma I)^{-H} with eigenvalue 1/conj(lam - sigma).
    t = time.perf_counter()
    # The factor is at sigma, so the shift ARPACK is told must be conj(sigma):
    # it maps the eigenvalues mu of B^{-H} back as conj(sigma) + 1/mu. Ask for
    # as many as the right solve found and pick the one paired with lam.
    lvals, lvecs = spla.eigs(
        A.conj().T,
        k=args.nev,
        ncv=ncv,
        sigma=np.conj(sigma),
        OPinv=opinv("H"),
        v0=np.conj(x),
        tol=args.eig_tol,
        which="LM",
    )
    left_s = time.perf_counter() - t
    jl = int(np.argmin(np.abs(np.conj(lvals) - lam)))
    y = lvecs[:, jl]
    left_res = float(
        np.linalg.norm(A.conj().T @ y - np.conj(lam) * y)
        / (np.linalg.norm(y) * abs(lam))
    )
    rec["eigen"] = {
        "lambda": [lam.real, lam.imag],
        "candidates": [[complex(v).real, complex(v).imag] for v in vals],
        "right_s": right_s,
        "left_s": left_s,
        "certify_s": cert_s,
        "solves_N": counts["N"],
        "solves_H": counts["H"],
        "certified_residual": res,
        "gate": gate,
        "certified": bool(res <= gate),
        "left_residual": left_res,
        "left_lambda_gap": float(abs(np.conj(lvals[jl]) - lam)),
        "yHx_over_norms": float(
            abs(np.vdot(y, x)) / (np.linalg.norm(x) * np.linalg.norm(y))
        ),
    }
    log(f"eigen {rec['eigen']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="d96")
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--mumps", nargs="*", default=["metis", "scotch", "amd", "pord"])
    p.add_argument("--superlu", nargs="*", default=["COLAMD", "MMD_AT_PLUS_A"])
    p.add_argument("--superlu-max-n", type=int, default=20000)
    p.add_argument("--memory-limit-mb", type=int, default=6000)
    p.add_argument("--pr3-rtols", type=float, nargs="*", default=[1e-6, 1e-10])
    p.add_argument("--pr3-maxiter", type=int, default=2400)
    p.add_argument("--pr3-restart", type=int, default=600)
    p.add_argument("--nev", type=int, default=16)
    p.add_argument("--eig-tol", type=float, default=1e-12)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--shift", type=complex, default=None)
    p.add_argument("--save-matrix", default=None, help="write B = A - sigma I (npz)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rec: dict = {
        "case": args.case,
        "dims": dict(zip(("Nz", "ntheta", "nperiod", "Nl", "Nm"), op.CASES[args.case])),
    }
    rec["environment"] = environment(sys.argv)
    case = op.Case(args.case, args.ky)
    rec |= {"n": case.n, "ky": case.ky}
    apply = jax.jit(case.apply_fn("full"))
    probe = jnp.asarray(np.random.default_rng(0).standard_normal(case.n) + 0j)
    t_mv = median_seconds(apply, probe)
    rec["t_matvec_s"] = t_mv
    log(f"case {args.case} n={case.n} t_mv={t_mv * 1e3:.3f} ms")

    t = time.perf_counter()
    A, st = op.assemble(case, "full")
    rec["assembly"] = st | {"wall_s": time.perf_counter() - t}
    rec["structure"] = op.structure(A)
    log(f"assembly {rec['assembly']} structure {rec['structure']}")

    t = time.perf_counter()
    sigma, v_init = shift_and_seed(case)
    rec["sigma_seed"] = [sigma.real, sigma.imag]
    if args.shift is not None:
        # A warm-start shift near the ITG branch: GKX's seed shift can land in
        # the marginal spectrum near the origin on the larger rungs, where
        # neither route is being asked the question an eigen solve asks.
        sigma = complex(args.shift)
    rec["sigma"] = [sigma.real, sigma.imag]
    rec["shift_seed_s"] = time.perf_counter() - t
    B = (A - sigma * sp.identity(case.n, dtype=np.complex128, format="csr")).tocsr()
    log(f"sigma {sigma}")
    if args.save_matrix:
        sp.save_npz(args.save_matrix, B, compressed=False)

    rng = np.random.default_rng(2)
    direct = {}
    keep = None
    for ordering in args.mumps:
        t = time.perf_counter()
        try:
            fac = Mumps(B, ordering, args.memory_limit_mb)
        except Exception as exc:  # recorded, not fatal
            direct[f"mumps-{ordering}"] = {"raised": f"{type(exc).__name__}: {exc}"}
            continue
        row = {
            "analysis_s": fac.analysis_s,
            "estimated_mb": fac.estimated_mb,
            "ordering_used": fac.ordering_used,
            "factored": fac.factored,
        }
        if fac.factored:
            row |= {
                "factor_s": fac.factor_s,
                "factor_entries": fac.factor_entries,
                "fill": fac.factor_entries / B.nnz,
                "effective_mb": fac.effective_mb,
            }
            row |= time_solves(fac, B, rng)
        direct[f"mumps-{ordering}"] = row
        log(f"mumps-{ordering} {row}")
        if ordering == args.mumps[0] and fac.factored:
            keep = fac
        else:
            fac.close()
    if case.n <= args.superlu_max_n:
        for ordering in args.superlu:
            fac = SuperLU(B, ordering)
            row = {
                "factor_s": fac.factor_s,
                "factor_entries": fac.factor_entries,
                "fill": fac.factor_entries / B.nnz,
                "effective_mb": fac.effective_mb,
            }
            row |= time_solves(fac, B, rng)
            direct[f"superlu-{ordering}"] = row
            log(f"superlu-{ordering} {row}")
            fac.close()
    rec["direct"] = direct

    if keep is not None and "eigen" not in args.skip:
        run_eigen(case, sigma, keep, A, rec, args)
    if "pr3" not in args.skip:
        run_pr3(case, sigma, v_init, B, rec, args)
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    rec["process_s"] = time.perf_counter() - T0_PROCESS
    text = json.dumps(rec, default=float)
    print("RESULT " + text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
