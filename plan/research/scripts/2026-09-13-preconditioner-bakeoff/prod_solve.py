# ruff: noqa: E402
"""Q7: matrix-free production-chain check of the structured candidates (host prototype).

Production linked Cyclone chain (Nz96 ntheta32 nperiod2, Nl16 Nm48, Nx1, ky=+.3,
n=73728). Nothing is assembled: z-local blocks come from Nl*Nm probes, streaming is
GKX's spectral Hermite-line solve, and GMRES (SciPy, unrestarted, right-preconditioned)
runs on GKX's matrix-free operator. The shift is #232's adaptive-route certified
eigenvalue plus 0.05 in growth, because the exact route cannot factor this size.

Part A: one RHS (the normalized runtime seed), iterations to 1e-5 / 1e-8 for
hermite-line and adi/pr2/pr3 with the balanced split (``-cm``) over alpha in {-1,-3,-10}.
Part B: shift-invert Arnoldi from the seed with the candidate of least measured work in
Part A; each step one preconditioned inner solve to ``--inner-rtol``; the Ritz pair is
certified with GKX's scale-invariant residual on the matrix-free operator.
Output: progress lines and one ``RESULT {json}``.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bakeoff as bk
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

REF_LAMBDA = complex(0.0930912, -0.2820327)  # #232 adaptive route, same deck and size


def z_mean_drift(case) -> np.ndarray:
    """(nb, bs) z-mean of the drift diagonal on linked rows, as the line solve uses it."""

    ns, nl, nm, ny, nx, nz = case.shape
    blocks = bk.zblocks_by_probing(
        case.operator(bk.with_terms(case.T_DB, end_damping=0), False),
        case.shape,
        batch=16,
    )
    diag = np.diagonal(blocks, axis1=1, axis2=2).reshape(ns, ny, nx, nz, -1).copy()
    del blocks
    mean = np.broadcast_to(diag.mean(axis=3, keepdims=True), diag.shape).copy()
    kx_cov = case.cov.reshape(case.shape)[0, 0, 0, 0, :, 0]
    mean[:, :, ~kx_cov] = 0.0
    return mean.reshape(case.nb, case.bs)


def gmres_solve(bmv_host, minv_host, rhs, rtol, cap):
    hist: list[float] = []
    op = spla.LinearOperator(
        (rhs.size, rhs.size),
        matvec=lambda v: bmv_host(minv_host(v)),
        dtype=np.complex128,
    )
    t = time.perf_counter()
    y, _ = spla.gmres(
        op,
        rhs,
        rtol=rtol,
        atol=0.0,
        restart=cap,
        maxiter=1,
        callback=hist.append,
        callback_type="pr_norm",
    )
    x = minv_host(y)
    seconds = time.perf_counter() - t
    true = float(np.linalg.norm(rhs - bmv_host(x)) / np.linalg.norm(rhs))
    h = np.asarray(hist, dtype=float)

    def first(tol):
        idx = np.flatnonzero(h <= tol)
        return int(idx[0]) + 1 if idx.size else None

    return x, {
        "its": int(h.size),
        "its_1e5": first(1e-5),
        "its_1e8": first(1e-8),
        "true": true,
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="-1,-3,-10")
    parser.add_argument("--cap", type=int, default=bk.CAP)
    parser.add_argument("--inner-rtol", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--certify", type=float, default=1e-8)
    parser.add_argument("--candidate", default=None, help="skip Part A choice")
    parser.add_argument("--skip-part-a", action="store_true")
    args = parser.parse_args()
    t_start = time.perf_counter()
    env = {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "loadavg": os.getloadavg(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bakeoff_sha256": hashlib.sha256(
            (HERE / "bakeoff.py").read_bytes()
        ).hexdigest(),
    } | bk.git_state()
    print("ENV " + json.dumps(env), flush=True)
    assert env["x64"], "complex128 required"

    case = bk.Case("prod", None)
    sigma = complex(REF_LAMBDA.real + bk.GROWTH_OFFSET, REF_LAMBDA.imag)
    rec: dict = {
        "n": case.n,
        "shape": case.shape,
        "ky": case.ky,
        "sigma": [sigma.real, sigma.imag],
    }
    bk.log(f"production chain n={case.n} ky={case.ky:+.3f} sigma={sigma:.6f}")

    def bmv_host(v):
        return np.asarray(case._mv(jnp.asarray(v, jnp.complex128))) - sigma * v

    def host(fn):
        return lambda v: np.asarray(fn(jnp.asarray(v, jnp.complex128)))

    t = time.perf_counter()
    mean = z_mean_drift(case)
    blocks = bk.zblocks_by_probing(case.operator(case.T_DC, True), case.shape, batch=16)
    idx = np.arange(case.bs)
    blocks[:, idx, idx] -= mean
    rec["probe_setup_s"] = time.perf_counter() - t
    bk.log(
        f"z-local blocks (Dc minus z-mean drift) by 2x{case.bs} probes: {rec['probe_setup_s']:.1f}s"
    )

    def build(kind, alpha):
        s1 = sigma / 2 - alpha
        sinv = case.sinv(s1, mean_drift=True)
        inv, t_inv, mem = case.dinv(blocks, s1)
        sweeps = {"adi-cm": 1, "pr2-cm": 2, "pr3-cm": 3}[kind]
        if sweeps == 1:
            fn = case.compose(lambda d: bk.adi(sinv, d, alpha), (inv,))
        else:
            fn = case.compose(
                lambda d: bk.peaceman_rachford(sinv, d, alpha, sweeps), (inv,)
            )
        return fn, t_inv, mem

    rng = np.random.default_rng(0)
    x_rand = jnp.asarray(rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n))
    rec["t_matvec_s"] = t_mv = bk.median_seconds(case._mv, x_rand, reps=9)
    part_a: dict = {}
    if not args.skip_part_a:
        hl = case.hl(sigma)
        rows = [("hl", None)] + [
            (kind, float(a))
            for kind in ("adi-cm", "pr2-cm", "pr3-cm")
            for a in args.alphas.split(",")
        ]
        for kind, alpha in rows:
            if kind == "hl":
                fn, t_inv, mem = hl, 0.0, 0
            else:
                fn, t_inv, mem = build(kind, alpha)
            t_apply = bk.median_seconds(fn, x_rand, reps=5)
            _, stats = gmres_solve(bmv_host, host(fn), case.b, 1e-9, args.cap)
            its = stats["its_1e8"]
            stats |= {
                "alpha": alpha,
                "apply_s": t_apply,
                "apply_over_matvec": t_apply / t_mv,
                "inverse_s": t_inv,
                "inverse_bytes": mem,
                "work_1e8_matvec_eq": None
                if its is None
                else its * (1 + t_apply / t_mv),
            }
            part_a[f"{kind}[{alpha}]"] = stats
            bk.log(f"A {kind}[{alpha}] {stats}")
            del fn
        rec["part_a"] = part_a

    if args.candidate:
        kind, alpha = args.candidate.split("@")
        alpha = float(alpha)
    else:
        # least work among rows that reach 1e-8; if none does, smallest final residual
        scored = [
            (v["work_1e8_matvec_eq"] is None, v["work_1e8_matvec_eq"] or v["true"], k)
            for k, v in part_a.items()
            if not k.startswith("hl")
        ]
        best = min(scored)[2]
        kind, alpha = best.split("[")[0], float(best.split("[")[1].rstrip("]"))
    t = time.perf_counter()
    fn, t_inv, mem = build(kind, alpha)
    minv = host(fn)
    minv(np.asarray(x_rand))  # compile
    setup_b = time.perf_counter() - t
    rec["part_b"] = {
        "candidate": kind,
        "alpha": alpha,
        "setup_s": setup_b,
        "inverse_bytes": mem,
    }
    bk.log(f"B shift-invert Arnoldi with {kind}[{alpha}] (setup {setup_b:.1f}s)")

    kmax = args.max_steps
    V = np.zeros((case.n, kmax + 1), dtype=np.complex128)
    H = np.zeros((kmax + 1, kmax), dtype=np.complex128)
    V[:, 0] = case.b / np.linalg.norm(case.b)
    steps = []
    t_arnoldi = time.perf_counter()
    certified = None
    for j in range(kmax):
        w, stats = gmres_solve(bmv_host, minv, V[:, j], args.inner_rtol, args.cap)
        for _ in range(2):
            h = V[:, : j + 1].conj().T @ w
            w = w - V[:, : j + 1] @ h
            H[: j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        V[:, j + 1] = w / H[j + 1, j]
        theta, Y = np.linalg.eig(H[: j + 1, : j + 1])
        i = int(np.argmax(np.abs(theta)))
        lam = sigma + 1.0 / theta[i]
        u = V[:, : j + 1] @ Y[:, i]
        Au = np.asarray(case._mv(jnp.asarray(u)))
        res = float(
            np.linalg.norm(Au - lam * u)
            / max(np.linalg.norm(Au), abs(lam) * np.linalg.norm(u), 1e-30)
        )
        steps.append(
            stats
            | {
                "step": j + 1,
                "lambda": [lam.real, lam.imag],
                "residual": res,
                "elapsed_s": time.perf_counter() - t_arnoldi,
            }
        )
        bk.log(f"B step {j + 1}: inner {stats} lambda {lam:.7f} residual {res:.2e}")
        if res <= args.certify:
            certified = j + 1
            break
    arnoldi_s = time.perf_counter() - t_arnoldi
    total_inner = sum(s["its"] for s in steps)
    rec["part_b"] |= {
        "steps": steps,
        "certified_at_step": certified,
        "arnoldi_s": arnoldi_s,
        "time_to_certified_pair_s": rec["probe_setup_s"] + setup_b + arnoldi_s,
        "total_inner_iterations": total_inner,
        "gamma_omega": [steps[-1]["lambda"][0], -steps[-1]["lambda"][1]],
    }
    bk.log(
        f"B certified at step {certified}: arnoldi {arnoldi_s:.1f}s, inner its {total_inner}, "
        f"time to certified pair (probes + inverse + compile + Arnoldi) "
        f"{rec['part_b']['time_to_certified_pair_s']:.1f}s"
    )
    rec["wall_s"] = time.perf_counter() - t_start
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
