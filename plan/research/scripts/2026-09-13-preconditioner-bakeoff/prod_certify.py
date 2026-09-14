# ruff: noqa: E402
"""Q7: cold time-to-certified-pair, matrix-free shift-invert at the production chain.

One fresh process. Linked Cyclone deck, Nx1, Nz96 (ntheta32, nperiod2), Nl16, Nm48,
ky=+.3, n=73728, complex128, single-threaded. The shift is the plan reference
eigenvalue (.09302951-.28199404j, the registered pilot shift) plus 0.05 in growth.

Setup: z-mean drift and Dc blocks by Nl*Nm probes each, symbol bounds, one dense
inverse of the z-local blocks (passed to the jitted inner solve as an argument).
Outer: shift-invert Arnoldi from the runtime seed; each step one SOLVAX
gcrot(m=20, k=10, "harmonic") solve of (A - sigma I) x = v_j, right-preconditioned by
two-parameter Peaceman-Rachford sweeps (``pr_kz.pr_two``), recycle space carried across
steps. Every step's Ritz pair nearest sigma is checked with GKX's
``_eigenpair_relative_residual`` on the matrix-free operator; the run records the
time to <= 1e-6 (shift-invert outer gate) and to <= ``--target`` and stops there, on
``--max-steps`` or at the caller's wall cap. Output: progress lines and ``RESULT {json}``.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bakeoff as bk
import jax
import jax.numpy as jnp
import numpy as np
import pr_kz as pk
import prod_solve as ps
import solvax

from gkx.solvers_linear_krylov import _eigenpair_relative_residual

REFERENCE = complex(0.09302951, -0.28199404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", default="kz-geo-best")
    parser.add_argument("--alpha-best", type=float, default=-10.0)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--inner-rtol", type=float, default=1e-9)
    parser.add_argument("--max-restarts", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--target", type=float, default=1e-9)
    args = parser.parse_args()
    env = {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "solvax": solvax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "loadavg": os.getloadavg(),
        "argv": sys.argv[1:],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pr_kz_sha256": hashlib.sha256((HERE / "pr_kz.py").read_bytes()).hexdigest(),
        "bakeoff_sha256": hashlib.sha256(
            (HERE / "bakeoff.py").read_bytes()
        ).hexdigest(),
    } | bk.git_state()
    print("ENV " + json.dumps(env), flush=True)
    assert env["x64"], "complex128 required"

    def since_start():
        return time.perf_counter() - T_PROCESS

    case = bk.Case("prod", None)
    sigma = complex(REFERENCE.real + bk.GROWTH_OFFSET, REFERENCE.imag)
    shape = case.shape
    rec: dict = {"n": case.n, "ky": case.ky, "sigma": [sigma.real, sigma.imag]}
    rec["t_context_s"] = since_start()
    bk.log(f"production chain n={case.n} ky={case.ky:+.3f} sigma={sigma:.6f}")

    t = time.perf_counter()
    mean = ps.z_mean_drift(case)
    blocks = bk.zblocks_by_probing(case.operator(case.T_DC, True), shape, batch=16)
    idx = np.arange(case.bs)
    blocks[:, idx, idx] -= mean
    line = pk.KzLine(case)
    d_rho = pk.spectral_radius(blocks)
    a_kz, b = pk.rule_parameters(args.rule, line, d_rho, args.alpha_best)
    inv, t_inv, mem = case.dinv(blocks, sigma / 2 - b)
    del blocks, mean
    rec["setup"] = {
        "probe_symbol_s": time.perf_counter() - t - t_inv,
        "inverse_s": t_inv,
        "inverse_bytes": mem,
        "d_rho": d_rho,
        "s1": line.s1,
        "s_max": float(line.s_kz.max()),
        "a_range": [float(a_kz.max()), float(a_kz.min())],
        "b": float(b),
    }
    bk.log(f"setup {rec['setup']}")

    mv = case._mv
    a_j = jnp.asarray(a_kz, dtype=jnp.complex128)

    def solve(inverse, rhs, recycle):
        def dsolve(z):
            return bk.from_blocks(
                jnp.einsum("bij,bj->bi", inverse, bk.to_blocks(z, shape)), shape
            )

        precond = pk.pr_two(line, sigma, a_j, b, dsolve, args.sweeps)
        return solvax.gcrot(
            lambda v: mv(v) - sigma * v,
            rhs,
            precond=precond,
            m=20,
            k=10,
            rtol=args.inner_rtol,
            max_restarts=args.max_restarts,
            recycle=recycle,
            recycle_strategy="harmonic",
        )

    solve_first = jax.jit(lambda inverse, rhs: solve(inverse, rhs, None))
    solve_next = jax.jit(solve)

    kmax = args.max_steps
    V = np.zeros((case.n, kmax + 1), dtype=np.complex128)
    H = np.zeros((kmax + 1, kmax), dtype=np.complex128)
    V[:, 0] = case.b / np.linalg.norm(case.b)
    steps = []
    recycle = None
    hits: dict = {}
    t_outer = time.perf_counter()
    for j in range(kmax):
        t = time.perf_counter()
        rhs = jnp.asarray(V[:, j])
        sol = (
            solve_first(inv, rhs) if recycle is None else solve_next(inv, rhs, recycle)
        )
        w = np.asarray(sol.x)
        recycle = sol.recycle
        true_inner = float(
            np.linalg.norm(V[:, j] - (np.asarray(mv(jnp.asarray(w))) - sigma * w))
            / np.linalg.norm(V[:, j])
        )
        inner_s = time.perf_counter() - t
        for _ in range(2):
            h = V[:, : j + 1].conj().T @ w
            w = w - V[:, : j + 1] @ h
            H[: j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        V[:, j + 1] = w / H[j + 1, j]
        theta, Y = np.linalg.eig(H[: j + 1, : j + 1])
        i = int(np.argmax(np.abs(theta)))
        lam = sigma + 1.0 / theta[i]
        u = (V[:, : j + 1] @ Y[:, i]).reshape(shape)
        res = _eigenpair_relative_residual(
            jnp.asarray(lam), jnp.asarray(u), case.cache, case.params, case.T0
        )
        step = {
            "step": j + 1,
            "inner_its": int(sol.iterations),
            "inner_converged": bool(sol.converged),
            "inner_true_rel": true_inner,
            "inner_s": inner_s,
            "lambda": [lam.real, lam.imag],
            "residual": res,
            "since_process_start_s": since_start(),
        }
        steps.append(step)
        bk.log(f"step {step}")
        for gate in (1e-6, args.target):
            if res <= gate and str(gate) not in hits:
                hits[str(gate)] = {
                    "step": j + 1,
                    "since_process_start_s": since_start(),
                    "total_inner_its": sum(s["inner_its"] for s in steps),
                    "lambda": [lam.real, lam.imag],
                    "residual": res,
                }
        if res <= args.target:
            break
    rec["steps"] = steps
    rec["gates"] = hits
    rec["outer_s"] = time.perf_counter() - t_outer
    rec["gamma_omega"] = [steps[-1]["lambda"][0], -steps[-1]["lambda"][1]]
    rec["wall_s"] = since_start()
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    bk.log(f"gates {hits} wall {rec['wall_s']:.1f}s")
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
