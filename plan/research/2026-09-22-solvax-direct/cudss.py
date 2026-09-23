"""cuDSS (through nvmath-python) on a saved B = A - sigma I: GPU factor and solve.

Usage: CUDA_VISIBLE_DEVICES=<free> python cudss.py <B.npz> [--rhs 16]
Prints one RESULT line: analysis, factorization, 1-RHS and multi-RHS solve times
(synchronized, second call warm) and the relative residuals on the host.
"""

import argparse
import json
import time

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import scipy.sparse as sp
from nvmath.sparse.advanced import DirectSolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("matrix")
    p.add_argument("--rhs", type=int, default=16)
    args = p.parse_args()
    B = sp.load_npz(args.matrix).tocsr()
    n = B.shape[0]
    rng = np.random.default_rng(0)
    b1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    bk = rng.standard_normal((n, args.rhs)) + 1j * rng.standard_normal((n, args.rhs))
    Bg = csp.csr_matrix(B)
    rec = {
        "n": n,
        "nnz": int(B.nnz),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    for label, rhs in (("1rhs", b1), (f"{args.rhs}rhs", bk)):
        # nvmath expects column-major multi-RHS: (n, k) Fortran order.
        bg = cp.asarray(np.asfortranarray(rhs))
        times = {}
        with DirectSolver(Bg, bg) as solver:
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            solver.plan()
            cp.cuda.Device().synchronize()
            times["analysis_s"] = time.perf_counter() - t
            t = time.perf_counter()
            solver.factorize()
            cp.cuda.Device().synchronize()
            times["factor_s"] = time.perf_counter() - t
            for k in range(2):
                t = time.perf_counter()
                x = solver.solve()
                cp.cuda.Device().synchronize()
                times[f"solve_s_{k}"] = time.perf_counter() - t
        xh = cp.asnumpy(x).reshape(rhs.shape)
        r = rhs - B @ xh
        times["relres"] = float(np.linalg.norm(r) / np.linalg.norm(rhs))
        times["peak_device_bytes"] = int(cp.get_default_memory_pool().total_bytes())
        rec[label] = times
    print("RESULT " + json.dumps(rec))


if __name__ == "__main__":
    main()
