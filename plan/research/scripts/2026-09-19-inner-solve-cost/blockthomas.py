# ruff: noqa: E402
"""Q21 candidate 4: the block-Thomas apply of the z-local D block (plan §5.1 L5).

Q7 measured that ``pr3-cm``'s apply is three (streaming line solve + dense
z-local block solve) pairs, and that the dense solve dominates. It noted, as
arithmetic only, that the z-local block is tridiagonal in the Laguerre index
plus a rank-one field part per (kx, z), so a block-Thomas solve with a
Sherman-Morrison correction would cost about 4x fewer flops and about 5x less
memory. This script checks that structure, builds the factorization, verifies it
against the dense inverse and **measures** the apply, so the projection rests on
measured seconds rather than on a flop count.

Structure, verified here on every rung it is run at:

* ``Dc`` with the field term removed is exactly block-tridiagonal in the Laguerre
  index with ``Nm x Nm`` blocks (off-tridiagonal entries are exactly zero);
* the field term is exactly rank one per (kx, z), so ``Dc = T + u w^T``.

``M = T + u w^T - s`` is then solved by block-Thomas on the tridiagonal part
(store ``Dinv_l``, ``F_l = Dinv_l A_l`` and ``Q_l = Dinv_l C_l``; one forward and
one back substitution cost three ``Nm x Nm`` matvecs per Laguerre index instead
of the dense ``(Nl Nm)^2``) plus Sherman-Morrison,
``M^-1 x = T^-1 x - T^-1 u (w^T T^-1 x) / (1 + w^T T^-1 u)``.

This is an exact solve of the same block, not an approximation, so the
preconditioner and therefore every iteration count is unchanged: only the apply
cost moves. Output: progress lines and one ``RESULT {json}`` line.
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
REPO = HERE.parents[3]
BAKEOFF = REPO / "plan" / "research" / "scripts" / "2026-09-13-preconditioner-bakeoff"
sys.path.insert(0, str(BAKEOFF))

import bakeoff as bk
import jax
import jax.numpy as jnp
import numpy as np
import pr_kz as pk
import prod_solve as ps

import dblock as db
from q21 import ALPHA_BEST, SIGMA, environment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="r96", choices=sorted(bk.CASES))
    p.add_argument("--sigma", type=complex, default=None)
    p.add_argument("--alpha-best", type=float, default=None)
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--cost-reps", type=int, default=9)
    p.add_argument(
        "--inner-iterations",
        type=int,
        default=0,
        help="inner iterations of the arm to project onto, 0 to skip",
    )
    p.add_argument("--outer-steps", type=int, default=0)
    p.add_argument("--label", default="")
    args = p.parse_args()

    env = environment(sys.argv[1:])
    env["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("ENV " + json.dumps(env), flush=True)
    if not env["x64"]:
        raise SystemExit("complex128 required")

    case = bk.Case(args.case, None)
    sigma = args.sigma if args.sigma is not None else SIGMA[args.case]
    alpha = args.alpha_best if args.alpha_best is not None else ALPHA_BEST[args.case]
    ns, nl, nm, ny, nx, nz = case.shape
    rec: dict = {
        "case": args.case,
        "label": args.label,
        "env": env,
        "n": case.n,
        "nb": case.nb,
        "nl": nl,
        "nm": nm,
        "sigma": [sigma.real, sigma.imag],
    }
    bk.log(f"{args.case} n={case.n} nb={case.nb} nl={nl} nm={nm}")

    # The matvec is timed here, before the dense inverse and the block-Thomas
    # factors are resident: at the production chain those are 0.91 GB and
    # 0.17 GB, and timing the matvec after them makes it several times slower
    # through memory pressure alone.
    t_mv = bk.median_seconds(case._mv, jnp.asarray(case.b), reps=args.cost_reps)
    rec["t_matvec_s_clean"] = t_mv
    bk.log(f"matvec (clean process) {t_mv * 1e3:.3f} ms")

    t = time.perf_counter()
    mean = ps.z_mean_drift(case)
    with_phi = bk.zblocks_by_probing(
        case.operator(case.T_DC, True), case.shape, batch=16
    )
    no_phi = bk.zblocks_by_probing(
        case.operator(case.T_DC, False), case.shape, batch=16
    )
    idx = np.arange(case.bs)
    with_phi[:, idx, idx] -= mean
    no_phi[:, idx, idx] -= mean
    rec["probe_s"] = time.perf_counter() - t
    rec["structure"] = db.check_structure(no_phi, with_phi, nl, nm)
    bk.log(f"structure {rec['structure']}")

    line = pk.KzLine(case)
    d_rho = pk.spectral_radius(with_phi)
    a_kz, b = pk.rule_parameters("scalar-best", line, d_rho, alpha)
    shift = sigma / 2 - b

    # dense reference, exactly as the si arm builds it
    t = time.perf_counter()
    dense_inv, t_inv, mem_dense = case.dinv(with_phi, shift)
    bk.log(f"dense inverse {t_inv:.2f}s {mem_dense / 1024**2:.0f} MB")
    rec["dense"] = {"setup_s": time.perf_counter() - t, "bytes": mem_dense}

    # block-Thomas factors plus the Sherman-Morrison vectors
    t = time.perf_counter()
    factors, make_bt, bt_info = db.build_solver(with_phi, no_phi, nl, nm, shift)
    bt_setup_s = time.perf_counter() - t
    mem_bt = bt_info["bytes"]
    rec["block_thomas"] = {
        "setup_s": bt_setup_s,
        "bytes": mem_bt,
        "memory_ratio_dense_over_bt": mem_dense / mem_bt,
        "setup_ratio_dense_over_bt": rec["dense"]["setup_s"] / bt_setup_s,
        "denominator_min_abs": bt_info["denominator_min_abs"],
    }
    bk.log(f"block-Thomas {rec['block_thomas']}")

    # Both applies take their factors as a jit *argument*, never as a captured
    # constant: that is how q21.py hands them to gcrot (Q7's convention), and a
    # constant-captured dense inverse compared against an argument-passed
    # block-Thomas would flatter the block-Thomas side by construction.
    shape = case.shape
    bt_jit = jax.jit(
        lambda f, x: bk.from_blocks(make_bt(f)(bk.to_blocks(x, shape)), shape)
    )
    dense_jit = jax.jit(
        lambda inverse, x: bk.from_blocks(
            jnp.einsum("bij,bj->bi", inverse, bk.to_blocks(x, shape)), shape
        )
    )

    def bt_apply(x):
        return bt_jit(factors, x)

    def dense_apply(x):
        return dense_jit(dense_inv, x)

    rng = np.random.default_rng(21)
    probe = rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n)
    probe = jnp.asarray(probe / np.linalg.norm(probe))
    a_ref = np.asarray(dense_apply(probe))
    a_bt = np.asarray(bt_apply(probe))
    rec["agreement"] = {
        "rel_difference": float(np.linalg.norm(a_bt - a_ref) / np.linalg.norm(a_ref)),
        "ref_norm": float(np.linalg.norm(a_ref)),
    }
    bk.log(f"agreement {rec['agreement']}")

    # Alternate dense/BT/BT/dense so the ratio does not inherit a cache order.
    t_dense_a = bk.median_seconds(dense_apply, probe, reps=args.cost_reps)
    t_bt_a = bk.median_seconds(bt_apply, probe, reps=args.cost_reps)
    t_bt_b = bk.median_seconds(bt_apply, probe, reps=args.cost_reps)
    t_dense_b = bk.median_seconds(dense_apply, probe, reps=args.cost_reps)
    t_dense = float(np.median([t_dense_a, t_dense_b]))
    t_bt = float(np.median([t_bt_a, t_bt_b]))
    rec["apply_repeats_s"] = {
        "dense": [t_dense_a, t_dense_b],
        "block_thomas": [t_bt_a, t_bt_b],
        "ratio_first_pair": t_dense_a / t_bt_a,
        "ratio_second_pair": t_dense_b / t_bt_b,
    }
    t_mv_late = bk.median_seconds(case._mv, jnp.asarray(case.b), reps=args.cost_reps)
    rec["t_matvec_s_late"] = t_mv_late
    s1_kz = jnp.asarray(sigma / 2 - a_kz, dtype=jnp.complex128)
    line_apply = jax.jit(lambda z: line.solve(z, s1_kz))
    t_line = bk.median_seconds(line_apply, probe, reps=args.cost_reps)
    c_dense, c_bt, c_line = t_dense / t_mv, t_bt / t_mv, t_line / t_mv
    rec["cost"] = {
        "t_matvec_s": t_mv,
        "t_line_s": t_line,
        "t_dense_s": t_dense,
        "t_block_thomas_s": t_bt,
        "c_line": c_line,
        "c_dense": c_dense,
        "c_block_thomas": c_bt,
        "apply_speedup_dense_over_bt": t_dense / t_bt,
        "c_precond_dense": args.sweeps * (c_line + c_dense),
        "c_precond_block_thomas": args.sweeps * (c_line + c_bt),
        "flop_ratio_expected": nl / 3.0,
    }
    bk.log(f"cost {rec['cost']}")

    if args.inner_iterations:
        its, steps = args.inner_iterations, max(args.outer_steps, 1)
        for name, c_p in (
            ("dense", rec["cost"]["c_precond_dense"]),
            ("block_thomas", rec["cost"]["c_precond_block_thomas"]),
        ):
            rec.setdefault("projection", {})[name] = {
                "inner": its * (1.0 + c_p),
                "outer_applies": 2 * steps,
                "note": "setup excluded; add the arm's own setup term",
            }
    rec["wall_s"] = time.perf_counter() - T_PROCESS
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    rec["loadavg_end"] = os.getloadavg()
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
