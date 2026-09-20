# ruff: noqa: E402
"""Q28 step 1: why ``src/``'s shift-invert leaves every inner solve unconverged.

Q26 (#257) recorded, on the shipped Cyclone deck, that
``KrylovConfig(method="shift_invert")`` leaves every inner FGMRES solve
unconverged at a maximum relative residual of order 30 against its 1e-4 inner
tolerance, with every solve sitting at the iteration budget cap, and that the
outer pair is then rejected at residual ~1. Raising the budget does not move it.
This script separates the two candidate causes on the same operator ``src/``
builds:

``budget``
    run ``dominant_eigenpair(method="shift_invert")`` through the shipped code
    path and record the ``EigenSolveStatus`` (outer residual, gate, inner
    summary), for a ladder of ``shift_maxiter`` values.
``inner``
    take the very first right-hand side the outer Arnoldi generates and, on
    that one system ``(A - sigma I) x = b``, report, all as relative residuals
    against ``||b||``:

    * ``r_zero``   the trivial guess ``x = 0`` (identically 1);
    * ``r_x0``     the shipped initial guess ``x0 = M^-1 b``;
    * ``r_gmres``  the shipped solve, ``x0 = M^-1 b``, right preconditioner
      ``M^-1``, ``restart`` and ``max_restarts`` exactly as
      ``_shift_invert_apply_factory`` computes them;
    * ``r_gmres_zero`` the same solve from ``x0 = 0``, which isolates how much
      of the reported residual is the initial guess and how much is the
      preconditioner;
    * ``r_unrestarted`` the same solve at ``restart = maxiter`` (one cycle).

    ``r_x0`` is the diagnostic that decides between the two causes. If the
    preconditioner is a usable approximate inverse, ``||b - (A - sigma) M^-1 b||
    / ||b||`` is below 1 and GMRES starts ahead of the trivial guess; if it is
    not, the shipped code starts the solve *behind* ``x = 0`` and the restarted
    budget cannot recover, which is a defect of the initial guess as well as of
    the preconditioner.

One fresh process per invocation. No GKX source is changed; this measures what
``src/`` does today. Output: progress lines and one ``RESULT {json}`` line.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "2026-09-13-preconditioner-bakeoff"))

import jax
import jax.numpy as jnp
import numpy as np

import bakeoff as bk  # noqa: E402  (evidence harness, reused unchanged)

import gkx  # noqa: E402
from gkx.solvers_linear_krylov import (  # noqa: E402
    KrylovConfig,
    _normalized_config,
    _shift_seed,
    dominant_eigenpair,
)
from gkx.solvers_linear_krylov_algorithms import (  # noqa: E402
    _apply_operator,
    _linked_covered_mode_mask,
    _projected_flat_operator,
    build_shift_invert_preconditioner,
)

# Diagnosis rungs. ``d96`` is the size Q26 recorded; ``prod`` is the shipped
# deck's own (Nl, Nm) = (16, 48) at the same (Nz, ntheta, nperiod).
bk.CASES = dict(bk.CASES) | {
    "d96": (1, 24, 96, 32, 2, 4, 8),
    "d48": (1, 24, 48, 16, 2, 4, 8),
    "d96l8": (1, 24, 96, 32, 2, 8, 24),
}


def env(argv: list[str]) -> dict:
    return {
        "argv": argv,
        "git": bk.git_state(),
        "gkx": str(Path(gkx.__file__).resolve()),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "env": {
            k: os.environ.get(k)
            for k in (
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
            )
        },
    }


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def run_budget(args, rec: dict) -> dict:
    case = bk.Case(args.case, None)
    rec |= {"n": case.n, "ky": case.ky, "shape": list(case.shape)}
    seed = jnp.asarray(case.seed)
    rows = []
    for maxiter in [int(v) for v in args.maxiter.split(",")]:
        t = time.perf_counter()
        row: dict = {"shift_maxiter": maxiter}
        try:
            _val, _vec, status = dominant_eigenpair(
                seed,
                case.cache,
                case.params,
                case.T0,
                method="shift_invert",
                shift_maxiter=maxiter,
                certify=True,
                return_status=True,
                status_callback=bk.log,
            )
            row |= {"raised": None, "status": status.as_dict()}
        except Exception as exc:  # the shipped gate raises on rejection
            row |= {"raised": f"{type(exc).__name__}: {exc}"}
        row["seconds"] = time.perf_counter() - t
        bk.log(f"budget {row}")
        rows.append(row)
    rec["budget"] = rows
    return rec


def _first_rhs(case, cfg: KrylovConfig):
    """sigma and the first right-hand side the outer Arnoldi actually solves."""

    seed = jnp.asarray(case.seed)
    covered = _linked_covered_mode_mask(case.cache)
    sigma, v_init = _shift_seed(
        seed,
        seed,
        case.cache,
        case.params,
        case.T0,
        cfg,
        None,
        select_overlap=False,
    )
    return sigma, v_init, covered


def run_inner(args, rec: dict) -> dict:
    from solvax import gmres

    case = bk.Case(args.case, None)
    rec |= {"n": case.n, "ky": case.ky, "shape": list(case.shape)}
    cfg = _normalized_config(
        dict(KrylovConfig(method="shift_invert", shift_maxiter=args.maxiter).__dict__)
    )
    sigma, v_init, covered = _first_rhs(case, cfg)
    sigma_val = jnp.asarray(sigma, dtype=v_init.dtype)
    rec["sigma"] = [float(np.real(sigma_val)), float(np.imag(sigma_val))]
    bk.log(f"sigma {complex(np.asarray(sigma_val))}")

    shape, size = v_init.shape, v_init.size
    results: dict = {}
    for mode in args.preconditioner.split(","):
        _p, precond_raw = build_shift_invert_preconditioner(
            v_init, case.cache, case.params, case.T0, sigma_val, mode
        )
        precond_op = _projected_flat_operator(precond_raw, covered, shape)

        def matvec(x_flat, _s=sigma_val):
            x = x_flat.reshape(shape)
            return (
                _apply_operator(x, case.cache, case.params, case.T0) - _s * x
            ).reshape(size)

        b = v_init.reshape(size)
        b = b / jnp.linalg.norm(b)
        b_norm = float(np.asarray(jnp.linalg.norm(b)))

        def rel(x):
            return float(
                np.asarray(jnp.linalg.norm(b - matvec(x)) / jnp.maximum(b_norm, 1e-300))
            )

        row: dict = {"b_norm": b_norm}
        x0 = precond_op(b) if precond_op is not None else b
        row["r_zero"] = rel(jnp.zeros_like(b))
        row["r_x0"] = rel(x0)
        # exactly what _shift_invert_apply_factory computes
        import math

        restart = min(max(cfg.shift_restart, 1), cfg.shift_maxiter, size)
        max_restarts = max(1, math.ceil(cfg.shift_maxiter / restart))
        row["restart"] = restart
        row["max_restarts"] = max_restarts
        t = time.perf_counter()
        sol = gmres(
            matvec,
            b,
            x0=x0,
            precond=precond_op,
            rtol=cfg.shift_tol,
            restart=restart,
            max_restarts=max_restarts,
        )
        row |= {
            "r_gmres": rel(sol.x),
            "gmres_reported": float(np.asarray(sol.residual_norm)) / b_norm,
            "gmres_iterations": int(np.asarray(sol.iterations)),
            "gmres_converged": bool(np.asarray(sol.converged)),
            "gmres_seconds": time.perf_counter() - t,
        }
        sol0 = gmres(
            matvec,
            b,
            x0=None,
            precond=precond_op,
            rtol=cfg.shift_tol,
            restart=restart,
            max_restarts=max_restarts,
        )
        row |= {
            "r_gmres_zero": rel(sol0.x),
            "gmres_zero_iterations": int(np.asarray(sol0.iterations)),
            "gmres_zero_converged": bool(np.asarray(sol0.converged)),
        }
        solu = gmres(
            matvec,
            b,
            x0=x0,
            precond=precond_op,
            rtol=cfg.shift_tol,
            restart=min(cfg.shift_maxiter, size),
            max_restarts=1,
        )
        row |= {
            "r_unrestarted": rel(solu.x),
            "unrestarted_iterations": int(np.asarray(solu.iterations)),
            "unrestarted_converged": bool(np.asarray(solu.converged)),
        }
        results[mode] = row
        bk.log(f"inner[{mode}] {row}")
    rec["inner"] = results
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="inner", choices=("budget", "inner"))
    p.add_argument("--case", default="d96")
    p.add_argument("--maxiter", default="50")
    p.add_argument(
        "--preconditioner", default="hermite-line", help="comma-separated names"
    )
    p.add_argument("--label", default="diagnose")
    args = p.parse_args()

    rec: dict = {"label": args.label, "mode": args.mode, "case": args.case}
    rec["environment"] = env(sys.argv)
    if args.mode == "budget":
        rec = run_budget(args, rec)
    else:
        args.maxiter = int(args.maxiter.split(",")[0])
        rec = run_inner(args, rec)
    rec["peak_rss_gib"] = peak_rss_gib()
    rec["process_seconds"] = time.perf_counter() - T_PROCESS
    print("RESULT " + json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
