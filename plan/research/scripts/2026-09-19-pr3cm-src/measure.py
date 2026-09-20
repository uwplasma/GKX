# ruff: noqa: E402
"""Q28 step 3: the two Q21 levers re-measured in ``src/``, against ``adaptive``.

Q21 (#255) measured both levers in a research harness; Q26 (#257) recorded that
neither had a landing site in ``src/``. With ``pr3-cm`` landed, both are
measurable through the shipped code path, and this harness measures them there.

Arms (``--arm``)
----------------
``adaptive``
    the runtime default and the control: ``run_runtime_linear`` with
    ``krylov_cfg=None``, with ``adaptive_propagator_eigenpair`` wrapped so the
    route's own ``operator_applications`` counter is recorded.
``si``
    ``KrylovConfig(method="shift_invert")`` through ``dominant_eigenpair``, with
    ``--preconditioner`` (``hermite-line`` is the shipped default, ``pr3-cm``
    the new one) and ``--block-solve`` (``block-thomas`` against the ``dense``
    control: the second lever, the same preconditioner and the same iteration
    counts with a cheaper apply and smaller factors).

Both arms certify with GKX's own ``_eigenpair_relative_residual`` against the
original matrix-free operator, and the gate is the shipped one.

Accounting
----------
Matvec-equivalents follow Q21's registered definition
(``plan/research/scripts/2026-09-19-inner-solve-cost/PREDICTIONS.txt``), so the
two rows can be read against each other:

* one matvec-equivalent is one application of the matrix-free operator at the
  arm's own size;
* one inner FGMRES iteration costs ``1 + c_P`` with ``c_P = t_P / t_mv``, both
  medians measured in this process;
* the outer loop costs one operator application per restart (the Rayleigh
  quotient) plus one for certification;
* setup seconds are converted at ``t_mv``;
* the ``adaptive`` arm is its own ``operator_applications`` plus one.

Wall time is recorded but is load-dependent; the matvec-equivalents, iteration
counts, factor bytes and residuals are not, and the verdict rests on those when
the host is contended. One fresh process per arm.
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

import bakeoff as bk

import gkx
import gkx.solvers_linear_krylov as klv
from gkx.solvers_linear_krylov import (
    _eigenpair_relative_residual,
    certifiable_residual_tolerance,
    dominant_eigenpair,
)
from gkx.solvers_linear_krylov_algorithms import (
    _linked_covered_mode_mask,
    _projected_flat_operator,
    build_shift_invert_preconditioner,
)
from gkx.solvers_linear_precond_pr3 import PR3_PRECOND_NAMES, build_pr3_factors

bk.CASES = dict(bk.CASES) | {
    "d96": (1, 24, 96, 32, 2, 4, 8),
    "d96l8": (1, 24, 96, 32, 2, 8, 24),
}


def environment(argv: list[str]) -> dict:
    return {
        "argv": argv,
        "git": bk.git_state(),
        "gkx": str(Path(gkx.__file__).resolve()),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "loadavg": os.getloadavg(),
        "env": {
            k: os.environ.get(k)
            for k in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS")
        },
    }


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def run_adaptive(args, rec: dict) -> dict:
    from gkx.runtime import run_runtime_linear

    case = bk.Case(args.case, None)
    rec |= {"n": case.n, "ky": case.ky}
    probe = jnp.asarray(case.b)
    t_mv = bk.median_seconds(case._mv, probe, reps=args.cost_reps)
    rec["cost"] = {"t_matvec_s": t_mv}
    bk.log(f"t_matvec {t_mv * 1e3:.3f} ms")

    Nx, Ny, Nz, ntheta, nperiod, Nl, Nm = bk.CASES[args.case]
    from dataclasses import replace

    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg0, _ = load_runtime_from_toml(REPO / bk.DECK)
    cfg = replace(
        cfg0,
        grid=replace(
            cfg0.grid, Nx=Nx, Ny=Ny, Nz=Nz, ntheta=ntheta, nperiod=nperiod, jtwist=1
        ),
        time=replace(cfg0.time, damp_ends_rate=0.1),
    )

    inner: dict = {}
    original = klv.adaptive_propagator_eigenpair

    def wrapped(*a, **kw):
        t = time.perf_counter()
        solution = original(*a, **kw)
        inner.update(
            operator_applications=int(np.asarray(solution.operator_applications)),
            restarts=int(np.asarray(solution.restarts)),
            adaptive_residual=float(np.asarray(solution.residual)),
            solver_s=time.perf_counter() - t,
        )
        return solution

    klv.adaptive_propagator_eigenpair = wrapped
    seed = np.asarray(case.seed)
    gate = certifiable_residual_tolerance(1.0e-9, jnp.asarray(seed).dtype)
    t = time.perf_counter()
    result = run_runtime_linear(
        cfg,
        ky_target=0.3,
        Nl=Nl,
        Nm=Nm,
        solver="krylov",
        krylov_cfg=None,
        return_state=True,
        initial_state=jnp.asarray(seed),
        status_callback=bk.log,
    )
    rec["route_s"] = time.perf_counter() - t
    klv.adaptive_propagator_eigenpair = original

    vec = jnp.asarray(result.state, dtype=jnp.complex128)
    lam = complex(result.gamma, -result.omega)
    residual = float(
        _eigenpair_relative_residual(
            jnp.asarray(lam, dtype=vec.dtype),
            vec.reshape(case.shape),
            case.cache,
            case.params,
            case.T0,
        )
    )
    applications = int(inner.get("operator_applications") or 0)
    rec |= {
        "adaptive": inner,
        "residual_gate": gate,
        "lambda": [lam.real, lam.imag],
        "gamma_omega": [float(result.gamma), float(result.omega)],
        "certified_residual": residual,
        "certified": bool(np.isfinite(residual) and residual <= gate),
        "matvec_equivalents": {
            "route_operator_applications": applications,
            "certification": 1,
            "total": applications + 1,
        },
    }
    return rec


def run_shift_invert(args, rec: dict) -> dict:
    case = bk.Case(args.case, None)
    rec |= {"n": case.n, "ky": case.ky}
    seed = jnp.asarray(case.seed)
    probe = jnp.asarray(case.b)
    t_mv = bk.median_seconds(case._mv, probe, reps=args.cost_reps)
    bk.log(f"t_matvec {t_mv * 1e3:.3f} ms")

    options = dict(
        method="shift_invert",
        shift=(complex(args.shift) if args.shift else None),
        shift_preconditioner=args.preconditioner,
        shift_precond_block_solve=args.block_solve,
        shift_maxiter=args.maxiter,
        shift_restart=args.restart,
        shift_tol=args.inner_rtol,
        krylov_dim=args.krylov_dim,
        restarts=args.restarts,
    )
    rec["options"] = dict(options)

    t = time.perf_counter()
    raised = None
    status = None
    lam = None
    try:
        value, vector, status = dominant_eigenpair(
            seed,
            case.cache,
            case.params,
            case.T0,
            return_status=True,
            status_callback=bk.log,
            **options,
        )
        lam = complex(np.asarray(value))
        rec["certified_residual"] = status.residual
        rec["certified"] = bool(status.certified)
        rec["residual_gate"] = status.tolerance
        rec["route"] = status.route
        rec["lambda"] = [lam.real, lam.imag]
        rec["gamma_omega"] = [lam.real, -lam.imag]
        del vector
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        rec["certified"] = False
    rec["route_s"] = time.perf_counter() - t
    rec["raised"] = raised
    rec["status"] = None if status is None else status.as_dict()

    # Cost model for this arm's own preconditioner, rebuilt and timed in this
    # same process, so c_P is measured under the conditions the arm ran in.
    from gkx.solvers_linear_krylov import _shift_seed

    factors = None
    meta: dict = {}
    cfg = klv._normalized_config(dict(klv.KrylovConfig(**options).__dict__))
    t = time.perf_counter()
    sigma, v_init = _shift_seed(
        seed, seed, case.cache, case.params, case.T0, cfg, None, select_overlap=False
    )
    if args.preconditioner in PR3_PRECOND_NAMES:
        factors, meta = build_pr3_factors(
            v_init,
            case.cache,
            case.params,
            case.T0,
            sigma,
            alpha=args.alpha,
            block_solve=args.block_solve,
        )
    setup_s = time.perf_counter() - t
    sigma_val = jnp.asarray(sigma, dtype=v_init.dtype)
    _p, raw = build_shift_invert_preconditioner(
        v_init,
        case.cache,
        case.params,
        case.T0,
        sigma_val,
        args.preconditioner,
        factors,
    )
    # jit the apply before timing it: inside the route it is traced into the
    # jitted Arnoldi, and an eager apply of three line solves and three block
    # solves is not the cost the solve pays.
    apply_p = jax.jit(
        _projected_flat_operator(
            raw, _linked_covered_mode_mask(case.cache), v_init.shape
        )
    )
    t_p = bk.median_seconds(apply_p, probe, reps=args.cost_reps)
    c_p = t_p / t_mv
    rec["cost"] = {"t_matvec_s": t_mv, "t_precond_s": t_p, "c_precond": c_p}
    rec["precond_setup"] = dict(meta, seconds=setup_s)
    bk.log(f"cost {rec['cost']}; setup {setup_s:.2f}s")

    inner = (rec["status"] or {}).get("inner") if rec["status"] else None
    iterations = int(inner["total_iterations"]) if inner else None
    if iterations is not None:
        counted = {
            "inner_iterations": iterations,
            "inner": iterations * (1.0 + c_p),
            "outer_applies": args.restarts + 1,
            "setup": setup_s / t_mv,
        }
        counted["total"] = (
            counted["inner"] + counted["outer_applies"] + counted["setup"]
        )
        rec["matvec_equivalents"] = counted
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="si", choices=("adaptive", "si"))
    p.add_argument("--case", default="d96")
    p.add_argument("--preconditioner", default="pr3-cm")
    p.add_argument("--block-solve", default="auto")
    p.add_argument("--alpha", type=float, default=None)
    # Q7's protocol placed the shift at the target eigenvalue plus 0.05 in
    # growth. The shipped default derives one from a short propagator run with
    # no user input, which is the configuration a first run takes; both are
    # measured, because the two answer different questions.
    p.add_argument("--shift", default=None, help="explicit complex shift")
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--restart", type=int, default=20)
    p.add_argument("--inner-rtol", type=float, default=1.0e-4)
    p.add_argument("--krylov-dim", type=int, default=24)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--cost-reps", type=int, default=7)
    p.add_argument("--label", default="arm")
    args = p.parse_args()

    rec: dict = {"label": args.label, "arm": args.arm, "case": args.case}
    rec["environment"] = environment(sys.argv)
    if args.arm == "adaptive":
        rec = run_adaptive(args, rec)
    else:
        rec = run_shift_invert(args, rec)
    rec["peak_rss_gib"] = peak_rss_gib()
    rec["process_seconds"] = time.perf_counter() - T_PROCESS
    print("RESULT " + json.dumps(rec, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
