# ruff: noqa: E402
"""Q21: the inner-solve cost of matrix-free shift-invert (plan §5.1 L5).

Q7 (#236) adopted ``pr3-cm`` as the L4 preconditioner but measured that
matrix-free shift-invert with it does not beat the runtime-default ``adaptive``
route at the production chain, because each outer Arnoldi step costs about 950
inner ``gcrot`` iterations at a fixed inner tolerance of 1e-9. This harness runs
the L5 levers against the same gate, reusing Q7's assembly, splits, line solve
and Peaceman-Rachford sweeps from
``plan/research/scripts/2026-09-13-preconditioner-bakeoff``.

Arms (``--arm``):

``adaptive``
    the runtime-default route through ``run_runtime_linear(krylov_cfg=None)``,
    with ``adaptive_propagator_eigenpair`` wrapped so the route's own
    ``operator_applications`` counter is recorded. This is the control.
``si``
    matrix-free shift-invert Arnoldi with ``pr3-cm`` and SOLVAX
    ``gcrot(m, k, "harmonic")`` recycling, as in Q7's ``prod_certify.py``, plus
    the two L5 levers:

    * ``--schedule outer`` sets the inner tolerance of step ``j`` to
      ``clip(kappa * res_{j-1}, floor, cap)`` (``res_0 = 1``) instead of a fixed
      ``--inner-rtol`` -- the Freitag-Spence inverse-iteration rule, loose while
      far; ``--schedule inexact`` uses the inexact-Krylov rule
      ``clip(kappa * target / res_{j-1}, floor, cap)``, which relaxes the *late*
      steps instead;
    * ``--dsolve block-thomas`` solves the z-local block exactly by block-Thomas
      in the Laguerre index plus Sherman-Morrison (see :mod:`dblock`) instead of
      by a dense inverse -- the same preconditioner, a cheaper apply;
    * ``--tuned K`` replaces the preconditioner ``P`` by the Freitag-Spence
      tuned ``M_i = P + (B - P) X_i X_i^H`` with ``X_i`` the ``K`` most recent
      Ritz vectors, applied by Woodbury
      ``M_i^-1 z = w - U (I + X^H U)^-1 X^H w``, ``w = P^-1 z``,
      ``U = P^-1 B X - X``.
``hks``
    thick-restart harmonic Rayleigh-Ritz on the original operator with no inner
    solves: a basis of ``--dim`` columns, harmonic extraction at the target
    ``tau`` from the pencil ``(((A - tau)V)^H (A - tau)V, ((A - tau)V)^H V)``,
    and a restart that keeps the ``--keep`` best harmonic Ritz vectors plus the
    last basis vector (no extra operator applications).

Every arm certifies with GKX's own ``_eigenpair_relative_residual`` against the
original matrix-free operator -- the gate is unchanged -- and every arm reports
matvec-equivalents under the accounting registered in ``PREDICTIONS.txt``.
Output: progress lines and one ``RESULT {json}`` line.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import replace
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
import dblock as db
import pr_kz as pk
import prod_solve as ps
import scipy
import scipy.linalg as sla
import solvax

import gkx
from gkx.solvers_linear_krylov import _eigenpair_relative_residual

REFERENCE = complex(0.09302951, -0.28199404)
# Shift per case: the certified eigenvalue plus GROWTH_OFFSET in growth, as Q7.
SIGMA = {
    "prod": complex(REFERENCE.real + bk.GROWTH_OFFSET, REFERENCE.imag),
    "r96": complex(0.1487757, -0.2769046),
    "r64": complex(0.16268013050461985, -0.27079063054988406),
}
ALPHA_BEST = {"prod": -10.0, "r96": -10.0, "r64": -3.0}


def since_start() -> float:
    return time.perf_counter() - T_PROCESS


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def environment(argv: list[str]) -> dict:
    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solvax": solvax.__version__,
        "gkx_file": str(gkx.__file__),
        "x64": bool(jax.config.jax_enable_x64),
        "cpu_count": os.cpu_count(),
        "loadavg_start": os.getloadavg(),
        "argv": argv,
        "env": {
            key: os.environ.get(key)
            for key in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "GKX_X64",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "PYTHONPATH",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bakeoff_sha256": hashlib.sha256(
            (BAKEOFF / "bakeoff.py").read_bytes()
        ).hexdigest(),
        "pr_kz_sha256": hashlib.sha256((BAKEOFF / "pr_kz.py").read_bytes()).hexdigest(),
    } | bk.git_state()


def deck_config(case_name: str):
    """The cfg ``bakeoff.prepare`` builds, so control and arms share one operator."""

    from gkx.workflows.runtime.toml import load_runtime_from_toml

    Nx, Ny, Nz, ntheta, nperiod, Nl, Nm = bk.CASES[case_name]
    cfg0, _ = load_runtime_from_toml(REPO / bk.DECK)
    cfg = replace(
        cfg0,
        grid=replace(
            cfg0.grid, Nx=Nx, Ny=Ny, Nz=Nz, ntheta=ntheta, nperiod=nperiod, jtwist=1
        ),
        time=replace(cfg0.time, damp_ends_rate=0.1),
    )
    return cfg, Nl, Nm


# --------------------------------------------------------------------------- #
# cost model                                                                    #
# --------------------------------------------------------------------------- #
def cost_model(case, applies: dict, reps: int) -> dict:
    """Median seconds of one matvec and of each named apply, in this process."""

    probe = jnp.asarray(case.b)
    out = {"t_matvec_s": bk.median_seconds(case._mv, probe, reps=reps)}
    for name, fn in applies.items():
        out[f"t_{name}_s"] = bk.median_seconds(fn, probe, reps=reps)
        out[f"c_{name}"] = out[f"t_{name}_s"] / out["t_matvec_s"]
    return out


# --------------------------------------------------------------------------- #
# arm: adaptive control                                                         #
# --------------------------------------------------------------------------- #
def run_adaptive(args, rec: dict) -> dict:
    import gkx.solvers_linear_krylov as klv
    from gkx.runtime import run_runtime_linear
    from gkx.solvers_linear_krylov import certifiable_residual_tolerance

    case = bk.Case(args.case, None)
    rec["n"] = case.n
    rec["ky"] = case.ky
    rec["cost"] = cost_model(case, {}, args.cost_reps)
    bk.log(f"cost model {rec['cost']}")
    seed = np.asarray(case.seed)

    cfg, Nl, Nm = deck_config(args.case)
    inner: dict = {}
    original = klv.adaptive_propagator_eigenpair

    def wrapped(*a, **kw):
        t = time.perf_counter()
        solution = original(*a, **kw)
        inner.update(
            operator_applications=int(np.asarray(solution.operator_applications)),
            restarts=int(np.asarray(solution.restarts)),
            filter_steps=int(solution.filter_steps),
            filter_dt=float(solution.filter_dt),
            adaptive_residual=float(np.asarray(solution.residual)),
            solver_s=time.perf_counter() - t,
            krylov_kwargs={
                k: int(v) for k, v in kw.items() if isinstance(v, int) and k != "tol"
            },
        )
        return solution

    klv.adaptive_propagator_eigenpair = wrapped
    statuses: list = []
    gate = certifiable_residual_tolerance(1.0e-9, jnp.asarray(seed).dtype)
    rec["residual_gate"] = gate
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
        status_callback=lambda m: statuses.append([round(since_start(), 2), m]),
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
    rec["state_native_dtype"] = str(np.asarray(result.state).dtype)
    rec["adaptive"] = inner
    rec["statuses"] = statuses
    rec["lambda"] = [lam.real, lam.imag]
    rec["gamma_omega"] = [float(result.gamma), float(result.omega)]
    rec["certified_residual"] = residual
    rec["certified"] = bool(np.isfinite(residual) and residual <= gate)
    rec["matvec_equivalents"] = {
        "route_operator_applications": inner.get("operator_applications"),
        "certification": 1,
        "total": (inner.get("operator_applications") or 0) + 1,
    }
    return rec


# --------------------------------------------------------------------------- #
# arm: matrix-free shift-invert with the L5 levers                              #
# --------------------------------------------------------------------------- #
def run_shift_invert(args, rec: dict) -> dict:
    case = bk.Case(args.case, None)
    sigma = args.sigma if args.sigma is not None else SIGMA[args.case]
    alpha_best = (
        args.alpha_best if args.alpha_best is not None else ALPHA_BEST[args.case]
    )
    shape, n = case.shape, case.n
    rec |= {"n": n, "ky": case.ky, "sigma": [sigma.real, sigma.imag]}
    bk.log(f"{args.case} n={n} ky={case.ky:+.3f} sigma={sigma:.6f} alpha={alpha_best}")

    t = time.perf_counter()
    mean = ps.z_mean_drift(case)
    blocks = bk.zblocks_by_probing(case.operator(case.T_DC, True), shape, batch=16)
    idx = np.arange(case.bs)
    blocks[:, idx, idx] -= mean
    line = pk.KzLine(case)
    d_rho = pk.spectral_radius(blocks)
    a_kz, b = pk.rule_parameters(args.rule, line, d_rho, alpha_best)
    nl, nm = shape[1], shape[2]
    structure = None
    if args.dsolve == "block-thomas":
        no_phi = bk.zblocks_by_probing(case.operator(case.T_DC, False), shape, batch=16)
        no_phi[:, idx, idx] -= mean
        structure = db.check_structure(no_phi, blocks, nl, nm)
        bk.log(f"structure {structure}")
        inv, make_dsolve, info = db.build_solver(blocks, no_phi, nl, nm, sigma / 2 - b)
        t_inv, mem = float("nan"), info["bytes"]
        del no_phi
    else:
        inv, t_inv, mem = case.dinv(blocks, sigma / 2 - b)

        def make_dsolve(inverse):
            return lambda xb: jnp.einsum("bij,bj->bi", inverse, xb)

    del blocks, mean
    setup_s = time.perf_counter() - t
    rec["setup"] = {
        "total_s": setup_s,
        "dsolve": args.dsolve,
        "inverse_s": t_inv,
        "inverse_bytes": mem,
        "d_rho": d_rho,
        "s1": line.s1,
        "b": float(b),
        "structure": structure,
    }
    bk.log(f"setup {rec['setup']}")

    mv = case._mv
    a_j = jnp.asarray(a_kz, dtype=jnp.complex128)
    tuned_k = int(args.tuned)

    def bmv(v):
        return mv(v) - sigma * v

    def base_precond(inverse):
        blocked = make_dsolve(inverse)

        def dsolve(z):
            return bk.from_blocks(blocked(bk.to_blocks(z, shape)), shape)

        return pk.pr_two(line, sigma, a_j, b, dsolve, args.sweeps)

    def tuned_apply(inverse, X, U, S, z):
        w = base_precond(inverse)(z)
        return w - U @ (S @ (X.conj().T @ w))

    precond_apply = jax.jit(lambda inverse, z: base_precond(inverse)(z))
    bmv_jit = jax.jit(bmv)

    def gcrot_call(matvec, precond, rhs, recycle, tol):
        return solvax.gcrot(
            matvec,
            rhs,
            precond=precond,
            m=args.gcrot_m,
            k=args.gcrot_k,
            rtol=0.0,
            atol=tol,
            max_restarts=args.max_restarts,
            recycle=recycle,
            recycle_strategy="harmonic",
        )

    solve_plain_first = jax.jit(
        lambda inverse, rhs, tol: gcrot_call(bmv, base_precond(inverse), rhs, None, tol)
    )
    solve_plain_next = jax.jit(
        lambda inverse, rhs, recycle, tol: gcrot_call(
            bmv, base_precond(inverse), rhs, recycle, tol
        )
    )
    solve_tuned_first = jax.jit(
        lambda inverse, rhs, tol, X, U, S: gcrot_call(
            bmv, lambda z: tuned_apply(inverse, X, U, S, z), rhs, None, tol
        )
    )
    solve_tuned_next = jax.jit(
        lambda inverse, rhs, recycle, tol, X, U, S: gcrot_call(
            bmv, lambda z: tuned_apply(inverse, X, U, S, z), rhs, recycle, tol
        )
    )

    # Cost model. Besides the whole preconditioner this measures its two parts
    # -- the spectral streaming line solve and the dense z-local block solve --
    # so the block-Thomas projection (L5 candidate 4) rests on this session's
    # numbers. Every probe is jitted, as the preconditioner is inside gcrot.
    s1_kz = jnp.asarray(sigma / 2 - a_j, dtype=jnp.complex128)
    line_apply = jax.jit(lambda z: line.solve(z, s1_kz))
    dsolve_apply = jax.jit(
        lambda inverse, z: bk.from_blocks(
            make_dsolve(inverse)(bk.to_blocks(z, shape)), shape
        )
    )
    applies = {
        "precond": lambda z: precond_apply(inv, z),
        "line": line_apply,
        "dsolve": lambda z: dsolve_apply(inv, z),
    }
    if tuned_k:
        X0 = jnp.asarray(
            np.linalg.qr(np.random.default_rng(0).standard_normal((n, tuned_k)) + 0j)[0]
        )
        U0 = jnp.zeros_like(X0)
        S0 = jnp.eye(tuned_k, dtype=jnp.complex128)
        tuned_probe = jax.jit(lambda z: tuned_apply(inv, X0, U0, S0, z))
        applies["tuned"] = tuned_probe
    rec["cost"] = cost_model(case, applies, args.cost_reps)
    bk.log(f"cost model {rec['cost']}")
    c_p = rec["cost"]["c_tuned" if tuned_k else "c_precond"]

    def tuning(Xnp):
        """(X, U, S) for the Freitag-Spence tuned preconditioner; 2k applies."""

        BX = np.stack(
            [np.asarray(bmv_jit(jnp.asarray(Xnp[:, i]))) for i in range(Xnp.shape[1])],
            axis=1,
        )
        PBX = np.stack(
            [
                np.asarray(precond_apply(inv, jnp.asarray(BX[:, i])))
                for i in range(Xnp.shape[1])
            ],
            axis=1,
        )
        U = PBX - Xnp
        S = np.linalg.inv(np.eye(Xnp.shape[1], dtype=np.complex128) + Xnp.conj().T @ U)
        return jnp.asarray(Xnp), jnp.asarray(U), jnp.asarray(S)

    kmax = args.max_steps
    V = np.zeros((n, kmax + 1), dtype=np.complex128)
    H = np.zeros((kmax + 1, kmax), dtype=np.complex128)
    V[:, 0] = case.b / np.linalg.norm(case.b)
    ritz: list[np.ndarray] = []
    steps: list[dict] = []
    recycle = None
    tune = None
    hits: dict = {}
    res_prev = 1.0
    extra_applies = 0
    inner_total = 0
    t_outer = time.perf_counter()
    for j in range(kmax):
        if args.schedule == "outer":
            # Freitag-Spence inverse-iteration rule: solve loosely while far.
            tol = float(np.clip(args.kappa * res_prev, args.rtol_floor, args.rtol_cap))
        elif args.schedule == "inexact":
            # Inexact-Krylov rule (Simoncini-Szyld): in a *Krylov* method the
            # perturbation allowance grows as the residual falls, so the early
            # matvecs are the ones that must be accurate and the late ones may
            # be relaxed -- the opposite of "outer".
            tol = float(
                np.clip(
                    args.kappa * args.target / max(res_prev, args.target),
                    args.rtol_floor,
                    args.rtol_cap,
                )
            )
        else:
            tol = float(args.inner_rtol)
        t = time.perf_counter()
        rhs = jnp.asarray(V[:, j])
        tol_j = jnp.asarray(tol, dtype=jnp.float64)
        if tuned_k and tune is not None:
            sol = (
                solve_tuned_first(inv, rhs, tol_j, *tune)
                if recycle is None
                else solve_tuned_next(inv, rhs, recycle, tol_j, *tune)
            )
        else:
            sol = (
                solve_plain_first(inv, rhs, tol_j)
                if recycle is None
                else solve_plain_next(inv, rhs, recycle, tol_j)
            )
        w = np.asarray(sol.x)
        recycle = sol.recycle
        inner_s = time.perf_counter() - t
        true_inner = float(
            np.linalg.norm(V[:, j] - (np.asarray(mv(jnp.asarray(w))) - sigma * w))
            / np.linalg.norm(V[:, j])
        )
        extra_applies += 1
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
        res = float(
            _eigenpair_relative_residual(
                jnp.asarray(lam),
                jnp.asarray(u.reshape(shape)),
                case.cache,
                case.params,
                case.T0,
            )
        )
        extra_applies += 1
        inner_total += int(sol.iterations)
        step = {
            "step": j + 1,
            "inner_tol": tol,
            "inner_its": int(sol.iterations),
            "inner_converged": bool(sol.converged),
            "inner_true_rel": true_inner,
            "inner_s": inner_s,
            "lambda": [lam.real, lam.imag],
            "residual": res,
            "cumulative_inner_its": inner_total,
            "since_process_start_s": since_start(),
        }
        steps.append(step)
        bk.log(f"step {step}")
        for gate in (1e-6, args.target):
            if res <= gate and str(gate) not in hits:
                hits[str(gate)] = {
                    "step": j + 1,
                    "since_process_start_s": since_start(),
                    "total_inner_its": inner_total,
                    "lambda": [lam.real, lam.imag],
                    "residual": res,
                }
        res_prev = res
        if res <= args.target:
            break
        if tuned_k:
            ritz.append(u / np.linalg.norm(u))
            Xnp = np.stack(ritz[-tuned_k:], axis=1)
            Xnp = np.linalg.qr(Xnp)[0]
            tune = tuning(Xnp)
            extra_applies += 2 * Xnp.shape[1]

    rec["steps"] = steps
    rec["gates"] = hits
    rec["outer_s"] = time.perf_counter() - t_outer
    rec["lambda"] = steps[-1]["lambda"]
    rec["gamma_omega"] = [steps[-1]["lambda"][0], -steps[-1]["lambda"][1]]
    rec["certified_residual"] = steps[-1]["residual"]
    rec["certified"] = steps[-1]["residual"] <= args.target
    # matvec-equivalent accounting, per PREDICTIONS.txt
    t_mv = rec["cost"]["t_matvec_s"]
    counted = {}
    for label, hit in hits.items():
        upto = hit["step"]
        its = sum(s["inner_its"] for s in steps[:upto])
        counted[label] = {
            "outer_steps": upto,
            "inner_iterations": its,
            "inner": its * (1.0 + c_p),
            "recycle_restore": (upto - 1) * args.gcrot_k,
            "outer_applies": 2 * upto
            + (2 * tuned_k * max(upto - 1, 0) if tuned_k else 0),
            "setup": setup_s / t_mv,
        }
        counted[label]["total"] = sum(
            v
            for key, v in counted[label].items()
            if key not in ("outer_steps", "inner_iterations")
        )
    rec["matvec_equivalents"] = counted
    rec["extra_applies_measured"] = extra_applies
    rec["c_precond"] = c_p
    return rec


# --------------------------------------------------------------------------- #
# arm: thick-restart harmonic Rayleigh-Ritz, no inner solves                     #
# --------------------------------------------------------------------------- #
def run_harmonic(args, rec: dict) -> dict:
    case = bk.Case(args.case, None)
    tau = args.sigma if args.sigma is not None else SIGMA[args.case]
    shape, n = case.shape, case.n
    rec |= {"n": n, "ky": case.ky, "tau": [tau.real, tau.imag]}
    rec["cost"] = cost_model(case, {}, args.cost_reps)
    t_mv = rec["cost"]["t_matvec_s"]
    bk.log(f"{args.case} n={n} tau={tau:.6f} dim={args.dim} keep={args.keep}")

    mv = case._mv
    dim, keep = args.dim, args.keep
    V = np.zeros((n, dim), dtype=np.complex128)
    G = np.zeros((n, dim), dtype=np.complex128)

    def apply(x):
        return np.asarray(mv(jnp.asarray(x)))

    V[:, 0] = case.b / np.linalg.norm(case.b)
    G[:, 0] = apply(V[:, 0])
    nv = 1
    matvecs = 1
    dense_s = 0.0
    rounds: list[dict] = []
    hits: dict = {}
    best = None
    t_outer = time.perf_counter()
    for r in range(args.restarts):
        while nv < dim:
            w = G[:, nv - 1].copy()
            for _ in range(2):
                w -= V[:, :nv] @ (V[:, :nv].conj().T @ w)
            beta = np.linalg.norm(w)
            if beta < 1e-12:
                break
            V[:, nv] = w / beta
            G[:, nv] = apply(V[:, nv])
            matvecs += 1
            nv += 1
        td = time.perf_counter()
        # Harmonic Rayleigh-Ritz on span(V): ((A-tau)V)^H[(A-tau)V y - mu V y] = 0.
        # With (A-tau)V = Qw Rw this is (Qw^H V Rw^-1) g = mu^-1 g, y = Rw^-1 g,
        # which avoids forming the ill-conditioned normal-equation pencil.
        W = G[:, :nv] - tau * V[:, :nv]
        Qw, Rw = np.linalg.qr(W)
        K = sla.solve_triangular(Rw, (Qw.conj().T @ V[:, :nv]).T, trans="T").T
        nu, Gv = np.linalg.eig(K)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 1.0 / nu
        Y = sla.solve_triangular(Rw, Gv)
        finite = np.isfinite(mu) & np.all(np.isfinite(Y), axis=0)
        order = np.argsort(np.where(finite, np.abs(mu), np.inf))
        dense_s += time.perf_counter() - td
        cand = []
        for c in order[: max(keep, 4)]:
            y = Y[:, c] / np.linalg.norm(Y[:, c])
            lam = tau + mu[c]
            u = V[:, :nv] @ y
            nu = np.linalg.norm(u)
            rel = np.linalg.norm(G[:, :nv] @ y - lam * u) / (abs(lam) * nu)
            cand.append((float(rel), complex(lam), c, u / nu))
        cand.sort(key=lambda z: z[0])
        rel, lam, _, u = cand[0]
        res = float(
            _eigenpair_relative_residual(
                jnp.asarray(lam),
                jnp.asarray(u.reshape(shape)),
                case.cache,
                case.params,
                case.T0,
            )
        )
        matvecs += 1
        rnd = {
            "restart": r + 1,
            "basis_dim": nv,
            "matvecs": matvecs,
            "lambda": [lam.real, lam.imag],
            "subspace_rel": rel,
            "residual": res,
            "since_process_start_s": since_start(),
        }
        rounds.append(rnd)
        bk.log(f"round {rnd}")
        if best is None or res < best["residual"]:
            best = rnd | {"vector_norm": float(np.linalg.norm(u))}
        for gate in (1e-6, args.target):
            if res <= gate and str(gate) not in hits:
                hits[str(gate)] = {
                    "restart": r + 1,
                    "matvecs": matvecs,
                    "since_process_start_s": since_start(),
                    "lambda": [lam.real, lam.imag],
                    "residual": res,
                }
        if res <= args.target:
            break
        # Thick restart: the retained harmonic Ritz vectors plus the Arnoldi
        # residual direction f (the vector the next Arnoldi step would produce),
        # so the continued space stays a genuine augmented Krylov space. Only
        # A f costs an operator application; the retained block's images come
        # from the stored G.
        td = time.perf_counter()
        sel = [c for _, _, c, _ in cand[:keep]]
        Yk = Y[:, sel]
        Yk = Yk / np.linalg.norm(Yk, axis=0, keepdims=True)
        Q, R = np.linalg.qr(V[:, :nv] @ Yk)
        Gk = (G[:, :nv] @ Yk) @ np.linalg.inv(R)
        f = G[:, nv - 1].copy()
        for _ in range(2):
            f -= V[:, :nv] @ (V[:, :nv].conj().T @ f)
        beta = np.linalg.norm(f)
        dense_s += time.perf_counter() - td
        if beta < 1e-12:
            bk.log("thick restart: residual direction exhausted (invariant subspace)")
            break
        f = f / beta
        for _ in range(2):
            f -= Q @ (Q.conj().T @ f)
        f = f / np.linalg.norm(f)
        kk = Q.shape[1]
        V[:, :kk], G[:, :kk] = Q, Gk
        V[:, kk] = f
        G[:, kk] = apply(f)
        matvecs += 1
        nv = kk + 1
    rec["rounds"] = rounds
    rec["gates"] = hits
    rec["best"] = best
    rec["matvecs"] = matvecs
    rec["dense_s"] = dense_s
    rec["outer_s"] = time.perf_counter() - t_outer
    rec["lambda"] = rounds[-1]["lambda"]
    rec["gamma_omega"] = [rounds[-1]["lambda"][0], -rounds[-1]["lambda"][1]]
    rec["certified_residual"] = best["residual"] if best else None
    rec["certified"] = bool(best and best["residual"] <= args.target)
    rec["matvec_equivalents"] = {
        label: {
            "matvecs": hit["matvecs"],
            "dense": dense_s / t_mv,
            "total": hit["matvecs"] + dense_s / t_mv,
        }
        for label, hit in hits.items()
    } or {
        "not_reached": {
            "matvecs": matvecs,
            "dense": dense_s / t_mv,
            "total": matvecs + dense_s / t_mv,
        }
    }
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("adaptive", "si", "hks"))
    p.add_argument("--case", default="prod", choices=sorted(bk.CASES))
    p.add_argument("--sigma", type=complex, default=None)
    p.add_argument("--alpha-best", type=float, default=None)
    p.add_argument("--rule", default="scalar-best")
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--inner-rtol", type=float, default=1e-9)
    p.add_argument("--schedule", default="none", choices=("none", "outer", "inexact"))
    p.add_argument("--kappa", type=float, default=1e-2)
    p.add_argument("--rtol-cap", type=float, default=1e-1)
    p.add_argument("--rtol-floor", type=float, default=1e-10)
    p.add_argument("--tuned", type=int, default=0)
    p.add_argument("--dsolve", default="dense", choices=("dense", "block-thomas"))
    p.add_argument("--gcrot-m", type=int, default=20)
    p.add_argument("--gcrot-k", type=int, default=10)
    p.add_argument("--max-restarts", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--restarts", type=int, default=40)
    p.add_argument("--target", type=float, default=1e-9)
    p.add_argument("--cost-reps", type=int, default=9)
    p.add_argument("--label", default="")
    args = p.parse_args()

    env = environment(sys.argv[1:])
    print("ENV " + json.dumps(env), flush=True)
    if not env["x64"]:
        raise SystemExit("complex128 required")
    if not str(gkx.__file__).startswith(str(REPO)):
        raise SystemExit(f"gkx imported from outside the worktree: {gkx.__file__}")

    rec: dict = {"arm": args.arm, "case": args.case, "label": args.label, "env": env}
    rec["t_context_s"] = since_start()
    if args.arm == "adaptive":
        rec = run_adaptive(args, rec)
    elif args.arm == "si":
        rec = run_shift_invert(args, rec)
    else:
        rec = run_harmonic(args, rec)
    rec["wall_s"] = since_start()
    rec["peak_rss_gib"] = peak_rss_gib()
    rec["loadavg_end"] = os.getloadavg()
    bk.log(f"gates {rec.get('gates')} wall {rec['wall_s']:.1f}s")
    print("RESULT " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
