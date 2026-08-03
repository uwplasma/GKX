"""Does Krylov recycling pay for itself inside shift-invert Arnoldi?

Shift-invert Arnoldi issues ``krylov_dim * restarts`` inner solves against the
*same* shifted operator ``A - sigma I``, varying only the right-hand side, and
GKX currently starts each one from scratch with ``jax.scipy.sparse.linalg.gmres``
-- this path predates the SOLVAX dependency and was never migrated
(``docs/solvers.rst``). That is the
textbook sequence-of-related-systems that Krylov recycling targets: the
directions that limited convergence on solve ``j`` limit solve ``j+1`` too, so
rediscovering them every time is wasted work (Parks et al. 2006).

SOLVAX ships ``gcrot`` with two recycle strategies. This tool measures whether
either beats the incumbent on GKX's real operator, in matrix-vector products --
the honest cost unit for a matrix-free solve, and the one that does not move
when the host is busy.

Two things this deliberately does not do. It does not replace the preconditioner
under test: the physics-aware Hermite-line inverse stays on, because a win
measured against a deliberately weak preconditioner is not a win. And it does not
assume recycling helps -- a recycle space costs O(nk) storage and one extra
operator application per restart to rebuild ``A U``, so a small iteration
saving is a real loss.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def build_case(
    n_laguerre: int,
    n_hermite: int,
    nz: int,
    *,
    preconditioner: str = "hermite-line",
    shift_offset: float = 1.0e-2,
) -> dict[str, Any]:
    """Assemble the shifted operator, preconditioner and target GKX actually uses."""

    import jax.numpy as jnp

    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import (
        LinearParams,
        LinearTerms,
        linear_terms_to_term_config,
    )
    from gkx.solvers.linear.krylov_algorithms import (
        _apply_operator,
        _build_shift_invert_precond,
    )

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=nz, Lx=6.0, Ly=12.0))
    grid = build_spectral_grid(cfg.grid)
    geometry = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    terms = LinearTerms()
    term_cfg = linear_terms_to_term_config(terms)
    cache = build_linear_cache(grid, geometry, params, Nl=n_laguerre, Nm=n_hermite)

    shape = (1, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    size = int(np.prod(shape))

    def operator(x_flat):
        return _apply_operator(x_flat.reshape(shape), cache, params, term_cfg).reshape(
            size
        )

    basis = np.eye(size, dtype=np.complex128)
    dense = np.stack([np.asarray(operator(jnp.asarray(col))) for col in basis], axis=1)
    spectrum = np.linalg.eigvals(dense)
    reference = complex(spectrum[np.argmax(spectrum.real)])

    # The shift is a CONTROLLED perturbation of the true eigenvalue, and the
    # offset is the experiment's independent variable.
    #
    # Two rejected alternatives, both measured. Using the exact eigenvalue makes
    # A - sigma I singular by construction: unpreconditioned GMRES stalls on it,
    # but a working preconditioner inverts the near-null direction and returns
    # NaN. Taking sigma from a genuinely coarser rung is what production
    # continuation does, but at the sizes that fit a dense reference there is no
    # room on the ladder -- coarsening (2,4) to (1,2) moved the eigenvalue 6.5
    # magnitudes, so every solver correctly converged to a different eigenvalue
    # and the run measured the shift rather than the solver.
    #
    # A stated offset separates the two questions: how good must a continuation
    # shift be, and which inner solver is best at a given shift quality.
    direction = np.exp(1j * 0.7)  # fixed, so runs are comparable
    sigma = reference + shift_offset * abs(reference) * direction

    seed = jnp.zeros(shape, dtype=jnp.complex128)
    # "hermite-line", not "auto": _build_shift_invert_precond matches its mode
    # against a whitelist and returns (None, None) for anything else, so a name
    # it does not know silently disables preconditioning instead of failing.
    # This benchmark spent a full round measuring an unpreconditioned solve while
    # reporting that the physics-aware inverse was on; assert it is active.
    _precond, precond_op = _build_shift_invert_precond(
        seed,
        cache,
        params,
        term_cfg,
        jnp.asarray(sigma, dtype=jnp.complex128),
        preconditioner,
    )
    if precond_op is None:
        raise RuntimeError(
            f"preconditioner {preconditioner!r} resolved to None -- GKX returns "
            "None for an unrecognised mode rather than raising, so this is "
            "almost certainly a name it does not know"
        )

    def shifted(x_flat):
        return operator(x_flat) - jnp.asarray(sigma, dtype=x_flat.dtype) * x_flat

    return {
        "shifted": shifted,
        "precond": precond_op,
        "size": size,
        "sigma": sigma,
        "spectral_radius": float(np.abs(spectrum).max()),
        "dense_rightmost": [float(reference.real), float(reference.imag)],
        "shift": [float(sigma.real), float(sigma.imag)],
        "shift_relative_offset": shift_offset,
    }


def arnoldi_with(
    case: dict[str, Any],
    solve: Callable,
    *,
    krylov_dim: int,
    restarts: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run shift-invert Arnoldi, counting every operator application.

    The right-hand sides are the genuine Arnoldi sequence rather than random
    vectors, because recycling is only useful to the extent that consecutive
    right-hand sides are related -- feeding it independent noise would understate
    it, and feeding it a repeated vector would overstate it.
    """

    import jax.numpy as jnp

    counter = {"matvecs": 0}

    def counted(x):
        counter["matvecs"] += 1
        return case["shifted"](x)

    generator = np.random.default_rng(seed)
    size = case["size"]
    v = generator.standard_normal(size) + 1j * generator.standard_normal(size)
    v = jnp.asarray(v / np.linalg.norm(v))

    recycle = None
    eigenvalue = float("nan")
    started = time.time()
    for _ in range(restarts):
        basis = [v]
        hessenberg = np.zeros((krylov_dim + 1, krylov_dim), dtype=np.complex128)
        for j in range(krylov_dim):
            w, recycle = solve(counted, case["precond"], basis[j], recycle)
            w = np.asarray(w)
            for i, bi in enumerate(basis):  # modified Gram-Schmidt
                hessenberg[i, j] = np.vdot(np.asarray(bi), w)
                w = w - hessenberg[i, j] * np.asarray(bi)
            hessenberg[j + 1, j] = np.linalg.norm(w)
            if hessenberg[j + 1, j] < 1e-14:
                break
            basis.append(jnp.asarray(w / hessenberg[j + 1, j]))
        square = hessenberg[: len(basis) - 1, : len(basis) - 1]
        theta, vectors = np.linalg.eig(square)
        # theta approximates 1/(lambda - sigma), so the eigenvalue nearest the
        # shift is the one of LARGEST |theta|. Selecting on max Re lambda
        # instead lets an inaccurate Ritz value with small |theta| map to a
        # spurious lambda far to the right of the true rightmost eigenvalue --
        # which is precisely the failure the exact-LU control exposed.
        pick = int(np.argmax(np.abs(theta)))
        lambdas = case["sigma"] + 1.0 / theta
        eigenvalue = complex(lambdas[pick])
        combination = (
            np.stack([np.asarray(b) for b in basis[: len(basis) - 1]], axis=1)
            @ vectors[:, pick]
        )
        v = jnp.asarray(combination / np.linalg.norm(combination))

    seconds = time.time() - started
    ritz = np.asarray(v)
    residual = float(
        np.linalg.norm(np.asarray(case["shifted"](jnp.asarray(ritz))) + 0.0)
    )
    del residual  # the accepted metric is agreement with the dense reference
    return {
        "matvecs": counter["matvecs"],
        "seconds": seconds,
        "eigenvalue": [float(eigenvalue.real), float(eigenvalue.imag)],
    }


def solver_variants(tol: float, maxiter: int, restart: int, recycle_k: int) -> dict:
    """The incumbent and the candidates, called identically.

    The incumbent is ``jax.scipy.sparse.linalg.gmres``: despite GKX depending on
    SOLVAX, this hot path never used it. So the comparison has three rungs, not
    two -- changing library and adding recycling are separate questions, and
    reporting them together would attribute any win to whichever we happened to
    prefer.
    """

    import jax.numpy as jnp
    import solvax
    from jax.scipy.sparse.linalg import gmres as jax_gmres

    _control_cache: dict[str, Any] = {}

    def exact(matvec, _precond, b, _recycle):
        """Dense LU of the shifted operator: a harness control, not a candidate.

        If the outer Arnoldi cannot reach the dense reference even when every
        inner solve is exact, then a failure by the iterative variants says
        nothing about those variants. This rung must converge, or no other
        number in the table is interpretable.
        """

        if "lu" not in _control_cache:
            size = int(b.shape[0])
            _control_cache["lu"] = np.stack(
                [
                    np.asarray(matvec(jnp.asarray(col)))
                    for col in np.eye(size, dtype=complex)
                ],
                axis=1,
            )
        return jnp.asarray(np.linalg.solve(_control_cache["lu"], np.asarray(b))), None

    def incumbent(matvec, precond, b, _recycle):
        x0 = precond(b) if precond is not None else b
        solution, _info = jax_gmres(
            matvec, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, M=precond
        )
        return solution, None

    def solvax_plain(matvec, precond, b, _recycle):
        solution = solvax.gmres(
            matvec,
            b,
            x0=precond(b) if precond is not None else None,
            precond=precond,
            restart=restart,
            rtol=tol,
            max_restarts=max(1, maxiter // restart),
        )
        return solution.x, None

    def recycling(strategy):
        def solve(matvec, precond, b, recycle):
            solution = solvax.gcrot(
                matvec,
                b,
                x0=precond(b) if precond is not None else None,
                precond=precond,
                m=restart,
                k=recycle_k,
                rtol=tol,
                max_restarts=max(1, maxiter // restart),
                recycle=recycle,
                recycle_strategy=strategy,
            )
            return solution.x, solution.recycle

        return solve

    return {
        "exact-lu (control)": exact,
        "jax-gmres": incumbent,
        "solvax-gmres": solvax_plain,
        "gcrot-fifo": recycling("fifo"),
        "gcrot-harmonic": recycling("harmonic"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-laguerre", type=int, default=4)
    parser.add_argument("--n-hermite", type=int, default=6)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--recycle-k", type=int, default=8)
    parser.add_argument("--preconditioner", default="hermite-line")
    parser.add_argument(
        "--shift-offset",
        type=float,
        default=1.0e-2,
        help="relative distance of the shift from the true eigenvalue, standing "
        "in for the accuracy of a production continuation shift",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    print(
        f"building case Nl={args.n_laguerre} Nm={args.n_hermite} nz={args.nz} "
        f"precond={args.preconditioner}"
    )
    case = build_case(
        args.n_laguerre,
        args.n_hermite,
        args.nz,
        preconditioner=args.preconditioner,
        shift_offset=args.shift_offset,
    )
    reference = complex(*case["dense_rightmost"])
    print(
        f"  n={case['size']}  sigma offset={case['shift_relative_offset']:.1e}  "
        f"spectral radius={case['spectral_radius']:.4g}  "
        f"ratio={case['spectral_radius'] / max(abs(reference), 1e-30):.0f}\n"
    )

    rows = []
    for name, solve in solver_variants(
        args.tol, args.maxiter, args.restart, args.recycle_k
    ).items():
        result = arnoldi_with(
            case, solve, krylov_dim=args.krylov_dim, restarts=args.restarts
        )
        value = complex(*result["eigenvalue"])
        result["name"] = name
        result["relative_error"] = abs(value - reference) / max(abs(reference), 1e-30)
        rows.append(result)
        print(
            f"  {name:16s} matvecs={result['matvecs']:>7d}  "
            f"{result['seconds']:>7.2f}s  rel.err={result['relative_error']:.2e}"
        )

    baseline = next(r for r in rows if r["name"] == "jax-gmres")
    print("\nversus the incumbent, in matrix-vector products:")
    for row in rows:
        if row["name"] == "jax-gmres":
            continue
        ratio = baseline["matvecs"] / max(row["matvecs"], 1)
        verdict = "cheaper" if ratio > 1.0 else "MORE EXPENSIVE"
        print(f"  {row['name']:16s} {ratio:5.2f}x  {verdict}")

    summary = {
        "kind": "shift_invert_recycling",
        "claim_level": "matvec_accounting_on_the_production_operator",
        "size": case["size"],
        "sigma": case["dense_rightmost"],
        "spectral_radius": case["spectral_radius"],
        "preconditioner": args.preconditioner,
        "krylov_dim": args.krylov_dim,
        "restarts": args.restarts,
        "results": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
