"""Where the campaign's GPU hours actually go, and the headroom available.

The linear growth rate is evaluated by ``_dominant_linear_branch``
(``gkx/objectives/core.py``), which builds the dense linear operator, calls
``lax_linalg.eig`` for **all** n eigenvalues and eigenvectors, and then keeps the
single one with the largest real part. The dense path is there for a reason --
it opts into ``enable_eigvec_derivs=True`` so the objective is differentiable --
but the cost is O(n^3) in time and O(n^2) in memory to obtain one eigenpair.

Measured on a QA boundary at ntheta = 32, so that n = n_laguerre * n_hermite *
ntheta:

    (N_l, N_m)      n      matrix     dense eig
      (2, 3)       192     0.6 MB       0.02 s
      (4, 6)       768     9.4 MB       0.92 s
      (6, 8)      1536    37.7 MB       4.24 s
      (8, 10)     2560   104.9 MB      14.36 s

Clean cubic scaling, which extrapolates badly to converged resolution. At
(12, 16) with ntheta = 64, n = 12288: about 26 minutes and 2.4 GB per
evaluation. At (16, 24), n = 24576: about 3.5 hours and 9.7 GB. A convergence
ladder needs several such evaluations per configuration, and the campaign needs
one ladder per configuration -- which is what makes a converged multi-device
study look unaffordable.

It is not actually unaffordable, because only one eigenpair is wanted. GKX
already ships a matrix-free Arnoldi (``gkx.solvers.linear.dominant_eigenpair``)
whose cost is O(n^2 k) with k Krylov vectors rather than O(n^3), and whose
memory is O(n k) rather than O(n^2). Measured here it is already 7.9x faster at
n = 2560, with the advantage growing as n^3 / (n^2 k) = n / k.

**Why the obvious substitution does not work, measured rather than guessed.**
The matrix-free RHS and the dense matrix agree to 3e-16 on the same vector, so
the operator is not in question. The spectrum is:

    max Re lambda    =   0.143     <- the wanted eigenvalue
    max |Im lambda|  =  80.15
    spectral radius  =  80.15

The target is an INTERIOR eigenvalue: its real part is ~560x smaller than the
spectral radius, which is dominated by fast oscillatory (large-|Im|) modes.
Plain Arnoldi converges to extremal |lambda| and therefore finds the |lambda| ~
80 modes, not this one -- measured values came back 30x to 300x too large and did
not improve with krylov_dim, which is the signature of converging to the wrong
part of the spectrum rather than of under-convergence.

**Superseded conclusion, kept because the measurements above still stand.**
This module originally argued that the propagator route was a dead end -- that
mu = exp(lambda dt) aliases once |Im lambda| dt approaches pi, and that a dt
small enough to avoid aliasing leaves a per-step separation exp(0.143 dt) that
converges too slowly to be useful. The premise is right and the conclusion was
wrong. Aliasing corrupts the *recovered* frequency, not the *selection*, because
|mu| = exp(Re lambda * T) does not wrap: a long horizon T separates modes by
growth alone, and the frequency is then recovered exactly from a Rayleigh
quotient against the original continuous operator rather than from log(mu). That
is the path ``gkx.solvers.linear.adaptive_propagator`` now takes, and it reaches
truncations the dense solve cannot represent at all.

Shift-invert was likewise reframed rather than dropped. It is not the cold
discovery method -- an unshifted rightmost sparse solve already exceeded 100 s at
n = 1536 -- but with a supplied or continuation shift it is the fast path, and
the inner solve became affordable only once the shift inverse exploited the
diagonal-plus-Hermite-streaming structure (FFT plus batched tridiagonal solve)
instead of a generic preconditioner.

See ``docs/differentiable_eigensolver.rst`` for the algorithm that replaced this
recommendation and the measurements that qualify it.

One note below survived intact:

* Gradients do not block the swap. The eigenvalue derivative of a simple
  eigenvalue is ``dlambda = (w^H dA v) / (w^H v)`` with w the left eigenvector,
  so a Krylov eigenpair carries an analytic implicit derivative instead of being
  differentiated through the iteration -- the same posture as the VMEC adjoint.
  Eigenvector-dependent observables need the bordered reduced-resolvent solve in
  addition, which is what ``solvax.eigenpair_reverse`` supplies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def measure(
    geometry,
    ladder: tuple[tuple[int, int], ...],
    *,
    ntheta: int,
    krylov: bool = True,
) -> list[dict]:
    import gkx
    import jax.numpy as jnp

    from gkx.config import GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import linear_params_for_geometry
    from gkx.solvers.linear import dominant_eigenpair

    rows = []
    for n_laguerre, n_hermite in ladder:
        started = time.time()
        matrix = np.asarray(
            gkx.solver_linear_operator_matrix_from_geometry(
                geometry, n_laguerre=n_laguerre, n_hermite=n_hermite
            )
        )
        build_seconds = time.time() - started

        started = time.time()
        eigenvalues = np.linalg.eigvals(matrix)
        dense_seconds = time.time() - started
        dense_value = float(eigenvalues.real.max())

        row = {
            "n_laguerre": n_laguerre,
            "n_hermite": n_hermite,
            "n": int(matrix.shape[0]),
            "matrix_megabytes": matrix.nbytes / 1e6,
            "build_seconds": build_seconds,
            "dense_eig_seconds": dense_seconds,
            "dense_max_real": dense_value,
        }

        if krylov:
            grid = build_spectral_grid(
                GridConfig(Nx=1, Ny=4, Nz=ntheta, Lx=6.0, Ly=12.0)
            )
            params = linear_params_for_geometry(geometry)
            cache = build_linear_cache(grid, geometry, params, n_laguerre, n_hermite)
            shape = (1, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
            generator = np.random.default_rng(0)
            seed = jnp.asarray(
                generator.normal(size=shape) + 1j * generator.normal(size=shape)
            )
            started = time.time()
            try:
                value, _ = dominant_eigenpair(
                    seed, cache, params, method="arnoldi", krylov_dim=32, restarts=3
                )
                row["krylov_seconds"] = time.time() - started
                row["krylov_real"] = float(jnp.real(value))
                row["speedup"] = dense_seconds / max(row["krylov_seconds"], 1e-9)
                row["relative_disagreement"] = abs(
                    row["krylov_real"] - dense_value
                ) / max(abs(dense_value), 1e-30)
            except Exception as err:
                row["krylov_error"] = f"{type(err).__name__}: {err}"[:160]

        rows.append(row)
        print(
            f"  (Nl,Nm)=({n_laguerre},{n_hermite})  n={row['n']:>6}  "
            f"dense {dense_seconds:>7.2f}s  "
            + (
                f"krylov {row.get('krylov_seconds', float('nan')):>6.2f}s  "
                f"speedup {row.get('speedup', float('nan')):>5.1f}x  "
                f"disagreement {row.get('relative_disagreement', float('nan')):.2e}"
                if krylov
                else ""
            ),
            flush=True,
        )
    return rows


def extrapolate(
    rows: list[dict], targets: tuple[tuple[int, int, int], ...]
) -> list[dict]:
    """Cubic extrapolation of the dense cost to converged resolutions."""

    sizes = np.array([r["n"] for r in rows], dtype=float)
    times = np.array([r["dense_eig_seconds"] for r in rows], dtype=float)
    usable = times > 0.05  # sub-0.05 s timings are dominated by dispatch overhead
    exponent, log_scale = np.polyfit(np.log(sizes[usable]), np.log(times[usable]), 1)
    scale = float(np.exp(log_scale))

    projections = []
    for n_laguerre, n_hermite, ntheta in targets:
        n = n_laguerre * n_hermite * ntheta
        projections.append(
            {
                "n_laguerre": n_laguerre,
                "n_hermite": n_hermite,
                "ntheta": ntheta,
                "n": n,
                "projected_dense_seconds": scale * n**exponent,
                "projected_matrix_gigabytes": (n**2 * 16) / 1e9,
            }
        )
    return [{"fitted_exponent": float(exponent), "fitted_scale": scale}] + projections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/eigensolver_cost_model.json"),
    )
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(args.input))
    geometry = turb.flux_tube_geometry(
        equilibrium.state,
        equilibrium.runtime,
        s_index=args.s_index,
        alpha=0.0,
        ntheta=args.ntheta,
    )

    print("measuring dense vs matrix-free cost", flush=True)
    rows = measure(geometry, ((2, 3), (4, 6), (6, 8), (8, 10)), ntheta=args.ntheta)
    projections = extrapolate(rows, ((12, 16, 64), (16, 24, 64), (32, 16, 64)))

    print(f"\nfitted dense scaling: t ~ n^{projections[0]['fitted_exponent']:.2f}")
    for item in projections[1:]:
        print(
            f"  ({item['n_laguerre']},{item['n_hermite']}) ntheta={item['ntheta']}: "
            f"n={item['n']}, {item['projected_dense_seconds'] / 60:.1f} min, "
            f"{item['projected_matrix_gigabytes']:.1f} GB per evaluation"
        )

    summary = {
        "kind": "eigensolver_cost_model",
        "claim_level": "cost_measurement_and_projection_not_a_validated_speedup",
        "ntheta": args.ntheta,
        "measurements": rows,
        "projections": projections,
        "note": (
            "the wanted eigenvalue is INTERIOR: max Re = 0.143 against a "
            "spectral radius of 80.15, so plain Arnoldi converges to the "
            "oscillatory extremes instead. Matrix-free and dense operators "
            "agree to 3e-16, so this is a spectrum-structure problem, not an "
            "operator mismatch. The method that resolved it is a long-horizon "
            "propagator Arnoldi with the frequency recovered from a Rayleigh "
            "quotient against the continuous operator, plus structure-aware "
            "shift-invert for continuation; see "
            "docs/differentiable_eigensolver.rst."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
