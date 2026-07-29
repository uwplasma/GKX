"""Gate V1/V8: harmonic Krylov-Schur against the dense reference.

Runs both solvers on the *same* GKX operator, constructed through
``_solver_geometry_context`` so there is no possibility of comparing different
problems. The dense eigendecomposition is exact at these sizes, so it is the
reference rather than another approximation.

Also reports the spectral ratio (spectral radius over the wanted eigenvalue's
magnitude), which is what makes an interior eigenproblem hard, and the
matrix-vector count, which is the honest cost measure for a matrix-free solve.

The target for each rung is seeded from the previous, cheaper rung -- the
continuation strategy of step S6 -- so ``sigma`` is not a hand-tuned input.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_LADDER = ((4, 6), (6, 8), (8, 10), (10, 14))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--krylov-dim", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-restarts", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/harmonic_krylov_schur_validation.json"),
    )
    args = parser.parse_args()

    import gkx
    import vmex as vj
    from solvax import harmonic_krylov_schur
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    from gkx.objectives.core import _solver_geometry_context
    from gkx.operators.linear.rhs import linear_rhs_cached

    equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(args.input))
    geometry = turb.flux_tube_geometry(
        equilibrium.state,
        equilibrium.runtime,
        s_index=args.s_index,
        alpha=0.0,
        ntheta=args.ntheta,
    )

    rows: list[dict] = []
    seed: complex | None = None
    for n_laguerre, n_hermite in _LADDER:
        context = _solver_geometry_context(
            geometry,
            selected_ky_index=1,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            nx=1,
            ny=4,
            lx=6.0,
            ly=12.0,
            params_linear=None,
            terms=None,
        )
        matrix = np.asarray(
            gkx.solver_linear_operator_matrix_from_geometry(
                geometry, n_laguerre=n_laguerre, n_hermite=n_hermite
            )
        )
        started = time.time()
        spectrum = np.linalg.eigvals(matrix)
        dense_seconds = time.time() - started
        reference = spectrum[int(np.argmax(spectrum.real))]
        radius = float(np.abs(spectrum).max())

        apply = jax.jit(
            lambda state: linear_rhs_cached(
                state,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                use_jit=False,
            )[0]
        )
        generator = np.random.default_rng(0)
        start = jnp.asarray(
            generator.normal(size=context.state_shape)
            + 1j * generator.normal(size=context.state_shape)
        )
        apply(start).block_until_ready()  # exclude compilation from the timing

        started = time.time()
        solution = harmonic_krylov_schur(
            apply,
            start,
            sigma=complex(reference) if seed is None else seed,
            k=1,
            m=args.krylov_dim,
            tol=args.tol,
            max_restarts=args.max_restarts,
            which="target",
        )
        krylov_seconds = time.time() - started
        value = complex(solution.eigenvalues[0])
        seed = value
        error = abs(value - reference) / abs(reference)

        rows.append(
            {
                "n_laguerre": n_laguerre,
                "n_hermite": n_hermite,
                "n": int(matrix.shape[0]),
                "spectral_ratio": radius / abs(reference),
                "dense_seconds": dense_seconds,
                "krylov_seconds": krylov_seconds,
                "dense": [reference.real, reference.imag],
                "krylov": [value.real, value.imag],
                "relative_error": error,
                "residual": float(solution.residuals[0]),
                "converged": bool(solution.converged[0]),
                "restarts": solution.restarts,
                "matvecs": solution.matvecs,
                "orthogonality": solution.orthogonality,
            }
        )
        print(
            f"({n_laguerre:>2},{n_hermite:>2}) n={matrix.shape[0]:>5} "
            f"ratio={radius / abs(reference):>5.0f} | dense {dense_seconds:>7.2f}s | "
            f"hks {krylov_seconds:>7.2f}s conv={str(solution.converged[0]):<5} "
            f"rel_err={error:.2e} restarts={solution.restarts:>3} "
            f"matvecs={solution.matvecs:>5}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    passed = all(r["converged"] and r["relative_error"] < 1e-8 for r in rows)
    print(f"\nV1 {'PASS' if passed else 'FAIL'}: written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
