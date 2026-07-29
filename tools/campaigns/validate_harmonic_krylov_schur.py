"""Gate V1/V8: harmonic Krylov-Schur against the dense reference.

Runs both solvers on the *same* GKX operator, constructed through
``_solver_geometry_context`` so there is no possibility of comparing different
problems. The dense eigendecomposition is exact at these sizes, so it is the
reference rather than another approximation.

Also reports the spectral ratio (spectral radius over the wanted eigenvalue's
magnitude), which is what makes an interior eigenproblem hard, and the
matrix-vector count, which is the honest cost measure for a matrix-free solve.

By default each rung uses the dense dominant eigenvalue as its target, isolating
eigensolver correctness from mode-tracking correctness. ``--target-mode
continuation`` runs the operational continuation audit instead and seeds each
rung only from a previously converged solve.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_LADDER = ((4, 6), (6, 8), (8, 10), (10, 14))


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _initial_vector(
    shape: tuple[int, ...],
    *,
    previous: jax.Array | None,
    seed: int,
) -> tuple[jax.Array, bool]:
    """Return a deterministic random or velocity-space-prolonged start."""

    generator = np.random.default_rng(seed)
    noise = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    if previous is None:
        return jnp.asarray(noise), False

    prolonged = np.zeros(shape, dtype=complex)
    previous_host = np.asarray(previous)
    common = tuple(
        slice(0, min(old_extent, new_extent))
        for old_extent, new_extent in zip(previous_host.shape, shape, strict=True)
    )
    prolonged[common] = previous_host[common]
    noise /= np.linalg.norm(noise)
    prolonged += 1e-6 * noise
    return jnp.asarray(prolonged), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--krylov-dim", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-restarts", type=int, default=400)
    parser.add_argument(
        "--target-mode",
        choices=("oracle", "continuation"),
        default="oracle",
        help=(
            "oracle uses each rung's dense dominant eigenvalue to isolate solver "
            "correctness; continuation uses the previous converged Krylov value "
            "and audits branch tracking"
        ),
    )
    parser.add_argument(
        "--start-mode",
        choices=("random", "continuation"),
        default="random",
        help="continuation prolongs the previous converged eigenvector in velocity space",
    )
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

    input_path = args.input.resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        input_label = str(input_path.relative_to(repository))
    except ValueError:
        input_label = str(input_path)
    rows: list[dict] = []
    seed: complex | None = None
    previous_vector: jax.Array | None = None
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
        start, recycled_start = _initial_vector(
            context.state_shape,
            previous=previous_vector if args.start_mode == "continuation" else None,
            seed=0,
        )
        apply(start).block_until_ready()  # exclude compilation from the timing

        started = time.time()
        target = (
            complex(reference) if args.target_mode == "oracle" or seed is None else seed
        )
        solution = harmonic_krylov_schur(
            apply,
            start,
            sigma=target,
            k=1,
            m=args.krylov_dim,
            tol=args.tol,
            max_restarts=args.max_restarts,
            which="target",
        )
        krylov_seconds = time.time() - started
        value = complex(solution.eigenvalues[0])
        residual = float(solution.residuals[0])
        if bool(solution.converged[0]):
            seed = value
        error = abs(value - reference) / abs(reference)
        recycle_residual_limit = 100.0 * args.tol
        recycle_eligible = np.isfinite(residual) and residual <= recycle_residual_limit
        if recycle_eligible:
            previous_vector = solution.eigenvectors[0]

        rows.append(
            {
                "n_laguerre": n_laguerre,
                "n_hermite": n_hermite,
                "n": int(matrix.shape[0]),
                "spectral_ratio": radius / abs(reference),
                "dense_seconds": dense_seconds,
                "krylov_seconds": krylov_seconds,
                "dense": [reference.real, reference.imag],
                "target": [target.real, target.imag],
                "recycled_start": recycled_start,
                "krylov": [value.real, value.imag],
                "relative_error": error,
                "residual": residual,
                "recycle_eligible": recycle_eligible,
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

    passed = all(r["converged"] and r["relative_error"] < 1e-8 for r in rows)
    artifact = {
        "schema_version": 2,
        "passed": passed,
        "provenance": {
            "input": input_label,
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "s_index": args.s_index,
            "ntheta": args.ntheta,
            "selected_ky_index": 1,
            "ladder": [list(rung) for rung in _LADDER],
            "target_mode": args.target_mode,
            "start_mode": args.start_mode,
            "krylov_dim": args.krylov_dim,
            "tolerance": args.tol,
            "recycle_residual_limit": 100.0 * args.tol,
            "max_restarts": args.max_restarts,
            "python": sys.version,
            "platform": platform.platform(),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "gkx": _package_version("gkx"),
            "solvax": _package_version("solvax"),
            "gkx_commit": _git_revision(repository),
            "solvax_commit": _git_revision(
                Path(__import__("solvax").__file__).parents[2]
            ),
            "jax_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(device) for device in jax.devices()],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nV1 {'PASS' if passed else 'FAIL'}: written to {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
