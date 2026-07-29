"""Gate V1/V8: matrix-free eigensolvers against the dense reference.

Runs both solvers on the *same* GKX operator, constructed through
``_solver_geometry_context`` so there is no possibility of comparing different
problems. The dense eigendecomposition is exact at these sizes, so it is the
reference rather than another approximation.

The harmonic solver reports its matrix-vector count. The rational solver reports
outer operator/subspace applications separately from its inner GMRES controls;
wall time is the comparable end-to-end cost until inner operator evaluations are
exposed directly by the shifted-solve API. The long-horizon propagator reports
its IMEX2 step count and certifies the returned Rayleigh pair against the
original continuous-time operator.

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


def _package_git_revision(name: str) -> str | None:
    """Return a revision only for a source checkout, never an enclosing venv."""

    module_path = Path(__import__(name).__file__).resolve()
    candidate = module_path.parents[2]
    if not (candidate / "pyproject.toml").is_file():
        return None
    if not (candidate / "src" / name).is_dir():
        return None
    return _git_revision(candidate)


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
    parser.add_argument(
        "--solver",
        choices=(
            "harmonic",
            "block-rational",
            "long-horizon",
            "adaptive-propagator",
            "block-propagator",
        ),
        default="harmonic",
    )
    rung_group = parser.add_mutually_exclusive_group()
    rung_group.add_argument(
        "--max-rungs",
        type=int,
        default=len(_LADDER),
        help="run only the first N velocity-space rungs",
    )
    rung_group.add_argument(
        "--rung-index",
        type=int,
        default=None,
        help="run only one 1-based velocity-space rung (oracle audits only)",
    )
    parser.add_argument("--krylov-dim", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-restarts", type=int, default=400)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--shift-offset",
        type=float,
        default=0.02,
        help=(
            "subtract this fraction of max(abs(target), 1e-3) from Re(target) "
            "for block-rational; a nonzero offset avoids a singular oracle shift"
        ),
    )
    parser.add_argument("--shift-tol", type=float, default=1e-7)
    parser.add_argument("--shift-maxiter", type=int, default=640)
    parser.add_argument("--shift-restart", type=int, default=320)
    parser.add_argument("--propagator-dt", type=float, default=1.0e-3)
    parser.add_argument("--propagator-steps", type=int, default=5000)
    parser.add_argument("--propagator-chunk-horizon", type=float, default=30.0)
    parser.add_argument("--stability-dimension", type=int, default=12)
    parser.add_argument("--stability-safety", type=float, default=0.9)
    parser.add_argument("--max-stability-retries", type=int, default=2)
    parser.add_argument(
        "--shift-solve-method",
        choices=("flexible", "batched", "incremental"),
        default="flexible",
        help="compatibility selector; shifted solves use right-preconditioned FGMRES",
    )
    parser.add_argument(
        "--shift-preconditioner",
        choices=(
            "field-corrected",
            "hermite-line",
            "damping",
            "none",
        ),
        default="field-corrected",
    )
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
        default=None,
    )
    args = parser.parse_args()
    if not 1 <= args.max_rungs <= len(_LADDER):
        parser.error(f"--max-rungs must lie in [1, {len(_LADDER)}]")
    if args.rung_index is not None and not 1 <= args.rung_index <= len(_LADDER):
        parser.error(f"--rung-index must lie in [1, {len(_LADDER)}]")
    if args.rung_index is not None and (
        args.target_mode == "continuation" or args.start_mode == "continuation"
    ):
        parser.error("--rung-index requires oracle targeting and a random start")
    if args.solver == "block-rational":
        if args.candidates < 1:
            parser.error("--candidates must be positive")
        if args.krylov_dim <= args.candidates:
            parser.error("--krylov-dim must exceed --candidates")
        if args.shift_offset <= 0.0:
            parser.error("--shift-offset must be positive for block-rational")
    if args.solver in {"long-horizon", "block-propagator"}:
        if args.propagator_dt <= 0.0:
            parser.error("--propagator-dt must be positive")
        if args.propagator_steps < 2:
            parser.error("--propagator-steps must be at least two")
    if args.solver == "adaptive-propagator":
        if args.propagator_chunk_horizon <= 0.0:
            parser.error("--propagator-chunk-horizon must be positive")
        if args.stability_dimension < 2:
            parser.error("--stability-dimension must be at least two")
        if not 0.0 < args.stability_safety < 1.0:
            parser.error("--stability-safety must lie in (0, 1)")
        if args.max_stability_retries < 0:
            parser.error("--max-stability-retries must be non-negative")
    if args.solver == "block-propagator":
        if args.candidates < 1:
            parser.error("--candidates must be positive")
        if args.krylov_dim <= args.candidates:
            parser.error("--krylov-dim must exceed --candidates")
    if args.output is None:
        stem = {
            "harmonic": "harmonic_krylov_schur_validation.json",
            "block-rational": "block_rational_eigensolver_validation.json",
            "long-horizon": "long_horizon_propagator_validation.json",
            "adaptive-propagator": "adaptive_propagator_validation.json",
            "block-propagator": "block_propagator_eigensolver_validation.json",
        }[args.solver]
        args.output = Path("docs/_static") / stem

    import gkx
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    if args.solver == "harmonic":
        from solvax import harmonic_krylov_schur

    from gkx.objectives.core import _solver_geometry_context
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.krylov import (
        adaptive_propagator_eigenpair,
        dominant_eigenpair,
        prepare_long_horizon_propagator,
        prepare_rational_shifted_inverse,
        propagator_eigenpairs,
        rational_eigenpairs,
    )

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
    ladder = (
        (_LADDER[args.rung_index - 1],)
        if args.rung_index is not None
        else _LADDER[: args.max_rungs]
    )
    for n_laguerre, n_hermite in ladder:
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

        target = (
            complex(reference) if args.target_mode == "oracle" or seed is None else seed
        )
        compile_seconds = 0.0
        if args.solver == "harmonic":
            solver_target = target
            started = time.time()
            solution = harmonic_krylov_schur(
                apply,
                start,
                sigma=solver_target,
                k=1,
                m=args.krylov_dim,
                tol=args.tol,
                max_restarts=args.max_restarts,
                which="target",
            )
        elif args.solver == "block-rational":
            # Scale the nonsingular oracle displacement to the eigenvalue. An
            # O(1) floor moves weakly damped stellarator branches past nearby
            # modes even though the intended shift is only a singularity guard.
            shift_scale = max(abs(target), 1.0e-3)
            solver_target = target - args.shift_offset * shift_scale
            preconditioner = (
                None
                if args.shift_preconditioner == "none"
                else args.shift_preconditioner
            )
            started = time.time()
            shifted_inverse = prepare_rational_shifted_inverse(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                shift=solver_target,
                shift_tol=args.shift_tol,
                shift_maxiter=args.shift_maxiter,
                shift_restart=args.shift_restart,
                shift_solve_method=args.shift_solve_method,
                shift_preconditioner=preconditioner,
            )
            shifted_inverse(start).block_until_ready()
            compile_seconds = time.time() - started
            started = time.time()
            solution = rational_eigenpairs(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                shift=solver_target,
                candidates=args.candidates,
                krylov_dim=args.krylov_dim,
                restarts=args.max_restarts,
                tol=args.tol,
                shift_tol=args.shift_tol,
                shift_maxiter=args.shift_maxiter,
                shift_restart=args.shift_restart,
                shift_solve_method=args.shift_solve_method,
                shift_preconditioner=preconditioner,
                shifted_inverse=shifted_inverse,
            )
        elif args.solver == "adaptive-propagator":
            solver_target = None
            started = time.time()
            compiled_solution = adaptive_propagator_eigenpair(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                krylov_dim=args.krylov_dim,
                max_restarts=args.max_restarts,
                tol=args.tol,
                chunk_horizon=args.propagator_chunk_horizon,
                stability_dimension=args.stability_dimension,
                stability_safety=args.stability_safety,
                max_stability_retries=args.max_stability_retries,
            )
            jax.tree.map(
                lambda item: item.block_until_ready()
                if hasattr(item, "block_until_ready")
                else item,
                compiled_solution,
            )
            compile_seconds = time.time() - started
            started = time.time()
            solution = adaptive_propagator_eigenpair(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                krylov_dim=args.krylov_dim,
                max_restarts=args.max_restarts,
                tol=args.tol,
                chunk_horizon=args.propagator_chunk_horizon,
                stability_dimension=args.stability_dimension,
                stability_safety=args.stability_safety,
                max_stability_retries=args.max_stability_retries,
            )
            solution.eigenvalue.block_until_ready()
            solution.eigenvector.block_until_ready()
        elif args.solver == "long-horizon":
            solver_target = None
            started = time.time()
            compiled_value, compiled_vector = dominant_eigenpair(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                method="propagator",
                krylov_dim=args.krylov_dim,
                restarts=args.max_restarts,
                power_dt=args.propagator_dt,
                propagator_steps=args.propagator_steps,
            )
            compiled_value.block_until_ready()
            compiled_vector.block_until_ready()
            compile_seconds = time.time() - started
            started = time.time()
            propagator_value, propagator_vector = dominant_eigenpair(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                method="propagator",
                krylov_dim=args.krylov_dim,
                restarts=args.max_restarts,
                power_dt=args.propagator_dt,
                propagator_steps=args.propagator_steps,
            )
            propagator_value.block_until_ready()
            propagator_vector.block_until_ready()
        else:
            solver_target = None
            started = time.time()
            propagator = prepare_long_horizon_propagator(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                dt=args.propagator_dt,
                steps=args.propagator_steps,
            )
            propagator(start).block_until_ready()
            compile_seconds = time.time() - started
            started = time.time()
            solution = propagator_eigenpairs(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                candidates=args.candidates,
                krylov_dim=args.krylov_dim,
                restarts=args.max_restarts,
                tol=args.tol,
                dt=args.propagator_dt,
                steps=args.propagator_steps,
                propagator=propagator,
            )
        krylov_seconds = time.time() - started
        if args.solver == "adaptive-propagator":
            value = complex(np.asarray(solution.eigenvalue))
            vector = solution.eigenvector
            residual = float(np.asarray(solution.residual))
            error = abs(value - reference) / abs(reference)
            converged = bool(solution.converged)
            selected = 0
            candidate_values = np.asarray([value])
            candidate_vectors = np.asarray([vector])
            candidate_residuals = np.asarray([residual])
            candidate_converged = np.asarray([converged])
        elif args.solver == "long-horizon":
            value = complex(np.asarray(propagator_value))
            vector = propagator_vector
            applied = apply(vector)
            residual = float(
                np.asarray(jnp.linalg.norm(applied - propagator_value * vector))
                / max(
                    abs(value) * float(np.asarray(jnp.linalg.norm(vector))),
                    np.finfo(float).tiny,
                )
            )
            error = abs(value - reference) / abs(reference)
            converged = residual < args.tol
            selected = 0
            candidate_values = np.asarray([value])
            candidate_vectors = np.asarray([vector])
            candidate_residuals = np.asarray([residual])
            candidate_converged = np.asarray([converged])
        else:
            candidate_values = np.asarray(solution.eigenvalues)
            candidate_vectors = np.asarray(solution.eigenvectors)
            candidate_residuals = np.asarray(solution.residuals)
            candidate_converged = np.asarray(solution.converged)
            if args.target_mode == "oracle":
                selected = int(np.nanargmin(np.abs(candidate_values - reference)))
            else:
                flattened_start = np.asarray(start).reshape(-1)
                flattened_candidates = candidate_vectors.reshape(
                    candidate_vectors.shape[0], -1
                )
                overlaps = np.abs(flattened_candidates.conj() @ flattened_start)
                selected = int(np.nanargmax(overlaps))
            value = complex(candidate_values[selected])
            residual = float(candidate_residuals[selected])
            converged = bool(candidate_converged[selected])
            error = abs(value - reference) / abs(reference)
        if converged:
            seed = value
        recycle_residual_limit = 100.0 * args.tol
        recycle_eligible = np.isfinite(residual) and residual <= recycle_residual_limit
        if recycle_eligible:
            previous_vector = jnp.asarray(candidate_vectors[selected])

        rows.append(
            {
                "n_laguerre": n_laguerre,
                "n_hermite": n_hermite,
                "n": int(matrix.shape[0]),
                "spectral_ratio": radius / abs(reference),
                "dense_seconds": dense_seconds,
                "compile_seconds": compile_seconds,
                "krylov_seconds": krylov_seconds,
                "dense": [reference.real, reference.imag],
                "target": [target.real, target.imag],
                "solver_target": (
                    None
                    if solver_target is None
                    else [solver_target.real, solver_target.imag]
                ),
                "recycled_start": recycled_start,
                "krylov": [value.real, value.imag],
                "selected_candidate": selected,
                "candidates": [
                    {
                        "eigenvalue": [
                            complex(candidate).real,
                            complex(candidate).imag,
                        ],
                        "residual": float(candidate_residuals[index]),
                        "converged": bool(candidate_converged[index]),
                    }
                    for index, candidate in enumerate(candidate_values)
                ],
                "relative_error": error,
                "residual": residual,
                "recycle_eligible": recycle_eligible,
                "converged": converged,
                "restarts": (
                    args.max_restarts
                    if args.solver == "long-horizon"
                    else solution.restarts
                ),
                "outer_applications": (
                    args.max_restarts * args.krylov_dim
                    if args.solver == "long-horizon"
                    else (
                        solution.restarts * args.krylov_dim
                        + args.stability_dimension
                        if args.solver == "adaptive-propagator"
                        else solution.matvecs
                    )
                ),
                "propagator_steps": (
                    args.max_restarts * args.krylov_dim * args.propagator_steps
                    if args.solver == "long-horizon"
                    else (
                        solution.restarts
                        * args.krylov_dim
                        * solution.filter_steps
                        if args.solver == "adaptive-propagator"
                        else None
                    )
                ),
                "original_operator_evaluations": (
                    4 * args.max_restarts * args.krylov_dim * args.propagator_steps + 1
                    if args.solver == "long-horizon"
                    else (
                        solution.operator_applications
                        if args.solver == "adaptive-propagator"
                        else None
                    )
                ),
                "selected_propagator_dt": (
                    solution.filter_dt
                    if args.solver == "adaptive-propagator"
                    else None
                ),
                "selected_propagator_steps": (
                    solution.filter_steps
                    if args.solver == "adaptive-propagator"
                    else None
                ),
                "filter_growth_defect": (
                    solution.filter_growth_defect
                    if args.solver == "adaptive-propagator"
                    else None
                ),
                "stability_passed": (
                    solution.stable
                    if args.solver == "adaptive-propagator"
                    else None
                ),
                "propagator_substeps_upper_bound": (
                    solution.matvecs * args.propagator_steps
                    if args.solver == "block-propagator"
                    else None
                ),
                "orthogonality": (
                    None
                    if args.solver in {"long-horizon", "adaptive-propagator"}
                    else solution.orthogonality
                ),
            }
        )
        print(
            f"({n_laguerre:>2},{n_hermite:>2}) n={matrix.shape[0]:>5} "
            f"ratio={radius / abs(reference):>5.0f} | dense {dense_seconds:>7.2f}s | "
            f"compile {compile_seconds:>7.2f}s | "
            f"{args.solver} {krylov_seconds:>7.2f}s conv={str(converged):<5} "
            f"rel_err={error:.2e} residual={residual:.2e} "
            f"restarts={rows[-1]['restarts']:>3} "
            f"outer={rows[-1]['outer_applications']:>5}",
            flush=True,
        )

    passed = all(r["converged"] and r["relative_error"] < 1e-8 for r in rows)
    artifact = {
        "schema_version": 5,
        "passed": passed,
        "provenance": {
            "input": input_label,
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "s_index": args.s_index,
            "ntheta": args.ntheta,
            "selected_ky_index": 1,
            "ladder": [list(rung) for rung in ladder],
            "solver": args.solver,
            "target_mode": args.target_mode,
            "start_mode": args.start_mode,
            "krylov_dim": args.krylov_dim,
            "candidates": (
                args.candidates
                if args.solver in {"block-rational", "block-propagator"}
                else 1
            ),
            "tolerance": args.tol,
            "recycle_residual_limit": 100.0 * args.tol,
            "max_restarts": args.max_restarts,
            "propagator_chunk_horizon": (
                args.propagator_chunk_horizon
                if args.solver == "adaptive-propagator"
                else None
            ),
            "stability_dimension": (
                args.stability_dimension
                if args.solver == "adaptive-propagator"
                else None
            ),
            "stability_safety": (
                args.stability_safety
                if args.solver == "adaptive-propagator"
                else None
            ),
            "max_stability_retries": (
                args.max_stability_retries
                if args.solver == "adaptive-propagator"
                else None
            ),
            "shift_offset": (
                args.shift_offset if args.solver == "block-rational" else None
            ),
            "shift_tolerance": (
                args.shift_tol if args.solver == "block-rational" else None
            ),
            "shift_maxiter": (
                args.shift_maxiter if args.solver == "block-rational" else None
            ),
            "shift_restart": (
                args.shift_restart if args.solver == "block-rational" else None
            ),
            "shift_solve_method": (
                args.shift_solve_method if args.solver == "block-rational" else None
            ),
            "shift_preconditioner": (
                args.shift_preconditioner if args.solver == "block-rational" else None
            ),
            "propagator_dt": (
                args.propagator_dt
                if args.solver in {"long-horizon", "block-propagator"}
                else None
            ),
            "propagator_steps_per_application": (
                args.propagator_steps
                if args.solver in {"long-horizon", "block-propagator"}
                else None
            ),
            "propagator_horizon": (
                args.propagator_dt * args.propagator_steps
                if args.solver in {"long-horizon", "block-propagator"}
                else None
            ),
            "cost_note": (
                "outer_applications includes original-operator and shifted-inverse "
                "calls, but not GMRES-internal operator evaluations"
                if args.solver == "block-rational"
                else (
                    "outer_applications is the Arnoldi propagator count; "
                    "propagator_steps is the total number of RK4 substeps"
                    if args.solver == "long-horizon"
                    else (
                        "outer_applications includes the stability sketch and "
                        "residual-driven Arnoldi restarts; original_operator_"
                        "evaluations includes all RK4 stages and certifications"
                        if args.solver == "adaptive-propagator"
                        else (
                        "outer_applications counts original-operator and "
                        "propagator calls; propagator_substeps_upper_bound is "
                        "conservative because SOLVAX does not expose that split"
                        if args.solver == "block-propagator"
                        else "outer_applications is the matrix-vector count"
                        )
                    )
                )
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "gkx": _package_version("gkx"),
            "solvax": _package_version("solvax"),
            "gkx_commit": _git_revision(repository),
            "solvax_commit": _package_git_revision("solvax"),
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
