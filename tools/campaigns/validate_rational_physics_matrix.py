"""Audit the rational eigensolver across branch, field, and geometry families.

Each row loads a shipped TOML case, preserves its species, field, normalization,
and geometry path, and reduces only spatial/velocity resolution so a dense
reference remains affordable. The rational candidate is certified with the
original GKX operator residual. Stable configurations are valid rows: the
solver must return the rightmost damped mode rather than invent instability.

This is an architecture/branch-selection gate, not a convergence study. The
full velocity-space ladder remains the responsibility of
``validate_harmonic_krylov_schur.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_CASES = (
    (
        "itg-circular",
        "ITG",
        Path("examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml"),
    ),
    (
        "etg-linked",
        "ETG",
        Path("examples/linear/axisymmetric/runtime_etg.toml"),
    ),
    (
        "tem-electromagnetic-linked",
        "TEM",
        Path("examples/linear/axisymmetric/runtime_tem.toml"),
    ),
    (
        "kbm-electromagnetic-linked",
        "KBM",
        Path("examples/linear/axisymmetric/runtime_kbm.toml"),
    ),
    (
        "itg-miller",
        "ITG",
        Path("examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml"),
    ),
    (
        "itg-qhs",
        "ITG",
        Path("examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml"),
    ),
    (
        "itg-qi",
        "ITG",
        Path(
            "examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml"
        ),
    ),
)


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
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


def _reduced_grid_config(grid_config, spatial_points: int):
    """Reduce a shipped grid without changing its boundary-condition family."""

    linked_points = spatial_points if grid_config.ntheta is not None else None
    linked_periods = 1 if grid_config.nperiod is not None else None
    return replace(
        grid_config,
        Nx=1,
        Nz=spatial_points,
        ntheta=linked_points,
        nperiod=linked_periods,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-points", type=int, default=8)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=4)
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--shift-offset", type=float, default=0.02)
    parser.add_argument("--shift-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case[0] for case in _CASES),
        default=None,
        help="run only the selected case; repeat for several cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/rational_eigensolver_physics_matrix.json"),
    )
    args = parser.parse_args()
    if args.spatial_points < 4:
        parser.error("--spatial-points must be at least 4")
    if args.n_laguerre < 1 or args.n_hermite < 2:
        parser.error("velocity-space dimensions must be positive")
    if args.candidates < 1 or args.krylov_dim <= args.candidates:
        parser.error("--krylov-dim must exceed the positive candidate count")
    if args.shift_offset <= 0.0:
        parser.error("--shift-offset must be positive")

    from gkx.core.grid import build_spectral_grid, select_ky_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import linear_terms_to_term_config
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )
    from gkx.solvers.linear.krylov import (
        prepare_rational_shifted_inverse,
        rational_eigenpairs,
    )
    from gkx.solvers.linear.krylov_algorithms import _apply_operator
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    repository = Path(__file__).resolve().parents[2]
    selected_names = set(args.case or ())
    selected_cases = tuple(
        case for case in _CASES if not selected_names or case[0] in selected_names
    )
    rows: list[dict] = []
    for case_index, (case_name, branch, relative_path) in enumerate(selected_cases):
        input_path = repository / relative_path
        runtime_config, raw_config = load_runtime_from_toml(input_path)
        runtime_config = replace(
            runtime_config,
            grid=_reduced_grid_config(
                runtime_config.grid,
                args.spatial_points,
            ),
        )
        geometry = build_runtime_geometry(runtime_config)
        full_grid = build_spectral_grid(runtime_config.grid)
        requested_ky = float(raw_config["run"]["ky"])
        ky_index = int(np.argmin(np.abs(np.asarray(full_grid.ky) - requested_ky)))
        grid = select_ky_grid(full_grid, ky_index)
        params = build_runtime_linear_params(
            runtime_config,
            Nm=args.n_hermite,
            geom=geometry,
        )
        terms = build_runtime_linear_terms(runtime_config)
        term_config = linear_terms_to_term_config(terms)
        cache = build_linear_cache(
            grid,
            geometry,
            params,
            Nl=args.n_laguerre,
            Nm=args.n_hermite,
        )
        state_shape = (
            len(runtime_config.species),
            args.n_laguerre,
            args.n_hermite,
            int(grid.ky.size),
            int(grid.kx.size),
            int(grid.z.size),
        )
        size = int(np.prod(state_shape))

        def apply(state: jax.Array) -> jax.Array:
            return _apply_operator(state, cache, params, term_config)

        basis = jnp.eye(size, dtype=jnp.complex128)
        started = time.time()
        matrix = np.asarray(
            jax.vmap(lambda vector: apply(vector.reshape(state_shape)).reshape(-1))(
                basis
            )
        ).T
        spectrum = np.linalg.eigvals(matrix)
        dense_seconds = time.time() - started
        reference = complex(spectrum[int(np.argmax(spectrum.real))])
        shift_scale = max(abs(reference), 1.0e-3)
        shift = reference - args.shift_offset * shift_scale
        generator = np.random.default_rng(case_index)
        start = jnp.asarray(
            generator.normal(size=state_shape) + 1j * generator.normal(size=state_shape)
        )
        inner_limit = size
        started = time.time()
        shifted_inverse = prepare_rational_shifted_inverse(
            start,
            cache,
            params,
            terms=terms,
            shift=shift,
            shift_tol=args.shift_tol,
            shift_maxiter=inner_limit,
            shift_restart=inner_limit,
            shift_preconditioner="field-corrected",
        )
        shifted_inverse(start).block_until_ready()
        compile_seconds = time.time() - started
        started = time.time()
        solution = rational_eigenpairs(
            start,
            cache,
            params,
            terms=terms,
            shift=shift,
            candidates=args.candidates,
            krylov_dim=args.krylov_dim,
            restarts=args.max_restarts,
            tol=args.tol,
            shift_tol=args.shift_tol,
            shift_maxiter=inner_limit,
            shift_restart=inner_limit,
            shift_preconditioner="field-corrected",
            shifted_inverse=shifted_inverse,
        )
        rational_seconds = time.time() - started
        values = np.asarray(solution.eigenvalues)
        selected = int(np.argmin(np.abs(values - reference)))
        value = complex(values[selected])
        residual = float(np.asarray(solution.residuals[selected]))
        relative_error = abs(value - reference) / max(
            abs(reference),
            np.finfo(float).tiny,
        )
        converged = bool(np.asarray(solution.converged[selected]))
        passed = converged and residual < args.tol and relative_error < 1.0e-8
        field_model = (
            "electromagnetic"
            if runtime_config.physics.electromagnetic
            else "electrostatic"
        )
        row = {
            "case": case_name,
            "branch": branch,
            "geometry_model": runtime_config.geometry.model,
            "boundary": runtime_config.grid.boundary,
            "field_model": field_model,
            "species": len(runtime_config.species),
            "input": str(relative_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "requested_ky": requested_ky,
            "actual_ky": float(np.asarray(grid.ky[0])),
            "state_shape": list(state_shape),
            "n": size,
            "dense": [reference.real, reference.imag],
            "rational": [value.real, value.imag],
            "relative_error": relative_error,
            "residual": residual,
            "converged": converged,
            "passed": passed,
            "compile_seconds": compile_seconds,
            "dense_seconds": dense_seconds,
            "rational_seconds": rational_seconds,
            "restarts": solution.restarts,
            "outer_applications": solution.matvecs,
        }
        rows.append(row)
        print(
            f"{case_name:>28} n={size:>4} {field_model:<15} "
            f"err={relative_error:.2e} residual={residual:.2e} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    passed = bool(rows) and all(row["passed"] for row in rows)
    artifact = {
        "schema_version": 1,
        "passed": passed,
        "scope": (
            "reduced-resolution architecture and branch-selection audit; "
            "not a velocity-space convergence claim"
        ),
        "provenance": {
            "spatial_points": args.spatial_points,
            "n_laguerre": args.n_laguerre,
            "n_hermite": args.n_hermite,
            "krylov_dim": args.krylov_dim,
            "candidates": args.candidates,
            "tolerance": args.tol,
            "shift_offset_fraction": args.shift_offset,
            "shift_tolerance": args.shift_tol,
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
    print(f"\nphysics matrix {'PASS' if passed else 'FAIL'}: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
