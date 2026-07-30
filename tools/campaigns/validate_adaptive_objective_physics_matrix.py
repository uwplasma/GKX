"""Gate adaptive objective AD across shipped branch and geometry families.

The seven TOML cases cover ITG, ETG, TEM, KBM, electrostatic and
electromagnetic fields, one and multiple kinetic species, circular and Miller
tokamaks, and linked QHS/QI stellarator grids. Each row perturbs all magnetic
drift profiles by one differentiable scalar and compares a phase-invariant
adaptive value and reverse gradient with the dense eigensolve on the exact same
reduced production operator.

This is an architecture and AD gate, not a velocity-space convergence claim.
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


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _reduced_grid(grid_config, spatial_points: int):
    return replace(
        grid_config,
        Nx=1,
        Nz=spatial_points,
        ntheta=spatial_points if grid_config.ntheta is not None else None,
        nperiod=1 if grid_config.nperiod is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-points", type=int, default=8)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=4)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case[0] for case in _CASES),
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/adaptive_objective_physics_matrix.json"),
    )
    args = parser.parse_args()
    if args.spatial_points < 4 or args.n_laguerre < 1 or args.n_hermite < 2:
        parser.error("spatial-points >= 4, n-laguerre >= 1, and n-hermite >= 2")

    from gkx.core.grid import build_spectral_grid, select_ky_grid
    from gkx.geometry.flux_tube import FluxTubeGeometryData, sample_flux_tube_geometry
    from gkx.objectives.core import solver_objective_vector_from_geometry
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    repository = Path(__file__).resolve().parents[2]
    selected = set(args.case or ())
    cases = tuple(case for case in _CASES if not selected or case[0] in selected)
    weights = jnp.asarray([1.0, 0.2, 0.1, 0.05, 0.0, 0.01], jnp.float64)
    rows: list[dict[str, object]] = []

    for name, branch, relative_path in cases:
        input_path = repository / relative_path
        runtime, raw = load_runtime_from_toml(input_path)
        runtime = replace(
            runtime,
            grid=_reduced_grid(runtime.grid, args.spatial_points),
        )
        runtime_geometry = build_runtime_geometry(runtime)
        full_grid = build_spectral_grid(runtime.grid)
        requested_ky = float(raw["run"]["ky"])
        ky_index = int(np.argmin(np.abs(np.asarray(full_grid.ky) - requested_ky)))
        grid = select_ky_grid(full_grid, ky_index)
        geometry = (
            runtime_geometry
            if isinstance(runtime_geometry, FluxTubeGeometryData)
            else sample_flux_tube_geometry(runtime_geometry, grid.z)
        )
        params = build_runtime_linear_params(
            runtime,
            Nm=args.n_hermite,
            geom=runtime_geometry,
        )
        terms = build_runtime_linear_terms(runtime)

        def objective(scale: jax.Array, method: str) -> jax.Array:
            perturbed = replace(
                geometry,
                cv_profile=geometry.cv_profile * (1.0 + scale),
                gb_profile=geometry.gb_profile * (1.0 + scale),
                cv0_profile=geometry.cv0_profile * (1.0 + scale),
                gb0_profile=geometry.gb0_profile * (1.0 + scale),
            )
            vector = solver_objective_vector_from_geometry(
                perturbed,
                spectral_grid=grid,
                n_laguerre=args.n_laguerre,
                n_hermite=args.n_hermite,
                params_linear=params,
                terms=terms,
                eigensolver=method,
            )
            return jnp.vdot(weights, vector)

        results = {}
        failure = None
        for method in ("dense", "adaptive-propagator"):
            started = time.perf_counter()
            try:
                value, gradient = jax.value_and_grad(
                    lambda scale, method=method: objective(scale, method)
                )(jnp.asarray(0.0, jnp.float64))
                value.block_until_ready()
                gradient.block_until_ready()
                results[method] = {
                    "value": float(value),
                    "gradient": float(gradient),
                    "seconds": time.perf_counter() - started,
                    "finite": bool(np.isfinite(value) and np.isfinite(gradient)),
                }
            except (RuntimeError, ValueError) as error:
                failure = f"{type(error).__name__}: {error}"
                results[method] = {
                    "value": None,
                    "gradient": None,
                    "seconds": time.perf_counter() - started,
                    "finite": False,
                }
                break

        dense = results["dense"]
        adaptive = results.get("adaptive-propagator", {})
        if failure is None:
            value_scale = max(abs(float(dense["value"])), np.finfo(float).tiny)
            gradient_scale = max(
                abs(float(dense["gradient"])),
                np.finfo(float).tiny,
            )
            value_error = (
                abs(float(adaptive["value"]) - float(dense["value"])) / value_scale
            )
            gradient_error = (
                abs(float(adaptive["gradient"]) - float(dense["gradient"]))
                / gradient_scale
            )
            passed = (
                bool(dense["finite"])
                and bool(adaptive["finite"])
                and value_error < 1.0e-8
                and gradient_error < 1.0e-6
            )
        else:
            value_error = None
            gradient_error = None
            passed = False
        species = int(np.asarray(params.density).size)
        row = {
            "case": name,
            "branch": branch,
            "field_model": (
                "electromagnetic"
                if runtime.physics.electromagnetic
                else "electrostatic"
            ),
            "geometry_model": runtime.geometry.model,
            "spectral_layout": (
                "linked"
                if runtime.grid.ntheta is not None or runtime.grid.nperiod is not None
                else "periodic"
            ),
            "species": species,
            "input": str(relative_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "state_size": (
                species
                * args.n_laguerre
                * args.n_hermite
                * int(grid.ky.size)
                * int(grid.kx.size)
                * int(grid.z.size)
            ),
            "dense": dense,
            "adaptive": adaptive,
            "value_relative_error": value_error,
            "gradient_relative_error": gradient_error,
            "failure": failure,
            "passed": passed,
        }
        rows.append(row)
        print(
            f"{name}: pass={passed} value_error={value_error} "
            f"gradient_error={gradient_error}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "scope": (
            "reduced-resolution branch/field/geometry matrix for value and "
            "reverse-AD architecture; not velocity-space convergence"
        ),
        "passed": all(bool(row["passed"]) for row in rows),
        "resolution": {
            "spatial_points": args.spatial_points,
            "n_laguerre": args.n_laguerre,
            "n_hermite": args.n_hermite,
        },
        "rows": rows,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "gkx_revision": _git_revision(repository),
            "solvax_revision": _git_revision(repository.parent / "solvax"),
            "devices": [str(device) for device in jax.devices()],
        },
    }
    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
