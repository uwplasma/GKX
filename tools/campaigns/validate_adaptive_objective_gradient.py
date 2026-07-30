"""Cold-process gate for matrix-free linear/quasilinear objective gradients.

Each eigensolver runs in a fresh Python process so compilation and first
execution are charged to the reported time. The adaptive value and reverse
gradient are compared with the dense right-eigenvector reference on the exact
same solver-ready geometry and scalar phase-invariant objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


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


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _worker(args: argparse.Namespace) -> int:
    import gkx
    from gkx.objectives.core import (
        AdaptiveLinearEigensolverConfig,
        solver_objective_vector_from_geometry,
    )
    from gkx.objectives.geometry import (
        default_solver_geometry_design_params,
        solver_ready_geometry_mapping,
    )

    theta = jnp.linspace(
        -jnp.pi,
        jnp.pi,
        args.ntheta,
        endpoint=False,
        dtype=jnp.float64,
    )
    base = default_solver_geometry_design_params().astype(jnp.float64)
    direction = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    weights = jnp.asarray(
        [1.0, 0.2, 0.1, 0.05, 0.0, 0.01],
        dtype=jnp.float64,
    )
    adaptive_config = AdaptiveLinearEigensolverConfig()

    def objective(offset: jax.Array) -> jax.Array:
        geometry = gkx.flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(base + offset * direction, theta),
            validate_finite=False,
        )
        vector = solver_objective_vector_from_geometry(
            geometry,
            n_laguerre=args.n_laguerre,
            n_hermite=args.n_hermite,
            ny=4,
            selected_ky_index=1,
            eigensolver=args.worker,
            adaptive_config=(
                adaptive_config if args.worker == "adaptive-propagator" else None
            ),
        )
        return jnp.vdot(weights, vector)

    started = time.perf_counter()
    value, gradient = jax.value_and_grad(objective)(jnp.asarray(0.0, jnp.float64))
    value.block_until_ready()
    gradient.block_until_ready()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "method": args.worker,
                "value": float(value),
                "directional_gradient": float(gradient),
                "cold_value_and_gradient_seconds": elapsed,
                "finite": bool(np.isfinite(value) and np.isfinite(gradient)),
                "adaptive_config": (
                    asdict(adaptive_config)
                    if args.worker == "adaptive-propagator"
                    else None
                ),
            }
        )
    )
    return 0


def _run_worker(
    repository: Path,
    method: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        method,
        "--ntheta",
        str(args.ntheta),
        "--n-laguerre",
        str(args.n_laguerre),
        "--n-hermite",
        str(args.n_hermite),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{method} worker produced no result")
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--n-laguerre", type=int, default=12)
    parser.add_argument("--n-hermite", type=int, default=16)
    parser.add_argument(
        "--require-cold-speedup",
        action="store_true",
        help="fail unless adaptive cold value-and-gradient beats dense",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/_static/adaptive_objective_gradient_cold_n6144_validation.json"
        ),
    )
    parser.add_argument(
        "--worker",
        choices=("dense", "adaptive-propagator"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.ntheta < 4 or args.n_laguerre < 1 or args.n_hermite < 1:
        parser.error("all resolution dimensions must be positive and ntheta >= 4")
    if args.worker is not None:
        return _worker(args)

    repository = Path(__file__).resolve().parents[2]
    adaptive = _run_worker(repository, "adaptive-propagator", args)
    dense = _run_worker(repository, "dense", args)
    value_scale = max(abs(float(dense["value"])), np.finfo(float).tiny)
    gradient_scale = max(
        abs(float(dense["directional_gradient"])),
        np.finfo(float).tiny,
    )
    value_relative_error = (
        abs(float(adaptive["value"]) - float(dense["value"])) / value_scale
    )
    gradient_relative_error = (
        abs(
            float(adaptive["directional_gradient"])
            - float(dense["directional_gradient"])
        )
        / gradient_scale
    )
    cold_speedup = float(dense["cold_value_and_gradient_seconds"]) / float(
        adaptive["cold_value_and_gradient_seconds"]
    )
    accuracy_passed = (
        bool(adaptive["finite"])
        and bool(dense["finite"])
        and value_relative_error < 1.0e-10
        and gradient_relative_error < 1.0e-7
    )
    speed_passed = cold_speedup > 1.0
    passed = accuracy_passed and (speed_passed if args.require_cold_speedup else True)
    sibling_solvax = repository.parent / "solvax"
    solvax_path = (
        sibling_solvax
        if (sibling_solvax / ".git").exists()
        else Path(importlib.metadata.distribution("solvax").locate_file(""))
    )
    report = {
        "schema_version": 1,
        "scope": (
            "fresh-process cold value-and-reverse-gradient comparison on one "
            "phase-invariant linear/quasilinear objective"
        ),
        "resolution": {
            "ntheta": args.ntheta,
            "n_laguerre": args.n_laguerre,
            "n_hermite": args.n_hermite,
            "operator_size": args.ntheta * args.n_laguerre * args.n_hermite,
        },
        "adaptive": adaptive,
        "dense": dense,
        "value_relative_error": value_relative_error,
        "gradient_relative_error": gradient_relative_error,
        "cold_speedup_over_dense": cold_speedup,
        "accuracy_passed": accuracy_passed,
        "cold_speed_passed": speed_passed,
        "cold_speed_required": args.require_cold_speedup,
        "passed": passed,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "gkx_revision": _git_revision(repository),
            "solvax_revision": _git_revision(solvax_path),
            "devices": [str(device) for device in jax.devices()],
        },
    }
    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
