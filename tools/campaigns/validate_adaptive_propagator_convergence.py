"""Certificate-only adaptive eigensolver ladders beyond dense-memory sizes.

The dense oracle campaign ends at ``n=4480``. This companion holds
``ntheta=64`` and advances the velocity-space ladder through ``(12, 16)``
without materializing the matrix. Every row must pass the continuous-operator
residual and RK4 stability gates; consecutive growth and complex-eigenvalue
changes are recorded rather than replaced by an unsupported convergence claim.
"""

from __future__ import annotations

import argparse
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

_DEVICES = {
    "qa": Path("examples/vmec/input.LandremanPaul2021_QA_lowres"),
    "qh": Path("examples/vmec/input.LandremanPaul2021_QH_reactorScale_lowres"),
    "qi": Path("examples/vmec/input.nfp3_QI_fixed_resolution_final"),
}
_DEFAULT_LADDER = ((8, 10), (10, 14), (12, 16))


def _resolution(raw: str) -> tuple[int, int]:
    try:
        n_laguerre, n_hermite = (int(token) for token in raw.split(",", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("resolution must be Nl,Nm") from error
    if n_laguerre < 1 or n_hermite < 2:
        raise argparse.ArgumentTypeError("resolution must satisfy Nl >= 1 and Nm >= 2")
    return n_laguerre, n_hermite


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


def _prolong(previous: jax.Array | None, shape: tuple[int, ...], seed: int) -> jax.Array:
    generator = np.random.default_rng(seed)
    noise = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    if previous is None:
        return jnp.asarray(noise)
    output = np.zeros(shape, dtype=complex)
    old = np.asarray(previous)
    common = tuple(
        slice(0, min(old_size, new_size))
        for old_size, new_size in zip(old.shape, shape, strict=True)
    )
    output[common] = old[common]
    output += 1.0e-8 * noise / np.linalg.norm(noise)
    return jnp.asarray(output)


def _changes(values: list[complex], *, growth_only: bool) -> list[float]:
    observable = [value.real if growth_only else value for value in values]
    return [
        float(
            abs(observable[index + 1] - observable[index])
            / max(abs(observable[index + 1]), np.finfo(float).tiny)
        )
        for index in range(len(observable) - 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        action="append",
        choices=tuple(_DEVICES),
        default=None,
    )
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--convergence-tol", type=float, default=0.05)
    parser.add_argument("--chunk-horizon", type=float, default=30.0)
    parser.add_argument("--stability-dimension", type=int, default=12)
    parser.add_argument("--stability-safety", type=float, default=0.9)
    parser.add_argument(
        "--resolution",
        action="append",
        type=_resolution,
        default=None,
        help="velocity resolution Nl,Nm; repeat to replace the default ladder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/adaptive_propagator_convergence.json"),
    )
    args = parser.parse_args()
    if args.ntheta < 8 or args.krylov_dim < 4:
        parser.error("--ntheta must be at least 8 and --krylov-dim at least 4")

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    from gkx.objectives.core import _solver_geometry_context
    from gkx.solvers.linear.krylov import adaptive_propagator_eigenpair

    repository = Path(__file__).resolve().parents[2]
    selected = args.device or list(_DEVICES)
    ladder = tuple(args.resolution or _DEFAULT_LADDER)
    if len(ladder) < 3:
        parser.error("the convergence ladder requires at least three resolutions")
    reports = []
    for device_index, name in enumerate(selected):
        input_path = repository / _DEVICES[name]
        equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(input_path))
        geometry = turb.flux_tube_geometry(
            equilibrium.state,
            equilibrium.runtime,
            s_index=args.s_index,
            alpha=0.0,
            ntheta=args.ntheta,
        )
        previous = None
        values: list[complex] = []
        rows = []
        for rung_index, (n_laguerre, n_hermite) in enumerate(ladder):
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
            start = _prolong(
                previous,
                context.state_shape,
                seed=100 * device_index + rung_index,
            )
            started = time.time()
            compiled = adaptive_propagator_eigenpair(
                start,
                context.cache,
                context.linear_params,
                terms=context.linear_terms,
                krylov_dim=args.krylov_dim,
                max_restarts=args.max_restarts,
                tol=args.tol,
                chunk_horizon=args.chunk_horizon,
                stability_dimension=args.stability_dimension,
                stability_safety=args.stability_safety,
            )
            compiled.eigenvalue.block_until_ready()
            compiled.eigenvector.block_until_ready()
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
                chunk_horizon=args.chunk_horizon,
                stability_dimension=args.stability_dimension,
                stability_safety=args.stability_safety,
            )
            solution.eigenvalue.block_until_ready()
            solution.eigenvector.block_until_ready()
            warm_seconds = time.time() - started
            value = complex(np.asarray(solution.eigenvalue))
            values.append(value)
            if solution.converged:
                previous = solution.eigenvector
            row = {
                "n_laguerre": n_laguerre,
                "n_hermite": n_hermite,
                "n": int(np.prod(context.state_shape)),
                "eigenvalue": [value.real, value.imag],
                "residual": float(np.asarray(solution.residual)),
                "converged": solution.converged,
                "stability_passed": solution.stable,
                "filter_growth_defect": solution.filter_growth_defect,
                "restarts": solution.restarts,
                "selected_propagator_dt": solution.filter_dt,
                "selected_propagator_steps": solution.filter_steps,
                "original_operator_evaluations": solution.operator_applications,
                "compile_seconds": compile_seconds,
                "warm_seconds": warm_seconds,
            }
            rows.append(row)
            print(
                f"{name.upper()} ({row['n_laguerre']:>2},{row['n_hermite']:>2}) "
                f"n={row['n']:>6} growth={value.real:+.8e} "
                f"res={row['residual']:.2e} warm={warm_seconds:.2f}s",
                flush=True,
            )
        growth_changes = _changes(values, growth_only=True)
        eigenvalue_changes = _changes(values, growth_only=False)
        certified = all(row["converged"] for row in rows)
        growth_converged = all(
            change < args.convergence_tol for change in growth_changes
        )
        reports.append(
            {
                "device": name,
                "input": str(_DEVICES[name]),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "certified": certified,
                "growth_converged": growth_converged,
                "growth_relative_changes": growth_changes,
                "eigenvalue_relative_changes": eigenvalue_changes,
                "rows": rows,
            }
        )

    certified = all(report["certified"] for report in reports)
    artifact = {
        "schema_version": 1,
        "certified": certified,
        "all_growth_converged": all(
            report["growth_converged"] for report in reports
        ),
        "scope": (
            "certificate-only ntheta=64 velocity-space ladder beyond the dense "
            "oracle memory range; convergence requires both consecutive growth "
            "changes to pass the declared tolerance"
        ),
        "provenance": {
            "ntheta": args.ntheta,
            "s_index": args.s_index,
            "selected_ky_index": 1,
            "ladder": [list(rung) for rung in ladder],
            "krylov_dim": args.krylov_dim,
            "max_restarts": args.max_restarts,
            "residual_tolerance": args.tol,
            "convergence_tolerance": args.convergence_tol,
            "chunk_horizon": args.chunk_horizon,
            "stability_dimension": args.stability_dimension,
            "stability_safety": args.stability_safety,
            "python": sys.version,
            "platform": platform.platform(),
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "gkx": _version("gkx"),
            "solvax": _version("solvax"),
            "gkx_commit": _git_revision(repository),
            "solvax_commit": _git_revision(
                Path(__import__("solvax").__file__).resolve().parents[2]
            ),
            "jax_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(device) for device in jax.devices()],
        },
        "devices": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\ncertificate {'PASS' if certified else 'FAIL'}: {args.output}")
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
