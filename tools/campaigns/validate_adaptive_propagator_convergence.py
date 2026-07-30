"""Certificate-only adaptive eigensolver ladders beyond dense-memory sizes.

The dense-oracle campaign ends at ``n=4480``. This companion holds ``ntheta``
fixed and advances a configurable velocity-space ladder without materializing
the matrix. Every cold row must pass the continuous-operator residual and RK4
stability gates. Growth, frequency, and full-complex-eigenvalue changes are
reported separately, and a convergence claim requires two consecutive changes
below the requested tolerance.
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


def _prolong(
    previous: jax.Array | None, shape: tuple[int, ...], seed: int
) -> jax.Array:
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


def _frequency_changes(values: list[complex]) -> list[float]:
    """Normalize frequency changes by the magnitude of the refined mode."""

    return [
        float(
            abs(values[index + 1].imag - values[index].imag)
            / max(abs(values[index + 1]), np.finfo(float).tiny)
        )
        for index in range(len(values) - 1)
    ]


def _plateau(
    changes: list[float],
    ladder: tuple[tuple[int, int], ...],
    tolerance: float,
) -> tuple[int, int] | None:
    for index in range(len(changes) - 1):
        if changes[index] < tolerance and changes[index + 1] < tolerance:
            return ladder[index + 2]
    return None


def _overlap(previous: jax.Array | None, current: jax.Array) -> float | None:
    if previous is None:
        return None
    old = np.asarray(previous)
    new = np.asarray(current)
    common = tuple(
        slice(0, min(old_size, new_size))
        for old_size, new_size in zip(old.shape, new.shape, strict=True)
    )
    old_flat = old[common].reshape(-1)
    new_flat = new[common].reshape(-1)
    return float(
        abs(np.vdot(old_flat, new_flat))
        / max(np.linalg.norm(old_flat) * np.linalg.norm(new_flat), np.finfo(float).tiny)
    )


def _write_checkpoint(
    output: Path,
    *,
    device: str,
    input_path: Path,
    ladder: tuple[tuple[int, int], ...],
    rows: list[dict[str, object]],
) -> Path:
    """Preserve completed expensive rungs without representing a certificate."""

    checkpoint = output.with_suffix(output.suffix + f".{device}.partial")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": False,
                "scope": "interrupted-run recovery only; not a convergence certificate",
                "device": device,
                "input": str(
                    input_path.relative_to(Path(__file__).resolve().parents[2])
                ),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "ladder": [list(rung) for rung in ladder],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return checkpoint


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
    parser.add_argument("--restart-krylov-dim", type=int, default=None)
    parser.add_argument("--adaptive-candidates", type=int, default=1)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--convergence-tol", type=float, default=0.05)
    parser.add_argument("--chunk-horizon", type=float, default=30.0)
    parser.add_argument("--stability-dimension", type=int, default=12)
    parser.add_argument("--stability-probe-count", type=int, default=2)
    parser.add_argument("--stability-safety", type=float, default=0.9)
    parser.add_argument(
        "--required-observable",
        choices=("residual", "growth", "eigenvalue", "frequency"),
        default="residual",
        help="gate that controls the process exit status",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=1,
        help="post-compilation timing repeats per rung; zero avoids a duplicate solve",
    )
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
    if args.stability_probe_count < 1:
        parser.error("--stability-probe-count must be positive")
    if args.warm_repeats < 0:
        parser.error("--warm-repeats must be non-negative")
    corrective_dimension = args.restart_krylov_dim or args.krylov_dim
    if corrective_dimension < 2:
        parser.error("--restart-krylov-dim must be at least two")
    if (
        not 1
        <= args.adaptive_candidates
        <= min(
            args.krylov_dim,
            corrective_dimension,
        )
    ):
        parser.error("--adaptive-candidates must fit every Krylov subspace")

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
    checkpoint_paths: list[Path] = []
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
                stability_probe_count=args.stability_probe_count,
                stability_safety=args.stability_safety,
                restart_krylov_dim=args.restart_krylov_dim,
                candidate_count=args.adaptive_candidates,
            )
            compiled.eigenvalue.block_until_ready()
            compiled.eigenvector.block_until_ready()
            compile_seconds = time.time() - started
            solution = compiled
            warm_samples = []
            for _repeat in range(args.warm_repeats):
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
                    stability_probe_count=args.stability_probe_count,
                    stability_safety=args.stability_safety,
                    restart_krylov_dim=args.restart_krylov_dim,
                    candidate_count=args.adaptive_candidates,
                )
                solution.eigenvalue.block_until_ready()
                solution.eigenvector.block_until_ready()
                warm_samples.append(time.time() - started)
            warm_seconds = min(warm_samples) if warm_samples else None
            value = complex(np.asarray(solution.eigenvalue))
            values.append(value)
            continuation_overlap = _overlap(previous, solution.eigenvector)
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
                "continuation_overlap": continuation_overlap,
                "compile_seconds": compile_seconds,
                "cold_seconds": compile_seconds,
                "warm_seconds": warm_seconds,
            }
            rows.append(row)
            checkpoint_path = _write_checkpoint(
                args.output,
                device=name,
                input_path=input_path,
                ladder=ladder,
                rows=rows,
            )
            if checkpoint_path not in checkpoint_paths:
                checkpoint_paths.append(checkpoint_path)
            print(
                f"{name.upper()} ({row['n_laguerre']:>2},{row['n_hermite']:>2}) "
                f"n={row['n']:>6} growth={value.real:+.8e} "
                f"res={row['residual']:.2e} cold={compile_seconds:.2f}s "
                f"warm={warm_seconds if warm_seconds is not None else 'skipped'}",
                flush=True,
            )
        growth_changes = _changes(values, growth_only=True)
        eigenvalue_changes = _changes(values, growth_only=False)
        frequency_changes = _frequency_changes(values)
        certified = all(row["converged"] for row in rows)
        growth_resolution = (
            _plateau(growth_changes, ladder, args.convergence_tol)
            if certified
            else None
        )
        eigenvalue_resolution = (
            _plateau(eigenvalue_changes, ladder, args.convergence_tol)
            if certified
            else None
        )
        frequency_resolution = (
            _plateau(frequency_changes, ladder, args.convergence_tol)
            if certified
            else None
        )
        reports.append(
            {
                "device": name,
                "input": str(_DEVICES[name]),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "certified": certified,
                "growth_converged": growth_resolution is not None,
                "growth_converged_resolution": (
                    list(growth_resolution) if growth_resolution is not None else None
                ),
                "eigenvalue_converged": eigenvalue_resolution is not None,
                "eigenvalue_converged_resolution": (
                    list(eigenvalue_resolution)
                    if eigenvalue_resolution is not None
                    else None
                ),
                "frequency_converged": frequency_resolution is not None,
                "frequency_converged_resolution": (
                    list(frequency_resolution)
                    if frequency_resolution is not None
                    else None
                ),
                "growth_relative_changes": growth_changes,
                "eigenvalue_relative_changes": eigenvalue_changes,
                "frequency_normalized_changes": frequency_changes,
                "rows": rows,
            }
        )

    certified = all(report["certified"] for report in reports)
    all_growth_converged = all(report["growth_converged"] for report in reports)
    all_eigenvalue_converged = all(report["eigenvalue_converged"] for report in reports)
    all_frequency_converged = all(report["frequency_converged"] for report in reports)
    passed = {
        "residual": certified,
        "growth": bool(certified and all_growth_converged),
        "eigenvalue": bool(certified and all_eigenvalue_converged),
        "frequency": bool(
            certified and all_eigenvalue_converged and all_frequency_converged
        ),
    }[args.required_observable]
    artifact = {
        "schema_version": 1,
        "passed": passed,
        "required_observable": args.required_observable,
        "certified": certified,
        "all_growth_converged": all_growth_converged,
        "all_eigenvalue_converged": all_eigenvalue_converged,
        "all_frequency_converged": all_frequency_converged,
        "scope": (
            "certificate-only fixed-ntheta velocity-space ladder beyond the dense "
            "oracle memory range; growth, full-eigenvalue, and frequency-only "
            "convergence separately require two consecutive normalized changes "
            "to pass tolerance"
        ),
        "provenance": {
            "ntheta": args.ntheta,
            "s_index": args.s_index,
            "selected_ky_index": 1,
            "ladder": [list(rung) for rung in ladder],
            "krylov_dim": args.krylov_dim,
            "restart_krylov_dim": args.restart_krylov_dim,
            "adaptive_candidates": args.adaptive_candidates,
            "max_restarts": args.max_restarts,
            "residual_tolerance": args.tol,
            "convergence_tolerance": args.convergence_tol,
            "chunk_horizon": args.chunk_horizon,
            "stability_dimension": args.stability_dimension,
            "stability_probe_count": args.stability_probe_count,
            "stability_safety": args.stability_safety,
            "warm_repeats": args.warm_repeats,
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
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.unlink(missing_ok=True)
    print(
        f"\n{args.required_observable} certificate "
        f"{'PASS' if passed else 'FAIL'}: {args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
