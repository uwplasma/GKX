"""Qualify matrix-free continuation through a real GKX growth-rate crossing.

The Cyclone operator has two spectrally distinct unstable modes whose growth
ordering exchanges near ``R/LTi=13.3`` at the reduced resolution used here.
The campaign starts on the dominant lower-frequency branch, then carries its
right and left eigenvectors across the exchange. Dense eigensystems are used
only as an oracle; matrix-free selection uses the previous adaptive left/right
pair and certified candidates from one propagator Arnoldi basis.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

_DEFAULT_SCAN = (12.0, 12.5, 13.0, 13.25, 13.5, 14.0)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _unit_columns(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=0)
    return vectors / np.maximum(norms, np.finfo(float).tiny)


def _relative_biorthogonal_scores(
    left: np.ndarray,
    anchor: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    anchor_projection = abs(np.vdot(left, anchor))
    candidate_norms = np.linalg.norm(candidates, axis=0)
    candidate_projections = np.abs(left.conj() @ candidates)
    return (
        candidate_projections / np.maximum(candidate_norms, np.finfo(float).tiny)
    ) / max(
        anchor_projection / max(np.linalg.norm(anchor), np.finfo(float).tiny),
        np.finfo(float).tiny,
    )


def _left_for_value(
    matrix: np.ndarray,
    value: complex,
) -> tuple[np.ndarray, float]:
    left_values, left_vectors = np.linalg.eig(matrix.conj().T)
    index = int(np.argmin(np.abs(left_values - np.conj(value))))
    residual = np.linalg.norm(
        matrix.conj().T @ left_vectors[:, index]
        - left_values[index] * left_vectors[:, index]
    )
    left = left_vectors[:, index] / np.linalg.norm(left_vectors[:, index])
    return left, float(residual)


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        abs(np.vdot(first.reshape(-1), second.reshape(-1)))
        / max(
            np.linalg.norm(first) * np.linalg.norm(second),
            np.finfo(float).tiny,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-over-lti", type=float, action="append", default=None)
    parser.add_argument("--ntheta", type=int, default=8)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--krylov-dim", type=int, default=24)
    parser.add_argument("--restart-krylov-dim", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--overlap-floor", type=float, default=0.9)
    parser.add_argument("--spectral-gap-floor", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/_static/adaptive_propagator_branch_crossing_validation.json"
        ),
    )
    args = parser.parse_args()
    scan = tuple(args.r_over_lti or _DEFAULT_SCAN)
    if len(scan) < 4 or any(right <= left for left, right in zip(scan, scan[1:])):
        parser.error("R/LTi scan must contain at least four increasing points")
    if not 2 <= args.candidates <= args.restart_krylov_dim:
        parser.error("candidate count must be between two and the restart dimension")

    from solvax import propagator_eigenpairs

    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid, select_ky_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.objectives.autodiff_validation import (
        explicit_complex_operator_matrix,
    )
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.krylov import adaptive_propagator_eigenpair

    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=4,
            Nz=args.ntheta,
            Lx=6.0,
            Ly=12.0,
        )
    )
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geometry = SAlphaGeometry.from_config(cfg.geometry)
    base_params = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=scan[0],
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    state_shape = (
        args.n_laguerre,
        args.n_hermite,
        int(grid.ky.size),
        int(grid.kx.size),
        int(grid.z.size),
    )
    operator_size = int(np.prod(state_shape))
    phase = jnp.arange(operator_size, dtype=jnp.float64) + 1.0
    broadband = jnp.reshape(
        jnp.exp(1j * phase * jnp.asarray(0.6180339887498948)),
        state_shape,
    )

    previous_dense_right: np.ndarray | None = None
    previous_dense_left: np.ndarray | None = None
    previous_adaptive_right: jax.Array | None = None
    previous_adaptive_left: jax.Array | None = None
    rows: list[dict[str, object]] = []
    for parameter in scan:
        params = replace(base_params, R_over_LTi=parameter)
        cache = build_linear_cache(
            grid,
            geometry,
            params,
            args.n_laguerre,
            args.n_hermite,
        )

        def apply(state: jax.Array) -> jax.Array:
            return linear_rhs_cached(
                state,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0]

        matrix = np.asarray(explicit_complex_operator_matrix(apply, state_shape))
        dense_values, dense_vectors = np.linalg.eig(matrix)
        dense_vectors = _unit_columns(dense_vectors)
        growth_order = np.argsort(dense_values.real)[::-1]
        if previous_dense_left is None or previous_dense_right is None:
            dense_index = int(growth_order[0])
            dense_selection_overlap = None
        else:
            dense_scores = _relative_biorthogonal_scores(
                previous_dense_left,
                previous_dense_right,
                dense_vectors,
            )
            dense_index = int(np.argmax(dense_scores))
            dense_selection_overlap = float(dense_scores[dense_index])
        dense_value = complex(dense_values[dense_index])
        dense_right = dense_vectors[:, dense_index]
        dense_left, dense_left_residual = _left_for_value(matrix, dense_value)
        dense_rank = int(np.flatnonzero(growth_order == dense_index)[0])
        other = np.arange(dense_values.size) != dense_index
        dense_spectral_gap = float(np.min(np.abs(dense_values[other] - dense_value)))

        started = time.time()
        solution = adaptive_propagator_eigenpair(
            broadband if previous_adaptive_right is None else previous_adaptive_right,
            cache,
            params,
            terms=terms,
            krylov_dim=args.krylov_dim,
            restart_krylov_dim=args.restart_krylov_dim,
            candidate_count=args.candidates,
            max_restarts=4,
            tol=args.tol,
            chunk_horizon=30.0,
            stability_dimension=min(12, operator_size - 1),
            stability_probe_count=2,
            stability_safety=0.9,
            continuation_vector=previous_adaptive_right,
            continuation_covector=previous_adaptive_left,
            continuation_overlap_floor=args.overlap_floor,
            continuation_spectral_gap_floor=args.spectral_gap_floor,
        )
        solution.eigenvalue.block_until_ready()
        elapsed = time.time() - started
        adaptive_value = complex(np.asarray(solution.eigenvalue))
        adaptive_right = solution.eigenvector

        transpose = jax.linear_transpose(apply, adaptive_right)

        @jax.jit
        def adjoint(vector: jax.Array) -> jax.Array:
            return jnp.conj(transpose(jnp.conj(vector))[0])

        left_candidates = propagator_eigenpairs(
            adjoint,
            broadband,
            dt=solution.filter_dt,
            steps=solution.filter_steps,
            krylov_dim=args.krylov_dim,
            candidates=args.candidates,
            tol=args.tol,
        )
        left_converged = np.asarray(left_candidates.converged, dtype=bool)
        left_values = np.asarray(left_candidates.eigenvalues)
        left_distances = np.where(
            left_converged,
            np.abs(left_values - np.conj(adaptive_value)),
            np.inf,
        )
        left_index = int(np.argmin(left_distances))
        adaptive_left = left_candidates.eigenvectors[left_index]
        adaptive_left_residual = float(
            np.asarray(left_candidates.residuals[left_index])
        )

        relative_error = float(
            abs(adaptive_value - dense_value)
            / max(abs(dense_value), np.finfo(float).tiny)
        )
        right_overlap = _overlap(
            np.asarray(adaptive_right),
            dense_right.reshape(state_shape),
        )
        row = {
            "R_over_LTi": parameter,
            "dense_tracked_eigenvalue": [dense_value.real, dense_value.imag],
            "dense_dominant_eigenvalue": [
                float(dense_values[growth_order[0]].real),
                float(dense_values[growth_order[0]].imag),
            ],
            "dense_tracked_growth_rank": dense_rank,
            "dense_selection_overlap": dense_selection_overlap,
            "dense_spectral_gap": dense_spectral_gap,
            "dense_left_residual": dense_left_residual,
            "adaptive_eigenvalue": [adaptive_value.real, adaptive_value.imag],
            "adaptive_relative_eigenvalue_error": relative_error,
            "adaptive_dense_right_overlap": right_overlap,
            "adaptive_residual": float(np.asarray(solution.residual)),
            "adaptive_left_residual": adaptive_left_residual,
            "adaptive_converged": bool(solution.converged),
            "adaptive_stable": bool(solution.stable),
            "adaptive_continued": bool(solution.continued),
            "adaptive_continuation_passed": bool(solution.continuation_passed),
            "adaptive_continuation_overlap": (
                None
                if not solution.continued
                else float(np.asarray(solution.continuation_overlap))
            ),
            "adaptive_selected_spectral_gap": float(
                np.asarray(solution.selected_spectral_gap)
            ),
            "adaptive_selected_candidate_index": int(solution.selected_candidate_index),
            "adaptive_candidate_eigenvalues": [
                [complex(value).real, complex(value).imag]
                for value in np.asarray(solution.candidate_eigenvalues)
            ],
            "adaptive_candidate_residuals": [
                float(value) for value in np.asarray(solution.candidate_residuals)
            ],
            "adaptive_candidate_overlaps": [
                None if not np.isfinite(value) else float(value)
                for value in np.asarray(solution.candidate_overlaps)
            ],
            "seconds": elapsed,
        }
        rows.append(row)
        print(
            f"R/LTi={parameter:5.2f} rank={dense_rank} "
            f"lambda={adaptive_value.real:+.8e}{adaptive_value.imag:+.8e}i "
            f"error={relative_error:.2e} "
            f"overlap={row['adaptive_continuation_overlap']} "
            f"candidate={solution.selected_candidate_index}",
            flush=True,
        )
        previous_dense_right = dense_right
        previous_dense_left = dense_left
        previous_adaptive_right = adaptive_right
        previous_adaptive_left = adaptive_left

    continuation_rows = rows[1:]
    exchange_rows = [row for row in rows if int(row["dense_tracked_growth_rank"]) > 0]
    selected_subdominant = [
        row
        for row in exchange_rows
        if int(row["adaptive_selected_candidate_index"]) > 0
    ]
    checks = {
        "dense_growth_order_exchange_observed": bool(exchange_rows),
        "adaptive_selected_subdominant_candidate": bool(selected_subdominant),
        "all_primal_pairs_certified": all(
            row["adaptive_converged"]
            and row["adaptive_stable"]
            and float(row["adaptive_residual"]) <= args.tol
            for row in rows
        ),
        "all_left_pairs_certified": all(
            float(row["adaptive_left_residual"]) <= args.tol for row in rows
        ),
        "dense_oracle_eigenvalue_agreement": all(
            float(row["adaptive_relative_eigenvalue_error"]) <= 1.0e-8 for row in rows
        ),
        "dense_oracle_eigenvector_agreement": all(
            float(row["adaptive_dense_right_overlap"]) >= 1.0 - 1.0e-8 for row in rows
        ),
        "continuation_overlap": all(
            row["adaptive_continuation_passed"]
            and float(row["adaptive_continuation_overlap"]) >= args.overlap_floor
            for row in continuation_rows
        ),
        "selected_branch_complex_isolation": all(
            float(row["adaptive_selected_spectral_gap"]) >= args.spectral_gap_floor
            for row in rows
        ),
    }
    passed = all(checks.values())
    repository = Path(__file__).resolve().parents[2]
    artifact = {
        "schema_version": 1,
        "passed": passed,
        "scope": (
            "real Cyclone linear-operator growth-order exchange; matrix-free "
            "selection uses the previous adaptive left/right eigenpair while "
            "dense eigensystems are retained only as a qualification oracle"
        ),
        "checks": checks,
        "thresholds": {
            "residual": args.tol,
            "relative_eigenvalue_error": 1.0e-8,
            "dense_eigenvector_overlap": 1.0 - 1.0e-8,
            "continuation_overlap": args.overlap_floor,
            "selected_complex_spectral_gap": args.spectral_gap_floor,
        },
        "exchange": {
            "parameter": "R_over_LTi",
            "scan": list(scan),
            "first_subdominant_parameter": (
                exchange_rows[0]["R_over_LTi"] if exchange_rows else None
            ),
            "subdominant_row_count": len(exchange_rows),
            "adaptive_subdominant_selection_count": len(selected_subdominant),
        },
        "provenance": {
            "geometry": "Cyclone s-alpha",
            "field_model": "electrostatic",
            "branch_family": "ITG",
            "n_laguerre": args.n_laguerre,
            "n_hermite": args.n_hermite,
            "ntheta": args.ntheta,
            "operator_size": operator_size,
            "selected_ky": float(np.asarray(grid.ky[0])),
            "krylov_dim": args.krylov_dim,
            "restart_krylov_dim": args.restart_krylov_dim,
            "candidate_count": args.candidates,
            "jax_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(device) for device in jax.devices()],
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
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        f"\nbranch-continuation certificate {'PASS' if passed else 'FAIL'}: {args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
