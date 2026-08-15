"""Compare finite-window AD, NILSAS, and multiple shooting on GKX turbulence.

This is an algorithm pilot, not a production infinite-time-gradient claim.  It
packs GKX's complex spectral state into a flat real vector, then applies the
generic matrix-free shadowing solvers to the configured projected production
map and physical heat flux.  Ladders in homogeneous-adjoint count (NILSAS) and
Schur-solve diagnostics (MSS) expose cost and conditioning before either method
is trusted in a stellarator optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.nonlinear_gradient_window import build_nonlinear_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
        ),
    )
    parser.add_argument("--saturated-state", type=Path, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument("--adjoint-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lyapunov-count", type=int, default=0)
    parser.add_argument("--lyapunov-warmup-intervals", type=int, default=4)
    parser.add_argument("--lyapunov-intervals", type=int, default=8)
    parser.add_argument("--lyapunov-steps-per-interval", type=int, default=4)
    parser.add_argument("--nilsas-regularization", type=float, default=1.0e-8)
    parser.add_argument("--run-mss", action="store_true")
    parser.add_argument("--mss-regularization", type=float, default=1.0e-6)
    parser.add_argument("--mss-cg-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--mss-cg-max-iterations", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.lyapunov_count < 0:
        raise SystemExit("--lyapunov-count must be non-negative")

    import jax
    import jax.numpy as jnp

    from gkx.solvers.nonlinear.sensitivity import (
        discrete_lyapunov_exponents,
        discrete_multiple_shooting_shadowing,
        discrete_nilsas,
    )
    from gkx.solvers.nonlinear.explicit import checkpointed_explicit_scan

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)
    archive = np.load(args.saturated_state)
    if not bool(archive["saturated"]):
        raise SystemExit("shadowing pilot requires a verified saturated state")
    state_complex = jnp.asarray(archive["state"])
    tau_ac = float(archive["tau_ac"]) if "tau_ac" in archive else None
    case = build_nonlinear_case(
        args.toml, {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz}
    )
    recorded_method = str(archive["method"]) if "method" in archive else None
    if recorded_method is not None and recorded_method != case["method"]:
        raise SystemExit(
            f"state was produced with {recorded_method}, but {args.toml.name} "
            f"configures {case['method']}"
        )
    if state_complex.shape != case["shape"]:
        raise SystemExit(
            f"state shape {state_complex.shape} does not match case {case['shape']}"
        )
    probe_dtype = case["rhs"](state_complex, 1.0).dtype
    state_complex = state_complex.astype(probe_dtype)
    dt = float(archive["adaptive_dt"])
    step_complex, objective_complex = case["window_functions"](
        dt, differentiable=bool(args.run_mss or args.lyapunov_count > 0)
    )
    shape = state_complex.shape
    complex_size = int(state_complex.size)

    def pack(values):
        flat = jnp.ravel(values)
        return jnp.concatenate((jnp.real(flat), jnp.imag(flat)))

    def unpack(values):
        return (values[:complex_size] + 1j * values[complex_size:]).reshape(shape)

    def step_real(state, parameters):
        return pack(step_complex(unpack(state), parameters))

    def objective_real(state, parameters):
        return objective_complex(unpack(state), parameters)

    state_real = pack(state_complex)
    parameters = jnp.asarray([1.0], dtype=state_real.dtype)
    total_steps = int(args.segments) * int(args.steps_per_segment)

    def finite_average(values):
        def advance(state, _unused):
            sample = objective_real(state, values)
            return step_real(state, values), sample

        _final, samples = checkpointed_explicit_scan(
            advance,
            jax.lax.stop_gradient(state_real),
            jnp.arange(total_steps),
            checkpoint=True,
        )
        return jnp.mean(samples)

    started = time.time()
    finite_value, finite_gradient = jax.value_and_grad(finite_average)(parameters)
    finite_seconds = time.time() - started
    print(
        f"finite AD N={total_steps}: Q={float(finite_value):.6e} "
        f"grad={float(finite_gradient[0]):+.6e} [{finite_seconds:.1f}s]",
        flush=True,
    )

    lyapunov_row = None
    if args.lyapunov_count > 0:
        started = time.time()
        result = discrete_lyapunov_exponents(
            step_real,
            state_real,
            parameters,
            vector_count=int(args.lyapunov_count),
            warmup_intervals=int(args.lyapunov_warmup_intervals),
            interval_count=int(args.lyapunov_intervals),
            steps_per_interval=int(args.lyapunov_steps_per_interval),
            step_size=dt,
        )
        exponents = np.asarray(result.exponents, dtype=float)
        elapsed = time.time() - started
        lyapunov_row = {
            "exponents_per_time": exponents.tolist(),
            "positive_exponent_count_lower_bound": int(np.sum(exponents > 0.0)),
            "orthogonality_residual": float(result.orthogonality_residual),
            "vector_count": int(args.lyapunov_count),
            "warmup_intervals": int(args.lyapunov_warmup_intervals),
            "interval_count": int(args.lyapunov_intervals),
            "steps_per_interval": int(args.lyapunov_steps_per_interval),
            "seconds": elapsed,
        }
        print(
            "Lyapunov exponents/time: "
            + ", ".join(f"{value:+.4e}" for value in exponents)
            + f" [{elapsed:.1f}s]",
            flush=True,
        )

    nilsas_rows: list[dict[str, Any]] = []
    for count in args.adjoint_counts:
        started = time.time()
        result = discrete_nilsas(
            step_real,
            objective_real,
            state_real,
            parameters,
            segment_count=int(args.segments),
            steps_per_segment=int(args.steps_per_segment),
            homogeneous_adjoint_count=int(count),
            regularization=float(args.nilsas_regularization),
        )
        elapsed = time.time() - started
        row = {
            "homogeneous_adjoint_count": int(count),
            "value": float(result.value),
            "gradient": float(result.gradient[0]),
            "constraint_residual": float(result.constraint_residual),
            "kkt_condition_number": float(result.kkt_condition_number),
            "max_boundary_inhomogeneous_norm": float(
                jnp.max(result.boundary_inhomogeneous_norms)
            ),
            "seconds": elapsed,
        }
        nilsas_rows.append(row)
        print(
            f"NILSAS M={count}: grad={row['gradient']:+.6e} "
            f"constraint={row['constraint_residual']:.2e} [{elapsed:.1f}s]",
            flush=True,
        )

    mss_row = None
    if args.run_mss:
        started = time.time()
        result = discrete_multiple_shooting_shadowing(
            step_real,
            objective_real,
            state_real,
            parameters,
            segment_count=int(args.segments),
            steps_per_segment=int(args.steps_per_segment),
            regularization=float(args.mss_regularization),
            cg_tolerance=float(args.mss_cg_tolerance),
            cg_max_iterations=int(args.mss_cg_max_iterations),
        )
        elapsed = time.time() - started
        mss_row = {
            "value": float(result.value),
            "gradient": float(result.gradient[0]),
            "normal_residual": float(result.normal_residual),
            "cg_iterations": int(result.cg_iterations),
            "seconds": elapsed,
        }
        print(
            f"MSS: grad={mss_row['gradient']:+.6e} "
            f"normal={mss_row['normal_residual']:.2e} "
            f"iterations={mss_row['cg_iterations']} [{elapsed:.1f}s]",
            flush=True,
        )

    payload = {
        "kind": "nonlinear_shadowing_algorithm_pilot",
        "claim_level": "reduced_resolution_algorithm_comparison_not_production_gradient",
        "case": args.toml.name,
        "grid": {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz},
        "state_shape": list(shape),
        "real_state_size": int(state_real.size),
        "dt": dt,
        "method": case["method"],
        "tau_ac": tau_ac,
        "segment_count": int(args.segments),
        "steps_per_segment": int(args.steps_per_segment),
        "total_steps": total_steps,
        "total_time": total_steps * dt,
        "total_time_in_tau_ac": (
            total_steps * dt / tau_ac
            if tau_ac is not None and np.isfinite(tau_ac) and tau_ac > 0.0
            else None
        ),
        "finite_window": {
            "value": float(finite_value),
            "gradient": float(finite_gradient[0]),
            "seconds": finite_seconds,
        },
        "lyapunov": lyapunov_row,
        "nilsas": nilsas_rows,
        "multiple_shooting": mss_row,
        "limitations": [
            "finite reduced-resolution trajectory",
            "the measured leading spectrum is only a lower bound on unstable dimension",
            "continuous-time neutral-direction projection/time dilation is not implemented",
            "the regularized MSS estimate must be checked for regularization convergence",
            "GKX state projection makes the discrete map non-invertible",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"written: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
