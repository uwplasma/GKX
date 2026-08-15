"""Measure nonlinear reverse-mode memory with and without block checkpointing.

This is a compiler-memory benchmark, not a turbulence result.  It uses GKX's
production nonlinear RHS on a deterministic low-amplitude state, compiles the
same discrete gradient under both checkpoint policies, and verifies primal and
adjoint parity before reporting XLA temporary memory and warmed execution time.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _memory_bytes(analysis: Any, name: str) -> int:
    value = getattr(analysis, name, 0)
    return int(0 if value is None else value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=0.03890582546591759)
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", args.precision == "64")

    try:
        from tools.campaigns.nonlinear_gradient_window import (
            build_nonlinear_case,
            fluctuation_energy,
            integrate,
        )
    except ModuleNotFoundError:  # Direct ``python tools/profiling/...`` execution.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.campaigns.nonlinear_gradient_window import (
            build_nonlinear_case,
            fluctuation_energy,
            integrate,
        )

    case = build_nonlinear_case(
        Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"),
        {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz},
    )
    real_dtype = np.float64 if args.precision == "64" else np.float32
    complex_dtype = np.complex128 if args.precision == "64" else np.complex64
    rng = np.random.default_rng(args.seed)
    state_np = 1.0e-7 * (
        rng.standard_normal(case["shape"]) + 1j * rng.standard_normal(case["shape"])
    )
    state = jnp.asarray(state_np.astype(complex_dtype))
    drive = jnp.asarray(case["drive"], dtype=real_dtype)

    rows = []
    results = []
    for checkpoint in (False, True):

        def objective(value):
            final = integrate(
                case["rhs"],
                state,
                value,
                args.dt,
                args.steps,
                checkpoint=checkpoint,
            )
            return fluctuation_energy(final)

        differentiated = jax.jit(jax.value_and_grad(objective))
        compile_started = time.perf_counter()
        compiled = differentiated.lower(drive).compile()
        compile_seconds = time.perf_counter() - compile_started
        analysis = compiled.memory_analysis()
        first = compiled(drive)
        jax.tree.map(lambda value: value.block_until_ready(), first)
        run_started = time.perf_counter()
        result = compiled(drive)
        jax.tree.map(lambda value: value.block_until_ready(), result)
        run_seconds = time.perf_counter() - run_started
        results.append(result)
        rows.append(
            {
                "checkpoint": "block" if checkpoint else "step",
                "compile_seconds": compile_seconds,
                "run_seconds": run_seconds,
                "temp_bytes": _memory_bytes(analysis, "temp_size_in_bytes"),
                "argument_bytes": _memory_bytes(analysis, "argument_size_in_bytes"),
                "output_bytes": _memory_bytes(analysis, "output_size_in_bytes"),
                "alias_bytes": _memory_bytes(analysis, "alias_size_in_bytes"),
                "objective": float(result[0]),
                "gradient": float(result[1]),
            }
        )

    parity_rtol = 2.0e-5 if args.precision == "32" else 2.0e-11
    np.testing.assert_allclose(
        np.asarray(results[0][0]), np.asarray(results[1][0]), rtol=parity_rtol
    )
    np.testing.assert_allclose(
        np.asarray(results[0][1]), np.asarray(results[1][1]), rtol=parity_rtol
    )
    summary = {
        "kind": "nonlinear_adjoint_checkpointing_profile",
        "claim_level": "compiler_memory_on_deterministic_production_rhs",
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "devices": [str(device) for device in jax.devices()],
        "grid_override": {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz},
        "state_shape": list(case["shape"]),
        "state_bytes": int(state.size * state.dtype.itemsize),
        "steps": args.steps,
        "dt": args.dt,
        "precision": args.precision,
        "rows": rows,
        "temp_reduction": rows[0]["temp_bytes"] / rows[1]["temp_bytes"],
        "runtime_ratio": rows[1]["run_seconds"] / rows[0]["run_seconds"],
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
