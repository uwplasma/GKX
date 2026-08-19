"""Measure nonlinear reverse-mode memory with and without block checkpointing.

This is a compiler-memory benchmark, not a turbulence result. It compiles the
*same* gradient of :func:`gkx.nonlinear_heat_flux_window` under both checkpoint
policies, verifies that the two agree in value and gradient, and only then
reports XLA temporary memory and warmed execution time. Reporting a memory
saving without that parity check would not distinguish a cheaper adjoint from a
different one.

It regenerates the memory claim in ``docs/nonlinear_autodiff.rst`` and the
left-hand panel of ``docs/_static/nonlinear_autodiff_validation.png``: at 2048
steps, temporary memory fell from 759 MB to 12.6 MB on CPU and from 11.88 GB to
168 MB on an RTX A4000, with runtime rising 1.54x and 1.67x. It replaces the
pre-1.7 profiler deleted in 612e1311, which drove a private sensitivity module
on a state-only functional; this one drives the shipped physical-heat-flux
entry point instead.

Example (the numbers above; ~25 min on an A4000, hours on a laptop CPU)::

    python tools/profiling/profile_nonlinear_adjoint_checkpointing.py \\
        --nx 16 --ny 16 --nz 16 --steps 2048 --precision 32 \\
        --output docs/_static/nonlinear_adjoint_checkpointing_gpu32.json

Cost scales linearly in ``--steps``; ``--steps 64`` is a ~1 min smoke test that
exercises every path, and the memory ratio it reports is already the shape of
the 2048-step one.
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
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
        ),
    )
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
            build_window_case,
            make_window,
        )
    except ModuleNotFoundError:  # Direct ``python tools/profiling/...`` execution.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.campaigns.nonlinear_gradient_window import (
            build_window_case,
            make_window,
        )

    case = build_window_case(
        args.toml, {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz}
    )
    complex_dtype = np.complex128 if args.precision == "64" else np.complex64
    rng = np.random.default_rng(args.seed)
    draw = rng.standard_normal(case["shape"]) + 1j * rng.standard_normal(case["shape"])
    state = jnp.asarray((1.0e-7 * draw).astype(complex_dtype))

    rows = []
    results = []
    for checkpoint in (False, True):
        objective = make_window(case, state, args.dt, args.steps, checkpoint=checkpoint)
        differentiated = jax.jit(jax.value_and_grad(objective))
        started = time.perf_counter()
        compiled = differentiated.lower(jnp.asarray(1.0)).compile()
        compile_seconds = time.perf_counter() - started
        analysis = compiled.memory_analysis()
        first = compiled(jnp.asarray(1.0))
        jax.tree.map(lambda value: value.block_until_ready(), first)
        started = time.perf_counter()
        result = compiled(jnp.asarray(1.0))
        jax.tree.map(lambda value: value.block_until_ready(), result)
        run_seconds = time.perf_counter() - started
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
        "claim_level": "compiler_memory_on_the_shipped_heat_flux_window",
        "entry_point": "gkx.nonlinear_heat_flux_window",
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "devices": [str(device) for device in jax.devices()],
        "case": case["case"],
        "grid_override": {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz},
        "state_shape": list(case["shape"]),
        "state_bytes": int(state.size * state.dtype.itemsize),
        "steps": args.steps,
        "dt": args.dt,
        "precision": args.precision,
        "rows": rows,
        "temp_reduction": rows[0]["temp_bytes"] / max(rows[1]["temp_bytes"], 1),
        "runtime_ratio": rows[1]["run_seconds"] / max(rows[0]["run_seconds"], 1e-12),
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
