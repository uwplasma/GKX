#!/usr/bin/env python3
"""Isolate the linked-chain gather and price it on both ``ky`` layouts.

Queue row Q27 asks whether ``grad_z_linked_fft`` lowers as well on a half
``ky`` axis as on a full one.  The nonlinear step ledger cannot answer that on
its own: it prices the whole RHS, and the two layouts differ there by the
Hermitian completion as well as by the chain gather.  This probe compiles the
linked derivative *alone*, on a cache built for one ``Ny``, and reports for
each layout:

* the optimized-HLO op counts, split into the bytes that own an output buffer
  and the bytes of instructions a fusion body emits as index arithmetic;
* XLA's own buffer assignment for the same executable.

It also sweeps ``Ny`` so the "the ky extent stopped being the grid's power of
two" reading recorded in #258 can be checked rather than assumed: ``Ny = 62``
gives ``Nyc = 32``, a power of two on the *half* axis and not on the full one,
and ``Ny = 32`` gives ``Nyc = 17``.  If the extent's factorization were the
cause, those two would swap places.

Every number here is a property of a compiled module. None of them depends on
machine load, and no wall-clock claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gkx.core_grid import build_spectral_grid  # noqa: E402
from gkx.geometry import apply_imported_geometry_grid_defaults  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.streaming import grad_z_linked_fft  # noqa: E402
from gkx.runtime import (  # noqa: E402
    build_runtime_geometry,
    build_runtime_linear_params,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402
from tools.profiling.profile_runtime_kernels import (  # noqa: E402
    _compiled_memory_stats,
    _hlo_op_counts,
)

REPORT = (
    "gather",
    "copy",
    "transpose",
    "concatenate",
    "reverse",
    "bytes_written",
    "materialized_bytes",
    "fused_interior_bytes",
)


def one(config: Path, Ny: int, Nx: int, Nz: int, layout: str, Nl: int, Nm: int) -> dict:
    from dataclasses import replace

    cfg, _ = load_runtime_from_toml(config)
    cfg = replace(cfg, grid=replace(cfg.grid, Nx=Nx, Ny=Ny, Nz=Nz))
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(
        apply_imported_geometry_grid_defaults(geom, cfg.grid), ky_layout=layout
    )
    params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    nky = int(grid.ky.size)
    state = jnp.zeros((1, Nm, Nl, nky, Nx, Nz), dtype=jnp.complex64)

    def apply(f: jnp.ndarray) -> jnp.ndarray:
        return grad_z_linked_fft(
            f,
            float(cache.dz),
            cache.linked_indices,
            cache.linked_kz,
            linked_inverse_permutation=cache.linked_inverse_permutation,
            linked_full_cover=bool(cache.linked_full_cover),
            linked_gather_map=cache.linked_gather_map,
            linked_gather_mask=cache.linked_gather_mask,
            linked_use_gather=bool(cache.linked_use_gather),
            ny_full=getattr(cache, "ny_full", None),
        )

    compiled = jax.jit(apply).lower(state).compile()
    counts = _hlo_op_counts(compiled.as_text())
    return {
        "Ny": Ny,
        "layout": layout,
        "nky": nky,
        "flat_modes": nky * Nx,
        "chain_classes": [list(np.asarray(i).shape) for i in cache.linked_indices],
        "state_bytes": int(state.size * state.dtype.itemsize),
        "counts": {key: counts[key] for key in REPORT},
        "memory_analysis": _compiled_memory_stats(compiled),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
        help="repo-relative, so the recorded `config` field names a path inside "
        "the repository rather than the machine this ran on",
    )
    parser.add_argument("--Ny", type=str, default="32,62,64")
    parser.add_argument("--Nx", type=int, default=32)
    parser.add_argument("--Nz", type=int, default=24)
    parser.add_argument("--Nl", type=int, default=2)
    parser.add_argument("--Nm", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = args.config if args.config.is_absolute() else ROOT / args.config
    rows = [
        one(config, int(ny), args.Nx, args.Nz, layout, args.Nl, args.Nm)
        for ny in args.Ny.split(",")
        if ny.strip()
        for layout in ("full", "half")
    ]
    summary = {
        "kind": "q27_linked_gather_probe",
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "grid": {"Nx": args.Nx, "Nz": args.Nz, "Nl": args.Nl, "Nm": args.Nm},
        "config": str(args.config),
        "rows": rows,
        "claim_scope": (
            "Op counts and buffer assignment of one compiled XLA module per row, "
            "for one jax version and backend. They locate materialized work; "
            "they are not a runtime claim."
        ),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
