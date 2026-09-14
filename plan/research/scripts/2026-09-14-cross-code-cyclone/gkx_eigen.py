#!/usr/bin/env python3
"""Q20 GKX certified eigenpair for one (geometry, ky, Nl, Nm).

    python gkx_eigen.py --geometry S|M --ky 0.30 --Nl 16 --Nm 48 --repo <source root>

S = examples/linear/axisymmetric/cyclone.toml (s-alpha); M = cyclone_miller_linear.toml next to this
script (same deck, circular Miller). Route: run_runtime_linear(solver="krylov", krylov_cfg=None), i.e. the
runtime default `adaptive` certified route. The original-operator relative residual of the returned
pair is recomputed here, and the Laguerre/Hermite free-energy spectra of the eigenvector are recorded.
Prints one line "RESULT {json}".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

T0 = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - T0:9.2f}s] {message}", flush=True)


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=("S", "M"), required=True)
parser.add_argument("--ky", type=float, required=True)
parser.add_argument("--Nl", type=int, required=True)
parser.add_argument("--Nm", type=int, required=True)
parser.add_argument("--repo", type=Path, default=Path.cwd())
args = parser.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import solvax  # noqa: E402

import gkx  # noqa: E402
from gkx.operators.linear.params import linear_terms_to_term_config  # noqa: E402
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear  # noqa: E402
from gkx.solvers_linear_krylov import (  # noqa: E402
    _eigenpair_relative_residual,
    certifiable_residual_tolerance,
)
from gkx.workflows.linear import _prepare_linear_runtime_context  # noqa: E402
from gkx.workflows.runtime.startup import _runtime_default_krylov_config  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

here = Path(__file__).resolve().parent
deck = (
    args.repo / "examples/linear/axisymmetric/cyclone.toml"
    if args.geometry == "S"
    else here / "cyclone_miller_linear.toml"
)
record: dict = {
    "geometry": args.geometry,
    "ky_target": args.ky,
    "Nl": args.Nl,
    "Nm": args.Nm,
    "deck": str(deck.name),
    "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
    "host": platform.node(),
    "python": sys.version.split()[0],
    "jax": jax.__version__,
    "numpy": np.__version__,
    "solvax": solvax.__version__,
    "gkx_file": gkx.__file__,
    "x64": bool(jax.config.jax_enable_x64),
    "env": {
        k: os.environ.get(k)
        for k in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "GKX_X64", "XLA_FLAGS", "OMP_NUM_THREADS")
    },
    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "loadavg_start": os.getloadavg(),
}
if not record["x64"] or not str(gkx.__file__).startswith(str(args.repo.resolve())):
    raise SystemExit(f"environment check failed: {record}")
log(f"env ok gkx={gkx.__file__} jax={jax.__version__} solvax={solvax.__version__}")

cfg, _ = load_runtime_from_toml(deck)
full = _runtime_linear_dispatch_deps().full_deps
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=full,
    ky_target=args.ky,
    n_laguerre=args.Nl,
    n_hermite=args.Nm,
    solver="krylov",
    fit_signal="auto",
    return_state=False,
    initial_state=None,
    status_callback=None,
)
seed = np.asarray(ctx.initial_state).astype(np.complex128)
record["n"] = int(seed.size)
record["shape"] = list(seed.shape)
resolved = _runtime_default_krylov_config(cfg)
record["krylov_cfg"] = {k: str(v) for k, v in asdict(resolved).items()}
gate = certifiable_residual_tolerance(1.0e-9, seed.dtype)
record["residual_gate"] = gate
statuses: list = []


def status(message: str) -> None:
    statuses.append([round(time.perf_counter() - T0, 3), message])
    log(f"status: {message}")


start = time.perf_counter()
try:
    result = run_runtime_linear(
        cfg,
        ky_target=args.ky,
        Nl=args.Nl,
        Nm=args.Nm,
        solver="krylov",
        krylov_cfg=None,
        return_state=True,
        initial_state=seed,
        status_callback=status,
    )
    record["route_s"] = time.perf_counter() - start
    record["gamma"] = float(result.gamma)
    record["omega"] = float(result.omega)
    record["result_ky"] = float(result.ky)
    vec = jnp.asarray(result.state)
    eig = jnp.asarray(complex(result.gamma, -result.omega), dtype=vec.dtype)
    cache = full.build_linear_cache(ctx.grid, ctx.geom, ctx.params, args.Nl, args.Nm)
    residual = _eigenpair_relative_residual(
        eig, vec, cache, ctx.params, linear_terms_to_term_config(ctx.terms)
    )
    record["residual"] = float(residual)
    record["certified"] = bool(np.isfinite(residual) and residual <= gate)
    power = np.abs(np.asarray(vec)) ** 2
    axes = tuple(range(power.ndim))
    spec_l = power.sum(axis=tuple(a for a in axes if a != 1))
    spec_m = power.sum(axis=tuple(a for a in axes if a != 2))
    spec_l = spec_l / spec_l.sum()
    spec_m = spec_m / spec_m.sum()
    record["laguerre_spectrum"] = [float(x) for x in spec_l]
    record["hermite_upper_quarter_fraction"] = float(spec_m[3 * args.Nm // 4 :].sum())
    record["laguerre_upper_quarter_fraction"] = float(spec_l[3 * args.Nl // 4 :].sum())
    record["error"] = None
except Exception as exc:  # a rejection is a result
    record["route_s"] = time.perf_counter() - start
    record["certified"] = False
    record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    record["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    log(f"route raised {record['error']}")
record["statuses"] = statuses
record["peak_rss_kib"] = peak_rss_kib()
record["loadavg_end"] = os.getloadavg()
print("RESULT " + json.dumps(record), flush=True)
