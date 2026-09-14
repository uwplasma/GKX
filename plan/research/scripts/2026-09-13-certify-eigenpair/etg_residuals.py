# ruff: noqa: E402
"""Q12: original-operator residual of every pair behind ledger row ``L-lin-etg``.

``L-lin-etg`` (``docs/_static/etg_mismatch_table.csv``) is written by
``tools/artifacts/make_tables.py --case etg`` (``ETG_SOLVER = "time"``): an RK4
runtime scan of ``examples/linear/axisymmetric/etg.toml`` at ``Nl=24, Nm=8`` over
the reference ky. ``ETG_KRYLOV_DEFAULT`` is not on that path. This script measures
both, one fresh process per (arm, ky):

* ``scan``            the generator's exact ``run_runtime_scan`` call (all ky, batched);
  reproduces the tracked CSV numbers.
* ``time``            the same time-integration controls per ky with
  ``return_state=True``; the final state is checked against the fitted
  eigenvalue and against its own Rayleigh quotient.
* ``etg_default``     ``krylov_cfg=ETG_KRYLOV_DEFAULT`` (raw propagator).
* ``runtime_default`` ``krylov_cfg=None`` (ETG runtime default: shift-invert with a
  gated Arnoldi fallback).
* ``adaptive``        ``KrylovConfig(method="adaptive")``.

Every pair is re-checked with ``_eigenpair_relative_residual`` against the
matrix-free operator. Output: progress lines and one ``RESULT {json}`` line.
"""

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
DECK = "examples/linear/axisymmetric/etg.toml"
N_LAGUERRE = 24
N_HERMITE = 8

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arm",
    required=True,
    choices=("scan", "time", "etg_default", "runtime_default", "adaptive"),
)
parser.add_argument("--ky", type=float, default=None)
parser.add_argument("--repo", type=Path, default=Path.cwd())
args = parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - T0:9.2f}s] {message}", flush=True)


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


import jax
import jax.numpy as jnp
import numpy as np
import solvax

import gkx
from gkx.benchmarking_shared import ETG_KRYLOV_DEFAULT, load_etg_reference
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear
from gkx.runtime import run_runtime_scan
from gkx.solvers_linear_krylov import (
    KrylovConfig,
    _apply_operator,
    _eigenpair_relative_residual,
    certifiable_residual_tolerance,
)
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.startup import _runtime_default_krylov_config
from gkx.workflows.runtime.toml import load_runtime_from_toml

record: dict = {
    "arm": args.arm,
    "ky_target": args.ky,
    "host": platform.node(),
    "python": sys.version.split()[0],
    "jax": jax.__version__,
    "numpy": np.__version__,
    "solvax": solvax.__version__,
    "gkx_file": gkx.__file__,
    "x64": bool(jax.config.jax_enable_x64),
    "env": {
        key: os.environ.get(key)
        for key in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "GKX_X64", "XLA_FLAGS")
    },
    "loadavg_start": os.getloadavg(),
}
if not record["x64"] or not str(gkx.__file__).startswith(str(args.repo)):
    raise SystemExit(f"environment check failed: {record}")
deck = args.repo / DECK
record["deck_sha256"] = hashlib.sha256(deck.read_bytes()).hexdigest()
cfg, _ = load_runtime_from_toml(deck)
ref = load_etg_reference()
time_options = dict(
    method=cfg.time.method,
    dt=cfg.time.dt,
    steps=int(round(cfg.time.t_max / cfg.time.dt)),
    sample_stride=cfg.time.sample_stride,
    auto_window=False,
    tmin=1.0,
    tmax=cfg.time.t_max,
    fit_signal="phi",
    mode_method="z_index",
)
record["time_options"] = time_options
log(f"env ok gkx={gkx.__file__} jax={jax.__version__} solvax={solvax.__version__}")

if args.arm == "scan":
    start = time.perf_counter()
    scan = run_runtime_scan(
        cfg,
        np.asarray(ref.ky, dtype=float),
        Nl=N_LAGUERRE,
        Nm=N_HERMITE,
        solver="time",
        batch_ky=True,
        show_progress=False,
        **time_options,
    )
    record["route_s"] = time.perf_counter() - start
    record["rows"] = [
        {"ky": float(k), "gamma": float(g), "omega": float(w)}
        for k, g, w in zip(scan.ky, scan.gamma, scan.omega, strict=True)
    ]
    record["process_peak_rss_kib"] = peak_rss_kib()
    record["loadavg_end"] = os.getloadavg()
    print("RESULT " + json.dumps(record), flush=True)
    sys.exit(0)

full = _runtime_linear_dispatch_deps().full_deps
solver = "time" if args.arm == "time" else "krylov"
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=full,
    ky_target=float(args.ky),
    n_laguerre=N_LAGUERRE,
    n_hermite=N_HERMITE,
    solver=solver,
    fit_signal="phi",
    return_state=False,
    initial_state=None,
    status_callback=None,
)
record["n"] = int(np.asarray(ctx.initial_state).size)
record["seed_dtype"] = str(np.asarray(ctx.initial_state).dtype)
idx = int(np.argmin(np.abs(np.asarray(ref.ky) - float(args.ky))))
record["reference"] = {"gamma": float(ref.gamma[idx]), "omega": float(ref.omega[idx])}

base_tol = 1.0e-6
if args.arm == "etg_default":
    kcfg = ETG_KRYLOV_DEFAULT
elif args.arm == "runtime_default":
    kcfg = None
elif args.arm == "adaptive":
    kcfg = KrylovConfig(method="adaptive")
    base_tol = 1.0e-9
else:
    kcfg = None
if solver == "krylov":
    resolved = kcfg if kcfg is not None else _runtime_default_krylov_config(cfg)
    record["krylov_cfg"] = {k: str(v) for k, v in asdict(resolved).items()}

statuses: list = []


def status(message: str) -> None:
    statuses.append([round(time.perf_counter() - T0, 3), message])
    log(f"status: {message}")


run_options = dict(time_options) if solver == "time" else {}
start = time.perf_counter()
try:
    result = run_runtime_linear(
        cfg,
        ky_target=float(args.ky),
        Nl=N_LAGUERRE,
        Nm=N_HERMITE,
        solver=solver,
        krylov_cfg=kcfg,
        return_state=True,
        status_callback=status,
        **run_options,
    )
    record["route_s"] = time.perf_counter() - start
    record["gamma"] = float(result.gamma)
    record["omega"] = float(result.omega)
    record["result_ky"] = float(result.ky)
    vec = jnp.asarray(result.state)
    record["state_dtype"] = str(vec.dtype)
    cache = full.build_linear_cache(
        ctx.grid, ctx.geom, ctx.params, N_LAGUERRE, N_HERMITE
    )
    term_cfg = linear_terms_to_term_config(ctx.terms)
    eig = jnp.asarray(complex(result.gamma, -result.omega), dtype=vec.dtype)
    residual = _eigenpair_relative_residual(eig, vec, cache, ctx.params, term_cfg)
    image = _apply_operator(vec, cache, ctx.params, term_cfg)
    rayleigh = jnp.vdot(vec, image) / jnp.vdot(vec, vec)
    rq_residual = _eigenpair_relative_residual(
        rayleigh, vec, cache, ctx.params, term_cfg
    )
    gate = certifiable_residual_tolerance(base_tol, vec.dtype)
    record["residual"] = float(residual)
    record["rayleigh"] = [float(jnp.real(rayleigh)), float(jnp.imag(rayleigh))]
    record["rayleigh_residual"] = float(rq_residual)
    record["gate"] = gate
    record["certified"] = bool(np.isfinite(residual) and residual <= gate)
    record["error"] = None
except Exception as exc:  # rejections are results, not failures
    record["route_s"] = time.perf_counter() - start
    record["certified"] = False
    record["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    record["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    log(f"route raised {record['error']}")
record["statuses"] = statuses
record["process_peak_rss_kib"] = peak_rss_kib()
record["loadavg_end"] = os.getloadavg()
print("RESULT " + json.dumps(record), flush=True)
