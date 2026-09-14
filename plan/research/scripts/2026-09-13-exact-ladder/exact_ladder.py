# ruff: noqa: E402
"""Q2: exact sparse shift-invert size ladder on the linked Cyclone deck.

One fresh process per (rung, arm), all through ``run_runtime_linear(solver="krylov")``
with an explicit complex128 seed built by the runtime's own initial condition:

* ``sparse``     ``KrylovConfig(method="sparse_shift_invert", shift=sigma)``
* ``default``    ``krylov_cfg=None`` (the runtime default; for the cyclone contract it
  resolves to ``KrylovConfig(method="adaptive")``, the certified adaptive propagator)
* ``propagator`` ``KrylovConfig()`` (dataclass default ``method="propagator"``; this
  branch has no certification gate, the residual is measured here)

The SOLVAX helpers the sparse branch imports at call time are wrapped at module
attribute level for timing and nnz only; no GKX or SOLVAX source is changed.
Every arm's eigenpair is re-certified against the original matrix-free operator with
``_eigenpair_relative_residual``. Output: progress lines and one ``RESULT {json}``.

``--summarize DIR`` prints the per-rung table from ``DIR/r*_*.txt`` and the
``/usr/bin/time -v`` files next to them.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

T0 = time.perf_counter()
RUNGS = {  # rung: (Nz, ntheta, nperiod, Nl, Nm)
    1: (32, 32, 1, 4, 8),
    2: (48, 16, 2, 8, 16),
    3: (96, 32, 2, 8, 24),
    4: (96, 32, 2, 16, 48),
}
REFERENCE_SHIFT = complex(0.09302951, -0.28199404)
DECK = "examples/linear/axisymmetric/cyclone.toml"
KY_TARGET = 0.3


def log(message: str) -> None:
    print(f"[{time.perf_counter() - T0:9.2f}s] {message}", flush=True)


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def summarize(directory: Path) -> None:
    rows = {}
    for path in sorted(directory.glob("r*_*.txt")):
        if path.name.endswith(".time.txt"):
            continue
        record = None
        for line in path.read_text().splitlines():
            if line.startswith("RESULT "):
                record = json.loads(line[len("RESULT ") :])
        time_path = path.with_name(path.stem + ".time.txt")
        max_rss = wall = exit_status = None
        if time_path.exists():
            for line in time_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("Maximum resident set size"):
                    max_rss = int(line.split(":")[-1])
                elif line.startswith("Elapsed (wall clock) time"):
                    wall = line.split("): ")[-1]
                elif line.startswith("Exit status"):
                    exit_status = int(line.split(":")[-1])
                elif line.startswith("Command terminated by signal"):
                    exit_status = line
        rows[path.stem] = (record, max_rss, wall, exit_status)
    print(
        "| rung (Nz,ntheta,nperiod,Nl,Nm) | arm | n | batches | nnz | nnz/row | "
        "L+U nnz (fill) | asm s | factor s | eigs s (LU solves) | route s | "
        "process wall | peak RSS GiB | residual | certified | gamma | omega | note |"
    )
    print("|" + "---|" * 18)
    for stem, (record, max_rss, wall, exit_status) in rows.items():
        rss = "—" if max_rss is None else f"{max_rss / 1024**2:.2f}"
        if record is None:
            print(
                f"| {stem} | — | — | — | — | — | — | — | — | — | — | {wall} | {rss} | "
                f"— | — | — | — | no RESULT line; exit {exit_status} |"
            )
            continue
        sparse = record.get("sparse", {})
        nnz = sparse.get("nnz")
        lu = sparse.get("lu_nnz")
        n = record["n"]

        def fmt(value, spec):
            return "—" if value is None else format(value, spec)

        fill = f"{lu} ({lu / nnz:.1f})" if lu and nnz else "—"
        eigs = (
            f"{sparse['eigs_s']:.2f} ({sparse['lu_solves']}, {sparse['lu_solve_s']:.2f})"
            if "eigs_s" in sparse
            else "—"
        )
        note = record.get("error") or ""
        note = note.replace("|", "/")[:160]
        print(
            f"| {record['rung']} {tuple(record['rung_dims'])} | {record['arm']} | {n} | "
            f"{fmt(sparse.get('probe_batches'), 'd')} | {fmt(nnz, 'd')} | "
            f"{fmt(None if nnz is None else nnz / n, '.1f')} | {fill} | "
            f"{fmt(sparse.get('assembly_s'), '.2f')} | {fmt(sparse.get('factor_s'), '.2f')} | "
            f"{eigs} | {fmt(record.get('route_s'), '.2f')} | {wall} | {rss} | "
            f"{fmt(record.get('residual'), '.2e')} | {record.get('certified')} | "
            f"{fmt(record.get('gamma'), '.6f')} | {fmt(record.get('omega'), '.6f')} | {note} |"
        )


parser = argparse.ArgumentParser()
parser.add_argument("--rung", type=int, choices=sorted(RUNGS))
parser.add_argument("--arm", choices=("sparse", "default", "propagator"))
parser.add_argument("--shift", type=complex, default=REFERENCE_SHIFT)
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--summarize", type=Path)
args = parser.parse_args()
if args.summarize is not None:
    summarize(args.summarize)
    sys.exit(0)

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import solvax

import gkx
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear
from gkx.solvers_linear_krylov import (
    KrylovConfig,
    _eigenpair_relative_residual,
    certifiable_residual_tolerance,
)
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.startup import _runtime_default_krylov_config
from gkx.workflows.runtime.toml import load_runtime_from_toml

Nz, ntheta, nperiod, Nl, Nm = RUNGS[args.rung]
record: dict = {
    "rung": args.rung,
    "rung_dims": [Nz, ntheta, nperiod, Nl, Nm],
    "arm": args.arm,
    "host": platform.node(),
    "python": sys.version.split()[0],
    "jax": jax.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "solvax": solvax.__version__,
    "gkx_file": gkx.__file__,
    "x64": bool(jax.config.jax_enable_x64),
    "cpu_count": os.cpu_count(),
    "env": {
        key: os.environ.get(key)
        for key in (
            "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES",
            "JAX_ENABLE_X64",
            "GKX_X64",
            "XLA_FLAGS",
            "PYTHONPATH",
        )
    },
    "loadavg_start": os.getloadavg(),
}
if not record["x64"] or not str(gkx.__file__).startswith(str(args.repo)):
    raise SystemExit(f"environment check failed: {record}")
log(f"env ok gkx={gkx.__file__} jax={jax.__version__} solvax={solvax.__version__}")

# ---- instrumentation of the helpers the sparse branch imports at call time ----
sparse_rec: dict = {}
_sparse_operator_matrix = solvax.sparse_operator_matrix
_sparse_eigenpairs = solvax.sparse_eigenpairs
_Splu = solvax.SpluFactorization


def timed_sparse_operator_matrix(apply, prototype, **kwargs):
    batch = int(kwargs.get("batch_size", 64))
    log(f"probe assembly start n={int(prototype.size)} batch_size={batch}")
    start = time.perf_counter()
    matrix = _sparse_operator_matrix(apply, prototype, **kwargs)
    sparse_rec.update(
        assembly_s=time.perf_counter() - start,
        nnz=int(matrix.nnz),
        probe_batch_size=batch,
        probe_batches=math.ceil(int(prototype.size) / batch),
        drop_tolerance=kwargs.get("drop_tolerance"),
        rss_after_assembly_kib=peak_rss_kib(),
    )
    log(f"probe assembly done nnz={matrix.nnz} {sparse_rec['assembly_s']:.2f}s")
    return matrix


class TimedSplu(_Splu):
    def __init__(self, matrix):
        log("SuperLU factor start")
        start = time.perf_counter()
        super().__init__(matrix)
        sparse_rec["factor_s"] = time.perf_counter() - start
        sparse_rec["lu_nnz"] = int(getattr(self._lu, "nnz", -1))
        sparse_rec["rss_after_factor_kib"] = peak_rss_kib()
        sparse_rec["lu_solves"] = 0
        sparse_rec["lu_solve_s"] = 0.0
        log(
            f"SuperLU factor done L+U nnz={sparse_rec['lu_nnz']} "
            f"{sparse_rec['factor_s']:.2f}s"
        )

    def _solve_numpy(self, b, *, trans="N"):
        start = time.perf_counter()
        out = super()._solve_numpy(b, trans=trans)
        sparse_rec["lu_solves"] += 1
        sparse_rec["lu_solve_s"] += time.perf_counter() - start
        return out


def timed_sparse_eigenpairs(matrix, **kwargs):
    log(f"ARPACK shift-invert start candidates={kwargs.get('candidates')}")
    start = time.perf_counter()
    modes = _sparse_eigenpairs(matrix, **kwargs)
    sparse_rec["eigs_s"] = time.perf_counter() - start
    sparse_rec["candidates"] = [
        {
            "eig": [float(v.real), float(v.imag)],
            "solvax_residual": float(r),
            "converged": bool(c),
        }
        for v, r, c in zip(
            np.asarray(modes.eigenvalues),
            np.asarray(modes.residuals),
            np.asarray(modes.converged),
            strict=True,
        )
    ]
    log(f"ARPACK done {sparse_rec['eigs_s']:.2f}s")
    return modes


solvax.sparse_operator_matrix = timed_sparse_operator_matrix
solvax.sparse_eigenpairs = timed_sparse_eigenpairs
solvax.SpluFactorization = TimedSplu

# ---- deck, grid, signed ky and complex128 seed ----
repo = args.repo
deck = repo / DECK
record["deck_sha256"] = hashlib.sha256(deck.read_bytes()).hexdigest()
cfg0, _ = load_runtime_from_toml(deck)
cfg = replace(cfg0, grid=replace(cfg0.grid, Nz=Nz, ntheta=ntheta, nperiod=nperiod))
full = _runtime_linear_dispatch_deps().full_deps
geom = full.build_runtime_geometry(cfg)
grid_full = full.build_spectral_grid(full.apply_geometry_grid_defaults(geom, cfg.grid))
ky_all = np.asarray(grid_full.ky, dtype=float)
ky_index = int(full.select_ky_index(ky_all, KY_TARGET))
ky_sel = float(ky_all[ky_index])
record["ky"] = {
    "grid_Ny": int(cfg.grid.Ny),
    "grid_size": int(ky_all.size),
    "selected_index": ky_index,
    "selected_ky": ky_sel,
    "signed_match": bool(np.sign(ky_sel) == np.sign(KY_TARGET)),
    "max_abs_ky": float(np.max(np.abs(ky_all))),
    "nyquist": bool(np.isclose(abs(ky_sel), np.max(np.abs(ky_all)))),
    "opposite_sign_partner": bool(np.any(np.isclose(ky_all, -ky_sel))),
    "grid_Nz": int(np.asarray(grid_full.z).size),
}
log(f"ky {record['ky']}")
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=full,
    ky_target=KY_TARGET,
    n_laguerre=Nl,
    n_hermite=Nm,
    solver="krylov",
    fit_signal="auto",
    return_state=False,
    initial_state=None,
    status_callback=None,
)
seed_native = np.asarray(ctx.initial_state)
seed = seed_native.astype(np.complex128)
record["seed"] = {
    "native_dtype": str(seed_native.dtype),
    "dtype": str(seed.dtype),
    "shape": list(seed.shape),
    "sha256": hashlib.sha256(np.ascontiguousarray(seed).tobytes()).hexdigest(),
}
n = int(seed.size)
record["n"] = n
record["shift"] = [args.shift.real, args.shift.imag]
log(f"n={n} shape={seed.shape} seed native dtype={seed_native.dtype}")

if args.arm == "sparse":
    kcfg = KrylovConfig(method="sparse_shift_invert", shift=args.shift)
    gate = certifiable_residual_tolerance(kcfg.shift_outer_residual_tol, seed.dtype)
elif args.arm == "default":
    kcfg = None
    gate = certifiable_residual_tolerance(1.0e-9, seed.dtype)  # _ADAPTIVE_BASE_TOL
else:
    kcfg = KrylovConfig()
    gate = certifiable_residual_tolerance(1.0e-6, seed.dtype)  # reported, not enforced
resolved = kcfg if kcfg is not None else _runtime_default_krylov_config(cfg)
record["krylov_cfg"] = {k: str(v) for k, v in asdict(resolved).items()}
record["residual_gate"] = gate

statuses: list = []


def status(message: str) -> None:
    statuses.append([round(time.perf_counter() - T0, 3), message])
    log(f"status: {message}")


record["rss_before_route_kib"] = peak_rss_kib()
start = time.perf_counter()
try:
    result = run_runtime_linear(
        cfg,
        ky_target=KY_TARGET,
        Nl=Nl,
        Nm=Nm,
        solver="krylov",
        krylov_cfg=kcfg,
        return_state=True,
        initial_state=seed,
        status_callback=status,
    )
    record["route_s"] = time.perf_counter() - start
    record["peak_rss_route_kib"] = peak_rss_kib()
    record["gamma"] = float(result.gamma)
    record["omega"] = float(result.omega)
    record["result_ky"] = float(result.ky)
    vec = jnp.asarray(result.state)
    eig = jnp.asarray(complex(result.gamma, -result.omega), dtype=vec.dtype)
    cache = full.build_linear_cache(ctx.grid, ctx.geom, ctx.params, Nl, Nm)
    term_cfg = linear_terms_to_term_config(ctx.terms)
    residual = _eigenpair_relative_residual(eig, vec, cache, ctx.params, term_cfg)
    record["state_dtype"] = str(vec.dtype)
    record["residual"] = float(residual)
    record["certified"] = bool(np.isfinite(residual) and residual <= gate)
    record["error"] = None
except Exception as exc:  # rejections are results, not failures
    record["route_s"] = time.perf_counter() - start
    record["peak_rss_route_kib"] = peak_rss_kib()
    record["certified"] = False
    record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    record["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    log(f"route raised {record['error']}")
record["sparse"] = sparse_rec
record["statuses"] = statuses
record["loadavg_end"] = os.getloadavg()
record["process_peak_rss_kib"] = peak_rss_kib()
print("RESULT " + json.dumps(record), flush=True)
