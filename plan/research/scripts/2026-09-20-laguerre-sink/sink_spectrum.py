# ruff: noqa: E402
"""Q31: certified eigenpairs of the Cyclone ky=.55 mode under a declared Laguerre sink.

Derived from Q16's ``eigen_spectrum.py`` (plan/research/scripts/
2026-09-18-eigen-laguerre-spectrum) with one addition: the deck's
``[collisions]`` block is overridden per case with a declared Laguerre
hypercollision sink -- ``hypercollisions_const = 1.0``, ``nu_hyper_l`` from
``--nu-hyper-l``, ``p_hyper_l`` from ``--p-hyper-l`` and
``nu_hyper_m_const = 0.0`` so the constant branch contributes *only* the
Laguerre channel and the shipped ``|k_z|`` Hermite hypercollisions stay exactly
as GX sets them.  Everything else -- deck, geometry, end-damping rate, seed,
route, re-certification and the free-energy spectra -- is Q16's, so a sink rung
is directly comparable with Q16's collisionless and species-nu rungs.

One fresh process per (nu, Nl) case. The route is the runtime ``adaptive``
eigensolve reached through ``run_runtime_linear(solver="krylov",
krylov_cfg=KrylovConfig())``; since #233 the dataclass default is
``method="adaptive"``, which always gates its pair on the original-operator
relative residual. The returned pair is re-certified here with
``_eigenpair_relative_residual`` on an independently built cache, so a route
that reported a pass is checked a second time outside the solver.

Reported per case:

* ``lambda`` as ``(gamma, omega)`` with the solver's own ``eigen_status``
  (residual, tolerance, certified, route) and this script's re-certification;
* the Laguerre and Hermite free-energy spectra of the eigenvector, from GKX's
  own ``distribution_free_energy_resolved`` measure (``Wg_lmst``, i.e. the
  Hermitian mode weight, the field-line volume weight and density*temperature),
  normalized to sum 1, plus tail statistics: the upper-quarter fraction, the
  cutoff-index fraction, the cutoff/peak ratio and a log-slope fitted over the
  top octave of indices;
* ``phi(z)`` of the eigenvector, phase-fixed at its largest-magnitude point, for
  resolution-independent cross-rung comparison.

The eigenvector is written to ``<out>/<key>.npy`` (complex128, phase- and
norm-fixed) for ``overlaps.py``; it is a scratch artifact, never committed, and
its SHA-256 is recorded instead. Output on stdout: progress lines and one
``RESULT {json}`` line.
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
from dataclasses import asdict, replace
from pathlib import Path

T0 = time.perf_counter()
DECK = "cyclone_salpha_itg_sink.toml"
# The float32 image of 0.55 that Q8 and Q3 resolved on this grid; kept so the
# selected ky index is identical to theirs.
KY_TARGET = 0.550000011920929


def log(message: str) -> None:
    print(f"[{time.perf_counter() - T0:9.2f}s] {message}", flush=True)


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


parser = argparse.ArgumentParser()
parser.add_argument("--nu", type=float, default=0.0)
parser.add_argument("--nu-hyper-l", type=float, required=True)
parser.add_argument("--p-hyper-l", type=float, default=6.0)
parser.add_argument(
    "--nu-hyper-m-const",
    type=float,
    default=0.0,
    help="const-branch Hermite rate; 0.0 keeps |k_z| as the only Hermite sink",
)
parser.add_argument("--nl", type=int, required=True)
parser.add_argument("--nm", type=int, default=96)
parser.add_argument("--key", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--deck-dir", type=Path, required=True)
parser.add_argument(
    "--save-vector",
    action="store_true",
    help="write the phase-fixed eigenvector to <out>/<key>.npy for overlaps.py",
)
args = parser.parse_args()

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import solvax

import gkx
from gkx.geometry.core import ensure_flux_tube_geometry_data
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.operators.moments import (
    distribution_free_energy_resolved,
    fieldline_quadrature_weights,
)
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear
from gkx.solvers_linear_krylov import KrylovConfig, _eigenpair_relative_residual
from gkx.terms.assembly import compute_fields_cached
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml

record: dict = {
    "key": args.key,
    "nu": args.nu,
    "nu_hyper_l_arg": args.nu_hyper_l,
    "Nl": args.nl,
    "Nm": args.nm,
    "platform": platform.system(),
    "python": sys.version.split()[0],
    "jax": jax.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "solvax": solvax.__version__,
    "gkx_file": gkx.__file__,
    "x64": bool(jax.config.jax_enable_x64),
    "devices": [str(d) for d in jax.devices()],
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

# ---- deck, collisionality, seed ----
deck = args.deck_dir / DECK
record["deck_sha256"] = hashlib.sha256(deck.read_bytes()).hexdigest()
cfg0, _ = load_runtime_from_toml(deck)
collisions = replace(
    cfg0.collisions,
    hypercollisions_const=1.0 if args.nu_hyper_l != 0.0 else 0.0,
    nu_hyper_l=float(args.nu_hyper_l),
    p_hyper_l=float(args.p_hyper_l),
    nu_hyper_m_const=float(args.nu_hyper_m_const),
)
cfg = replace(
    cfg0,
    species=tuple(replace(s, nu=float(args.nu)) for s in cfg0.species),
    collisions=collisions,
)
record["resolved_species_nu"] = [float(s.nu) for s in cfg.species]
record["nu_hyper_l"] = float(args.nu_hyper_l)
record["p_hyper_l"] = float(args.p_hyper_l)
record["resolved_collisions"] = cfg.collisions.to_dict()
record["resolved_damp_ends_rate"] = cfg.time.damp_ends_rate
record["resolved_grid"] = {
    "Nx": int(cfg.grid.Nx),
    "Ny": int(cfg.grid.Ny),
    "Nz": int(cfg.grid.Nz),
    "ntheta": int(cfg.grid.ntheta),
    "nperiod": int(cfg.grid.nperiod),
    "boundary": str(cfg.grid.boundary),
}

full = _runtime_linear_dispatch_deps().full_deps
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=full,
    ky_target=KY_TARGET,
    n_laguerre=args.nl,
    n_hermite=args.nm,
    solver="krylov",
    fit_signal="auto",
    return_state=True,
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
record["n"] = int(seed.size)
record["selected_ky"] = float(ctx.grid.ky[ctx.selection.ky_index])
record["operator_params"] = {
    "nu_hyper_l": float(ctx.params.nu_hyper_l),
    "p_hyper_l": float(ctx.params.p_hyper_l),
    "nu_hyper_m": float(ctx.params.nu_hyper_m),
    "nu_hyper_m_const": (
        None
        if ctx.params.nu_hyper_m_const is None
        else float(ctx.params.nu_hyper_m_const)
    ),
    "const_branch_nu_hyper_m": float(ctx.params.const_branch_nu_hyper_m()),
    "p_hyper_m": float(ctx.params.p_hyper_m),
    "hypercollisions_const": float(ctx.params.hypercollisions_const),
    "hypercollisions_kz": float(ctx.params.hypercollisions_kz),
    "top_laguerre_rate": float(args.nl * ctx.params.nu_hyper_l),
}
log(f"n={record['n']} shape={seed.shape} ky={record['selected_ky']}")

kcfg = KrylovConfig()
record["krylov_cfg"] = {k: str(v) for k, v in asdict(kcfg).items()}

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
        Nl=args.nl,
        Nm=args.nm,
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
    eigen_status = result.eigen_status
    record["eigen_status"] = None if eigen_status is None else eigen_status.as_dict()
    record["error"] = None

    # ---- independent re-certification on a freshly built cache ----
    vec = jnp.asarray(result.state)
    cache = full.build_linear_cache(ctx.grid, ctx.geom, ctx.params, args.nl, args.nm)
    term_cfg = linear_terms_to_term_config(ctx.terms)
    eig = jnp.asarray(complex(result.gamma, -result.omega), dtype=vec.dtype)
    residual = _eigenpair_relative_residual(eig, vec, cache, ctx.params, term_cfg)
    record["state_dtype"] = str(vec.dtype)
    record["recertified_residual"] = float(residual)

    # ---- normalization: unit 2-norm, phase fixed at the largest component ----
    v = np.asarray(vec, dtype=np.complex128)
    flat = v.reshape(-1)
    pivot = int(np.argmax(np.abs(flat)))
    phase = flat[pivot] / abs(flat[pivot])
    v = (v / phase) / np.linalg.norm(flat)
    record["pivot_index"] = pivot
    record["vector_sha256"] = hashlib.sha256(
        np.ascontiguousarray(v).tobytes()
    ).hexdigest()

    # ---- Laguerre / Hermite free-energy spectra (GKX's own measure) ----
    geom_eff = ensure_flux_tube_geometry_data(ctx.geom, ctx.grid.z)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom_eff, ctx.grid)
    wg = distribution_free_energy_resolved(
        jnp.asarray(v), ctx.grid, ctx.params, vol_fac
    )
    wg_lmst = np.asarray(wg[5])  # (species, Nm, Nl)
    per_species = wg_lmst.sum(axis=(1, 2))
    total = float(per_species.sum())
    laguerre = wg_lmst.sum(axis=(0, 1)) / total  # (Nl,)
    hermite = wg_lmst.sum(axis=(0, 2)) / total  # (Nm,)

    def tail_stats(spectrum: np.ndarray, label: str) -> dict:
        n = int(spectrum.size)
        quarter = float(spectrum[(3 * n) // 4 :].sum())
        peak = int(np.argmax(spectrum))
        top = float(spectrum[-1])
        octave = spectrum[max(n // 2, 1) :]
        idx = np.arange(max(n // 2, 1), n, dtype=float)
        positive = octave > 0.0
        slope = None
        if positive.sum() >= 3:
            slope = float(np.polyfit(idx[positive], np.log10(octave[positive]), 1)[0])
        return {
            "axis": label,
            "size": n,
            "peak_index": peak,
            "peak_value": float(spectrum[peak]),
            "upper_quarter_fraction": quarter,
            "cutoff_fraction": top,
            "cutoff_over_peak": float(top / spectrum[peak]),
            "decades_peak_to_cutoff": (
                None if top <= 0.0 else float(np.log10(spectrum[peak] / top))
            ),
            "log10_slope_per_index_top_half": slope,
        }

    record["laguerre_spectrum"] = [float(x) for x in laguerre]
    record["hermite_spectrum"] = [float(x) for x in hermite]

    def interior_maxima(spectrum: np.ndarray) -> dict:
        """Locate the cutoff pile-up: strict interior local maxima of W(l)."""

        idx = [
            int(i)
            for i in range(1, int(spectrum.size) - 1)
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]
        ]
        last = idx[-1] if idx else None
        return {
            "count": len(idx),
            "indices": idx,
            "last_index": last,
            "last_fraction_of_Nl": (
                None if last is None else float(last) / float(spectrum.size)
            ),
            "monotone_decaying": len(idx) == 0,
        }

    record["laguerre_interior_maxima"] = interior_maxima(laguerre)
    record["laguerre_tail"] = tail_stats(laguerre, "laguerre")
    record["hermite_tail"] = tail_stats(hermite, "hermite")
    record["free_energy_total"] = total

    # ---- phi(z), phase-fixed, for resolution-independent overlaps ----
    fields = compute_fields_cached(jnp.asarray(v), cache, ctx.params, terms=term_cfg)
    phi = np.asarray(fields.phi, dtype=np.complex128).reshape(-1)
    pivot_phi = int(np.argmax(np.abs(phi)))
    phi = (phi / (phi[pivot_phi] / abs(phi[pivot_phi]))) / np.linalg.norm(phi)
    record["phi_z_real"] = [float(x) for x in phi.real]
    record["phi_z_imag"] = [float(x) for x in phi.imag]
    record["z"] = [float(x) for x in np.asarray(ctx.grid.z).reshape(-1)]

    if args.save_vector:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.key}.npy"
        np.save(path, v)
        record["vector_path"] = str(path)
        record["vector_bytes"] = int(path.stat().st_size)
except Exception as exc:  # rejections are results, not failures
    record["route_s"] = time.perf_counter() - start
    record["peak_rss_route_kib"] = peak_rss_kib()
    record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    record["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
    log(f"route raised {record['error']}")

record["statuses"] = statuses
record["loadavg_end"] = os.getloadavg()
record["process_peak_rss_kib"] = peak_rss_kib()
print("RESULT " + json.dumps(record), flush=True)
