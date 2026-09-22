"""PERF-LIT profile: forward step, window adjoint and linear eigen derivative.

One process per measurement. Each subcommand prints one ``RESULT {json}`` line
and, with ``--out``, writes the same record. Timings separate lowering+compile
from warm execution (median of ``--repeats`` after one discarded call, every
call synchronized). Memory is XLA's own buffer assignment
(``compiled.memory_analysis()``), device ``peak_bytes_in_use`` where the
backend reports it, and host peak RSS. HLO counts use the op-name ledger of
``tools/profiling/profile_runtime_kernels.py`` (``materialized_bytes`` counts
only instructions that own a buffer, per queue row Q27 / PR #259).

Subcommands
-----------
forward   one nonlinear RHS, its field solve and linear part, and one RK step
window    value and value+gradient of gkx.nonlinear_heat_flux_window with
          respect to (tprim scale, geometry arrays)
eigen     value+gradient of the dense linear growth-rate objective VMEX calls
fftfloor  time the exact FFT instructions one compiled RHS contains

Run from the repository root with ``PYTHONPATH=src``. Not a turbulence result.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TOML = ROOT / "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"


def _env() -> dict[str, Any]:
    import jax
    import jaxlib

    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001
        sha, dirty = None, None
    return {
        "git_sha": sha,
        "src_dirty": dirty,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "loadavg_start": list(os.getloadavg()),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
    }


def _host_peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _device_peak() -> int | None:
    import jax

    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return int(stats.get("peak_bytes_in_use", 0))


def _block(tree: Any) -> Any:
    import jax

    jax.tree.map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, tree)
    return tree


def _measure(fn, args: tuple, repeats: int, hlo: bool = True) -> dict[str, Any]:
    """Compile once, run once (discarded), then time ``repeats`` warm calls."""

    from tools.profiling.profile_runtime_kernels import _hlo_op_counts

    t0 = time.perf_counter()
    lowered = fn.lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    mem = compiled.memory_analysis()
    record: dict[str, Any] = {
        "lower_s": t1 - t0,
        "compile_s": t2 - t1,
        "temp_bytes": int(getattr(mem, "temp_size_in_bytes", 0) or 0),
        "argument_bytes": int(getattr(mem, "argument_size_in_bytes", 0) or 0),
        "output_bytes": int(getattr(mem, "output_size_in_bytes", 0) or 0),
        "generated_code_bytes": int(getattr(mem, "generated_code_size_in_bytes", 0) or 0),
    }
    if hlo:
        text = compiled.as_text()
        counts = _hlo_op_counts(text)
        record["hlo"] = {
            k: counts[k]
            for k in (
                "fft", "transpose", "copy", "concatenate", "gather", "scatter",
                "dot", "fusion", "while", "materialized_bytes",
                "fused_interior_bytes", "constant_bytes",
            )
            if k in counts
        }
        record["hlo"]["fft_shapes"] = _fft_shapes(text)
    t3 = time.perf_counter()
    out = _block(compiled(*args))
    record["first_call_s"] = time.perf_counter() - t3
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = _block(compiled(*args))
        times.append(time.perf_counter() - t)
    record["warm_s"] = times
    record["warm_median_s"] = statistics.median(times)
    record["device_peak_bytes"] = _device_peak()
    record["host_peak_bytes"] = _host_peak_bytes()
    return record, out


_FFT = re.compile(
    r"=\s*(?P<dtype>[a-z0-9]+)\[(?P<shape>[0-9,]*)\][^ ]*\s+fft\((?P<operand>[^)]*)\).*?fft_type=(?P<type>[A-Z0-9]+).*?fft_length=\{(?P<len>[0-9,]+)\}"
)


def _fft_shapes(text: str) -> list[dict[str, Any]]:
    tally: dict[tuple, int] = {}
    for line in text.splitlines():
        m = _FFT.search(line)
        if not m:
            continue
        key = (m.group("type"), m.group("dtype"), m.group("shape"), m.group("len"))
        tally[key] = tally.get(key, 0) + 1
    return [
        {"type": k[0], "out_dtype": k[1], "out_shape": k[2], "fft_length": k[3], "count": v}
        for k, v in sorted(tally.items())
    ]


# Differentiated geometry arrays. gds21/gds22 (with s_hat) fix the integer
# twist-shift link map on a linked deck and are refused under a trace.
GEOMETRY_FIELDS = (
    "bmag_profile", "bgrad_profile", "gds2_profile", "cv_profile", "gb_profile",
    "cv0_profile", "gb0_profile", "jacobian_profile", "grho_profile",
)


def _with_geometry(geometry, values):
    return dataclasses.replace(geometry, **dict(zip(GEOMETRY_FIELDS, values)))


def _hlo_source(text: str, name: str) -> str | None:
    """op_name/source line of the HLO instruction a trace kernel is named after."""

    m = re.search(r"%" + re.escape(name) + r" = [^\n]*", text)
    if not m:
        return None
    line = m.group(0)
    meta = re.search(r'op_name="([^"]*)".*?source_file="([^"]*)" source_line=(\d+)', line)
    if meta:
        return f"{meta.group(1)[-80:]} @ {meta.group(2).split('/src/')[-1]}:{meta.group(3)}"
    calls = re.search(r"calls=(%[\w.\-]+)", line)
    return f"{line[:160]}" if not calls else f"fusion {calls.group(1)}"


def _trace_top_ops(fn, args: tuple, trace_dir: Path, calls: int) -> dict[str, Any]:
    """Perfetto trace of ``calls`` warm calls; total duration by event name.

    Device (GPU) kernels are grouped by kernel name; on CPU the XLA thunk
    events are grouped the same way. Durations are summed over ``calls``.
    """

    import glob
    import gzip

    import jax

    _block(fn(*args))
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for _ in range(calls):
            _block(fn(*args))
    files = sorted(glob.glob(str(trace_dir / "**/*.trace.json.gz"), recursive=True), key=os.path.getmtime)
    if not files:
        return {"error": "no perfetto trace written"}
    data = json.loads(gzip.open(files[-1]).read())
    events = data["traceEvents"] if isinstance(data, dict) else data
    pnames = {e["pid"]: e["args"].get("name", "") for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    device = [p for p, n in pnames.items() if "/device:" in n and "CPU" not in n]
    keep = set(device) if device else set(pnames)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in keep:
            continue
        name = e.get("name", "?")
        if not device and (name.startswith("$") or "Executor" in name or "PjRt" in name or "Thunk" in name.split(":")[0]):
            continue  # host-side Python frames and executor wrappers
        totals[name] = totals.get(name, 0.0) + float(e.get("dur", 0.0))
        counts[name] = counts.get(name, 0) + 1
    grand = sum(totals.values())
    hlo_text = fn.lower(*args).compile().as_text()
    cats: dict[str, float] = {}
    for name, us in totals.items():
        low = name.lower()
        if "fft" in low:
            cat = "fft"
        elif "concatenate" in low:
            cat = "concatenate"
        elif "transpose" in low or "copy" in low:
            cat = "transpose_copy"
        elif "gather" in low or "scatter" in low or "dynamic" in low:
            cat = "gather_scatter"
        elif "dot" in low or "gemm" in low or "cublas" in low or "matmul" in low:
            cat = "matmul"
        elif "reduce" in low:
            cat = "reduce"
        elif "pjit" in low or "memcpy" in low:
            cat = "dispatch_memcpy"
        else:
            cat = "other_fusion"
        cats[cat] = cats.get(cat, 0.0) + us
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:30]
    return {
        "calls": calls, "scope": "device" if device else "all processes",
        "total_us": grand,
        "by_category": {k: {"us": v, "frac": v / grand if grand else 0.0} for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
        "top": [
            {"name": k[:120], "us": v, "frac": v / grand if grand else 0.0, "count": counts[k],
             "source": _hlo_source(hlo_text, k)}
            for k, v in top
        ],
    }


def build_case(nx: int, ny: int, nz: int, nl: int | None, nm: int | None) -> dict[str, Any]:
    """The shipped Cyclone nonlinear deck at an overridden grid and moment count."""

    from gkx.core_grid import build_spectral_grid
    from gkx.geometry import ensure_flux_tube_geometry_data
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(TOML)
    runtime = dataclasses.replace(
        runtime,
        grid=dataclasses.replace(runtime.grid, Nx=nx, Ny=ny, Nz=nz, ntheta=None),
    )
    grid = build_spectral_grid(runtime.grid)
    geometry = ensure_flux_tube_geometry_data(build_runtime_geometry(runtime), grid.z)
    term_cfg = build_runtime_term_config(runtime)
    assert float(term_cfg.nonlinear) != 0.0
    n_l = int(nl or raw["run"]["Nl"])
    n_m = int(nm or raw["run"]["Nm"])
    params = build_runtime_linear_params(runtime, Nm=n_m, geom=geometry)
    shape = (len(runtime.species), n_l, n_m, grid.ky.size, grid.kx.size, grid.z.size)
    return {
        "grid": grid, "geometry": geometry, "params": params, "terms": term_cfg,
        "method": str(runtime.time.method), "Nl": n_l, "Nm": n_m, "shape": shape,
    }


def _state(case: dict[str, Any], amplitude: float, seed: int):
    import jax.numpy as jnp

    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.linked import mask_supplied_state

    import jax

    cdtype = np.complex128 if jax.config.jax_enable_x64 else np.complex64
    rng = np.random.default_rng(seed)
    shape = case["shape"]
    draw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    cache = build_linear_cache(
        case["grid"], case["geometry"], case["params"], Nl=case["Nl"], Nm=case["Nm"]
    )
    state = mask_supplied_state(jnp.asarray((amplitude * draw).astype(cdtype)), cache)
    return state, cache


def cmd_forward(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gkx.operators.nonlinear.projection import (
        hermitian_projector_for_signature,
        hermitian_projector_signature,
    )
    from gkx.solvers_nonlinear_explicit import advance_explicit_nonlinear_state
    from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
    from gkx.terms.assembly import compute_fields_cached

    case = build_case(args.nx, args.ny, args.nz, args.nl, args.nm)
    state, cache = _state(case, args.amplitude, args.seed)
    params, terms = case["params"], case["terms"]
    linear_terms = dataclasses.replace(terms, nonlinear=0.0)
    method = args.method or case["method"]
    project = hermitian_projector_for_signature(
        hermitian_projector_signature(np.asarray(case["grid"].ky), int(np.asarray(case["grid"].kx).size))
    )
    real = jnp.real(state).dtype

    def rhs_with(tc):
        def rhs(G, cache, params):
            return nonlinear_rhs_cached(
                G, cache, params, tc, compressed_real_fft=True,
                laguerre_mode="grid", differentiable=True,
            )
        return rhs

    full = rhs_with(terms)
    lin = rhs_with(linear_terms)

    def step(G, cache, params):
        def rhs(x):
            return full(x, cache, params)
        G = project(G)
        dG, _ = rhs(G)
        return advance_explicit_nonlinear_state(
            G, dG, jnp.asarray(args.dt, dtype=real), method=method,
            rhs_fn=rhs, project_state=project, state_dtype=G.dtype,
        )

    graphs = {
        "fields": jax.jit(lambda G, c, p: compute_fields_cached(G, c, p, terms=terms).phi),
        "linear_rhs": jax.jit(lambda G, c, p: lin(G, c, p)[0]),
        "nonlinear_rhs": jax.jit(lambda G, c, p: full(G, c, p)[0]),
        "hermitian_projector": jax.jit(lambda G, c, p: project(G)),
        f"{method}_step": jax.jit(step),
    }
    rec: dict[str, Any] = {
        "command": "forward", "method": method, "shape": list(case["shape"]),
        "state_bytes": int(state.size * state.dtype.itemsize), "graphs": {},
    }
    for name, fn in graphs.items():
        rec["graphs"][name], _ = _measure(fn, (state, cache, params), args.repeats)
    if args.trace_dir is not None:
        rec["trace_top_ops"] = _trace_top_ops(
            graphs[f"{method}_step"], (state, cache, params), args.trace_dir, calls=5
        )
    g = rec["graphs"]
    rec["derived"] = {
        "bracket_s_estimate": g["nonlinear_rhs"]["warm_median_s"] - g["linear_rhs"]["warm_median_s"],
        "step_over_rhs": g[f"{method}_step"]["warm_median_s"] / g["nonlinear_rhs"]["warm_median_s"],
    }
    return rec


def cmd_fftfloor(args) -> dict[str, Any]:
    """Time the FFT instructions one nonlinear RHS contains, alone and batched."""

    import jax
    import jax.numpy as jnp

    rec = json.loads(Path(args.forward_json).read_text())
    shapes = rec["graphs"]["nonlinear_rhs"]["hlo"]["fft_shapes"]
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    rdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    ops = []
    for s in shapes:
        out_shape = tuple(int(v) for v in s["out_shape"].split(",") if v)
        length = tuple(int(v) for v in s["fft_length"].split(","))
        kind = s["type"]
        if kind == "RFFT":
            in_shape = out_shape[: -len(length)] + length
            x = jnp.ones(in_shape, rdtype)
            f = lambda x, n=len(length): jnp.fft.rfftn(x, axes=tuple(range(-n, 0)))  # noqa: E731
        elif kind == "IRFFT":
            in_shape = out_shape[:-1] + (length[-1] // 2 + 1,)
            x = jnp.ones(in_shape, cdtype)
            f = lambda x, L=length: jnp.fft.irfftn(x, s=L, axes=tuple(range(-len(L), 0)))  # noqa: E731
        else:
            x = jnp.ones(out_shape, cdtype)
            fn = jnp.fft.fftn if kind == "FFT" else jnp.fft.ifftn
            f = lambda x, n=len(length), fn=fn: fn(x, axes=tuple(range(-n, 0)))  # noqa: E731
        ops.append((s, jax.jit(f), x))
    rows, total = [], 0.0
    for s, f, x in ops:
        r, _ = _measure(f, (x,), args.repeats, hlo=False)
        rows.append({**s, "warm_median_s": r["warm_median_s"]})
        total += r["warm_median_s"] * s["count"]
    rhs = rec["graphs"]["nonlinear_rhs"]["warm_median_s"]
    return {
        "command": "fftfloor", "source": str(args.forward_json), "rows": rows,
        "fft_seconds_per_rhs": total, "rhs_seconds": rhs, "fft_fraction_upper": total / rhs,
    }


def cmd_window(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gkx.solvers_nonlinear_state_integration import nonlinear_heat_flux_window

    if args.inner_remat == "off":
        # Experiment, not shipped behaviour: keep the O(sqrt N) block checkpoint
        # but store each block's per-step residuals instead of rematerializing
        # every step inside the block (one forward recompute instead of two).
        import gkx.solvers_nonlinear_explicit as explicit

        explicit.checkpoint_explicit_step = lambda step, checkpoint: step
    case = build_case(args.nx, args.ny, args.nz, args.nl, args.nm)
    state, _ = _state(case, args.amplitude, args.seed)
    params, geometry = case["params"], case["geometry"]
    base = jnp.asarray(params.tprim)
    method = args.method or case["method"]

    geo_vars = [jnp.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS]

    def objective(scale, geo_vars, checkpoint):
        geom = _with_geometry(geometry, geo_vars)
        p = dataclasses.replace(params, tprim=base * scale)
        return nonlinear_heat_flux_window(
            state, case["grid"], geom, p, dt=args.dt, steps=args.steps,
            method=method, terms=case["terms"], checkpoint=checkpoint,
            divergence_knee_steps=None,
        )

    one = jnp.asarray(1.0, dtype=jnp.real(state).dtype)
    rec: dict[str, Any] = {
        "command": "window", "method": method, "inner_remat": args.inner_remat, "steps": args.steps, "dt": args.dt,
        "shape": list(case["shape"]), "state_bytes": int(state.size * state.dtype.itemsize),
        "geometry_fields": list(GEOMETRY_FIELDS),
        "geometry_values": int(sum(v.size for v in geo_vars)), "arms": {},
    }
    value_fn = jax.jit(lambda s, g: objective(s, g, True))
    rec["arms"]["value"], v = _measure(value_fn, (one, geo_vars), args.repeats, hlo=False)
    rec["arms"]["value"]["objective"] = float(v)
    for ckpt in args.checkpoint:
        flag = ckpt == "block"
        vg = jax.jit(jax.value_and_grad(lambda s, g: objective(s, g, flag), argnums=(0, 1)))
        r, out = _measure(vg, (one, geo_vars), args.repeats, hlo=False)
        val, (gs, gg) = out
        r["objective"] = float(val)
        r["grad_tprim_scale"] = float(gs)
        r["grad_geometry_norm"] = float(sum(jnp.linalg.norm(x) ** 2 for x in gg) ** 0.5)
        rec["arms"][f"value_and_grad_{ckpt}"] = r
    t_val = rec["arms"]["value"]["warm_median_s"]
    rec["derived"] = {
        f"{k}_over_value": a["warm_median_s"] / t_val
        for k, a in rec["arms"].items() if k != "value"
    }
    return rec


def cmd_eigen(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gkx import solver_growth_rate_from_geometry

    case = build_case(1, 4, args.nz, None, None)
    geometry = case["geometry"]
    geo_vars = [jnp.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS]

    def gamma(geo_vars):
        return solver_growth_rate_from_geometry(
            _with_geometry(geometry, geo_vars), n_laguerre=args.nl, n_hermite=args.nm,
        )

    n = 1 * args.nl * args.nm * args.nz
    rec: dict[str, Any] = {"command": "eigen", "Nl": args.nl, "Nm": args.nm, "Nz": args.nz, "operator_size_estimate": n, "arms": {}}
    rec["arms"]["value"], v = _measure(jax.jit(gamma), (geo_vars,), args.repeats, hlo=False)
    rec["arms"]["value"]["gamma"] = float(v)
    r, out = _measure(jax.jit(jax.value_and_grad(gamma)), (geo_vars,), args.repeats, hlo=False)
    r["gamma"] = float(out[0])
    r["grad_norm"] = float(sum(jnp.linalg.norm(x) ** 2 for x in out[1]) ** 0.5)
    rec["arms"]["value_and_grad"] = r
    rec["derived"] = {"grad_over_value": r["warm_median_s"] / rec["arms"]["value"]["warm_median_s"]}
    return rec


def cmd_eigen_adaptive(args) -> dict[str, Any]:
    """Matrix-free certified eigenpair + implicit (left-eigenvector) gradient.

    This route is host-driven (restarts, certification, fail-closed raises), so
    it is timed eagerly: call 1 includes tracing/compiles, call 2 is warm.
    """

    import jax
    import jax.numpy as jnp

    from gkx import solver_objective_vector_from_geometry

    case = build_case(1, 4, args.nz, None, None)
    geometry = case["geometry"]
    geo_vars = [jnp.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS]

    def gamma(geo_vars):
        return solver_objective_vector_from_geometry(
            _with_geometry(geometry, geo_vars), n_laguerre=args.nl,
            n_hermite=args.nm, eigensolver="adaptive-propagator",
        )[0]

    rec: dict[str, Any] = {"command": "eigen-adaptive", "Nl": args.nl, "Nm": args.nm, "Nz": args.nz, "arms": {}}
    for name, fn in (("value", gamma), ("value_and_grad", jax.value_and_grad(gamma))):
        calls = []
        for _ in range(2):
            t = time.perf_counter()
            out = _block(fn(geo_vars))
            calls.append(time.perf_counter() - t)
        arm = {"call_s": calls, "host_peak_bytes": _host_peak_bytes(), "device_peak_bytes": _device_peak()}
        if name == "value":
            arm["gamma"] = float(out)
        else:
            arm["gamma"] = float(out[0])
            arm["grad_norm"] = float(sum(jnp.linalg.norm(x) ** 2 for x in out[1]) ** 0.5)
        rec["arms"][name] = arm
    rec["derived"] = {"warm_grad_over_value": rec["arms"]["value_and_grad"]["call_s"][1] / rec["arms"]["value"]["call_s"][1]}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("forward", "window", "eigen", "eigen-adaptive", "fftfloor"))
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nz", type=int, default=24)
    ap.add_argument("--nl", type=int, default=None)
    ap.add_argument("--nm", type=int, default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--dt", type=float, default=0.03890582546591759)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--checkpoint", nargs="+", default=["block"], choices=("block", "none"))
    ap.add_argument("--inner-remat", choices=("on", "off"), default="on")
    ap.add_argument("--amplitude", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--precision", choices=("32", "64"), default="32")
    ap.add_argument("--forward-json", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", args.precision == "64")
    env = _env()
    started = time.perf_counter()
    rec = {"forward": cmd_forward, "window": cmd_window, "eigen": cmd_eigen, "eigen-adaptive": cmd_eigen_adaptive, "fftfloor": cmd_fftfloor}[args.command](args)
    rec = {"label": args.label, "argv": sys.argv[1:], "precision": args.precision, "environment": env, **rec}
    rec["environment"]["loadavg_end"] = list(os.getloadavg())
    rec["process_s"] = time.perf_counter() - started
    rec["host_peak_bytes"] = _host_peak_bytes()
    text = json.dumps(rec, sort_keys=True, default=str)
    print("RESULT " + text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
