#!/usr/bin/env python3
"""Profile runtime kernels and full RHS graphs from one command surface.

Subcommands:
- ``cyclone``: end-to-end nonlinear runtime warm/profile pass.
- ``nonlinear-step-split``: field solve, nonlinear bracket, linear RHS, full RHS split.
- ``full-linear-rhs``: HLO/Perfetto/memory triage for the fused linear RHS.
- ``full-nonlinear-rhs``: HLO/Perfetto/memory triage for the fused nonlinear RHS.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tools.profiling._profiler_options import git_source_state, make_profile_options
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _profiler_options import (  # type: ignore[import-not-found,no-redef]
        git_source_state,
        make_profile_options,
    )

from gkx.core.grid import build_spectral_grid
from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached
from gkx.runtime import (
    _build_initial_condition,
    _runtime_external_phi,
    _select_nonlinear_mode_indices,
    build_runtime_nonlinear_diagnostics_kwargs,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    build_runtime_term_config,
)
from gkx.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from gkx.terms.nonlinear import nonlinear_em_contribution
from gkx.workflows.runtime.toml import load_runtime_from_toml

HLO_TOKENS = (
    "fusion",
    "fft",
    "reduce",
    "gather",
    "scatter",
    "transpose",
    "broadcast",
    "reshape",
    "slice",
    "concatenate",
    "convert",
    "multiply",
    "add",
    "subtract",
    "divide",
    "select",
)
ROOT = Path(__file__).resolve().parents[2]


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _time_call(fn: Callable[..., Any], *args: Any) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn(*args)
    _block_tree(out)
    return time.perf_counter() - t0, out


def _time_callable(
    fn: Callable[..., Any], *args: Any, repeats: int
) -> tuple[float, Any]:
    out = fn(*args)
    _block_tree(out)
    t0 = time.perf_counter()
    last = out
    for _ in range(repeats):
        last = fn(*args)
        _block_tree(last)
    t1 = time.perf_counter()
    return (t1 - t0) / float(repeats), last


def _z_variation_norm(state: jnp.ndarray) -> float:
    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return float(np.asarray(jnp.linalg.norm(state - mean_z)))


def _inject_z_wave(
    state: jnp.ndarray,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> jnp.ndarray:
    """Inject a deterministic parallel wave so linked-z paths are active."""

    state = jnp.asarray(state)
    nz = state.shape[-1]
    nm = state.shape[-4]
    m_index = min(max(1, nm - 1), 3)
    z = jnp.arange(nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - runtime state builder controls dimensionality.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _hlo_token_counts(
    hlo_text: str, tokens: tuple[str, ...] = HLO_TOKENS
) -> dict[str, int]:
    """Count coarse HLO tokens used for trace triage."""

    lower = hlo_text.lower()
    return {token: lower.count(token) for token in tokens}


def _write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _field_norm(value: jnp.ndarray | None) -> float:
    if value is None:
        return 0.0
    return float(np.asarray(jnp.linalg.norm(value)))


def _array_fingerprint(value: Any) -> dict[str, Any]:
    """Return compact numerical identity metadata for one profiled array."""

    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "finite_fraction": float(np.mean(finite)) if array.size else 1.0,
        "l2_norm": float(np.linalg.norm(finite_values)) if finite_values.size else 0.0,
        "max_abs": float(np.max(np.abs(finite_values))) if finite_values.size else 0.0,
        "sum_real": float(np.sum(np.real(finite_values), dtype=np.float64)),
        "sum_imag": float(np.sum(np.imag(finite_values), dtype=np.float64)),
    }


def _prepared_result_summary(result: Any) -> dict[str, Any]:
    """Fingerprint the state and diagnostics paired with prepared timings."""

    final_state, diagnostics, dt_series, fields = result
    return {
        "final_state": _array_fingerprint(final_state),
        "phi": _array_fingerprint(fields.phi),
        "heat_flux": _array_fingerprint(diagnostics.heat_flux_t),
        "dt": _array_fingerprint(dt_series),
    }


def _peak_rss_bytes(peak_rss: int, *, system: str | None = None) -> int:
    """Normalize ``ru_maxrss`` to bytes across macOS and Linux."""

    platform_name = platform.system() if system is None else system
    return int(peak_rss) if platform_name == "Darwin" else int(peak_rss) * 1024


def _runtime_memory_summary() -> dict[str, Any]:
    """Return host peak RSS and available JAX device allocator metrics."""

    device = jax.devices()[0]
    raw_stats = device.memory_stats() or {}
    device_stats = {
        key: int(raw_stats[key])
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if key in raw_stats
    }
    return {
        "host_peak_rss_bytes": _peak_rss_bytes(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "device": str(device),
        "device_stats": device_stats,
    }


def _configure_xla(args: argparse.Namespace) -> None:
    if getattr(args, "xla_dump_dir", None) is None:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = [
        f"--xla_dump_to={args.xla_dump_dir}",
        "--xla_dump_hlo_as_text",
        f"--xla_dump_hlo_pass_re={args.xla_hlo_pass_re}",
    ]
    os.environ["XLA_FLAGS"] = " ".join([flags] + dump_flags).strip()


def build_cyclone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile nonlinear Cyclone runtime.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--resolved-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-prepared-simulation",
        action="store_true",
        help="Prepare one explicit scan and reuse its compiled executable across repeats.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument("--xla-hlo-pass-re", type=str, default=".*")
    parser.add_argument("--python-tracer-level", type=int, default=0)
    parser.add_argument("--host-tracer-level", type=int, default=0)
    parser.add_argument("--warmup-only", action="store_true", default=False)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main_cyclone(argv: list[str] | None = None) -> int:
    args = build_cyclone_parser().parse_args(argv)
    _configure_xla(args)
    source_state = git_source_state(ROOT)

    from jax import profiler

    from gkx.solvers.nonlinear.diagnostic_integration import prepare_nonlinear_explicit_diagnostics
    from gkx.runtime import run_runtime_nonlinear

    cfg, _data = load_runtime_from_toml(args.config)
    dt_effective = float(cfg.time.dt if args.dt is None else args.dt)
    method_effective = str(cfg.time.method if args.method is None else args.method)
    sample_stride_effective = int(
        cfg.time.sample_stride if args.sample_stride is None else args.sample_stride
    )
    diagnostics_stride_effective = int(
        cfg.time.diagnostics_stride
        if args.diagnostics_stride is None
        else args.diagnostics_stride
    )

    if args.reuse_prepared_simulation:
        if args.steps is None:
            raise ValueError("--reuse-prepared-simulation requires --steps")
        geom = build_runtime_geometry(cfg)
        grid = build_spectral_grid(
            apply_imported_geometry_grid_defaults(geom, cfg.grid)
        )
        params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
        terms = build_runtime_term_config(cfg)
        ky_index, kx_index = _select_nonlinear_mode_indices(
            grid,
            ky_target=args.ky,
            kx_target=None,
            use_dealias_mask=bool(cfg.time.nonlinear_dealias),
        )
        initial_state = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=args.Nl,
            Nm=args.Nm,
            nspecies=len(cfg.species),
        )
        prepared_kwargs = build_runtime_nonlinear_diagnostics_kwargs(
            cfg,
            dt=dt_effective,
            steps=int(args.steps),
            method=method_effective,
            term_config=terms,
            sample_stride=sample_stride_effective,
            diagnostics_stride=diagnostics_stride_effective,
            laguerre_mode=str(cfg.time.laguerre_nonlinear_mode),
            ky_index=ky_index,
            kx_index=kx_index,
            fixed_dt=bool(cfg.time.fixed_dt),
            fixed_mode_ky_index=(
                cfg.expert.iky_fixed if cfg.expert.fixed_mode else None
            ),
            fixed_mode_kx_index=(
                cfg.expert.ikx_fixed if cfg.expert.fixed_mode else None
            ),
            external_phi=_runtime_external_phi(cfg),
            resolved_diagnostics=bool(args.resolved_diagnostics),
            show_progress=False,
        )
        prepared = prepare_nonlinear_explicit_diagnostics(
            initial_state,
            grid,
            geom,
            params,
            **prepared_kwargs,
        )

        def _run():
            result = prepared.run()
            _block_tree(result)
            return result

    else:

        def _run():
            result = run_runtime_nonlinear(
                cfg,
                ky_target=args.ky,
                Nl=args.Nl,
                Nm=args.Nm,
                dt=args.dt,
                steps=args.steps,
                method=args.method,
                sample_stride=args.sample_stride,
                diagnostics_stride=args.diagnostics_stride,
                diagnostics=True,
                resolved_diagnostics=args.resolved_diagnostics,
            )
            _block_tree(result)
            return result

    t0 = time.perf_counter()
    with profiler.TraceAnnotation("gkx_warmup"):
        last_result = _run()
    t1 = time.perf_counter()

    if args.warmup_only:
        print(f"warmup_time_s={t1 - t0:.3f}")
        return 0

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.start_trace(
            str(args.trace_dir),
            profiler_options=make_profile_options(
                python_tracer_level=args.python_tracer_level,
                host_tracer_level=args.host_tracer_level,
            ),
        )
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")
    run_times: list[float] = []
    try:
        with profiler.TraceAnnotation("gkx_profiled_run"):
            for _ in range(args.repeats):
                elapsed, last_result = _time_call(_run)
                run_times.append(elapsed)
    finally:
        if args.trace_dir is not None:
            profiler.stop_trace()

    if args.memory_profile is not None:
        with profiler.TraceAnnotation("gkx_memory_snapshot"):
            profiler.save_device_memory_profile(str(args.memory_profile))

    run_median = float(np.median(np.asarray(run_times, dtype=float)))
    print(
        f"warmup_time_s={t1 - t0:.3f} run_time_s={run_median:.3f} "
        f"run_times_s={','.join(f'{value:.6f}' for value in run_times)}"
    )
    if args.out is not None:
        payload = {
            **source_state,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "config": str(args.config),
            "nl": int(args.Nl),
            "nm": int(args.Nm),
            "steps": args.steps,
            "dt": dt_effective,
            "method": method_effective,
            "fixed_dt": bool(cfg.time.fixed_dt),
            "sample_stride": sample_stride_effective,
            "diagnostics_stride": diagnostics_stride_effective,
            "resolved_diagnostics": bool(args.resolved_diagnostics),
            "reuse_prepared_simulation": bool(args.reuse_prepared_simulation),
            "software": {
                "python": sys.version.split()[0],
                "jax": str(getattr(jax, "__version__", "unknown")),
                "numpy": str(np.__version__),
            },
            "warmup_time_s": float(t1 - t0),
            "run_times_s": run_times,
            "run_median_s": run_median,
            "memory_summary": _runtime_memory_summary(),
        }
        if args.reuse_prepared_simulation:
            payload["result_summary"] = _prepared_result_summary(last_result)
        _write_summary_json(payload, args.out)
    return 0


def build_nonlinear_step_split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile nonlinear field solve vs bracket vs full RHS."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--laguerre-mode", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main_nonlinear_step_split(argv: list[str] | None = None) -> int:
    args = build_nonlinear_step_split_parser().parse_args(argv)
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    laguerre_mode = (
        cfg.time.laguerre_nonlinear_mode
        if args.laguerre_mode is None
        else str(args.laguerre_mode)
    )

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    g0 = jnp.asarray(g0)

    field_fn = jax.jit(
        lambda state: compute_fields_cached(state, cache, params, terms=term_cfg)
    )
    fields = field_fn(g0)
    _block_tree(fields)

    nonlinear_fn = jax.jit(
        lambda state, phi, apar, bpar: nonlinear_em_contribution(
            state,
            phi=phi,
            apar=apar,
            bpar=bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=jnp.asarray(
                term_cfg.nonlinear,
                dtype=jnp.real(jnp.empty((), dtype=state.dtype)).dtype,
            ),
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            compressed_real_fft=bool(cfg.time.compressed_real_fft),
            laguerre_mode=laguerre_mode,
        )
    )
    linear_terms = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = jax.jit(
        lambda state: assemble_rhs_cached_jit(state, cache, params, linear_terms)
    )
    rhs_fn = jax.jit(
        lambda state: nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            compressed_real_fft=bool(cfg.time.compressed_real_fft),
            laguerre_mode=laguerre_mode,
        )
    )

    field_time, fields = _time_callable(field_fn, g0, repeats=args.repeats)
    nl_time, nl_out = _time_callable(
        nonlinear_fn, g0, fields.phi, fields.apar, fields.bpar, repeats=args.repeats
    )
    linear_rhs_time, linear_rhs_out = _time_callable(
        linear_rhs_fn, g0, repeats=args.repeats
    )
    rhs_time, rhs_out = _time_callable(rhs_fn, g0, repeats=args.repeats)
    linear_rhs_state, _linear_rhs_fields = linear_rhs_out
    rhs_state, rhs_fields = rhs_out

    rows = [
        {
            "kernel": "field_solve",
            "seconds": field_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(fields.phi))),
        },
        {
            "kernel": "nonlinear_bracket",
            "seconds": nl_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(nl_out))),
        },
        {
            "kernel": "linear_rhs",
            "seconds": linear_rhs_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(linear_rhs_state))),
        },
        {
            "kernel": "full_rhs",
            "seconds": rhs_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(rhs_state))),
        },
    ]

    for row in rows:
        print(f"{row['kernel']}: seconds={row['seconds']:.6f} norm={row['norm']:.6e}")
    print(
        "rhs_fields:",
        f"phi_norm={float(np.asarray(jnp.linalg.norm(rhs_fields.phi))):.6e}",
        f"apar_norm={_field_norm(rhs_fields.apar):.6e}",
        f"bpar_norm={_field_norm(rhs_fields.bpar):.6e}",
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["kernel", "seconds", "repeats", "norm"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")
    return 0


def _add_full_rhs_common_args(
    parser: argparse.ArgumentParser, *, nonlinear: bool
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml"
        ),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--state", choices=("initial", "z_wave"), default="initial")
    parser.add_argument("--z-mode", type=int, default=1)
    parser.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    if nonlinear:
        parser.add_argument("--laguerre-mode", type=str, default=None)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "docs/_static/full_nonlinear_rhs_trace_summary.json"
            if nonlinear
            else "docs/_static/full_linear_rhs_trace_summary.json"
        ),
    )
    parser.add_argument("--hlo-out", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--python-tracer-level", type=int, default=0)
    parser.add_argument("--host-tracer-level", type=int, default=0)


def build_full_linear_rhs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace the fused full-linear-RHS graph."
    )
    _add_full_rhs_common_args(parser, nonlinear=False)
    return parser


def build_full_nonlinear_rhs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace the fused full-nonlinear-RHS graph."
    )
    _add_full_rhs_common_args(parser, nonlinear=True)
    return parser


def _build_summary(
    *,
    config: str,
    backend: str,
    nl: int,
    nm: int,
    repeats: int,
    state: str,
    z_variation_norm: float,
    compile_execute_seconds: float,
    warm_seconds: float,
    rhs_norm: float,
    phi_norm: float,
    hlo_text: str,
    trace_dir: Path | None,
    memory_profile: Path | None,
    hlo_out: Path | None,
    force_electrostatic_fields: bool,
    source: str,
) -> dict[str, Any]:
    """Build a machine-readable full-linear-RHS trace summary."""

    return {
        "kind": "full_linear_rhs_trace_summary",
        "case": Path(config).stem,
        "config": config,
        "backend": backend,
        "Nl": int(nl),
        "Nm": int(nm),
        "repeats": int(repeats),
        "state": state,
        "z_variation_norm": float(z_variation_norm),
        "compile_execute_seconds": float(compile_execute_seconds),
        "warm_seconds": float(warm_seconds),
        "rhs_norm": float(rhs_norm),
        "phi_norm": float(phi_norm),
        "hlo_line_count": len(hlo_text.splitlines()),
        "hlo_bytes": len(hlo_text.encode("utf-8")),
        "hlo_token_counts": _hlo_token_counts(hlo_text),
        "trace_dir": None if trace_dir is None else str(trace_dir),
        "memory_profile": None if memory_profile is None else str(memory_profile),
        "hlo_out": None if hlo_out is None else str(hlo_out),
        "force_electrostatic_fields": bool(force_electrostatic_fields),
        "source": str(source),
        "claim_scope": (
            "Full fused linear-RHS graph triage for one runtime state. Use this to choose "
            "kernel-level optimization targets; do not treat it as a standalone runtime claim."
        ),
    }


def _build_nonlinear_summary(
    *,
    config: str,
    backend: str,
    nl: int,
    nm: int,
    repeats: int,
    state: str,
    laguerre_mode: str,
    compressed_real_fft: bool,
    z_variation_norm: float,
    compile_execute_seconds: float,
    warm_seconds: float,
    rhs_norm: float,
    phi_norm: float,
    apar_norm: float,
    bpar_norm: float,
    hlo_text: str,
    trace_dir: Path | None,
    memory_profile: Path | None,
    hlo_out: Path | None,
    electrostatic_specialized: bool,
) -> dict[str, Any]:
    """Build a machine-readable full-nonlinear-RHS trace summary."""

    return {
        "kind": "full_nonlinear_rhs_trace_summary",
        "case": Path(config).stem,
        "config": config,
        "backend": backend,
        "Nl": int(nl),
        "Nm": int(nm),
        "repeats": int(repeats),
        "state": state,
        "laguerre_mode": laguerre_mode,
        "compressed_real_fft": bool(compressed_real_fft),
        "z_variation_norm": float(z_variation_norm),
        "compile_execute_seconds": float(compile_execute_seconds),
        "warm_seconds": float(warm_seconds),
        "rhs_norm": float(rhs_norm),
        "phi_norm": float(phi_norm),
        "apar_norm": float(apar_norm),
        "bpar_norm": float(bpar_norm),
        "hlo_line_count": len(hlo_text.splitlines()),
        "hlo_bytes": len(hlo_text.encode("utf-8")),
        "hlo_token_counts": _hlo_token_counts(hlo_text),
        "trace_dir": None if trace_dir is None else str(trace_dir),
        "memory_profile": None if memory_profile is None else str(memory_profile),
        "hlo_out": None if hlo_out is None else str(hlo_out),
        "electrostatic_specialized": bool(electrostatic_specialized),
        "claim_scope": (
            "Full fused nonlinear-RHS graph triage for one runtime state. Use this to choose "
            "kernel-level optimization targets; do not treat it as a standalone transport "
            "runtime claim."
        ),
    }


def main_full_linear_rhs(argv: list[str] | None = None) -> int:
    args = build_full_linear_rhs_parser().parse_args(argv)
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    linear_terms = build_runtime_linear_terms(cfg)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    g0 = jnp.asarray(g0)
    if args.state == "z_wave":
        g0 = _inject_z_wave(
            g0,
            ky_index=int(ky_index),
            kx_index=int(kx_index),
            amplitude=float(args.z_wave_amplitude),
            z_mode=int(args.z_mode),
        )

    force_electrostatic_fields = _is_static_zero(linear_terms.apar) and _is_static_zero(
        linear_terms.bpar
    )
    rhs_fn = jax.jit(
        lambda state: linear_rhs_cached(
            state,
            cache,
            params,
            terms=linear_terms,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    )
    compile_execute_seconds, first_out = _time_call(rhs_fn, g0)
    rhs, phi = first_out

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(
            str(args.trace_dir),
            profiler_options=make_profile_options(
                python_tracer_level=int(args.python_tracer_level),
                host_tracer_level=int(args.host_tracer_level),
            ),
        )
    try:
        warm_t0 = time.perf_counter()
        for _ in range(int(args.repeats)):
            rhs, phi = rhs_fn(g0)
            _block_tree((rhs, phi))
        warm_seconds = (time.perf_counter() - warm_t0) / float(args.repeats)
    finally:
        if args.trace_dir is not None:
            jax.profiler.stop_trace()

    if args.memory_profile is not None:
        args.memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(args.memory_profile))

    hlo_ir = rhs_fn.lower(g0).compiler_ir(dialect="hlo")
    if hlo_ir is None:
        raise RuntimeError("failed to lower full linear RHS to HLO")
    hlo_text = hlo_ir.as_hlo_text()
    if args.hlo_out is not None:
        args.hlo_out.parent.mkdir(parents=True, exist_ok=True)
        args.hlo_out.write_text(hlo_text, encoding="utf-8")

    summary = _build_summary(
        config=str(args.config),
        backend=jax.default_backend(),
        nl=int(args.Nl),
        nm=int(args.Nm),
        repeats=int(args.repeats),
        state=str(args.state),
        z_variation_norm=_z_variation_norm(g0),
        compile_execute_seconds=float(compile_execute_seconds),
        warm_seconds=float(warm_seconds),
        rhs_norm=float(np.asarray(jnp.linalg.norm(rhs))),
        phi_norm=float(np.asarray(jnp.linalg.norm(phi))),
        hlo_text=hlo_text,
        trace_dir=args.trace_dir,
        memory_profile=args.memory_profile,
        hlo_out=args.hlo_out,
        force_electrostatic_fields=force_electrostatic_fields,
        source="gkx.operators.linear.rhs.linear_rhs_cached",
    )
    _write_summary_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main_full_nonlinear_rhs(argv: list[str] | None = None) -> int:
    args = build_full_nonlinear_rhs_parser().parse_args(argv)
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    laguerre_mode = (
        cfg.time.laguerre_nonlinear_mode
        if args.laguerre_mode is None
        else str(args.laguerre_mode)
    )

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    g0 = jnp.asarray(g0)
    if args.state == "z_wave":
        g0 = _inject_z_wave(
            g0,
            ky_index=int(ky_index),
            kx_index=int(kx_index),
            amplitude=float(args.z_wave_amplitude),
            z_mode=int(args.z_mode),
        )

    rhs_fn = jax.jit(
        lambda state: nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            compressed_real_fft=bool(cfg.time.compressed_real_fft),
            laguerre_mode=laguerre_mode,
        )
    )
    compile_execute_seconds, first_out = _time_call(rhs_fn, g0)
    rhs, fields = first_out

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(
            str(args.trace_dir),
            profiler_options=make_profile_options(
                python_tracer_level=int(args.python_tracer_level),
                host_tracer_level=int(args.host_tracer_level),
            ),
        )
    try:
        warm_t0 = time.perf_counter()
        for _ in range(int(args.repeats)):
            rhs, fields = rhs_fn(g0)
            _block_tree((rhs, fields))
        warm_seconds = (time.perf_counter() - warm_t0) / float(args.repeats)
    finally:
        if args.trace_dir is not None:
            jax.profiler.stop_trace()

    if args.memory_profile is not None:
        args.memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(args.memory_profile))

    hlo_text = rhs_fn.lower(g0).compiler_ir(dialect="hlo").as_hlo_text()
    if args.hlo_out is not None:
        args.hlo_out.parent.mkdir(parents=True, exist_ok=True)
        args.hlo_out.write_text(hlo_text, encoding="utf-8")

    summary = _build_nonlinear_summary(
        config=str(args.config),
        backend=jax.default_backend(),
        nl=int(args.Nl),
        nm=int(args.Nm),
        repeats=int(args.repeats),
        state=str(args.state),
        laguerre_mode=str(laguerre_mode),
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        z_variation_norm=_z_variation_norm(g0),
        compile_execute_seconds=float(compile_execute_seconds),
        warm_seconds=float(warm_seconds),
        rhs_norm=float(np.asarray(jnp.linalg.norm(rhs))),
        phi_norm=_field_norm(fields.phi),
        apar_norm=_field_norm(fields.apar),
        bpar_norm=_field_norm(fields.bpar),
        hlo_text=hlo_text,
        trace_dir=args.trace_dir,
        memory_profile=args.memory_profile,
        hlo_out=args.hlo_out,
        electrostatic_specialized=_is_static_zero(term_cfg.apar)
        and _is_static_zero(term_cfg.bpar),
    )
    _write_summary_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


SUBCOMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "cyclone": main_cyclone,
    "full-linear-rhs": main_full_linear_rhs,
    "full-nonlinear-rhs": main_full_nonlinear_rhs,
    "nonlinear-step-split": main_nonlinear_step_split,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(SUBCOMMANDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    try:
        handler = SUBCOMMANDS[command]
    except KeyError:
        build_parser().parse_args([command])
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
