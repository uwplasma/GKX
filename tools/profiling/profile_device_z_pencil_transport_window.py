#!/usr/bin/env python3
"""Profile the device-z-sharded nonlinear transport-window route."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_device_z_pencil_transport_cpu4_profile"
)
DEFAULT_RHS_OUT_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_device_z_pencil_rhs_cpu4_profile"
)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(raw).split(",") if str(item).strip())


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _time_repeated(fn: Any, *, warmups: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(int(warmups)):
        _block_until_ready(fn())
    last: Any = None
    times: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        last = _block_until_ready(fn())
        times.append(float(time.perf_counter() - start))
    return last, times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _max_abs_rel(
    candidate: Any, reference: Any, *, floor: float
) -> tuple[float, float]:
    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    max_abs = float(np.max(np.abs(candidate_arr - reference_arr)))
    scale = max(float(np.max(np.abs(reference_arr))), float(floor))
    return max_abs, float(max_abs / scale)


def _hlo_text(lowered: Any) -> str:
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    if hasattr(compiler_ir, "as_hlo_text"):
        return str(compiler_ir.as_hlo_text())
    return str(compiler_ir)


def _hlo_keyword_counts(hlo_text: str) -> dict[str, int]:
    lowered = hlo_text.lower()
    return {
        "fft": lowered.count("fft"),
        "all_to_all": lowered.count("all-to-all") + lowered.count("all_to_all"),
        "collective_permute": lowered.count("collective-permute")
        + lowered.count("collective_permute"),
        "all_reduce": lowered.count("all-reduce") + lowered.count("all_reduce"),
        "copy": lowered.count("copy"),
        "fusion": lowered.count("fusion"),
    }


def _write_hlo(path: Path, lowered: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _hlo_text(lowered)
    path.write_text(text, encoding="utf-8")
    return {
        "path": str(path),
        "line_count": text.count("\n") + 1,
        **_hlo_keyword_counts(text),
    }


def _run_trace(fn: Any, trace_dir: Path) -> dict[str, Any]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    try:
        _block_until_ready(fn())
    finally:
        jax.profiler.stop_trace()
    trace_files = sorted(path.name for path in trace_dir.rglob("*") if path.is_file())
    return {
        "requested": True,
        "trace_dir": str(trace_dir),
        "file_count": len(trace_files),
        "sample_files": trace_files[:8],
    }


def build_profile(
    *,
    shape: tuple[int, int, int, int, int],
    device_counts: tuple[int, ...],
    steps: int,
    dt: float,
    warmups: int,
    repeats: int,
    observable_repeats: int,
    atol: float,
    rtol: float,
    min_speedup: float,
    z_chunk_size: int | None,
    auto_z_chunk_size: bool,
    max_fft_batch_count: int,
    observable_mode: str,
    trace_dir: Path | None,
    trace_device_count: int | None,
    hlo_prefix: Path | None,
) -> dict[str, Any]:
    from gkx.operators.nonlinear.device_z import (
        _device_z_pencil_shard_map_rhs_fn,
    )
    from gkx.operators.nonlinear.parallel import (  # type: ignore[import-untyped]
        deterministic_nonlinear_spectral_state,
        device_z_pencil_fft_batch_pressure_model,
        device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
    )
    from gkx.operators.nonlinear.spectral_core import (
        _host_staged_array_for_sharding,
        _serial_nonlinear_spectral_rhs,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    devices = tuple(jax.devices())
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state).dtype)
    if int(observable_repeats) < 0:
        raise ValueError("observable_repeats must be non-negative")
    if observable_mode not in {"host_gather", "sharded_reduce"}:
        raise ValueError("observable_mode must be 'host_gather' or 'sharded_reduce'")
    requested_parallel_counts = tuple(
        int(count) for count in device_counts if int(count) > 1
    )
    batch_model: dict[str, Any] | None = None
    if requested_parallel_counts:
        target_count = max(requested_parallel_counts)
        model = device_z_pencil_fft_batch_pressure_model(
            shape,
            device_count=target_count,
            max_fft_batch_count=int(max_fft_batch_count),
            z_chunk_size=z_chunk_size,
        )
        if bool(auto_z_chunk_size) and z_chunk_size is None:
            z_chunk_size = model.suggested_z_chunk_size
            model = device_z_pencil_fft_batch_pressure_model(
                shape,
                device_count=target_count,
                max_fft_batch_count=int(max_fft_batch_count),
                z_chunk_size=z_chunk_size,
            )
        batch_model = model.to_dict()

    def serial_route(item: jax.Array) -> jax.Array:
        out = item
        for _ in range(int(steps)):
            _field, _bracket, rhs = _serial_nonlinear_spectral_rhs(out)
            out = out + dt_array * rhs
        return out

    serial_jit = jax.jit(serial_route)
    serial_out, serial_times = _time_repeated(
        lambda: serial_jit(state),
        warmups=int(warmups),
        repeats=int(repeats),
    )
    serial_stats = _stats(serial_times)
    hlo_reports: dict[str, Any] = {}
    if hlo_prefix is not None:
        hlo_reports["serial"] = _write_hlo(
            Path(f"{hlo_prefix}_serial.hlo.txt"),
            serial_jit.lower(state),
        )
    rows: list[dict[str, Any]] = [
        {
            "device_count": 1,
            "active": True,
            "identity_passed": True,
            "transport_window_identity_passed": True,
            "final_state_max_abs_error": 0.0,
            "final_state_max_rel_error": 0.0,
            "physical_flux_trace_max_abs_error": 0.0,
            "physical_flux_trace_max_rel_error": 0.0,
            "median_s": serial_stats["median"],
            "speedup_vs_serial": 1.0,
            "speedup_gate_passed": False,
            "blocked_reasons": [],
            "stats": serial_stats,
            "timing_scope": "compute_only_final_state_update",
            "identity_gate_elapsed_s": None,
            "observable_gate_median_s": None,
            "observable_gate_overhead_vs_compute": None,
            "observable_gate_stats_s": {},
        }
    ]
    trace_report: dict[str, Any] = {"requested": False}

    for count in device_counts:
        count = int(count)
        if count <= 1:
            continue
        if len(devices) < count:
            rows.append(
                {
                    "device_count": count,
                    "active": False,
                    "identity_passed": False,
                    "transport_window_identity_passed": False,
                    "final_state_max_abs_error": None,
                    "final_state_max_rel_error": None,
                    "physical_flux_trace_max_abs_error": None,
                    "physical_flux_trace_max_rel_error": None,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": ["not_enough_devices"],
                    "stats": {},
                    "timing_scope": "compute_only_final_state_update",
                    "identity_gate_elapsed_s": None,
                    "observable_gate_median_s": None,
                    "observable_gate_overhead_vs_compute": None,
                    "observable_gate_stats_s": {},
                }
            )
            continue

        identity_start = time.perf_counter()
        report = device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
            state,
            devices=devices[:count],
            steps=int(steps),
            dt=float(dt),
            atol=float(atol),
            rtol=float(rtol),
            z_chunk_size=z_chunk_size,
            observable_mode=observable_mode,  # type: ignore[arg-type]
        )
        identity_gate_elapsed_s = float(time.perf_counter() - identity_start)
        observable_stats: dict[str, float] = {}
        if int(observable_repeats) > 0:
            _last_observable_report, observable_times = _time_repeated(
                lambda: (
                    device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
                        state,
                        devices=devices[:count],
                        steps=int(steps),
                        dt=float(dt),
                        atol=float(atol),
                        rtol=float(rtol),
                        z_chunk_size=z_chunk_size,
                        observable_mode=observable_mode,  # type: ignore[arg-type]
                    )
                ),
                warmups=0,
                repeats=int(observable_repeats),
            )
            observable_stats = _stats(observable_times)
        if not report.decomposed_path_enabled:
            rows.append(
                {
                    "device_count": count,
                    "active": bool(report.device_sharding_active),
                    "identity_passed": bool(report.identity_passed),
                    "transport_window_identity_passed": bool(report.identity_passed),
                    "final_state_max_abs_error": report.final_state_max_abs_error,
                    "final_state_max_rel_error": report.final_state_max_rel_error,
                    "physical_flux_trace_max_abs_error": (
                        report.physical_flux_trace_max_abs_error
                    ),
                    "physical_flux_trace_max_rel_error": (
                        report.physical_flux_trace_max_rel_error
                    ),
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": list(report.blocked_reasons),
                    "stats": {},
                    "timing_scope": "compute_only_final_state_update",
                    "identity_gate_elapsed_s": identity_gate_elapsed_s,
                    "observable_gate_median_s": (
                        observable_stats.get("median") if observable_stats else None
                    ),
                    "observable_gate_overhead_vs_compute": None,
                    "observable_gate_stats_s": observable_stats,
                    "transport_window_report": report.to_dict(),
                }
            )
            continue

        mesh = Mesh(np.asarray(devices[:count]), ("z",))
        state_spec = PartitionSpec(None, None, None, None, "z")
        sharding = NamedSharding(mesh, state_spec)
        with mesh:
            sharded_state = jax.device_put(
                _host_staged_array_for_sharding(state),
                sharding,
            )
            sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
                mesh,
                axis_name="z",
                z_chunk_size=z_chunk_size,
            )

            def sharded_route(item: jax.Array) -> jax.Array:
                out = item
                for _ in range(int(steps)):
                    out = out + dt_array * sharded_rhs_fn(out)
                return out

            sharded_jit = jax.jit(sharded_route)
            if hlo_prefix is not None:
                hlo_reports[f"device_{count}"] = _write_hlo(
                    Path(f"{hlo_prefix}_device{count}.hlo.txt"),
                    sharded_jit.lower(sharded_state),
                )
            if trace_dir is not None and (
                trace_device_count is None or int(trace_device_count) == count
            ):
                trace_report = _run_trace(
                    lambda: sharded_jit(sharded_state),
                    trace_dir / f"device{count}",
                )
            sharded_out, sharded_times = _time_repeated(
                lambda: sharded_jit(sharded_state),
                warmups=int(warmups),
                repeats=int(repeats),
            )

        state_abs, state_rel = _max_abs_rel(sharded_out, serial_out, floor=float(atol))
        sharded_stats = _stats(sharded_times)
        speedup = serial_stats["median"] / sharded_stats["median"]
        observable_gate_median = (
            observable_stats.get("median") if observable_stats else None
        )
        observable_gate_overhead = (
            float(observable_gate_median) / sharded_stats["median"]
            if observable_gate_median is not None and sharded_stats["median"] > 0.0
            else None
        )
        timing_identity_passed = bool(
            state_abs <= float(atol) or state_rel <= float(rtol)
        )
        row_blockers = list(report.blocked_reasons)
        if not timing_identity_passed:
            row_blockers.append("timed_device_z_pencil_transport_identity_failed")
        if speedup < float(min_speedup):
            row_blockers.append("speedup_below_gate")
        rows.append(
            {
                "device_count": count,
                "active": True,
                "identity_passed": timing_identity_passed,
                "transport_window_identity_passed": bool(report.identity_passed),
                "final_state_max_abs_error": state_abs,
                "final_state_max_rel_error": state_rel,
                "physical_flux_trace_max_abs_error": (
                    report.physical_flux_trace_max_abs_error
                ),
                "physical_flux_trace_max_rel_error": (
                    report.physical_flux_trace_max_rel_error
                ),
                "median_s": sharded_stats["median"],
                "speedup_vs_serial": speedup,
                "speedup_gate_passed": bool(
                    timing_identity_passed
                    and report.identity_passed
                    and speedup >= float(min_speedup)
                ),
                "blocked_reasons": sorted(set(row_blockers)),
                "stats": sharded_stats,
                "timing_scope": "compute_only_final_state_update",
                "identity_gate_elapsed_s": identity_gate_elapsed_s,
                "observable_gate_median_s": observable_gate_median,
                "observable_gate_overhead_vs_compute": observable_gate_overhead,
                "observable_gate_stats_s": observable_stats,
                "transport_window_report": report.to_dict(),
            }
        )

    active_rows = [row for row in rows if row["active"] and row["device_count"] > 1]
    measured_speedups = [
        float(row["speedup_vs_serial"])
        for row in active_rows
        if row.get("speedup_vs_serial") is not None
    ]
    max_speedup = max(measured_speedups, default=1.0)
    all_active_identity = all(bool(row["identity_passed"]) for row in active_rows)
    speedup_gate = any(bool(row["speedup_gate_passed"]) for row in active_rows)
    if speedup_gate:
        status = "transport_window_speedup_candidate"
    elif active_rows and not all_active_identity:
        status = "identity_failed_no_transport_speedup"
    else:
        status = "identity_timed_no_transport_speedup"
    if not active_rows:
        status = "skipped_no_multidevice"

    return _json_clean(
        {
            "kind": "nonlinear_device_z_pencil_transport_window_profile",
            "claim_scope": (
                "device z-sharded shard_map nonlinear transport-window timing; "
                "serial-vs-sharded final-state and scalar-trace identity are "
                "required before any micro-route speedup claim, and this is not "
                "yet a full production nonlinear turbulent-transport solve"
            ),
            "backend": str(jax.default_backend()),
            "device_count_available": int(len(devices)),
            "shape": shape,
            "steps": int(steps),
            "dt": float(dt),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "observable_repeats": int(observable_repeats),
            "observable_mode": str(observable_mode),
            "atol": float(atol),
            "rtol": float(rtol),
            "min_speedup": float(min_speedup),
            "z_chunk_size": None if z_chunk_size is None else int(z_chunk_size),
            "auto_z_chunk_size": bool(auto_z_chunk_size),
            "max_fft_batch_count": int(max_fft_batch_count),
            "fft_batch_pressure_model": batch_model,
            "serial_stats_s": serial_stats,
            "timing_scope": (
                "speedup rows time compute-only fixed-step final-state updates; "
                "observable_gate_* fields optionally time the scalar identity/"
                "transport diagnostics separately and are not included in the "
                "speedup gate"
            ),
            "rows": rows,
            "trace": trace_report,
            "hlo": hlo_reports,
            "summary": {
                "status": status,
                "all_active_identity_passed": bool(all_active_identity),
                "max_speedup_vs_serial": float(max_speedup),
                "transport_window_speedup_claim_allowed": bool(speedup_gate),
                "full_solver_speedup_claim_allowed": False,
            },
        }
    )


def build_rhs_profile(
    *,
    shape: tuple[int, int, int, int, int],
    device_counts: tuple[int, ...],
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
    min_speedup: float,
) -> dict[str, Any]:
    """Profile only the fused nonlinear RHS route.

    The transport-window profiler is the paper-facing diagnostic because it
    checks state and scalar transport identity. This RHS-only mode is kept as a
    narrow microkernel gate for profiling the bracket/FFT path without
    duplicating a second executable script.
    """

    from gkx.operators.nonlinear.parallel import (  # type: ignore[import-untyped]
        deterministic_nonlinear_spectral_state,
        device_z_pencil_nonlinear_spectral_rhs,
    )
    from gkx.operators.nonlinear.spectral_core import (
        _pencil_nonlinear_spectral_rhs,
        _serial_nonlinear_spectral_rhs,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    devices = tuple(jax.devices())
    serial_jit = jax.jit(lambda item: _serial_nonlinear_spectral_rhs(item)[2])
    serial_out, serial_times = _time_repeated(
        lambda: serial_jit(state),
        warmups=int(warmups),
        repeats=int(repeats),
    )
    serial_stats = _stats(serial_times)
    rows: list[dict[str, Any]] = [
        {
            "device_count": 1,
            "active": True,
            "identity_passed": True,
            "rhs_max_abs_error": 0.0,
            "rhs_max_rel_error": 0.0,
            "median_s": serial_stats["median"],
            "speedup_vs_serial": 1.0,
            "speedup_gate_passed": False,
            "blocked_reasons": [],
            "stats": serial_stats,
        }
    ]

    for count in device_counts:
        count = int(count)
        if count <= 1:
            continue
        if len(devices) < count:
            rows.append(
                {
                    "device_count": count,
                    "active": False,
                    "identity_passed": False,
                    "rhs_max_abs_error": None,
                    "rhs_max_rel_error": None,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": ["not_enough_devices"],
                    "stats": {},
                }
            )
            continue
        routed_rhs, report = device_z_pencil_nonlinear_spectral_rhs(
            state,
            devices=devices[:count],
            atol=atol,
            rtol=rtol,
        )
        _block_until_ready(routed_rhs)
        if not report.decomposed_path_enabled:
            rows.append(
                {
                    "device_count": count,
                    "active": bool(report.device_sharding_active),
                    "identity_passed": bool(report.identity_passed),
                    "rhs_max_abs_error": report.rhs_max_abs_error,
                    "rhs_max_rel_error": report.rhs_max_rel_error,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": list(report.blocked_reasons),
                    "stats": {},
                }
            )
            continue

        mesh = Mesh(np.asarray(devices[:count]), ("z",))
        state_spec = PartitionSpec(None, None, None, None, "z")
        sharding = NamedSharding(mesh, state_spec)
        with mesh:
            sharded_state = jax.device_put(
                np.asarray(jax.device_get(state)),
                sharding,
            )
            sharded_jit = jax.jit(
                jax.shard_map(
                    lambda item: _pencil_nonlinear_spectral_rhs(item)[2],
                    mesh=mesh,
                    in_specs=state_spec,
                    out_specs=state_spec,
                    check_vma=False,
                )
            )
            sharded_out, sharded_times = _time_repeated(
                lambda: sharded_jit(sharded_state),
                warmups=int(warmups),
                repeats=int(repeats),
            )
        rhs_abs, rhs_rel = _max_abs_rel(sharded_out, serial_out, floor=atol)
        sharded_stats = _stats(sharded_times)
        speedup = serial_stats["median"] / sharded_stats["median"]
        timing_identity_passed = bool(rhs_abs <= atol and rhs_rel <= rtol)
        row_blockers: list[str] = []
        if not timing_identity_passed:
            row_blockers.append("timed_device_z_pencil_rhs_identity_failed")
        if speedup < min_speedup:
            row_blockers.append("speedup_below_gate")
        rows.append(
            {
                "device_count": count,
                "active": True,
                "identity_passed": timing_identity_passed,
                "rhs_max_abs_error": rhs_abs,
                "rhs_max_rel_error": rhs_rel,
                "median_s": sharded_stats["median"],
                "speedup_vs_serial": speedup,
                "speedup_gate_passed": bool(
                    speedup >= min_speedup and timing_identity_passed
                ),
                "blocked_reasons": row_blockers,
                "stats": sharded_stats,
            }
        )

    active_rows = [row for row in rows if row["active"] and row["device_count"] > 1]
    measured_speedups = [
        float(row["speedup_vs_serial"])
        for row in active_rows
        if row.get("speedup_vs_serial") is not None
    ]
    max_speedup = max(measured_speedups, default=1.0)
    all_active_identity = all(bool(row["identity_passed"]) for row in active_rows)
    speedup_gate = any(bool(row["speedup_gate_passed"]) for row in active_rows)
    if speedup_gate:
        status = "production_speedup_candidate"
    elif active_rows and not all_active_identity:
        status = "identity_failed_no_production_speedup"
    else:
        status = "identity_timed_no_production_speedup"
    if not active_rows:
        status = "skipped_no_multidevice"

    return _json_clean(
        {
            "kind": "nonlinear_device_z_pencil_rhs_profile",
            "claim_scope": (
                "device z-sharded shard_map fused pencil nonlinear RHS timing; FFT axes are local "
                "per z shard, no global spectral reconstruction is used, and no production speedup "
                "claim is allowed unless the speedup gate passes"
            ),
            "backend": str(jax.default_backend()),
            "device_count_available": int(len(devices)),
            "shape": shape,
            "warmups": int(warmups),
            "repeats": int(repeats),
            "atol": float(atol),
            "rtol": float(rtol),
            "min_speedup": float(min_speedup),
            "serial_stats_s": serial_stats,
            "rows": rows,
            "summary": {
                "status": status,
                "all_active_identity_passed": bool(all_active_identity),
                "max_speedup_vs_serial": float(max_speedup),
                "production_speedup_claim_allowed": bool(speedup_gate),
            },
        }
    )


def _write_rhs_artifacts(summary: dict[str, Any], out_prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = list(summary["rows"])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device_count",
                "active",
                "identity_passed",
                "rhs_max_abs_error",
                "rhs_max_rel_error",
                "median_s",
                "speedup_vs_serial",
                "speedup_gate_passed",
                "blocked_reasons",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    counts = [int(row["device_count"]) for row in rows]
    speedups = [float(row["speedup_vs_serial"] or 0.0) for row in rows]
    errors = [float(row["rhs_max_abs_error"] or 0.0) for row in rows]
    axes[0].plot(counts, speedups, "o-", lw=2.0, color="#1b6ca8")
    axes[0].axhline(
        float(summary["min_speedup"]),
        color="0.25",
        ls=":",
        lw=1.2,
        label="promotion gate",
    )
    axes[0].set_xlabel("logical CPU devices")
    axes[0].set_ylabel("speedup vs serial")
    axes[0].set_title("z-sharded fused pencil RHS")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(
        [str(count) for count in counts], np.maximum(errors, 1.0e-16), color="#b65f23"
    )
    axes[1].axhline(float(summary["atol"]), color="0.25", ls=":", lw=1.2, label="atol")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("logical CPU devices")
    axes[1].set_ylabel("max |RHS error|")
    axes[1].set_title("serial-vs-sharded identity")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25, axis="y")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> None:
    if summary.get("kind") == "nonlinear_device_z_pencil_rhs_profile":
        _write_rhs_artifacts(summary, out_prefix)
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = list(summary["rows"])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device_count",
                "active",
                "identity_passed",
                "transport_window_identity_passed",
                "final_state_max_abs_error",
                "final_state_max_rel_error",
                "physical_flux_trace_max_abs_error",
                "physical_flux_trace_max_rel_error",
                "median_s",
                "speedup_vs_serial",
                "speedup_gate_passed",
                "timing_scope",
                "identity_gate_elapsed_s",
                "observable_gate_median_s",
                "observable_gate_overhead_vs_compute",
                "blocked_reasons",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    set_plot_style()
    has_observable_timing = any(
        row.get("observable_gate_overhead_vs_compute") is not None for row in rows
    )
    ncols = 3 if has_observable_timing else 2
    fig, axes_arr = plt.subplots(
        1,
        ncols,
        figsize=(13.2 if has_observable_timing else 9.2, 3.6),
        constrained_layout=True,
    )
    axes = list(np.ravel(axes_arr))
    counts = [int(row["device_count"]) for row in rows]
    speedups = [float(row["speedup_vs_serial"] or 0.0) for row in rows]
    errors = [float(row["final_state_max_abs_error"] or 0.0) for row in rows]
    axes[0].plot(counts, speedups, "o-", lw=2.0, color="#1b6ca8")
    axes[0].axhline(
        float(summary["min_speedup"]), color="0.25", ls=":", lw=1.2, label="gate"
    )
    axes[0].set_xlabel("local devices")
    axes[0].set_ylabel("speedup vs serial")
    axes[0].set_title("z-sharded transport window")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(
        [str(count) for count in counts], np.maximum(errors, 1.0e-16), color="#b65f23"
    )
    axes[1].axhline(float(summary["atol"]), color="0.25", ls=":", lw=1.2, label="atol")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("local devices")
    axes[1].set_ylabel("max final-state error")
    axes[1].set_title("transport-window identity")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25, axis="y")

    if has_observable_timing:
        observable_counts = [
            int(row["device_count"])
            for row in rows
            if row.get("observable_gate_overhead_vs_compute") is not None
        ]
        observable_overheads = [
            float(row["observable_gate_overhead_vs_compute"])
            for row in rows
            if row.get("observable_gate_overhead_vs_compute") is not None
        ]
        axes[2].bar(
            [str(count) for count in observable_counts],
            observable_overheads,
            color="#6a7f3f",
        )
        if observable_overheads and max(observable_overheads) > 10.0:
            axes[2].set_yscale("log")
        axes[2].axhline(1.0, color="0.25", ls=":", lw=1.2, label="compute parity")
        axes[2].set_xlabel("local devices")
        axes[2].set_ylabel("observable gate / compute")
        axes[2].set_title("diagnostic overhead")
        axes[2].legend(frameon=False, fontsize=8, loc="lower right")
        axes[2].grid(True, alpha=0.25, axis="y")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("transport-window", "rhs"),
        default="transport-window",
        help=(
            "profile the paper-facing transport window by default, or the "
            "narrow fused-RHS microkernel with 'rhs'"
        ),
    )
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--shape", type=_parse_int_tuple, default=(4, 16, 96, 96, 32))
    parser.add_argument("--device-counts", type=_parse_int_tuple, default=(1, 2, 4))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--observable-repeats",
        type=int,
        default=0,
        help=(
            "time the scalar transport-window identity/observable gate this many "
            "times after the required identity pass; not included in speedup gates"
        ),
    )
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--z-chunk-size", type=int)
    parser.add_argument(
        "--auto-z-chunk-size",
        action="store_true",
        help="choose z_chunk_size from the cuFFT batch-pressure preflight model",
    )
    parser.add_argument("--max-fft-batch-count", type=int, default=65_536)
    parser.add_argument(
        "--observable-mode",
        choices=("host_gather", "sharded_reduce"),
        default="host_gather",
        help=(
            "choose how the identity gate records scalar observables; "
            "sharded_reduce avoids full-state host gathers"
        ),
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-device-count", type=int)
    parser.add_argument("--hlo-prefix", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shape = tuple(int(item) for item in args.shape)
    if len(shape) != 5:
        raise ValueError("--shape must contain five comma-separated integers")
    device_counts = tuple(int(item) for item in args.device_counts)
    if args.mode == "rhs":
        summary = build_rhs_profile(
            shape=shape,  # type: ignore[arg-type]
            device_counts=device_counts,
            warmups=int(args.warmups),
            repeats=int(args.repeats),
            atol=float(args.atol),
            rtol=float(args.rtol),
            min_speedup=float(args.min_speedup),
        )
        out_prefix = args.out_prefix or DEFAULT_RHS_OUT_PREFIX
    else:
        summary = build_profile(
            shape=shape,  # type: ignore[arg-type]
            device_counts=device_counts,
            steps=int(args.steps),
            dt=float(args.dt),
            warmups=int(args.warmups),
            repeats=int(args.repeats),
            observable_repeats=int(args.observable_repeats),
            atol=float(args.atol),
            rtol=float(args.rtol),
            min_speedup=float(args.min_speedup),
            z_chunk_size=args.z_chunk_size,
            auto_z_chunk_size=bool(args.auto_z_chunk_size),
            max_fft_batch_count=int(args.max_fft_batch_count),
            observable_mode=str(args.observable_mode),
            trace_dir=args.trace_dir,
            trace_device_count=args.trace_device_count,
            hlo_prefix=args.hlo_prefix,
        )
        out_prefix = args.out_prefix or DEFAULT_OUT_PREFIX
    write_artifacts(summary, out_prefix)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
