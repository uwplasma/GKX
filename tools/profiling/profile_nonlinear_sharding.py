#!/usr/bin/env python3
"""Profile fixed-step nonlinear state sharding with a numerical identity gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import statistics
import sys
import tempfile
import time
from pathlib import Path
import re
import subprocess
from typing import Any

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

from gkx._version import __version__ as gkx_version
from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.solvers.nonlinear.state_integration import integrate_nonlinear_cached, nonlinear_rhs_cached
from gkx.parallel.integrators import integrate_nonlinear_sharded
from gkx.parallel.state import resolve_state_sharding
from gkx.terms.config import TermConfig

try:
    from tools.profiling._profiler_options import git_source_state
except ModuleNotFoundError:  # Direct ``python tools/profiling/...`` execution.
    from _profiler_options import git_source_state


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "_static" / "nonlinear_sharding_profile.json"
DEFAULT_SWEEP_PREFIX = ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling"
OFFICE_GPU_XLARGE_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_gpu_xlarge"
)
PROFILE_TOOL = Path(__file__).resolve()
CPU_WHOLE_STATE_SHARDING_SKIP_REASON = (
    "skipped: cpu_whole_state_pjit_sharding_unsafe_for_fft_layout; "
    "use --allow-unsafe-cpu-state-sharding only for bounded debugging"
)


def _artifact_path_for_contract(path: Path) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _profile_command_argv(argv: list[str] | None) -> list[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *[str(item) for item in raw_args],
    ]


def _profile_command(argv: list[str] | None) -> str:
    return shlex.join(_profile_command_argv(argv))


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "gkx": str(gkx_version),
        "jax": str(getattr(jax, "__version__", "unknown")),
        "jaxlib": str(getattr(jaxlib, "__version__", "unknown")),
        "numpy": str(np.__version__),
    }


def _git_source_state() -> dict[str, Any]:
    """Return reproducible source provenance without requiring Git at runtime."""

    return git_source_state(ROOT)


def _source_contract(
    args: argparse.Namespace,
    argv: list[str] | None,
    *,
    backend: str | None = None,
    device_count: int | None = None,
) -> dict[str, Any]:
    timing_warmup_repeat = {"warmups": int(args.warmups), "repeats": int(args.repeats)}
    return {
        "source_contract_version": 1,
        "backend": str(jax.default_backend() if backend is None else backend),
        "device_count": int(
            jax.device_count() if device_count is None else device_count
        ),
        "sharding_axis": str(args.sharding),
        "profile_command": _profile_command(argv),
        "profile_command_argv": _profile_command_argv(argv),
        "source_artifact": _artifact_path_for_contract(Path(args.out_json)),
        "software_versions": _software_versions(),
        **_git_source_state(),
        "timing_warmup_repeat": timing_warmup_repeat,
        "allow_unsafe_cpu_state_sharding": bool(args.allow_unsafe_cpu_state_sharding),
    }


def _block_until_ready(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _build_problem(args: argparse.Namespace) -> tuple[jnp.ndarray, Any, LinearParams]:
    grid_cfg = GridConfig(
        Nx=int(args.nx),
        Ny=int(args.ny),
        Nz=int(args.nz),
        Lx=6.0,
        Ly=6.0,
        boundary="periodic",
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    shape = (int(args.nl), int(args.nm), grid.ky.size, grid.kx.size, grid.z.size)
    G0 = jnp.zeros(shape, dtype=jnp.complex64)
    z = jnp.arange(grid.z.size, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * z / max(int(grid.z.size), 1)
    amplitude = jnp.asarray(args.amplitude, dtype=G0.real.dtype)
    if grid.ky.size < 3 or grid.kx.size < 3:
        raise ValueError("nonlinear profile requires at least three ky and kx modes")
    G0 = G0.at[0, 0, 1, 1, :].set(amplitude * (1.0 + 0.25j) * jnp.exp(1j * phase))
    G0 = G0.at[0, 0, 2, 2, :].set(0.7 * amplitude * (1.0 - 0.4j) * jnp.exp(-2j * phase))
    if int(args.nl) > 1 and int(args.nm) > 1:
        G0 = G0.at[1, 1, 1, 2, :].set(
            0.35 * amplitude * (1.0 + 0.1j) * jnp.exp(3j * phase)
        )
    cache = build_linear_cache(grid, geom, params, int(args.nl), int(args.nm))
    return G0, cache, params


def _time_call(fn: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    out = _block_until_ready(fn())
    return out, time.perf_counter() - start


def _time_repeated(fn: Any, *, warmups: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(int(warmups)):
        _block_until_ready(fn())
    last: Any = None
    times: list[float] = []
    for _ in range(int(repeats)):
        last, elapsed = _time_call(fn)
        times.append(float(elapsed))
    return last, times


def _time_stats(times: list[float]) -> dict[str, float]:
    if not times:
        raise ValueError("at least one timing sample is required")
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _max_abs_rel_error(
    candidate: Any, reference: Any, *, floor: float = 1.0e-30
) -> tuple[float, float]:
    """Return max absolute and scale-normalized errors for two array-like values."""

    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    err = candidate_arr - reference_arr
    max_abs = float(np.max(np.abs(err)))
    scale = max(float(np.max(np.abs(reference_arr))), float(floor))
    return max_abs, float(max_abs / scale)


def _nonlinear_diagnostic_identity_metrics(
    reference_state: Any,
    candidate_state: Any,
    cache: Any,
    params: LinearParams,
    terms: TermConfig,
    *,
    compressed_real_fft: bool,
    laguerre_mode: str,
) -> dict[str, float]:
    """Compare field solve and nonlinear RHS diagnostics on final states."""

    reference_rhs, reference_fields = nonlinear_rhs_cached(
        jnp.asarray(reference_state),
        cache,
        params,
        terms,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )
    candidate_rhs, candidate_fields = nonlinear_rhs_cached(
        jnp.asarray(candidate_state),
        cache,
        params,
        terms,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )
    _block_until_ready(
        (reference_rhs, reference_fields, candidate_rhs, candidate_fields)
    )
    rhs_abs, rhs_rel = _max_abs_rel_error(candidate_rhs, reference_rhs)
    phi_abs, phi_rel = _max_abs_rel_error(candidate_fields.phi, reference_fields.phi)
    return {
        "max_abs_rhs_error": rhs_abs,
        "max_rel_rhs_error": rhs_rel,
        "max_abs_phi_error": phi_abs,
        "max_rel_phi_error": phi_rel,
    }


def _initial_nonlinear_activity(
    state: Any, cache: Any, params: LinearParams, *, laguerre_mode: str
) -> float:
    """Return the maximum pure nonlinear RHS magnitude for profile admission."""

    nonlinear_only = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=1.0,
    )
    rhs, _fields = nonlinear_rhs_cached(
        state,
        cache,
        params,
        nonlinear_only,
        compressed_real_fft=True,
        laguerre_mode=laguerre_mode,
    )
    _block_until_ready(rhs)
    return float(np.max(np.abs(np.asarray(rhs))))


def _sharding_specs(primary: str, extra: str | None) -> list[str]:
    specs: list[str] = []
    for raw in (primary, extra):
        if raw is None:
            continue
        for item in str(raw).split(","):
            key = item.strip()
            if key and key not in specs:
                specs.append(key)
    return specs


def _best_identity_preserving_candidate(
    sharded_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the fastest identity-preserving candidate from a profile payload."""

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for spec, result in sharded_results.items():
        speedup = result.get("engineering_speedup_median")
        if (
            not bool(result.get("state_sharding_active", False))
            or not bool(result.get("identity_gate_pass", False))
            or speedup is None
        ):
            continue
        speedup_float = float(speedup)
        if math.isfinite(speedup_float):
            candidates.append((speedup_float, str(spec), result))
    if not candidates:
        return {
            "spec": None,
            "engineering_speedup_median": None,
            "state_sharding_active": False,
            "identity_gate_pass": False,
        }
    speedup, spec, result = max(candidates, key=lambda item: item[0])
    return {
        "spec": spec,
        "engineering_speedup_median": float(speedup),
        "state_sharding_active": bool(result.get("state_sharding_active", False)),
        "identity_gate_pass": bool(result.get("identity_gate_pass", False)),
    }


def _skip_unsafe_cpu_state_sharding(
    *,
    backend: str,
    device_count: int,
    state_sharding_active: bool,
    allow_unsafe_cpu_state_sharding: bool,
) -> bool:
    """Return True when a pjit whole-state shard should be skipped on CPU.

    Multi-device CPU pjit sharding of the nonlinear state can feed non-monotonic
    layouts into XLA FFT thunks and abort the process before Python can catch an
    exception. The production-speedup lane must therefore fail closed unless a
    developer explicitly opts into this unsafe diagnostic path.
    """

    return bool(
        str(backend).lower() == "cpu"
        and int(device_count) > 1
        and bool(state_sharding_active)
        and not bool(allow_unsafe_cpu_state_sharding)
    )


def _candidate_failure(
    *,
    state_sharding_active: bool,
    error: str,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Return the standard fail-closed candidate row."""

    row: dict[str, Any] = {
        "state_sharding_active": bool(state_sharding_active),
        "times_s": [],
        "stats_s": None,
        "engineering_speedup_median": None,
        "max_abs_state_error": None,
        "max_rel_state_error": None,
        "max_abs_rhs_error": None,
        "max_rel_rhs_error": None,
        "max_abs_phi_error": None,
        "max_rel_phi_error": None,
        "diagnostic_identity_gate_pass": False,
        "identity_gate_pass": False,
        "error": str(error),
    }
    if skip_reason is not None:
        row["skip_reason"] = str(skip_reason)
    return row


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


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _append_xla_flag(existing: str, flag: str) -> str:
    key = flag.split("=")[0]
    if key == "--xla_force_host_platform_device_count":
        cleaned = re.sub(
            r"--xla_force_host_platform_device_count=\S+", "", existing
        ).strip()
        return f"{cleaned} {flag}".strip()
    if key in existing:
        return existing
    return f"{existing} {flag}".strip()


def _tail_text(value: Any, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-int(limit) :]


def _device_env(
    base_env: dict[str, str], *, backend: str, devices: int
) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = _append_xla_flag(
            env.get("XLA_FLAGS", ""),
            f"--xla_force_host_platform_device_count={int(devices)}",
        )
    elif backend_key in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(int(devices))
        )
    else:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return env


def _sweep_profile_command(
    *,
    out_json: Path,
    trace_dir: Path | None,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    dt: float,
    steps: int,
    method: str,
    sharding: str,
    sharding_options: str,
    laguerre_mode: str,
    warmups: int,
    repeats: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROFILE_TOOL),
        "--out-json",
        str(out_json),
        "--nx",
        str(int(nx)),
        "--ny",
        str(int(ny)),
        "--nz",
        str(int(nz)),
        "--nl",
        str(int(nl)),
        "--nm",
        str(int(nm)),
        "--dt",
        str(float(dt)),
        "--steps",
        str(int(steps)),
        "--method",
        str(method),
        "--sharding",
        str(sharding),
        "--sharding-options",
        str(sharding_options),
        "--laguerre-mode",
        str(laguerre_mode),
        "--warmups",
        str(int(warmups)),
        "--repeats",
        str(int(repeats)),
    ]
    if trace_dir is not None:
        cmd.extend(["--trace-dir", str(trace_dir)])
    return cmd


def _select_parallel_row(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    best = dict(payload.get("best_identity_preserving_candidate") or {})
    best_spec = best.get("spec")
    if best_spec is not None:
        result = dict(payload["sharded_results"][str(best_spec)])
        if (
            bool(result.get("identity_gate_pass", False))
            and result.get("stats_s") is not None
        ):
            return str(best_spec), result
    requested = str(payload.get("state_sharding_requested", "auto"))
    return requested, dict(payload["sharded_results"][requested])


def _row_from_payload(
    payload: dict[str, Any], *, requested_devices: int
) -> dict[str, Any]:
    spec, result = _select_parallel_row(payload)
    stats = result.get("stats_s")
    parallel_median = float(stats["median"]) if stats else math.nan
    serial_median = float(payload["serial_stats_s"]["median"])
    return {
        "requested_devices": int(requested_devices),
        "actual_devices": int(payload["device_count"]),
        "backend": str(payload["default_backend"]),
        "state_shape": tuple(int(x) for x in payload["state_shape"]),
        "best_spec": spec,
        "state_sharding_active": bool(result.get("state_sharding_active", False)),
        "identity_gate_pass": bool(result.get("identity_gate_pass", False)),
        "serial_median_s": serial_median,
        "parallel_median_s": parallel_median,
        "same_process_speedup": serial_median / parallel_median
        if parallel_median > 0.0
        else math.nan,
        "strong_speedup_vs_1_device": math.nan,
        "max_abs_state_error": result.get("max_abs_state_error"),
        "max_rel_state_error": result.get("max_rel_state_error"),
        "profile_json": str(payload.get("_profile_json", "")),
        "source_contract_version": payload.get("source_contract_version"),
        "profile_command": payload.get("profile_command"),
        "profile_command_argv": payload.get("profile_command_argv"),
        "source_artifact": payload.get("source_artifact"),
        "software_versions": payload.get("software_versions"),
        "timing_warmup_repeat": payload.get("timing_warmup_repeat"),
        "profile_backend": payload.get("backend", payload.get("default_backend")),
        "profile_device_count": payload.get("device_count"),
        "profile_sharding_axis": payload.get(
            "sharding_axis", payload.get("state_sharding_requested")
        ),
        "error": result.get("error"),
    }


def _failure_row(
    *,
    requested_devices: int,
    backend: str,
    profile_json: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "requested_devices": int(requested_devices),
        "actual_devices": None,
        "backend": str(backend),
        "state_shape": None,
        "best_spec": None,
        "state_sharding_active": False,
        "identity_gate_pass": False,
        "serial_median_s": math.nan,
        "parallel_median_s": math.nan,
        "same_process_speedup": math.nan,
        "strong_speedup_vs_1_device": math.nan,
        "max_abs_state_error": None,
        "max_rel_state_error": None,
        "profile_json": str(profile_json),
        "error": str(error),
    }


def _speedup_status(rows: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    parallel_rows = [row for row in rows if int(row.get("requested_devices") or 0) > 1]
    speedup_threshold = 1.0
    speedup_blockers: list[str] = []
    for row in parallel_rows:
        requested_devices = int(row.get("requested_devices") or 0)
        if not bool(row.get("identity_gate_pass", False)):
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_identity_failed"
            )
            continue
        speedup = row.get("strong_speedup_vs_1_device")
        speedup_value = float(speedup) if speedup is not None else math.nan
        if not math.isfinite(speedup_value):
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_speedup_missing"
            )
        elif speedup_value < speedup_threshold:
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_speedup_{speedup_value:.3g}_below_{speedup_threshold:.3g}"
            )
    speedup_passed = bool(parallel_rows) and not speedup_blockers
    return {
        "speedup_passed": speedup_passed,
        "speedup_threshold_vs_1_device": speedup_threshold,
        "speedup_blockers": speedup_blockers,
        "status": "identity_and_speedup"
        if speedup_passed
        else "diagnostic_identity_only",
    }


def run_sweep(
    *,
    backend: str,
    devices: list[int],
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    dt: float,
    steps: int,
    method: str,
    sharding: str,
    sharding_options: str,
    laguerre_mode: str,
    warmups: int,
    repeats: int,
    timeout_s: float,
    trace: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="gkx-nonlinear-scaling-"
    ) as tmp_name:
        tmp = Path(tmp_name)
        for requested_devices in devices:
            profile_json = tmp / f"{backend}_{requested_devices}devices.json"
            trace_dir = (
                tmp / f"trace_{backend}_{requested_devices}devices" if trace else None
            )
            env = _device_env(
                os.environ, backend=backend, devices=int(requested_devices)
            )
            cmd = _sweep_profile_command(
                out_json=profile_json,
                trace_dir=trace_dir,
                nx=nx,
                ny=ny,
                nz=nz,
                nl=nl,
                nm=nm,
                dt=dt,
                steps=steps,
                method=method,
                sharding=sharding,
                sharding_options=sharding_options,
                laguerre_mode=laguerre_mode,
                warmups=warmups,
                repeats=repeats,
            )
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=float(timeout_s),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                output_tail = _tail_text(exc.stderr) or _tail_text(exc.stdout)
                rows.append(
                    _failure_row(
                        requested_devices=int(requested_devices),
                        backend=backend,
                        profile_json=profile_json,
                        error=(
                            f"profile timed out after {float(timeout_s):.3g} s"
                            + (f"\n{output_tail}" if output_tail else "")
                        ),
                    )
                )
                continue
            if profile_json.exists():
                try:
                    payload = json.loads(profile_json.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    payload["_profile_json"] = str(profile_json)
                    payload["profile_returncode"] = int(completed.returncode)
                    profiles[str(requested_devices)] = payload
                    row = _row_from_payload(
                        payload, requested_devices=int(requested_devices)
                    )
                    row["profile_returncode"] = int(completed.returncode)
                    rows.append(row)
                    continue
            if completed.returncode != 0:
                rows.append(
                    _failure_row(
                        requested_devices=int(requested_devices),
                        backend=backend,
                        profile_json=profile_json,
                        error=_tail_text(completed.stderr)
                        or _tail_text(completed.stdout),
                    )
                )
                continue
            rows.append(
                _failure_row(
                    requested_devices=int(requested_devices),
                    backend=backend,
                    profile_json=profile_json,
                    error="profile returned success but did not write a valid JSON artifact",
                )
            )

    valid_rows = [
        row
        for row in rows
        if bool(row.get("identity_gate_pass")) and float(row["parallel_median_s"]) > 0.0
    ]
    baseline_row = next(
        (row for row in valid_rows if int(row["requested_devices"]) == 1), None
    )
    baseline = float(baseline_row["parallel_median_s"]) if baseline_row else math.nan
    for row in rows:
        current = (
            float(row["parallel_median_s"])
            if row["parallel_median_s"] is not None
            else math.nan
        )
        row["strong_speedup_vs_1_device"] = (
            baseline / current if baseline > 0.0 and current > 0.0 else math.nan
        )

    passed = all(bool(row.get("identity_gate_pass")) for row in rows)
    speedup_status = _speedup_status(rows, backend=backend)
    return _json_clean(
        {
            "kind": "nonlinear_sharding_strong_scaling_sweep",
            "backend": str(backend),
            "devices": [int(device) for device in devices],
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Nz": int(nz),
                "Nl": int(nl),
                "Nm": int(nm),
            },
            "dt": float(dt),
            "steps": int(steps),
            "method": str(method),
            "sharding": str(sharding),
            "sharding_options": str(sharding_options),
            "laguerre_mode": str(laguerre_mode),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "timeout_s": float(timeout_s),
            "identity_passed": passed,
            **speedup_status,
            "claim_scope": (
                "large fixed-step nonlinear state-sharding strong-scaling artifact with numerical identity gates; "
                "speedup is a separate fail-closed field, and this artifact is profiler evidence unless "
                "speedup_passed is true, not as a broad production speedup claim"
            ),
            "rows": rows,
            "profiles": profiles,
        }
    )


def write_sweep_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    fieldnames = [
        "requested_devices",
        "actual_devices",
        "backend",
        "state_shape",
        "best_spec",
        "state_sharding_active",
        "identity_gate_pass",
        "serial_median_s",
        "parallel_median_s",
        "same_process_speedup",
        "strong_speedup_vs_1_device",
        "max_abs_state_error",
        "max_rel_state_error",
        "profile_json",
        "source_contract_version",
        "profile_command",
        "profile_command_argv",
        "source_artifact",
        "software_versions",
        "timing_warmup_repeat",
        "profile_backend",
        "profile_device_count",
        "profile_sharding_axis",
        "profile_returncode",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    valid = [
        row
        for row in rows
        if row["parallel_median_s"] is not None
        and math.isfinite(float(row["parallel_median_s"]))
    ]
    x = np.asarray([int(row["requested_devices"]) for row in valid], dtype=float)
    y = np.asarray(
        [float(row["strong_speedup_vs_1_device"]) for row in valid], dtype=float
    )
    times = np.asarray([float(row["parallel_median_s"]) for row in valid], dtype=float)
    errors = np.asarray(
        [
            float(row["max_rel_state_error"])
            if row["max_rel_state_error"] is not None
            else np.nan
            for row in valid
        ],
        dtype=float,
    )

    axes[0].plot(x, y, "o-", lw=2.2, color="#276b8e", label="measured")
    if x.size:
        axes[0].plot(x, x, ":", lw=1.5, color="0.35", label="ideal")
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("speedup vs 1 device")
    axes[0].set_title(f"{str(summary['backend']).upper()} nonlinear strong scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        x,
        np.maximum(times, 1.0e-16),
        "s-",
        lw=2.0,
        color="#b45f06",
        label="median time",
    )
    ax_err = axes[1].twinx()
    ax_err.semilogy(
        x + 0.03,
        np.maximum(errors, 1.0e-16),
        "^-",
        lw=1.7,
        color="#4f7f2d",
        label="rel. error",
    )
    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("median time [s]")
    ax_err.set_ylabel("max relative state error")
    axes[1].set_title("Timing and identity")
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_err.get_legend_handles_labels()
    axes[1].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--office-gpu-xlarge",
        action="store_true",
        help=(
            "Use the canonical office two-GPU nonlinear sharding profile: "
            "gpu backend, devices 1,2, Nx=48, Ny=96, Nz=128, Nl=4, Nm=8, steps=12, trace enabled."
        ),
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_SWEEP_PREFIX)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2])
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--nl", type=int, default=3)
    parser.add_argument("--nm", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sharding", default="auto")
    parser.add_argument("--sharding-options", default="auto,kx")
    parser.add_argument("--laguerre-mode", default="grid")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--trace", action="store_true")
    return parser


def apply_sweep_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply named profiling presets after argparse has filled defaults."""

    if not bool(getattr(args, "office_gpu_xlarge", False)):
        return args
    args.backend = "gpu"
    args.devices = [1, 2]
    args.nx = 48
    args.ny = 96
    args.nz = 128
    args.nl = 4
    args.nm = 8
    args.steps = 12
    args.sharding = "auto"
    args.sharding_options = "auto,kx"
    args.out_prefix = OFFICE_GPU_XLARGE_PREFIX
    args.trace = True
    return args


def main_sweep(argv: list[str] | None = None) -> int:
    args = apply_sweep_preset(build_sweep_parser().parse_args(argv))
    summary = run_sweep(
        backend=str(args.backend),
        devices=list(args.devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        dt=float(args.dt),
        steps=int(args.steps),
        method=str(args.method),
        sharding=str(args.sharding),
        sharding_options=str(args.sharding_options),
        laguerre_mode=str(args.laguerre_mode),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        timeout_s=float(args.timeout_s),
        trace=bool(args.trace),
    )
    paths = write_sweep_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "identity_passed": summary["identity_passed"],
                "speedup_passed": summary["speedup_passed"],
                "status": summary["status"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sharding", default="auto")
    parser.add_argument(
        "--sharding-options",
        default=None,
        help=(
            "Comma-separated candidate nonlinear state-decomposition axes to gate. "
            "Use auto,kx for the release-gated path; z is exploratory FFT-axis decomposition."
        ),
    )
    parser.add_argument("--laguerre-mode", default="grid")
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional JAX profiler trace directory. The trace is not a runtime claim; it localizes hot paths.",
    )
    parser.add_argument(
        "--allow-unsafe-cpu-state-sharding",
        action="store_true",
        help=(
            "Allow multi-device CPU pjit whole-state nonlinear sharding. This can abort in XLA FFT "
            "layout/collective code on current CPU backends and is intended only for bounded debugging."
        ),
    )
    return parser


def main_profile(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    G0, cache, params = _build_problem(args)
    G0_host = np.asarray(G0)
    terms = TermConfig(
        nonlinear=1.0, collisions=0.0, hypercollisions=0.0, apar=0.0, bpar=0.0
    )
    nonlinear_activity = _initial_nonlinear_activity(
        G0, cache, params, laguerre_mode=str(args.laguerre_mode)
    )
    if not math.isfinite(nonlinear_activity) or nonlinear_activity <= 0.0:
        raise RuntimeError(
            "nonlinear sharding profile requires a finite, nonzero nonlinear RHS"
        )
    sharding_specs = _sharding_specs(str(args.sharding), args.sharding_options)

    def serial_run():
        return integrate_nonlinear_cached(
            jnp.asarray(G0_host),
            cache,
            params,
            dt=float(args.dt),
            steps=int(args.steps),
            method=str(args.method),
            terms=terms,
            compressed_real_fft=True,
            laguerre_mode=str(args.laguerre_mode),
            return_fields=False,
        )

    def make_sharded_run(spec: str):
        state_sharding = resolve_state_sharding(G0, spec)

        def sharded_run():
            return integrate_nonlinear_sharded(
                jnp.asarray(G0_host),
                cache,
                params,
                dt=float(args.dt),
                steps=int(args.steps),
                method=str(args.method),
                terms=terms,
                state_sharding=state_sharding,
                compressed_real_fft=True,
                laguerre_mode=str(args.laguerre_mode),
                return_fields=False,
            )

        return state_sharding, sharded_run

    trace_status: dict[str, Any] = {
        "requested": args.trace_dir is not None,
        "path": None,
        "error": None,
    }
    sharded_fns: dict[str, Any] = {}
    sharded_state_active: dict[str, bool] = {}
    for spec in sharding_specs:
        state_sharding, fn = make_sharded_run(spec)
        sharded_fns[spec] = fn
        sharded_state_active[spec] = state_sharding is not None

    if args.trace_dir is not None:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_status["path"] = str(trace_dir)
        try:
            jax.profiler.start_trace(str(trace_dir))
            _block_until_ready(serial_run())
            for spec, fn in sharded_fns.items():
                if _skip_unsafe_cpu_state_sharding(
                    backend=str(jax.default_backend()),
                    device_count=int(jax.device_count()),
                    state_sharding_active=bool(sharded_state_active[spec]),
                    allow_unsafe_cpu_state_sharding=bool(
                        args.allow_unsafe_cpu_state_sharding
                    ),
                ):
                    continue
                _block_until_ready(fn())
            jax.profiler.stop_trace()
        except (
            Exception
        ) as exc:  # pragma: no cover - profiler availability is platform-specific.
            trace_status["error"] = repr(exc)
            try:
                jax.profiler.stop_trace()
            except Exception:
                pass

    serial_final, serial_times = _time_repeated(
        serial_run, warmups=int(args.warmups), repeats=int(args.repeats)
    )
    serial_stats = _time_stats(serial_times)

    scale = max(float(np.max(np.abs(np.asarray(serial_final)))), 1.0e-30)
    sharded_results: dict[str, dict[str, Any]] = {}
    identity_passes: list[bool] = []
    for spec, fn in sharded_fns.items():
        if _skip_unsafe_cpu_state_sharding(
            backend=str(jax.default_backend()),
            device_count=int(jax.device_count()),
            state_sharding_active=bool(sharded_state_active[spec]),
            allow_unsafe_cpu_state_sharding=bool(args.allow_unsafe_cpu_state_sharding),
        ):
            identity_pass = False
            sharded_results[spec] = _candidate_failure(
                state_sharding_active=bool(sharded_state_active[spec]),
                error=CPU_WHOLE_STATE_SHARDING_SKIP_REASON,
                skip_reason="cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
            )
            identity_passes.append(identity_pass)
            continue
        try:
            sharded_final, sharded_times = _time_repeated(
                fn, warmups=int(args.warmups), repeats=int(args.repeats)
            )
            sharded_stats = _time_stats(sharded_times)
            err = np.asarray(sharded_final - serial_final)
            max_abs = float(np.max(np.abs(err)))
            max_rel = float(max_abs / scale)
            diagnostic_metrics = _nonlinear_diagnostic_identity_metrics(
                serial_final,
                sharded_final,
                cache,
                params,
                terms,
                compressed_real_fft=True,
                laguerre_mode=str(args.laguerre_mode),
            )
            diagnostic_pass = bool(
                diagnostic_metrics["max_abs_rhs_error"] <= 1.0e-5
                and diagnostic_metrics["max_rel_rhs_error"] <= 1.0e-5
                and diagnostic_metrics["max_abs_phi_error"] <= 1.0e-5
                and diagnostic_metrics["max_rel_phi_error"] <= 1.0e-5
            )
            identity_pass = bool(
                max_abs <= 1.0e-5 and max_rel <= 1.0e-5 and diagnostic_pass
            )
            sharded_results[spec] = {
                "state_sharding_active": bool(sharded_state_active[spec]),
                "times_s": sharded_times,
                "stats_s": sharded_stats,
                "engineering_speedup_median": float(
                    serial_stats["median"] / sharded_stats["median"]
                )
                if sharded_stats["median"] > 0.0
                else None,
                "max_abs_state_error": max_abs,
                "max_rel_state_error": max_rel,
                **diagnostic_metrics,
                "diagnostic_identity_gate_pass": diagnostic_pass,
                "identity_gate_pass": identity_pass,
                "error": None,
            }
        except Exception as exc:
            identity_pass = False
            sharded_results[spec] = _candidate_failure(
                state_sharding_active=bool(sharded_state_active[spec]),
                error=repr(exc),
            )
        identity_passes.append(identity_pass)

    primary_spec = sharding_specs[0]
    primary = sharded_results[primary_spec]
    best_candidate = _best_identity_preserving_candidate(sharded_results)
    source_contract = _source_contract(args, argv)
    payload = {
        "case": "cyclone_nonlinear_fixed_step",
        **source_contract,
        "devices": [str(device) for device in jax.devices()],
        "default_backend": str(source_contract["backend"]),
        "state_shape": list(map(int, G0.shape)),
        "state_sharding_requested": str(args.sharding),
        "state_sharding_primary": primary_spec,
        "state_sharding_active": bool(primary["state_sharding_active"]),
        "sharding_options": sharding_specs,
        "dt": float(args.dt),
        "steps": int(args.steps),
        "method": str(args.method),
        "laguerre_mode": str(args.laguerre_mode),
        "initial_nonlinear_rhs_max": nonlinear_activity,
        "warmups": int(source_contract["timing_warmup_repeat"]["warmups"]),
        "repeats": int(source_contract["timing_warmup_repeat"]["repeats"]),
        "serial_times_s": serial_times,
        "serial_stats_s": serial_stats,
        "sharded_results": sharded_results,
        "best_identity_preserving_candidate": best_candidate,
        "profiler_trace": trace_status,
        "serial_warm_s": serial_stats["median"],
        "sharded_warm_s": primary["stats_s"]["median"] if primary["stats_s"] else None,
        "engineering_speedup": (
            primary["engineering_speedup_median"]
            if bool(primary["state_sharding_active"])
            else None
        ),
        "max_abs_state_error": primary["max_abs_state_error"],
        "max_rel_state_error": primary["max_rel_state_error"],
        "identity_gate_pass": bool(all(identity_passes)),
        "claim_scope": (
            "Profiler/identity artifact for fixed-step nonlinear state sharding and candidate "
            "state-axis decompositions. Do not use as a published runtime claim without a larger "
            "matched CPU/GPU sweep and profiler trace review."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["identity_gate_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "sweep":
        return main_sweep(tokens[1:])
    return main_profile(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
