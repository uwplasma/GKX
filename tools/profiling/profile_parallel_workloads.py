#!/usr/bin/env python3
"""Profile independent-work parallel workloads with identity gates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEPENDENT_KY_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling"
)
DEFAULT_QUASILINEAR_UQ_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling"
)
# Backward-compatible default for tests that bind this module as the independent-ky tool.
DEFAULT_PREFIX = DEFAULT_INDEPENDENT_KY_PREFIX
_GAMMA_RTOL = 1.0e-7
_OMEGA_ATOL = 1.0e-7


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


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    return values


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all device counts must be positive")
    return values


def _split_ky(ky_values: np.ndarray, n_parts: int) -> list[np.ndarray]:
    ky = np.asarray(ky_values, dtype=float)
    if ky.ndim != 1 or ky.size == 0:
        raise ValueError("ky_values must be a nonempty one-dimensional array")
    parts = min(int(n_parts), int(ky.size))
    return [chunk for chunk in np.array_split(ky, parts) if chunk.size > 0]


def _split_values(values: np.ndarray, n_parts: int) -> list[np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    parts = min(int(n_parts), int(arr.size))
    return [chunk for chunk in np.array_split(arr, parts) if chunk.size > 0]


def _worker_env(
    base_env: dict[str, str], *, backend: str, worker_index: int
) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    elif backend_key in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = str(int(worker_index))
    else:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return env


def _time_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    return {
        "min": float(min(samples)),
        "median": float(statistics.median(samples)),
        "mean": float(statistics.fmean(samples)),
        "max": float(max(samples)),
        "std": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def args_global_gamma_rtol() -> float:
    return _GAMMA_RTOL


def args_global_omega_atol() -> float:
    return _OMEGA_ATOL


def _run_solver_chunk(args: argparse.Namespace) -> dict[str, Any]:
    from gkx.runtime import run_runtime_scan
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    ky = np.asarray(args.worker_ky, dtype=float)
    base_cfg, _ = load_runtime_from_toml(
        REPO_ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    cfg = replace(
        base_cfg,
        grid=replace(
            base_cfg.grid,
            Nx=int(args.nx),
            Ny=int(args.ny),
            Nz=int(args.nz),
            ntheta=int(args.nz),
            nperiod=1,
            y0=10.0,
        )
    )
    tmin = float(args.fit_start_fraction) * float(args.dt) * int(args.steps)
    tmax = float(args.fit_end_fraction) * float(args.dt) * int(args.steps)
    samples: list[float] = []
    last = None
    for repeat_index in range(int(args.warmups) + int(args.repeats)):
        start = time.perf_counter()
        result = run_runtime_scan(
            cfg,
            ky,
            Nl=int(args.nl),
            Nm=int(args.nm),
            dt=float(args.dt),
            steps=int(args.steps),
            solver="time",
            method=str(args.method),
            sample_stride=int(args.sample_stride),
            fit_signal="phi",
            auto_window=False,
            tmin=tmin,
            tmax=tmax,
            min_points=int(args.min_points),
            batch_ky=False,
            require_positive=False,
        )
        elapsed = time.perf_counter() - start
        last = result
        if repeat_index >= int(args.warmups):
            samples.append(float(elapsed))
    assert last is not None
    payload = {
        "ky": np.asarray(last.ky, dtype=float).tolist(),
        "gamma": np.asarray(last.gamma, dtype=float).tolist(),
        "omega": np.asarray(last.omega, dtype=float).tolist(),
        "samples_s": samples,
        "stats_s": _time_stats(samples),
    }
    Path(args.worker_out).write_text(
        json.dumps(_json_clean(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _independent_worker_command(
    args: argparse.Namespace, *, ky: np.ndarray, out_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "independent-ky",
        "--worker",
        "--worker-out",
        str(out_path),
        "--worker-ky",
        ",".join(f"{float(value):.16g}" for value in ky),
        "--nx",
        str(int(args.nx)),
        "--ny",
        str(int(args.ny)),
        "--nz",
        str(int(args.nz)),
        "--nl",
        str(int(args.nl)),
        "--nm",
        str(int(args.nm)),
        "--dt",
        str(float(args.dt)),
        "--steps",
        str(int(args.steps)),
        "--method",
        str(args.method),
        "--sample-stride",
        str(int(args.sample_stride)),
        "--fit-start-fraction",
        str(float(args.fit_start_fraction)),
        "--fit-end-fraction",
        str(float(args.fit_end_fraction)),
        "--min-points",
        str(int(args.min_points)),
        "--warmups",
        str(int(args.warmups)),
        "--repeats",
        str(int(args.repeats)),
    ]


def _run_independent_device_count(
    args: argparse.Namespace,
    *,
    ky_values: np.ndarray,
    device_count: int,
    tmp: Path,
) -> dict[str, Any]:
    chunks = _split_ky(ky_values, int(device_count))
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    start_wall = time.perf_counter()
    for worker_index, chunk in enumerate(chunks):
        out_path = tmp / f"{args.backend}_{device_count}_worker_{worker_index}.json"
        env = _worker_env(
            os.environ, backend=str(args.backend), worker_index=worker_index
        )
        proc = subprocess.Popen(
            _independent_worker_command(args, ky=chunk, out_path=out_path),
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((worker_index, out_path, proc))

    worker_payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for worker_index, out_path, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=float(args.timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            errors.append(
                f"worker {worker_index} timed out after {args.timeout_s}s\n{stderr[-2000:]}"
            )
            continue
        if proc.returncode != 0:
            errors.append(
                f"worker {worker_index} failed with code {proc.returncode}\n{stderr[-4000:]}\n{stdout[-1000:]}"
            )
            continue
        worker_payloads.append(json.loads(out_path.read_text(encoding="utf-8")))
    wall_s = time.perf_counter() - start_wall

    if errors:
        return {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "wall_s": wall_s,
            "timed_wall_s": math.nan,
            "ky": [],
            "gamma": [],
            "omega": [],
            "worker_stats": worker_payloads,
            "error": "\n".join(errors),
        }

    ky_all = np.concatenate(
        [np.asarray(payload["ky"], dtype=float) for payload in worker_payloads]
    )
    gamma_all = np.concatenate(
        [np.asarray(payload["gamma"], dtype=float) for payload in worker_payloads]
    )
    omega_all = np.concatenate(
        [np.asarray(payload["omega"], dtype=float) for payload in worker_payloads]
    )
    order = np.argsort(ky_all)
    timed_wall_s = max(
        float(payload["stats_s"]["median"]) for payload in worker_payloads
    )
    return _json_clean(
        {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "wall_s": float(wall_s),
            "timed_wall_s": float(timed_wall_s),
            "ky": ky_all[order].tolist(),
            "gamma": gamma_all[order].tolist(),
            "omega": omega_all[order].tolist(),
            "worker_stats": worker_payloads,
            "error": None,
        }
    )


def _identity_metrics(
    reference: dict[str, Any], row: dict[str, Any]
) -> dict[str, float | bool]:
    if row.get("error") is not None:
        return {
            "max_gamma_abs_error": math.nan,
            "max_gamma_rel_error": math.nan,
            "max_omega_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    ref_ky = np.asarray(reference["ky"], dtype=float)
    ky = np.asarray(row["ky"], dtype=float)
    if ref_ky.shape != ky.shape or not np.allclose(ref_ky, ky, rtol=0.0, atol=1.0e-14):
        return {
            "max_gamma_abs_error": math.nan,
            "max_gamma_rel_error": math.nan,
            "max_omega_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    ref_gamma = np.asarray(reference["gamma"], dtype=float)
    gamma = np.asarray(row["gamma"], dtype=float)
    ref_omega = np.asarray(reference["omega"], dtype=float)
    omega = np.asarray(row["omega"], dtype=float)
    gamma_abs = np.abs(gamma - ref_gamma)
    gamma_rel = gamma_abs / np.maximum(np.abs(ref_gamma), 1.0e-12)
    omega_abs = np.abs(omega - ref_omega)
    return {
        "max_gamma_abs_error": float(np.max(gamma_abs)),
        "max_gamma_rel_error": float(np.max(gamma_rel)),
        "max_omega_abs_error": float(np.max(omega_abs)),
        "identity_gate_pass": bool(
            float(np.max(gamma_rel)) <= float(args_global_gamma_rtol())
            and float(np.max(omega_abs)) <= float(args_global_omega_atol())
        ),
    }


def run_independent_ky_sweep(args: argparse.Namespace) -> dict[str, Any]:
    global _GAMMA_RTOL, _OMEGA_ATOL
    _GAMMA_RTOL = float(args.gamma_rtol)
    _OMEGA_ATOL = float(args.omega_atol)
    ky_values = np.asarray(args.ky, dtype=float)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="gkx-independent-ky-scaling-"
    ) as tmp_name:
        tmp = Path(tmp_name)
        for device_count in list(args.devices):
            rows.append(
                _run_independent_device_count(
                    args, ky_values=ky_values, device_count=int(device_count), tmp=tmp
                )
            )

    reference = next(
        (
            row
            for row in rows
            if int(row["requested_devices"]) == 1 and row.get("error") is None
        ),
        None,
    )
    baseline = float(reference["timed_wall_s"]) if reference is not None else math.nan
    for row in rows:
        metrics = (
            _identity_metrics(reference, row)
            if reference is not None
            else {
                "max_gamma_abs_error": math.nan,
                "max_gamma_rel_error": math.nan,
                "max_omega_abs_error": math.nan,
                "identity_gate_pass": False,
            }
        )
        row.update(metrics)
        current = (
            float(row["timed_wall_s"])
            if row.get("timed_wall_s") is not None
            else math.nan
        )
        row["strong_speedup_vs_1_device"] = (
            baseline / current if baseline > 0.0 and current > 0.0 else math.nan
        )
        row["parallel_efficiency"] = (
            row["strong_speedup_vs_1_device"] / float(row["requested_devices"])
            if math.isfinite(float(row["strong_speedup_vs_1_device"]))
            else math.nan
        )
    return _json_clean(
        {
            "kind": "independent_ky_scan_strong_scaling",
            "backend": str(args.backend),
            "devices": [int(value) for value in args.devices],
            "ky": ky_values.tolist(),
            "grid": {
                "Nx": int(args.nx),
                "Ny": int(args.ny),
                "Nz": int(args.nz),
                "Nl": int(args.nl),
                "Nm": int(args.nm),
            },
            "time": {
                "dt": float(args.dt),
                "steps": int(args.steps),
                "method": str(args.method),
                "sample_stride": int(args.sample_stride),
                "fit_start_fraction": float(args.fit_start_fraction),
                "fit_end_fraction": float(args.fit_end_fraction),
            },
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "gamma_rtol": float(args.gamma_rtol),
            "omega_atol": float(args.omega_atol),
            "identity_passed": all(bool(row.get("identity_gate_pass")) for row in rows),
            "claim_scope": (
                "solver-backed independent ky scan strong-scaling artifact. This validates the "
                "preferred production parallelization path for independent scans/UQ ensembles; "
                "it is not a nonlinear domain-decomposition speedup claim."
            ),
            "rows": rows,
        }
    )


def _quasilinear_reduced_observables(
    ky: np.ndarray, gamma: np.ndarray, omega: np.ndarray
) -> dict[str, Any]:
    from gkx.diagnostics.quasilinear_transport import quasilinear_feature_objective

    ky_arr = np.asarray(ky, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)
    kperp_eff2 = np.maximum(ky_arr**2, 1.0e-12)
    linear_weight = np.maximum(gamma_arr, 0.0)
    features = np.stack([gamma_arr, kperp_eff2, linear_weight], axis=-1)
    spectrum = np.asarray(
        quasilinear_feature_objective(features, rule="mixing_length"), dtype=float
    )
    finite = np.isfinite(spectrum)
    heat_proxy = float(np.sum(np.where(finite, spectrum, 0.0)))
    weighted_growth = float(np.sum(np.maximum(gamma_arr, 0.0)))
    omega_span = (
        float(np.nanmax(omega_arr) - np.nanmin(omega_arr)) if omega_arr.size else 0.0
    )
    return {
        "heat_flux_proxy": heat_proxy,
        "weighted_growth": weighted_growth,
        "omega_span": omega_span,
        "spectrum": spectrum.tolist(),
        "kperp_eff2": kperp_eff2.tolist(),
        "linear_weight": linear_weight.tolist(),
    }


def _run_ensemble_chunk(args: argparse.Namespace) -> dict[str, Any]:
    from gkx.runtime import run_runtime_scan
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    gradients = np.asarray(args.worker_gradients, dtype=float)
    ky = np.asarray(args.ky, dtype=float)
    runtime_cfg, _ = load_runtime_from_toml(
        REPO_ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    base_cfg = replace(
        runtime_cfg,
        grid=replace(
            runtime_cfg.grid,
            Nx=int(args.nx),
            Ny=int(args.ny),
            Nz=int(args.nz),
            ntheta=int(args.nz),
            nperiod=1,
            y0=10.0,
        )
    )
    tmin = float(args.fit_start_fraction) * float(args.dt) * int(args.steps)
    tmax = float(args.fit_end_fraction) * float(args.dt) * int(args.steps)
    samples: list[float] = []
    last_members: list[dict[str, Any]] = []
    for repeat_index in range(int(args.warmups) + int(args.repeats)):
        start = time.perf_counter()
        members: list[dict[str, Any]] = []
        for gradient in gradients:
            cfg = replace(
                base_cfg,
                species=(replace(base_cfg.species[0], tprim=float(gradient)),),
            )
            result = run_runtime_scan(
                cfg,
                ky,
                Nl=int(args.nl),
                Nm=int(args.nm),
                dt=float(args.dt),
                steps=int(args.steps),
                solver="time",
                method=str(args.method),
                sample_stride=int(args.sample_stride),
                fit_signal="phi",
                auto_window=False,
                tmin=tmin,
                tmax=tmax,
                min_points=int(args.min_points),
                batch_ky=False,
                require_positive=False,
            )
            ql = _quasilinear_reduced_observables(result.ky, result.gamma, result.omega)
            members.append(
                {
                    "tprim": float(gradient),
                    "ky": np.asarray(result.ky, dtype=float).tolist(),
                    "gamma": np.asarray(result.gamma, dtype=float).tolist(),
                    "omega": np.asarray(result.omega, dtype=float).tolist(),
                    **ql,
                }
            )
        elapsed = time.perf_counter() - start
        last_members = members
        if repeat_index >= int(args.warmups):
            samples.append(float(elapsed))
    heat = np.asarray(
        [member["heat_flux_proxy"] for member in last_members], dtype=float
    )
    payload = {
        "gradients": gradients.tolist(),
        "members": last_members,
        "ensemble_mean_heat_flux_proxy": float(np.mean(heat)),
        "ensemble_std_heat_flux_proxy": float(np.std(heat, ddof=0)),
        "samples_s": samples,
        "stats_s": _time_stats(samples),
    }
    Path(args.worker_out).write_text(
        json.dumps(_json_clean(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _quasilinear_worker_command(
    args: argparse.Namespace, *, gradients: np.ndarray, out_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "quasilinear-uq",
        "--worker",
        "--worker-out",
        str(out_path),
        "--worker-gradients",
        ",".join(f"{float(value):.16g}" for value in gradients),
        "--ky",
        ",".join(f"{float(value):.16g}" for value in args.ky),
        "--nx",
        str(int(args.nx)),
        "--ny",
        str(int(args.ny)),
        "--nz",
        str(int(args.nz)),
        "--nl",
        str(int(args.nl)),
        "--nm",
        str(int(args.nm)),
        "--dt",
        str(float(args.dt)),
        "--steps",
        str(int(args.steps)),
        "--method",
        str(args.method),
        "--sample-stride",
        str(int(args.sample_stride)),
        "--fit-start-fraction",
        str(float(args.fit_start_fraction)),
        "--fit-end-fraction",
        str(float(args.fit_end_fraction)),
        "--min-points",
        str(int(args.min_points)),
        "--warmups",
        str(int(args.warmups)),
        "--repeats",
        str(int(args.repeats)),
    ]


def _run_quasilinear_device_count(
    args: argparse.Namespace,
    *,
    gradients: np.ndarray,
    tmp: Path,
    device_count: int,
) -> dict[str, Any]:
    chunks = _split_values(gradients, int(device_count))
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    start_wall = time.perf_counter()
    for worker_index, chunk in enumerate(chunks):
        out_path = tmp / f"{args.backend}_{device_count}_worker_{worker_index}.json"
        env = _worker_env(
            os.environ, backend=str(args.backend), worker_index=worker_index
        )
        proc = subprocess.Popen(
            _quasilinear_worker_command(args, gradients=chunk, out_path=out_path),
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((worker_index, out_path, proc))

    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for worker_index, out_path, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=float(args.timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            errors.append(
                f"worker {worker_index} timed out after {args.timeout_s}s\n{stderr[-2000:]}"
            )
            continue
        if proc.returncode != 0:
            errors.append(
                f"worker {worker_index} failed with code {proc.returncode}\n{stderr[-4000:]}\n{stdout[-1000:]}"
            )
            continue
        payloads.append(json.loads(out_path.read_text(encoding="utf-8")))
    wall_s = time.perf_counter() - start_wall
    if errors:
        return {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "timed_wall_s": math.nan,
            "wall_s": float(wall_s),
            "members": [],
            "worker_stats": payloads,
            "error": "\n".join(errors),
        }

    members = [member for payload in payloads for member in payload["members"]]
    members = sorted(members, key=lambda item: float(item["tprim"]))
    heat = np.asarray([member["heat_flux_proxy"] for member in members], dtype=float)
    timed_wall_s = max(float(payload["stats_s"]["median"]) for payload in payloads)
    return _json_clean(
        {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "timed_wall_s": float(timed_wall_s),
            "wall_s": float(wall_s),
            "ensemble_mean_heat_flux_proxy": float(np.mean(heat)),
            "ensemble_std_heat_flux_proxy": float(np.std(heat, ddof=0)),
            "members": members,
            "worker_stats": payloads,
            "error": None,
        }
    )


def _quasilinear_identity_metrics(
    reference: dict[str, Any],
    row: dict[str, Any],
    *,
    value_rtol: float,
    value_atol: float,
) -> dict[str, Any]:
    if row.get("error") is not None:
        return {
            "max_heat_flux_proxy_rel_error": math.nan,
            "max_heat_flux_proxy_abs_error": math.nan,
            "max_gamma_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    ref_members = {
        float(member["tprim"]): member for member in reference["members"]
    }
    row_members = {float(member["tprim"]): member for member in row["members"]}
    if set(ref_members) != set(row_members):
        return {
            "max_heat_flux_proxy_rel_error": math.nan,
            "max_heat_flux_proxy_abs_error": math.nan,
            "max_gamma_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    heat_abs: list[float] = []
    heat_rel: list[float] = []
    gamma_abs: list[float] = []
    for key in sorted(ref_members):
        ref = ref_members[key]
        cur = row_members[key]
        ref_heat = float(ref["heat_flux_proxy"])
        cur_heat = float(cur["heat_flux_proxy"])
        heat_abs.append(abs(cur_heat - ref_heat))
        heat_rel.append(abs(cur_heat - ref_heat) / max(abs(ref_heat), 1.0e-12))
        gamma_abs.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(cur["gamma"], dtype=float)
                        - np.asarray(ref["gamma"], dtype=float)
                    )
                )
            )
        )
    max_heat_abs = float(max(heat_abs)) if heat_abs else 0.0
    max_heat_rel = float(max(heat_rel)) if heat_rel else 0.0
    max_gamma_abs = float(max(gamma_abs)) if gamma_abs else 0.0
    return {
        "max_heat_flux_proxy_abs_error": max_heat_abs,
        "max_heat_flux_proxy_rel_error": max_heat_rel,
        "max_gamma_abs_error": max_gamma_abs,
        "identity_gate_pass": bool(
            max_heat_abs <= value_atol
            and max_heat_rel <= value_rtol
            and max_gamma_abs <= value_atol
        ),
    }


def run_quasilinear_uq_sweep(args: argparse.Namespace) -> dict[str, Any]:
    gradients = np.asarray(args.gradients, dtype=float)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gkx-ql-uq-scaling-") as tmp_name:
        tmp = Path(tmp_name)
        for device_count in args.devices:
            rows.append(
                _run_quasilinear_device_count(
                    args, gradients=gradients, tmp=tmp, device_count=int(device_count)
                )
            )
    reference = next(
        (
            row
            for row in rows
            if int(row["requested_devices"]) == 1 and row.get("error") is None
        ),
        None,
    )
    baseline = float(reference["timed_wall_s"]) if reference is not None else math.nan
    for row in rows:
        metrics = (
            _quasilinear_identity_metrics(
                reference,
                row,
                value_rtol=float(args.value_rtol),
                value_atol=float(args.value_atol),
            )
            if reference is not None
            else {
                "max_heat_flux_proxy_rel_error": math.nan,
                "max_heat_flux_proxy_abs_error": math.nan,
                "max_gamma_abs_error": math.nan,
                "identity_gate_pass": False,
            }
        )
        row.update(metrics)
        current = (
            float(row["timed_wall_s"])
            if row.get("timed_wall_s") is not None
            else math.nan
        )
        row["strong_speedup_vs_1_device"] = (
            baseline / current if baseline > 0.0 and current > 0.0 else math.nan
        )
        row["parallel_efficiency"] = (
            row["strong_speedup_vs_1_device"] / float(row["requested_devices"])
            if math.isfinite(float(row["strong_speedup_vs_1_device"]))
            else math.nan
        )
    return _json_clean(
        {
            "kind": "quasilinear_uq_ensemble_scaling",
            "backend": str(args.backend),
            "devices": [int(value) for value in args.devices],
            "gradients": gradients.tolist(),
            "ky": list(map(float, args.ky)),
            "grid": {
                "Nx": int(args.nx),
                "Ny": int(args.ny),
                "Nz": int(args.nz),
                "Nl": int(args.nl),
                "Nm": int(args.nm),
            },
            "time": {
                "dt": float(args.dt),
                "steps": int(args.steps),
                "method": str(args.method),
                "sample_stride": int(args.sample_stride),
                "fit_start_fraction": float(args.fit_start_fraction),
                "fit_end_fraction": float(args.fit_end_fraction),
            },
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "value_rtol": float(args.value_rtol),
            "value_atol": float(args.value_atol),
            "identity_passed": all(bool(row.get("identity_gate_pass")) for row in rows),
            "claim_scope": (
                "solver-backed quasilinear/UQ ensemble scaling artifact using a reduced "
                "mixing-length feature observable from real linear scans; not an absolute "
                "nonlinear heat-flux validation claim"
            ),
            "rows": rows,
        }
    )


def _write_scaling_plot(
    summary: dict[str, Any], out_prefix: Path, *, mode: str
) -> dict[str, str]:
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
    if mode == "independent-ky":
        fieldnames = [
            "requested_devices",
            "actual_workers",
            "timed_wall_s",
            "wall_s",
            "strong_speedup_vs_1_device",
            "parallel_efficiency",
            "max_gamma_abs_error",
            "max_gamma_rel_error",
            "max_omega_abs_error",
            "identity_gate_pass",
            "error",
        ]
    else:
        fieldnames = [
            "requested_devices",
            "actual_workers",
            "timed_wall_s",
            "wall_s",
            "strong_speedup_vs_1_device",
            "parallel_efficiency",
            "ensemble_mean_heat_flux_proxy",
            "ensemble_std_heat_flux_proxy",
            "max_heat_flux_proxy_abs_error",
            "max_heat_flux_proxy_rel_error",
            "max_gamma_abs_error",
            "identity_gate_pass",
            "error",
        ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    x = np.asarray([int(row["requested_devices"]) for row in rows], dtype=float)
    speedup = np.asarray(
        [float(row["strong_speedup_vs_1_device"]) for row in rows], dtype=float
    )
    elapsed = np.asarray([float(row["timed_wall_s"]) for row in rows], dtype=float)
    axes[0].plot(x, speedup, "o-", lw=2.2, color="#276b8e", label="measured")
    axes[0].plot(x, x, ":", lw=1.3, color="0.35", label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title(f"{str(summary['backend']).upper()} {mode} scaling")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogy(x, np.maximum(elapsed, 1.0e-16), "s-", lw=2.0, color="#b45f06")
    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median worker wall time [s]")
    axes[1].set_title("Timed throughput")
    if mode == "independent-ky":
        gamma_rel = np.asarray(
            [float(row["max_gamma_rel_error"]) for row in rows], dtype=float
        )
        omega_abs = np.asarray(
            [float(row["max_omega_abs_error"]) for row in rows], dtype=float
        )
        axes[2].semilogy(
            x, np.maximum(gamma_rel, 1.0e-16), "o-", lw=2.0, label=r"$\gamma$ rel."
        )
        axes[2].semilogy(
            x + 0.03,
            np.maximum(omega_abs, 1.0e-16),
            "s-",
            lw=2.0,
            label=r"$\omega$ abs.",
        )
        axes[2].axhline(float(summary["gamma_rtol"]), color="#276b8e", ls=":", lw=1.2)
        axes[2].axhline(float(summary["omega_atol"]), color="#b45f06", ls=":", lw=1.2)
    else:
        heat_rel = np.asarray(
            [float(row["max_heat_flux_proxy_rel_error"]) for row in rows], dtype=float
        )
        gamma_abs = np.asarray(
            [float(row["max_gamma_abs_error"]) for row in rows], dtype=float
        )
        axes[2].semilogy(
            x, np.maximum(heat_rel, 1.0e-16), "o-", lw=2.0, label="QL proxy rel."
        )
        axes[2].semilogy(
            x + 0.04,
            np.maximum(gamma_abs, 1.0e-16),
            "s--",
            lw=1.8,
            label=r"$\gamma$ abs.",
        )
        axes[2].axhline(float(summary["value_rtol"]), ls=":", color="#276b8e", lw=1.2)
        axes[2].axhline(float(summary["value_atol"]), ls=":", color="#b45f06", lw=1.2)
    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("Serial identity")
    axes[2].legend(frameon=False, fontsize=8)
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


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    return _write_scaling_plot(summary, out_prefix, mode="independent-ky")


def write_quasilinear_artifacts(
    summary: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    return _write_scaling_plot(summary, out_prefix, mode="quasilinear-uq")


def build_independent_ky_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile independent-ky scan scaling.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-ky", type=_parse_float_list, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_INDEPENDENT_KY_PREFIX
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2, 4])
    parser.add_argument(
        "--ky",
        type=_parse_float_list,
        default=[
            0.08,
            0.12,
            0.16,
            0.20,
            0.24,
            0.28,
            0.32,
            0.36,
            0.40,
            0.44,
            0.48,
            0.52,
        ],
    )
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--fit-start-fraction", type=float, default=0.35)
    parser.add_argument("--fit-end-fraction", type=float, default=0.9)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-7)
    parser.add_argument("--omega-atol", type=float, default=1.0e-7)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser


def build_quasilinear_uq_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile quasilinear/UQ ensemble scaling."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-gradients",
        type=_parse_float_list,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_QUASILINEAR_UQ_PREFIX
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2, 4])
    parser.add_argument(
        "--gradients",
        type=_parse_float_list,
        default=[2.20, 2.40, 2.60, 2.80, 3.00, 3.20],
    )
    parser.add_argument(
        "--ky", type=_parse_float_list, default=[0.10, 0.20, 0.30, 0.40, 0.50]
    )
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--nl", type=int, default=3)
    parser.add_argument("--nm", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--fit-start-fraction", type=float, default=0.5)
    parser.add_argument("--fit-end-fraction", type=float, default=0.95)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--value-rtol", type=float, default=1.0e-7)
    parser.add_argument("--value-atol", type=float, default=1.0e-7)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser


# Backward-compatible parser binding for tests that use this module as independent_ky.
def build_parser() -> argparse.ArgumentParser:
    return build_independent_ky_parser()


def main_independent_ky(argv: list[str] | None = None) -> int:
    args = build_independent_ky_parser().parse_args(argv)
    if bool(args.worker):
        if args.worker_out is None or args.worker_ky is None:
            raise ValueError("--worker requires --worker-out and --worker-ky")
        _run_solver_chunk(args)
        return 0
    summary = run_independent_ky_sweep(args)
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


def main_quasilinear_uq(argv: list[str] | None = None) -> int:
    args = build_quasilinear_uq_parser().parse_args(argv)
    if bool(args.worker):
        if args.worker_out is None or args.worker_gradients is None:
            raise ValueError("--worker requires --worker-out and --worker-gradients")
        _run_ensemble_chunk(args)
        return 0
    summary = run_quasilinear_uq_sweep(args)
    paths = write_quasilinear_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


SUBCOMMANDS = {
    "independent-ky": main_independent_ky,
    "quasilinear-uq": main_quasilinear_uq,
}


def build_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(SUBCOMMANDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_dispatch_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    try:
        handler = SUBCOMMANDS[command]
    except KeyError:
        build_dispatch_parser().parse_args([command])
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
