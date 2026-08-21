#!/usr/bin/env python3
"""Generate a solver-backed ky-batch parallelization identity gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

from gkx.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from gkx.artifacts.plotting import set_plot_style
from gkx.runtime import RuntimeLinearScanResult, run_runtime_scan
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KY_SCAN_PREFIX = REPO_ROOT / "docs" / "_static" / "parallel_ky_scan_gate"


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


def _summary_rows(summary: Mapping[str, object]) -> list[dict[str, Any]]:
    raw_rows = summary.get("rows", [])
    if not isinstance(raw_rows, list):
        raise TypeError("parallel identity artifact summary must contain list rows")
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(raw_rows):
        raise TypeError("parallel identity artifact rows must be dictionaries")
    if not rows:
        raise ValueError("parallel identity artifact rows cannot be empty")
    return rows


def _summary_float(summary: Mapping[str, object], key: str) -> float:
    value = summary[key]
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"summary field {key!r} must be numeric")
    return float(value)


def _timed_cyclone_scan(
    ky_values: np.ndarray,
    *,
    ky_batch: int,
    cfg: RuntimeConfig,
    steps: int,
    dt: float,
    nlaguerre: int,
    nhermite: int,
) -> tuple[RuntimeLinearScanResult, float]:
    start = time.perf_counter()
    result = run_runtime_scan(
        cfg,
        ky_values,
        Nl=nlaguerre,
        Nm=nhermite,
        dt=dt,
        steps=steps,
        solver="time",
        method="rk2",
        sample_stride=2,
        fit_signal="phi",
        auto_window=False,
        tmin=0.2 * dt * steps,
        tmax=0.9 * dt * steps,
        min_points=4,
        batch_ky=ky_batch > 1,
        require_positive=False,
    )
    return result, time.perf_counter() - start


def build_parallel_ky_scan_gate(
    *,
    ky_values: np.ndarray,
    serial_batch: int,
    parallel_batch: int,
    gamma_rtol: float,
    omega_atol: float,
    steps: int,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    nlaguerre: int,
    nhermite: int,
) -> dict[str, object]:
    """Run serial and batched Cyclone scans and build identity-gate metadata."""

    base_cfg, _ = load_runtime_from_toml(
        REPO_ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    cfg = replace(
        base_cfg,
        grid=replace(
            base_cfg.grid, Nx=nx, Ny=ny, Nz=nz, ntheta=nz, nperiod=1, y0=10.0
        ),
    )
    serial, serial_elapsed = _timed_cyclone_scan(
        ky_values,
        ky_batch=serial_batch,
        cfg=cfg,
        steps=steps,
        dt=dt,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    batched, batch_elapsed = _timed_cyclone_scan(
        ky_values,
        ky_batch=parallel_batch,
        cfg=cfg,
        steps=steps,
        dt=dt,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    if not np.allclose(serial.ky, batched.ky, rtol=0.0, atol=0.0):
        raise ValueError("serial and batched scans returned different ky ordering")

    gamma_denom = np.maximum(np.abs(serial.gamma), 1.0e-12)
    gamma_rel = np.abs(batched.gamma - serial.gamma) / gamma_denom
    omega_abs = np.abs(batched.omega - serial.omega)
    max_gamma_rel = float(np.max(gamma_rel))
    max_omega_abs = float(np.max(omega_abs))
    identity_passed = bool(max_gamma_rel <= gamma_rtol and max_omega_abs <= omega_atol)
    speedup = float(serial_elapsed / batch_elapsed) if batch_elapsed > 0.0 else math.inf

    rows = []
    for idx, ky in enumerate(serial.ky):
        rows.append(
            {
                "ky": float(ky),
                "serial_gamma": float(serial.gamma[idx]),
                "batched_gamma": float(batched.gamma[idx]),
                "gamma_rel_error": float(gamma_rel[idx]),
                "serial_omega": float(serial.omega[idx]),
                "batched_omega": float(batched.omega[idx]),
                "omega_abs_error": float(omega_abs[idx]),
            }
        )

    return _json_clean(
        {
            "case": "Cyclone ITG linear ky-batch scan",
            "source": "GKX real linear solver; serial ky_batch=1 vs fixed-shape ky_batch>1",
            "physics_anchor": "Cyclone Base Case ITG linear scan",
            "ky_values": ky_values.tolist(),
            "grid": {"Nx": nx, "Ny": ny, "Nz": nz, "Nl": nlaguerre, "Nm": nhermite},
            "time": {"dt": dt, "steps": steps, "sample_stride": 2},
            "serial_batch": serial_batch,
            "parallel_batch": parallel_batch,
            "serial_elapsed_s": serial_elapsed,
            "batched_elapsed_s": batch_elapsed,
            "observed_speedup": speedup,
            "max_gamma_rel_error": max_gamma_rel,
            "max_omega_abs_error": max_omega_abs,
            "gamma_rtol": gamma_rtol,
            "omega_atol": omega_atol,
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "This is a numerical-identity gate for independent ky batching. "
                "Speedup is reported for engineering tracking but is not the acceptance gate."
            ),
        }
    )


def write_parallel_ky_scan_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _summary_rows(summary)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_gamma = np.asarray([row["serial_gamma"] for row in rows], dtype=float)
    batched_gamma = np.asarray([row["batched_gamma"] for row in rows], dtype=float)
    serial_omega = np.asarray([row["serial_omega"] for row in rows], dtype=float)
    batched_omega = np.asarray([row["batched_omega"] for row in rows], dtype=float)
    gamma_rel = np.asarray([row["gamma_rel_error"] for row in rows], dtype=float)
    omega_abs = np.asarray([row["omega_abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)

    axes[0].plot(ky, serial_gamma, "o-", lw=2.0, color="#1b6ca8", label="serial gamma")
    axes[0].plot(
        ky, batched_gamma, "s--", lw=1.8, color="#c03a2b", label="batched gamma"
    )
    axes[0].plot(ky, serial_omega, "o-", lw=2.0, color="#5f8f2d", label="serial omega")
    axes[0].plot(
        ky, batched_omega, "s--", lw=1.8, color="#7d3c98", label="batched omega"
    )
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel(r"$\gamma, \omega$")
    axes[0].set_title("Cyclone linear scan")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].semilogy(
        ky,
        np.maximum(gamma_rel, 1.0e-16),
        "o-",
        lw=2.0,
        color="#1b6ca8",
        label=r"$\gamma$ rel.",
    )
    axes[1].semilogy(
        ky,
        np.maximum(omega_abs, 1.0e-16),
        "s-",
        lw=2.0,
        color="#c03a2b",
        label=r"$\omega$ abs.",
    )
    axes[1].axhline(_summary_float(summary, "gamma_rtol"), color="#1b6ca8", ls=":", lw=1.2)
    axes[1].axhline(_summary_float(summary, "omega_atol"), color="#c03a2b", ls=":", lw=1.2)
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel("identity error")
    axes[1].set_title("Numerical identity gate")
    axes[1].legend(frameon=False, fontsize=8)

    serial_elapsed = _summary_float(summary, "serial_elapsed_s")
    batch_elapsed = _summary_float(summary, "batched_elapsed_s")
    axes[2].bar(
        ["serial", "batched"],
        [serial_elapsed, batch_elapsed],
        color=["#1b6ca8", "#c03a2b"],
        alpha=0.82,
    )
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_ylabel("wall time [s]")
    axes[2].set_title("Batch timing")
    axes[2].text(
        0.04,
        0.95,
        f"speedup = {_summary_float(summary, 'observed_speedup'):.2f}x\nidentity gate: {status}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)}


def _build_ky_scan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a solver-backed ky-batch parallelization identity gate.")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_KY_SCAN_PREFIX)
    parser.add_argument("--ky", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--serial-batch", type=int, default=1)
    parser.add_argument("--parallel-batch", type=int, default=2)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-8)
    parser.add_argument("--omega-atol", type=float, default=1.0e-8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--nlaguerre", type=int, default=3)
    parser.add_argument("--nhermite", type=int, default=4)
    return parser


def _main_ky_scan(argv: list[str] | None = None) -> int:
    args = _build_ky_scan_parser().parse_args(argv)
    summary = build_parallel_ky_scan_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        serial_batch=args.serial_batch,
        parallel_batch=args.parallel_batch,
        gamma_rtol=args.gamma_rtol,
        omega_atol=args.omega_atol,
        steps=args.steps,
        dt=args.dt,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nlaguerre=args.nlaguerre,
        nhermite=args.nhermite,
    )
    paths = write_parallel_ky_scan_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0 if bool(summary["identity_passed"]) else 1


DEFAULT_LOGICAL_CPU_PREFIX = REPO_ROOT / "docs" / "_static" / "logical_cpu_parallel_scan_gate"
def _block_until_ready(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timed_scan_model(
    ky_values: np.ndarray,
    *,
    batch_size: int,
    devices: list[Any],
) -> tuple[dict[str, np.ndarray], float]:
    """Evaluate a tiny JAX-native linear-scan model through ``batch_map``."""

    import jax
    import jax.numpy as jnp

    from gkx.parallel import batch_map

    ky_jax = jnp.asarray(
        ky_values,
        dtype=jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32,
    )

    def model(ky):
        kperp2 = 0.08 + ky**2
        drive = 0.52 * ky * jnp.exp(-0.75 * ky**2)
        damping = 0.045 + 0.02 * ky**2
        operator = jnp.asarray(
            [
                [drive - damping - 0.18j * ky, 0.08 + 0.02j * ky],
                [0.04 - 0.01j * ky, -0.12 - 0.04 * kperp2 + 0.06j * ky],
            ],
            dtype=jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64,
        )
        eigvals = jnp.linalg.eigvals(operator)
        mode = eigvals[jnp.argmax(jnp.real(eigvals))]
        gamma = jnp.real(mode)
        omega = jnp.imag(mode)
        return {
            "gamma": gamma,
            "omega": omega,
            "kperp2": kperp2,
            "ql_proxy": jnp.maximum(gamma, 0.0) / kperp2,
        }

    start = time.perf_counter()
    output = batch_map(model, ky_jax, batch_size=batch_size, devices=devices)
    _block_until_ready(output)
    elapsed = time.perf_counter() - start
    arrays = jax.tree_util.tree_map(lambda leaf: np.asarray(leaf, dtype=float), output)
    return dict(arrays), elapsed


def _select_devices(requested_devices: int) -> list[Any]:
    import jax

    devices = list(jax.devices("cpu"))
    if not devices:
        raise RuntimeError("No CPU JAX devices are available")
    return devices[: max(1, min(int(requested_devices), len(devices)))]


def build_logical_cpu_parallel_scan_gate(
    *,
    ky_values: np.ndarray,
    serial_batch: int,
    parallel_batch: int,
    requested_devices: int,
    gamma_rtol: float,
    omega_atol: float,
    ql_rtol: float,
) -> dict[str, object]:
    """Compare serial and logical-CPU device-batched independent scans."""

    from gkx.workflows.runtime.config import RuntimeParallelConfig

    devices = _select_devices(requested_devices)
    serial_config = RuntimeParallelConfig(
        strategy="serial", axis="ky", batch_size=serial_batch, num_devices=1
    )
    parallel_config = RuntimeParallelConfig(
        strategy="device_batch",
        axis="ky",
        batch_size=parallel_batch,
        num_devices=len(devices),
        strict_identity=True,
    )
    serial, serial_elapsed = _timed_scan_model(
        ky_values, batch_size=serial_batch, devices=devices[:1]
    )
    batched, batch_elapsed = _timed_scan_model(
        ky_values, batch_size=parallel_batch, devices=devices
    )

    gamma_denom = np.maximum(np.abs(serial["gamma"]), 1.0e-12)
    ql_denom = np.maximum(np.abs(serial["ql_proxy"]), 1.0e-12)
    gamma_rel = np.abs(batched["gamma"] - serial["gamma"]) / gamma_denom
    omega_abs = np.abs(batched["omega"] - serial["omega"])
    ql_rel = np.abs(batched["ql_proxy"] - serial["ql_proxy"]) / ql_denom

    max_gamma_rel = float(np.max(gamma_rel))
    max_omega_abs = float(np.max(omega_abs))
    max_ql_rel = float(np.max(ql_rel))
    identity_passed = bool(
        max_gamma_rel <= gamma_rtol
        and max_omega_abs <= omega_atol
        and max_ql_rel <= ql_rtol
    )
    speedup = float(serial_elapsed / batch_elapsed) if batch_elapsed > 0.0 else math.inf

    rows = []
    for idx, ky in enumerate(np.asarray(ky_values, dtype=float)):
        rows.append(
            {
                "ky": float(ky),
                "serial_gamma": float(serial["gamma"][idx]),
                "batched_gamma": float(batched["gamma"][idx]),
                "gamma_rel_error": float(gamma_rel[idx]),
                "serial_omega": float(serial["omega"][idx]),
                "batched_omega": float(batched["omega"][idx]),
                "omega_abs_error": float(omega_abs[idx]),
                "serial_ql_proxy": float(serial["ql_proxy"][idx]),
                "batched_ql_proxy": float(batched["ql_proxy"][idx]),
                "ql_rel_error": float(ql_rel[idx]),
            }
        )

    return _json_clean(
        {
            "case": "Logical-CPU independent ky scan",
            "source": "JAX-native tiny non-Hermitian linear scan model through gkx.batch_map",
            "claim_scope": "parallel interface and numerical identity gate, not a gyrokinetic physics validation",
            "ky_values": np.asarray(ky_values, dtype=float),
            "serial_parallel_config": serial_config.to_dict(),
            "device_parallel_config": parallel_config.to_dict(),
            "requested_devices": int(requested_devices),
            "actual_devices": len(devices),
            "serial_elapsed_s": serial_elapsed,
            "batched_elapsed_s": batch_elapsed,
            "observed_speedup": speedup,
            "max_gamma_rel_error": max_gamma_rel,
            "max_omega_abs_error": max_omega_abs,
            "max_ql_rel_error": max_ql_rel,
            "gamma_rtol": gamma_rtol,
            "omega_atol": omega_atol,
            "ql_rtol": ql_rtol,
            "identity_passed": identity_passed,
            "rows": rows,
        }
    )


def write_logical_cpu_parallel_scan_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
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
    rows = _summary_rows(summary)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_gamma = np.asarray([row["serial_gamma"] for row in rows], dtype=float)
    batched_gamma = np.asarray([row["batched_gamma"] for row in rows], dtype=float)
    serial_ql = np.asarray([row["serial_ql_proxy"] for row in rows], dtype=float)
    batched_ql = np.asarray([row["batched_ql_proxy"] for row in rows], dtype=float)
    gamma_rel = np.asarray([row["gamma_rel_error"] for row in rows], dtype=float)
    ql_rel = np.asarray([row["ql_rel_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    axes[0].plot(ky, serial_gamma, "o-", lw=2.0, label="serial gamma")
    axes[0].plot(ky, batched_gamma, "s--", lw=1.8, label="parallel gamma")
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel(r"$\gamma$")
    axes[0].set_title("Linear-scan observable")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(ky, serial_ql, "o-", lw=2.0, label="serial QL proxy")
    axes[1].plot(ky, batched_ql, "s--", lw=1.8, label="parallel QL proxy")
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel(r"$\max(\gamma,0)/k_\perp^2$")
    axes[1].set_title("Structured pytree output")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].semilogy(
        ky, np.maximum(gamma_rel, 1.0e-16), "o-", lw=2.0, label=r"$\gamma$ rel."
    )
    axes[2].semilogy(ky, np.maximum(ql_rel, 1.0e-16), "s-", lw=2.0, label="QL rel.")
    axes[2].axhline(_summary_float(summary, "gamma_rtol"), ls=":", lw=1.1)
    axes[2].axhline(_summary_float(summary, "ql_rtol"), ls=":", lw=1.1)
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_xlabel(r"$k_y \rho_i$")
    axes[2].set_ylabel("identity error")
    axes[2].set_title(f"Gate {status}, {_summary_float(summary, 'observed_speedup'):.2f}x")
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


def _build_logical_cpu_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a logical-CPU independent-scan parallelization identity gate.")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_LOGICAL_CPU_PREFIX)
    parser.add_argument(
        "--ky", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    parser.add_argument("--serial-batch", type=int, default=1)
    parser.add_argument("--parallel-batch", type=int, default=2)
    parser.add_argument("--logical-devices", type=int, default=4)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-6)
    parser.add_argument("--omega-atol", type=float, default=1.0e-6)
    parser.add_argument("--ql-rtol", type=float, default=1.0e-6)
    return parser


def _configure_logical_cpu_devices(count: int) -> None:
    if int(count) <= 1:
        return
    flag = f"--xla_force_host_platform_device_count={int(count)}"
    current = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


def _main_logical_cpu(argv: list[str] | None = None) -> int:
    args = _build_logical_cpu_parser().parse_args(argv)
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_logical_cpu_parallel_scan_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        serial_batch=args.serial_batch,
        parallel_batch=args.parallel_batch,
        requested_devices=args.logical_devices,
        gamma_rtol=args.gamma_rtol,
        omega_atol=args.omega_atol,
        ql_rtol=args.ql_rtol,
    )
    paths = write_logical_cpu_parallel_scan_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0 if bool(summary["identity_passed"]) else 1




DEFAULT_QUASILINEAR_RUNTIME_PREFIX = REPO_ROOT / "docs" / "_static" / "quasilinear_runtime_parallel_gate"
def _default_runtime_config(*, nx: int, ny: int, nz: int) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=nx, Ny=ny, Nz=nz, ntheta=nz, nperiod=1, y0=10.0, boundary="linked"
        ),
        time=TimeConfig(
            t_max=0.2, dt=0.02, method="rk2", use_diffrax=False, sample_stride=1
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
        species=(RuntimeSpeciesConfig(name="ion"),),
        quasilinear=RuntimeQuasilinearConfig(
            enabled=True,
            mode="saturated",
            saturation_rule="mixing_length",
            amplitude_normalization="phi_rms",
            csat=1.0,
        ),
    )


def _timed_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: np.ndarray,
    *,
    workers: int,
    solver: str,
    nlaguerre: int,
    nhermite: int,
) -> tuple[RuntimeLinearScanResult, float]:
    start = time.perf_counter()
    result = run_runtime_scan(
        cfg,
        ky_values=ky_values.tolist(),
        Nl=nlaguerre,
        Nm=nhermite,
        solver=solver,
        workers=workers,
        parallel_executor="thread",
        show_progress=False,
    )
    return result, time.perf_counter() - start


def _ql_column(payloads: tuple[dict[str, Any], ...], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in payloads], dtype=float)


def build_quasilinear_runtime_parallel_gate(
    *,
    ky_values: np.ndarray,
    workers: int,
    rtol: float,
    atol: float,
    solver: str,
    nx: int,
    ny: int,
    nz: int,
    nlaguerre: int,
    nhermite: int,
) -> dict[str, Any]:
    """Compare serial and worker-parallel quasilinear runtime scans."""

    cfg = _default_runtime_config(nx=nx, ny=ny, nz=nz)
    serial, serial_elapsed = _timed_runtime_scan(
        cfg,
        ky_values,
        workers=1,
        solver=solver,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    parallel, parallel_elapsed = _timed_runtime_scan(
        cfg,
        ky_values,
        workers=workers,
        solver=solver,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    if serial.quasilinear is None or parallel.quasilinear is None:
        raise RuntimeError("quasilinear runtime scan did not return spectrum payloads")
    if len(serial.quasilinear) != len(parallel.quasilinear):
        raise RuntimeError(
            "serial and worker scans returned different quasilinear spectrum lengths"
        )
    if not np.allclose(serial.ky, parallel.ky, rtol=0.0, atol=0.0):
        raise RuntimeError("serial and worker scans returned different ky ordering")

    columns = (
        "gamma",
        "omega",
        "kperp_eff2",
        "heat_flux_weight_total",
        "particle_flux_weight_total",
        "amplitude2",
        "saturated_heat_flux_total",
        "saturated_particle_flux_total",
    )
    rows: list[dict[str, Any]] = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    for idx, ky in enumerate(serial.ky):
        row: dict[str, Any] = {"ky": float(ky)}
        for column in columns:
            serial_values = _ql_column(serial.quasilinear, column)
            parallel_values = _ql_column(parallel.quasilinear, column)
            serial_value = float(serial_values[idx])
            parallel_value = float(parallel_values[idx])
            abs_error = abs(parallel_value - serial_value)
            denom = max(abs(serial_value), float(atol))
            rel_error = abs_error / denom
            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)
            row[f"serial_{column}"] = serial_value
            row[f"parallel_{column}"] = parallel_value
            row[f"{column}_abs_error"] = abs_error
            row[f"{column}_rel_error"] = rel_error
        rows.append(row)

    identity_passed = bool(max_abs_error <= float(atol) or max_rel_error <= float(rtol))
    speedup = (
        float(serial_elapsed / parallel_elapsed) if parallel_elapsed > 0.0 else math.inf
    )
    return _json_clean(
        {
            "kind": "quasilinear_runtime_parallel_gate",
            "case": "Cyclone ITG runtime quasilinear scan",
            "source": "run_runtime_scan serial workers=1 vs independent ky workers>1",
            "claim_level": "quasilinear_state_extraction_identity_not_speedup_claim",
            "ky_values": ky_values.tolist(),
            "grid": {"Nx": nx, "Ny": ny, "Nz": nz, "Nl": nlaguerre, "Nm": nhermite},
            "solver": solver,
            "workers": int(workers),
            "serial_elapsed_s": serial_elapsed,
            "parallel_elapsed_s": parallel_elapsed,
            "observed_speedup": speedup,
            "rtol": float(rtol),
            "atol": float(atol),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "identity_passed": identity_passed,
            "serial_parallel_metadata": parallel.parallel,
            "rows": rows,
            "notes": (
                "Acceptance is exact ordered quasilinear-spectrum identity. "
                "Timing is reported for tracking only and should not be used as a production speedup claim."
            ),
        }
    )


def write_quasilinear_runtime_parallel_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for the quasilinear identity gate."""

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(
        json.dumps(_json_clean(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(summary["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_heat = np.asarray(
        [row["serial_heat_flux_weight_total"] for row in rows], dtype=float
    )
    parallel_heat = np.asarray(
        [row["parallel_heat_flux_weight_total"] for row in rows], dtype=float
    )
    serial_sat = np.asarray(
        [row["serial_saturated_heat_flux_total"] for row in rows], dtype=float
    )
    parallel_sat = np.asarray(
        [row["parallel_saturated_heat_flux_total"] for row in rows], dtype=float
    )
    heat_err = np.asarray(
        [row["heat_flux_weight_total_abs_error"] for row in rows], dtype=float
    )
    sat_err = np.asarray(
        [row["saturated_heat_flux_total_abs_error"] for row in rows], dtype=float
    )

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    axes[0].plot(ky, serial_heat, "o-", lw=2.0, color="#1b6f8f", label="serial weight")
    axes[0].plot(
        ky, parallel_heat, "s--", lw=1.8, color="#b55a30", label="worker weight"
    )
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel("linear heat-flux weight")
    axes[0].set_title("QL state extraction")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        ky, serial_sat, "o-", lw=2.0, color="#386641", label="serial saturated"
    )
    axes[1].plot(
        ky, parallel_sat, "s--", lw=1.8, color="#92400e", label="worker saturated"
    )
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel("saturated heat-flux estimate")
    axes[1].set_title("Spectrum ordering")
    axes[1].legend(frameon=False, fontsize=8)

    floor = 1.0e-18
    axes[2].semilogy(
        ky,
        np.maximum(heat_err, floor),
        "o-",
        lw=2.0,
        color="#1b6f8f",
        label="weight abs. error",
    )
    axes[2].semilogy(
        ky,
        np.maximum(sat_err, floor),
        "s-",
        lw=2.0,
        color="#b55a30",
        label="sat. abs. error",
    )
    axes[2].axhline(
        float(summary["atol"]), color="#44403c", ls=":", lw=1.2, label="atol"
    )
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_xlabel(r"$k_y \rho_i$")
    axes[2].set_ylabel("absolute error")
    axes[2].set_title(f"Identity gate: {status}")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Quasilinear runtime scan parallelization identity",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def _build_quasilinear_runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a quasilinear runtime-scan worker identity gate.")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_QUASILINEAR_RUNTIME_PREFIX)
    parser.add_argument("--ky", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument(
        "--solver",
        choices=("krylov", "time", "explicit_time", "auto"),
        default="krylov",
    )
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--nlaguerre", type=int, default=2)
    parser.add_argument("--nhermite", type=int, default=2)
    return parser


def _main_quasilinear_runtime(argv: list[str] | None = None) -> int:
    args = _build_quasilinear_runtime_parser().parse_args(argv)
    summary = build_quasilinear_runtime_parallel_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        workers=args.workers,
        rtol=args.rtol,
        atol=args.atol,
        solver=args.solver,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nlaguerre=args.nlaguerre,
        nhermite=args.nhermite,
    )
    paths = write_quasilinear_runtime_parallel_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0 if bool(summary["identity_passed"]) else 1


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "ky-scan": _main_ky_scan,
        "logical-cpu": _main_logical_cpu,
        "quasilinear-runtime": _main_quasilinear_runtime,
    }
    if not raw or raw[0] in {"-h", "--help"}:
        print("usage: generate_parallel_identity_gate.py {ky-scan,logical-cpu,quasilinear-runtime} [options]")
        print("\nsubcommands:")
        print("  ky-scan              real Cyclone solver ky-batch identity gate")
        print("  logical-cpu          JAX batch_map logical-CPU identity gate")
        print("  quasilinear-runtime  runtime quasilinear worker identity gate")
        return 0
    command = raw.pop(0)
    try:
        runner = commands[command]
    except KeyError as exc:
        raise SystemExit(f"unknown parallel identity gate subcommand: {command}") from exc
    return runner(raw)


if __name__ == "__main__":
    raise SystemExit(main())
