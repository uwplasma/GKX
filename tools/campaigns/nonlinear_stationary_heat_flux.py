"""Measure stationary heat flux after perturbing the temperature-gradient drive.

This campaign complements ``nonlinear_gradient_window.py``.  The latter gives
the exact discrete derivative of a finite trajectory; this one estimates the
long-time response by restarting the same verified saturated state at a chosen
multiplier of ``tprim`` and discarding a configurable transient.  Symmetric
runs can then be combined into a centered finite difference with uncertainty.

The restart goes through ``run_runtime_nonlinear`` and therefore retains the
production CFL controller, state projection, dealiasing, and diagnostics.  The
reported primary mean is trapezoid-weighted in time because adaptive steps make
an unweighted sample mean subtly dependent on the CFL history.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def time_weighted_mean(time_values: np.ndarray, values: np.ndarray) -> float:
    """Return the trapezoid-weighted mean of samples on a nonuniform time grid."""

    time_values = np.asarray(time_values, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_values.size != values.size or time_values.size < 2:
        raise ValueError("time and values must have the same length >= 2")
    span = float(time_values[-1] - time_values[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("time must span a finite positive interval")
    trapezoid = getattr(np, "trapezoid", None)
    integral = (
        trapezoid(values, time_values)
        if trapezoid is not None
        else np.trapz(values, time_values)
    )
    return float(integral / span)


def _late_trace(
    times: np.ndarray, values: np.ndarray, transient_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= transient_fraction < 1.0:
        raise ValueError("transient_fraction must be in [0, 1)")
    cutoff = float(times[0]) + transient_fraction * float(times[-1] - times[0])
    mask = np.asarray(times) >= cutoff
    return np.asarray(times)[mask], np.asarray(values)[mask]


def _json_trace(times: np.ndarray, flux: np.ndarray) -> list[dict[str, float]]:
    return [
        {"t": float(t_value), "heat_flux": float(q_value)}
        for t_value, q_value in zip(times, flux, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
        ),
    )
    parser.add_argument("--saturated-state", type=Path, required=True)
    parser.add_argument("--drive-scale", type=float, required=True)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--t-max", type=float, default=200.0)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not np.isfinite(args.drive_scale) or args.drive_scale <= 0.0:
        raise SystemExit("--drive-scale must be finite and positive")
    if not np.isfinite(args.t_max) or args.t_max <= 0.0:
        raise SystemExit("--t-max must be finite and positive")

    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    from gkx import run_runtime_nonlinear
    from gkx.diagnostics.transport_windows import (
        NonlinearWindowConvergenceConfig,
        nonlinear_window_convergence_report,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    archive = np.load(args.saturated_state)
    initial_state = np.asarray(archive["state"])
    saturated = bool(archive["saturated"])
    if not saturated:
        raise SystemExit(
            "refusing a stationary-response run from a state not marked saturated"
        )

    runtime, raw = load_runtime_from_toml(args.toml)
    grid_override = {
        key: value
        for key, value in (("Nx", args.nx), ("Ny", args.ny), ("Nz", args.nz))
        if value is not None
    }
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )
    n_laguerre = int(raw.get("run", {}).get("Nl", 4))
    n_hermite = int(raw.get("run", {}).get("Nm", 8))
    expected_shape = (
        len(runtime.species),
        n_laguerre,
        n_hermite,
        runtime.grid.Ny,
        runtime.grid.Nx,
        runtime.grid.Nz,
    )
    if initial_state.shape != expected_shape:
        raise SystemExit(
            f"state shape {initial_state.shape} does not match runtime {expected_shape}"
        )

    sample_stride = (
        runtime.time.sample_stride
        if args.sample_stride is None
        else int(args.sample_stride)
    )
    runtime = dataclasses.replace(
        runtime,
        species=tuple(
            dataclasses.replace(species, tprim=species.tprim * args.drive_scale)
            for species in runtime.species
        ),
        time=dataclasses.replace(
            runtime.time,
            t_max=float(args.t_max),
            sample_stride=sample_stride,
            diagnostics_stride=sample_stride,
        ),
    )

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="gkx-stationary-response-") as temp_dir:
        # The production restart reader intentionally uses the portable GX-compatible
        # complex64 format.  A temporary file keeps this campaign on the public
        # runtime path instead of reaching into an integration implementation.
        restart_path = Path(temp_dir) / "restart.bin"
        np.asarray(initial_state, dtype=np.complex64).tofile(restart_path)
        runtime = dataclasses.replace(
            runtime,
            init=dataclasses.replace(
                runtime.init,
                init_file=str(restart_path),
                init_file_scale=1.0,
                init_file_mode="replace",
                init_amp=0.0,
            ),
        )
        result = run_runtime_nonlinear(
            runtime,
            Nl=n_laguerre,
            Nm=n_hermite,
            diagnostics=True,
            return_state=True,
        )
    elapsed = time.time() - started

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise RuntimeError("production run returned no diagnostics")
    times = np.asarray(diagnostics.t, dtype=float)
    flux = np.asarray(diagnostics.heat_flux_t, dtype=float)
    steps_dt = np.asarray(diagnostics.dt_t, dtype=float)
    late_t, late_q = _late_trace(times, flux, float(args.transient_fraction))
    weighted_mean = time_weighted_mean(late_t, late_q)
    convergence = nonlinear_window_convergence_report(
        times,
        flux,
        case=f"{args.toml.stem}_tprim_scale_{args.drive_scale:.8g}",
        observable="heat_flux",
        source_artifact=str(args.output),
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=float(args.transient_fraction)
        ),
    )

    summary: dict[str, Any] = {
        "kind": "nonlinear_stationary_heat_flux_response_sample",
        "claim_level": "production_long_window_temperature_gradient_continuation",
        "case": args.toml.name,
        "drive_parameter": "uniform_tprim_scale",
        "drive_scale": float(args.drive_scale),
        "grid_override": grid_override,
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "initial_state": str(args.saturated_state),
        "initial_state_saturated": saturated,
        "t_max": float(args.t_max),
        "transient_fraction": float(args.transient_fraction),
        "late_time_weighted_mean_heat_flux": weighted_mean,
        "wall_seconds": elapsed,
        "samples": int(times.size),
        "late_samples": int(late_t.size),
        "adaptive_dt": {
            "min": float(np.nanmin(steps_dt)),
            "median": float(np.nanmedian(steps_dt)),
            "max": float(np.nanmax(steps_dt)),
        },
        "window_convergence": convergence,
        "trace": _json_trace(times, flux),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.state_out is not None and result.state is not None:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        final_state = np.asarray(result.state)
        final_state_saturated = saturated and bool(np.all(np.isfinite(final_state)))
        np.savez_compressed(
            args.state_out,
            state=final_state,
            # Convergence of one statistics window is recorded separately and
            # must not turn a finite continuation of a verified saturated state
            # back into an unsaturated startup state.
            saturated=final_state_saturated,
            stationary_window_passed=convergence["passed"],
            t_end=float(times[-1]),
            adaptive_dt=float(np.nanmedian(steps_dt)),
            method=str(runtime.time.method),
            tau_ac=float("nan"),
        )
        summary["final_state"] = str(args.state_out)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")

    stats = convergence["statistics"]
    print(
        f"scale={args.drive_scale:.8g}  Q_time={weighted_mean:.8e}  "
        f"Q_sample={stats['late_mean']:.8e}  sem={stats['sem']:.3e}  "
        f"window_passed={convergence['passed']}  [{elapsed:.1f}s]",
        flush=True,
    )
    print(f"written: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
