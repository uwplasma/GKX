"""Reach a genuinely saturated nonlinear state, using GKX's own production stepper.

The nonlinear heat-flux adjoint needs a
trajectory that has actually saturated. A hand-rolled fixed-step integrator
cannot supply one: the E x B nonlinearity imposes a CFL condition that tightens
as the amplitude grows, so a step size that is stable through the linear phase
goes unstable exactly when the nonlinearity would otherwise bite. Measured, that
looked like a run whose energy grew by 3.3e49 over t=600 and never saturated.

Production solves this with machinery the hand-rolled loop skipped entirely:
CFL-adaptive stepping (``fixed_dt=false``, ``dt_max``, ``cfl``), a state
projector enforcing the reality and fixed-mode constraints, and a dealias mask.
Rather than reimplement any of that, this tool drives ``run_runtime_nonlinear``
-- the same entry point the CLI uses -- and saves the final state.

Saturation is judged by the production stop policy on the production heat-flux
trace, with ``Wphi`` as its stationarity guard. ``Wg`` is reported as an
additional audit guard without changing the runtime decision. A state that
fails is written out anyway, flagged, so the failure is inspectable rather than
silent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
        ),
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--nl", type=int, default=None)
    parser.add_argument("--nm", type=int, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument(
        "--run-to",
        choices=("saturation", "t_max"),
        default=None,
        help="override the deck stop policy; use t_max for held-out audits",
    )
    parser.add_argument(
        "--vmec-file",
        type=Path,
        default=None,
        help="override [geometry].vmec_file after loading the deck",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="optional npz continuation state; unlike the gradient campaign, "
        "this may be marked unsaturated because the purpose is to finish it",
    )
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="compact npz with scalar traces and available kx/ky spectra",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    from gkx import build_spectral_grid, run_runtime_nonlinear
    from gkx.diagnostics.saturation import (
        SaturationStopConfig,
        saturation_stop_decision,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(args.toml)
    grid_override = {
        k: v for k, v in (("Nx", args.nx), ("Ny", args.ny)) if v is not None
    }
    if args.nz is not None:
        grid_override["Nz"] = args.nz
        if runtime.grid.ntheta is not None:
            periods = runtime.grid.zp
            if periods is None:
                periods = (
                    2 * int(runtime.grid.nperiod) - 1
                    if runtime.grid.nperiod is not None
                    else 1
                )
            if args.nz % periods:
                raise SystemExit(
                    f"--nz={args.nz} must be divisible by the field-line period "
                    f"factor {periods}"
                )
            grid_override["ntheta"] = args.nz // periods
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )
    time_override = {}
    if args.t_max is not None:
        time_override["t_max"] = args.t_max
    if args.run_to is not None:
        time_override["run_to"] = args.run_to
    if time_override:
        runtime = dataclasses.replace(
            runtime, time=dataclasses.replace(runtime.time, **time_override)
        )
    if args.vmec_file is not None:
        runtime = dataclasses.replace(
            runtime,
            geometry=dataclasses.replace(
                runtime.geometry, vmec_file=str(args.vmec_file.resolve())
            ),
        )

    time_cfg = runtime.time
    spectral_grid = build_spectral_grid(runtime.grid)
    grid_shape = {
        "Nx": int(spectral_grid.kx.size),
        "Ny": int(spectral_grid.ky.size),
        "Nz": int(spectral_grid.z.size),
    }
    print(
        f"case: {args.toml.name}  grid={grid_shape}  "
        f"t_max={time_cfg.t_max}  fixed_dt={time_cfg.fixed_dt}  "
        f"dt={time_cfg.dt} dt_max={time_cfg.dt_max} cfl={time_cfg.cfl}",
        flush=True,
    )
    if time_cfg.fixed_dt:
        print(
            "  NOTE: this config uses a fixed step. The nonlinear CFL tightens "
            "with amplitude, so saturation is not guaranteed.",
            flush=True,
        )

    n_laguerre = int(args.nl or raw.get("run", {}).get("Nl", 4))
    n_hermite = int(args.nm or raw.get("run", {}).get("Nm", 8))
    previous_t_end = 0.0

    def run(config):
        return run_runtime_nonlinear(
            config,
            Nl=n_laguerre,
            Nm=n_hermite,
            sample_stride=args.sample_stride,
            diagnostics_stride=args.diagnostics_stride,
            return_state=True,
            diagnostics=True,
            show_progress=args.progress,
        )

    started = time.time()
    if args.initial_state is None:
        result = run(runtime)
    else:
        archive = np.load(args.initial_state)
        continuation_state = np.asarray(archive["state"])
        expected_shape = (
            len(runtime.species),
            n_laguerre,
            n_hermite,
            grid_shape["Ny"],
            grid_shape["Nx"],
            grid_shape["Nz"],
        )
        if continuation_state.shape != expected_shape:
            raise SystemExit(
                f"continuation shape {continuation_state.shape} != {expected_shape}"
            )
        previous_t_end = float(archive["t_end"]) if "t_end" in archive else 0.0
        with tempfile.TemporaryDirectory(prefix="gkx-saturation-continuation-") as temp:
            restart_file = Path(temp) / "restart.bin"
            np.asarray(continuation_state, dtype=np.complex64).tofile(restart_file)
            continuation_runtime = dataclasses.replace(
                runtime,
                init=dataclasses.replace(
                    runtime.init,
                    init_file=str(restart_file),
                    init_file_scale=1.0,
                    init_file_mode="replace",
                    init_amp=0.0,
                ),
            )
            result = run(continuation_runtime)
    elapsed = time.time() - started

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise RuntimeError("run returned no diagnostics; cannot judge saturation")
    times = np.asarray(diagnostics.t, dtype=float)
    flux = np.asarray(diagnostics.heat_flux_t, dtype=float)
    wphi = np.asarray(diagnostics.Wphi_t, dtype=float)
    wg = np.asarray(diagnostics.Wg_t, dtype=float)
    steps_dt = np.asarray(diagnostics.dt_t, dtype=float)
    absolute_times = previous_t_end + times

    stop_config = SaturationStopConfig(
        rel_sem=float(time_cfg.saturation_rel_sem),
        min_window=time_cfg.saturation_min_window,
    )
    report = saturation_stop_decision(
        times,
        flux,
        guard=wphi,
        config=stop_config,
    )
    wg_report = saturation_stop_decision(times, flux, guard=wg, config=stop_config)
    report["Wg_guard_stationary"] = wg_report["guard_stationary"]
    print(
        f"\nran {times.size} samples to t={absolute_times[-1]:.1f} in {elapsed:.1f}s",
        flush=True,
    )
    print(
        f"  adaptive dt: min={np.nanmin(steps_dt):.4g} "
        f"median={np.nanmedian(steps_dt):.4g} max={np.nanmax(steps_dt):.4g}",
        flush=True,
    )
    if report.get("tau_ac") is not None:
        tau_ac = float(report["tau_ac"])
        window_span = float(report["window_span"])
        print(
            f"  window={report['window_tmin']:.2f}--{report['window_tmax']:.2f}  "
            f"tau_ac={tau_ac:.2f}  window/tau_ac={window_span / tau_ac:.1f}  "
            f"rel_sem={report['rel_sem']:.1%}  mean flux={report['mean']:.4e}",
            flush=True,
        )
    print(
        f"  Wphi stationary={report['guard_stationary']}  "
        f"Wg stationary={report['Wg_guard_stationary']}",
        flush=True,
    )
    if report["reasons"]:
        print(f"  failed gates: {', '.join(report['reasons'])}", flush=True)
    print(f"  -> {'SATURATED' if report['saturated'] else 'NOT SATURATED'}", flush=True)

    if args.trace_out is not None:
        payload = {
            "time": absolute_times,
            "dt": steps_dt,
            "kx": np.asarray(spectral_grid.kx),
            "ky": np.asarray(spectral_grid.ky),
            "heat_flux": flux,
            "Wphi": wphi,
            "Wg": wg,
            "Nl": np.asarray(n_laguerre),
            "Nm": np.asarray(n_hermite),
            "elapsed_seconds": np.asarray(elapsed),
            "previous_t_end": np.asarray(previous_t_end),
        }
        resolved = diagnostics.resolved
        if resolved is not None:
            for name in (
                "Phi2_kxt",
                "Phi2_kyt",
                "HeatFlux_kxst",
                "HeatFlux_kyst",
            ):
                value = getattr(resolved, name)
                if value is not None:
                    payload[name] = np.asarray(value)
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace_out, **payload)
        print(f"trace written: {args.trace_out}", flush=True)

    if args.state_out is not None and result.state is not None:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_out,
            state=np.asarray(result.state),
            saturated=report["saturated"],
            t_end=absolute_times[-1],
            # The step the trajectory was actually produced with. Without it the
            # consumer has to be told by hand, and a mismatched --dt silently
            # rescales its time axis: every t/tau_ac it reports would be wrong
            # while every number still looked plausible.
            adaptive_dt=float(np.nanmedian(steps_dt)),
            method=str(time_cfg.method),
            tau_ac=(
                float(report["tau_ac"])
                if report.get("tau_ac") is not None
                else float("nan")
            ),
        )
        print(f"state written: {args.state_out}", flush=True)

    summary = {
        "kind": "nonlinear_saturated_state",
        "claim_level": "production_saturation_stop_policy_audit",
        "case": args.toml.name,
        "initial_state": None
        if args.initial_state is None
        else str(args.initial_state),
        "previous_t_end": previous_t_end,
        "grid": grid_shape,
        "grid_override": grid_override,
        "t_max": float(time_cfg.t_max),
        "fixed_dt": bool(time_cfg.fixed_dt),
        "adaptive_dt": {
            "min": float(np.nanmin(steps_dt)),
            "median": float(np.nanmedian(steps_dt)),
            "max": float(np.nanmax(steps_dt)),
        },
        "wall_seconds": elapsed,
        "samples": int(times.size),
        "report": report,
        "trace": [
            {
                "t": float(t),
                "heat_flux": float(q),
                "Wphi": float(phi),
                "Wg": float(g),
            }
            for t, q, phi, g in zip(absolute_times, flux, wphi, wg)
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}", flush=True)
    return 0 if report["saturated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
