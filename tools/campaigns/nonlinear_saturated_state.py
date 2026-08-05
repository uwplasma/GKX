"""Reach a genuinely saturated nonlinear state, using GKX's own production stepper.

The gradient-window measurement (``nonlinear_gradient_window.py``) needs a
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

Saturation is then judged on the **production heat-flux trace**, not on a proxy:
the run is accepted only if the late-time flux has stopped trending and the
window spans enough correlation times to mean anything, reusing the tau_ac
definition from ``heat_flux_autocorrelation.py``. A state that fails is written
out anyway, flagged, so the failure is inspectable rather than silent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def integrated_autocorrelation_time(
    signal: np.ndarray, dt: float
) -> tuple[float, bool]:
    """Integrated autocorrelation time, truncated at the first zero crossing.

    Same definition as ``heat_flux_autocorrelation.py`` so the numbers here are
    comparable to the ones measured on the committed traces. The boolean reports
    whether the trace was long enough to resolve its own correlation time.
    """

    fluctuation = np.asarray(signal, dtype=float)
    fluctuation = fluctuation - fluctuation.mean()
    if fluctuation.size < 4 or not np.any(fluctuation):
        return 0.0, False
    size = int(2 ** np.ceil(np.log2(2 * fluctuation.size)))
    spectrum = np.fft.rfft(fluctuation, n=size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: fluctuation.size]
    rho = correlation / correlation[0]
    negative = np.nonzero(rho < 0.0)[0]
    cut = int(negative[0]) if negative.size else rho.size
    tau = float(np.trapezoid(rho[:cut], dx=dt)) if cut > 1 else 0.0
    return tau, bool(cut < rho.size)


def saturation_report(
    times: np.ndarray, flux: np.ndarray, *, min_tau_multiples: float
) -> dict[str, Any]:
    """Judge saturation on the production heat-flux trace."""

    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux)
    times, flux = times[finite], flux[finite]
    if times.size < 8:
        return {"saturated": False, "reason": "trace too short to judge"}

    half = times.size // 2
    late_t, late_q = times[half:], flux[half:]
    dt = float(np.median(np.diff(late_t))) if late_t.size > 1 else 0.0
    tau, resolved = integrated_autocorrelation_time(late_q, dt)
    span = float(late_t[-1] - late_t[0])
    windows = span / tau if tau > 0 else float("inf")

    quarter = max(1, late_q.size // 4)
    drift = abs(
        late_q[-quarter:].mean() - late_q[-2 * quarter : -quarter].mean()
    ) / max(abs(late_q[-quarter:].mean()), 1e-300)
    grown = float(flux[-1]) / max(
        abs(float(flux[: max(1, flux.size // 20)].mean())), 1e-300
    )

    checks = {
        "flux_is_finite": bool(np.all(np.isfinite(flux))),
        "stopped_trending": bool(drift < 0.25),
        "grew_from_seed": bool(grown > 10.0),
        "tau_resolved": bool(resolved),
        "window_long_enough": bool(windows >= min_tau_multiples),
    }
    return {
        "saturated": all(checks.values()),
        "checks": checks,
        "tau_ac": tau,
        "window_in_tau_ac": windows,
        "late_drift": float(drift),
        "growth_over_start": float(grown),
        "late_mean_flux": float(late_q.mean()),
    }


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
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--min-tau-multiples", type=float, default=10.0)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    from gkx import run_runtime_nonlinear
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(args.toml)
    grid_override = {
        k: v
        for k, v in (("Nx", args.nx), ("Ny", args.ny), ("Nz", args.nz))
        if v is not None
    }
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )
    if args.t_max is not None:
        runtime = dataclasses.replace(
            runtime, time=dataclasses.replace(runtime.time, t_max=args.t_max)
        )

    time_cfg = runtime.time
    print(
        f"case: {args.toml.name}  grid={grid_override or 'as shipped'}  "
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

    started = time.time()
    result = run_runtime_nonlinear(
        runtime,
        Nl=int(raw.get("run", {}).get("Nl", 4)),
        Nm=int(raw.get("run", {}).get("Nm", 8)),
        return_state=True,
        diagnostics=True,
    )
    elapsed = time.time() - started

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise RuntimeError("run returned no diagnostics; cannot judge saturation")
    times = np.asarray(diagnostics.t, dtype=float)
    flux = np.asarray(diagnostics.heat_flux_t, dtype=float)
    steps_dt = np.asarray(diagnostics.dt_t, dtype=float)

    report = saturation_report(times, flux, min_tau_multiples=args.min_tau_multiples)
    print(
        f"\nran {times.size} samples to t={times[-1]:.1f} in {elapsed:.1f}s", flush=True
    )
    print(
        f"  adaptive dt: min={np.nanmin(steps_dt):.4g} "
        f"median={np.nanmedian(steps_dt):.4g} max={np.nanmax(steps_dt):.4g}",
        flush=True,
    )
    for name, passed in report.get("checks", {}).items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}", flush=True)
    if "tau_ac" in report:
        print(
            f"  tau_ac={report['tau_ac']:.2f}  "
            f"window={report['window_in_tau_ac']:.1f} tau_ac  "
            f"late drift={report['late_drift']:.1%}  "
            f"mean flux={report['late_mean_flux']:.4e}",
            flush=True,
        )
    print(f"  -> {'SATURATED' if report['saturated'] else 'NOT SATURATED'}", flush=True)

    if args.state_out is not None and result.state is not None:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_out,
            state=np.asarray(result.state),
            saturated=report["saturated"],
            t_end=times[-1],
            # The step the trajectory was actually produced with. Without it the
            # consumer has to be told by hand, and a mismatched --dt silently
            # rescales its time axis: every t/tau_ac it reports would be wrong
            # while every number still looked plausible.
            adaptive_dt=float(np.nanmedian(steps_dt)),
            method=str(time_cfg.method),
            tau_ac=float(report.get("tau_ac", float("nan"))),
        )
        print(f"state written: {args.state_out}", flush=True)

    summary = {
        "kind": "nonlinear_saturated_state",
        "claim_level": "production_adaptive_stepper_saturation_check",
        "case": args.toml.name,
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
        "trace": [{"t": float(a), "heat_flux": float(b)} for a, b in zip(times, flux)],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}", flush=True)
    return 0 if report["saturated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
