"""How long is a nonlinear window, measured in units the turbulence sets?

Every transport window in GKX is declared in code-time units (``tmin``,
``tmax``). Nothing checks that window against the only timescale that decides
whether an average means anything: the heat-flux autocorrelation time
``tau_ac``. A window shorter than a few ``tau_ac`` contains a handful of
independent samples no matter how many output steps it holds, and its error bar
is optimistic by the ratio of samples to independent samples.

This matters right now because the production gradient gate is blocked on
``gradient_uncertainty_rel = 1.806`` against a maximum of ``0.5``. Closing that
by averaging alone costs ``(1.806/0.5)^2 ~ 13x`` longer windows. Before paying
that, it is worth knowing how many independent samples the current windows
actually contain -- and ``tau_ac`` is what converts an output-step count into
that number.

``tau_ac`` is also what bounds a windowed adjoint: the usable backpropagation
window is set by the dynamical memory of the turbulence, not by a solver
tolerance, and a code that has never measured ``tau_ac`` cannot know which side
of that bound it is on.

Definition used here
--------------------

The normalized autocorrelation of the post-transient heat-flux fluctuation,

    rho(k) = <(q_i - qbar)(q_{i+k} - qbar)> / <(q_i - qbar)^2>,

integrated by the trapezoid rule up to its **first zero crossing** (the standard
initial-positive-sequence truncation; summing past the crossing accumulates
noise rather than signal). ``tau_ac`` is reported in code time units, and the
window is reported as a multiple of it.

The effective independent-sample count for a window of ``n`` samples at spacing
``dt`` is

    n_eff = min(n, n dt / (2 tau_ac)),

which returns ``n`` at the independent-sample floor ``tau_ac=dt/2``.

This tool only reads committed trace CSVs. It runs no simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Promoted into the package so the run-to-saturation runtime stop check and
# this post-hoc campaign tool share one estimator.
from gkx.diagnostics.saturation import (
    sokal_autocorrelation_time as autocorrelation_time,
)

REQUIRED_COLUMNS = ("t", "heat_flux")


def analyse(path: Path) -> dict[str, Any] | None:
    """Autocorrelation summary for one committed trace.

    Returns ``None`` for a CSV that is not a heat-flux time series -- the
    ``*.traces.csv`` suffix is also used for unrelated diagnostics such as the
    zonal-response panel, and silently treating one of those as a flux trace
    would produce a confident number for a quantity nobody asked about.
    """

    table = np.genfromtxt(path, delimiter=",", names=True)
    if table.dtype.names is None or not all(
        column in table.dtype.names for column in REQUIRED_COLUMNS
    ):
        return None
    time = np.asarray(table["t"], dtype=float)
    flux = np.asarray(table["heat_flux"], dtype=float)

    # Restrict to the saturated half: an autocorrelation computed across the
    # linear-growth transient measures the growth, not the turbulence.
    start = time.size // 2
    time, flux = time[start:], flux[start:]
    dt = float(np.median(np.diff(time)))

    tau, cut, rho = autocorrelation_time(flux, dt)
    span = float(time[-1] - time[0])
    resolved = cut < rho.size

    n_eff = (
        min(float(flux.size), flux.size * dt / (2.0 * tau))
        if tau > 0.0 and dt > 0.0
        else float(flux.size)
    )
    mean = float(flux.mean())
    naive_error = float(flux.std(ddof=1) / np.sqrt(flux.size))
    corrected_error = float(flux.std(ddof=1) / np.sqrt(max(n_eff, 1.0)))

    return {
        "trace": path.name,
        "samples_in_window": int(flux.size),
        "output_dt": dt,
        "window_span": span,
        "tau_ac": tau,
        "tau_ac_resolved": bool(resolved),
        "window_in_tau_ac": span / tau if tau > 0 else float("inf"),
        "independent_samples": float(n_eff),
        "heat_flux_mean": mean,
        "naive_standard_error": naive_error,
        "correlation_corrected_standard_error": corrected_error,
        "error_bar_understated_by": corrected_error / naive_error
        if naive_error > 0
        else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="*",
        default=sorted(Path("docs/_static").glob("*.traces.csv")),
    )
    parser.add_argument("--min-tau-multiples", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    candidates = {path: analyse(path) for path in args.traces}
    skipped = [path.name for path, row in candidates.items() if row is None]
    rows = [row for row in candidates.values() if row is not None]
    rows.sort(key=lambda row: row["window_in_tau_ac"])
    if skipped:
        print(f"skipped {len(skipped)} non-flux trace(s): {', '.join(skipped)}\n")

    print(f"{'trace':52s} {'tau_ac':>8s} {'win/tau':>8s} {'n_eff':>7s} {'err x':>6s}")
    for row in rows:
        flag = "" if row["tau_ac_resolved"] else "  <- tau_ac UNRESOLVED"
        print(
            f"{row['trace'][:52]:52s} {row['tau_ac']:8.2f} "
            f"{row['window_in_tau_ac']:8.1f} {row['independent_samples']:7.1f} "
            f"{row['error_bar_understated_by']:6.2f}{flag}"
        )

    short = [r for r in rows if r["window_in_tau_ac"] < args.min_tau_multiples]
    unresolved = [r for r in rows if not r["tau_ac_resolved"]]
    print(
        f"\n{len(short)}/{len(rows)} windows are shorter than "
        f"{args.min_tau_multiples:g} correlation times"
    )
    if unresolved:
        print(
            f"{len(unresolved)}/{len(rows)} traces never cross zero: too short to "
            "resolve their own correlation time"
        )

    summary = {
        "kind": "heat_flux_autocorrelation",
        "claim_level": "post_processing_of_committed_traces_no_new_simulation",
        "definition": (
            "integrated normalized autocorrelation of the post-transient heat "
            "flux, trapezoid rule truncated at the first zero crossing; "
            "n_eff = min(n, n dt / (2 tau_ac))"
        ),
        "min_tau_multiples": args.min_tau_multiples,
        "windows_shorter_than_threshold": len(short),
        "traces_with_unresolved_tau": len(unresolved),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
