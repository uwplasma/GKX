"""A4, step one: where does GKX's linear ITG threshold sit?

The Dimits shift is the gap between the gradient at which the linear mode first
grows and the gradient at which nonlinear transport actually turns on. Measuring
it needs both ends, and the linear end is cheap: an eigenvalue per gradient,
seconds on a CPU, against a nonlinear point that costs a saturated run.

So this locates the linear threshold first. It sets the scan window for the
nonlinear pass and it is a check in its own right -- if GKX's linear critical
gradient disagrees with the Cyclone base case, the nonlinear number would be
uninterpretable and no GPU time should be spent on it.

Convention: the shipped Cyclone TOML carries ``tprim = a/L_T = 2.49``. This tool
scans the drive as a multiple of the shipped value so the reported threshold is
independent of whether a reader thinks in ``a/L_T`` or ``R/L_T``; both the
multiplier and the resulting ``R_over_LTi`` are recorded.

What this deliberately does not do: call any point "the threshold" from a single
sign change. Growth rates near marginality are small and the eigensolver's own
branch selection can wander, so the threshold is found by fitting the growth rate
against the drive over the points that are unambiguously unstable and
extrapolating to zero -- and the fit residual is reported so a bad fit is visible
rather than silently producing a confident number.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np


def linear_growth_scan(
    toml_path: Path, multipliers: np.ndarray, *, ky_indices: tuple[int, ...]
):
    """Growth rate maximised over ``k_y`` at each drive.

    The critical gradient is a property of the *most unstable* mode, not of one
    arbitrary wavenumber. Scanning a fixed index reports the threshold of that
    index, which sits above the true one whenever another ``k_y`` goes unstable
    first -- an error that looks like a clean measurement.
    """

    import jax.numpy as jnp

    from gkx.geometry.flux_tube import sample_flux_tube_geometry
    from gkx.objectives.core import solver_objective_vector_from_geometry
    from gkx.runtime import build_runtime_geometry, build_runtime_linear_params
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(toml_path)
    analytic = build_runtime_geometry(runtime)
    # The objective needs a solver-ready flux tube; build_runtime_geometry
    # returns the analytic model for a Cyclone case, so sample it onto the
    # runtime's own theta grid rather than inventing one.
    theta = jnp.linspace(
        float(runtime.grid.z_min),
        float(runtime.grid.z_max),
        int(runtime.grid.ntheta),
        endpoint=False,
    )
    geometry = sample_flux_tube_geometry(analytic, theta)
    n_laguerre = int(raw["run"]["Nl"])
    n_hermite = int(raw["run"]["Nm"])
    base = build_runtime_linear_params(runtime, Nm=n_hermite, geom=analytic)
    base_drive = jnp.asarray(base.R_over_LTi)

    rows = []
    for multiplier in multipliers:
        params = dataclasses.replace(base, R_over_LTi=base_drive * float(multiplier))
        started = time.time()
        per_ky = []
        for index in ky_indices:
            values = solver_objective_vector_from_geometry(
                geometry,
                selected_ky_index=int(index),
                n_laguerre=n_laguerre,
                n_hermite=n_hermite,
                # The case's own perpendicular box, not the objective's default:
                # ky spacing is 2 pi / Ly, so a default Ly puts the scanned
                # wavenumbers somewhere the ITG mode does not live.
                ny=int(runtime.grid.Ny),
                ly=float(runtime.grid.Ly),
                params_linear=params,
            )
            per_ky.append((float(values[0]), float(values[1]), int(index)))
        gamma, omega, best = max(per_ky, key=lambda item: item[0])
        rows.append(
            {
                "multiplier": float(multiplier),
                "R_over_LTi": float(np.asarray(base_drive).ravel()[0] * multiplier),
                "gamma": gamma,
                "omega": omega,
                "ky_index": best,
                "gamma_by_ky": [g for g, _, _ in per_ky],
                "seconds": time.time() - started,
            }
        )
        print(
            f"  x{multiplier:5.3f}  gamma={gamma:+.6e}  omega={omega:+.6e}  "
            f"ky#{best}  [{rows[-1]['seconds']:.1f}s]",
            flush=True,
        )
    return rows


def threshold_from_scan(rows: list[dict], *, near_fraction: float = 0.5) -> dict:
    """Extrapolate to gamma = 0 through a square-root threshold law.

    Near a linear instability threshold the growth rate rises as
    ``gamma ~ C sqrt(x - x_crit)``, so ``gamma^2`` is *linear* in the drive with
    its root at the critical value. Fitting ``gamma`` itself with a straight line
    -- as a first version of this tool did -- is the wrong model: well above
    threshold the growth rate flattens, the line is dragged by those points, and
    the extrapolated root comes out low. On the first Cyclone scan that produced
    a relative residual of 0.269 and a critical multiplier of 0.403 that the gate
    correctly refused.

    Only points in the lower ``near_fraction`` of the unstable range are used,
    because the square-root law is a near-threshold expansion and does not
    describe the saturated regime.
    """

    x = np.array([r["multiplier"] for r in rows])
    g = np.array([r["gamma"] for r in rows])
    peak = float(np.nanmax(g)) if g.size else 0.0
    if peak <= 0.0:
        return {"resolved": False, "reason": "no unstable point in the scan"}

    near = (g > 0.02 * peak) & (g < near_fraction * peak)
    if near.sum() < 3:
        return {
            "resolved": False,
            "reason": f"only {int(near.sum())} points in the near-threshold band; "
            "refine the grid around the crossing",
        }

    slope, intercept = np.polyfit(x[near], g[near] ** 2, 1)
    predicted = np.polyval([slope, intercept], x[near])
    residual = float(
        np.std(g[near] ** 2 - predicted) / max(np.std(g[near] ** 2), 1e-30)
    )
    critical = float(-intercept / slope) if slope != 0 else float("nan")
    return {
        "resolved": bool(np.isfinite(critical) and residual < 0.15),
        "critical_multiplier": critical,
        "fit_model": "gamma^2 linear in drive (square-root threshold law)",
        "relative_fit_residual": residual,
        "points_used": int(near.sum()),
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
    parser.add_argument(
        "--ky-indices",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 11, 14],
        help="k_y indices to maximise over at each drive. The critical gradient "
        "belongs to the most unstable mode, so a single index reports that "
        "index's threshold rather than the case's",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=2,
        help="refinement passes concentrating points around the crossing; the "
        "square-root law only holds near threshold, so a uniform span spends "
        "most of its budget where the fit cannot use it",
    )
    parser.add_argument("--min-multiplier", type=float, default=0.3)
    parser.add_argument("--max-multiplier", type=float, default=1.4)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    multipliers = np.linspace(args.min_multiplier, args.max_multiplier, args.points)
    print(
        f"linear drive scan on {args.toml.name}, "
        f"x{args.min_multiplier} to x{args.max_multiplier}",
        flush=True,
    )
    ky_indices = tuple(int(i) for i in args.ky_indices)
    rows = linear_growth_scan(args.toml, multipliers, ky_indices=ky_indices)
    threshold = threshold_from_scan(rows)

    # Refine around the crossing. The square-root law is a near-threshold
    # expansion, so points far above it cannot inform the fit however many there
    # are; concentrating the budget where gamma is small is what resolves it.
    for pass_index in range(args.refine):
        gammas = np.array([r["gamma"] for r in rows])
        order = np.argsort([r["multiplier"] for r in rows])
        xs = np.array([rows[i]["multiplier"] for i in order])
        gs = gammas[order]
        peak = float(np.nanmax(gs))
        if peak <= 0:
            break
        below = np.nonzero(gs <= 0.02 * peak)[0]
        above = np.nonzero(gs > 0.02 * peak)[0]
        if not below.size or not above.size:
            break
        lo, hi = xs[below[-1]], xs[above[0]]
        extra = np.linspace(lo, hi, 6)[1:-1]
        print(
            f"\nrefinement {pass_index + 1}: bracketing x{lo:.3f} to x{hi:.3f}",
            flush=True,
        )
        rows.extend(linear_growth_scan(args.toml, extra, ky_indices=ky_indices))
        rows.sort(key=lambda r: r["multiplier"])
        threshold = threshold_from_scan(rows)

    if threshold.get("resolved"):
        print(
            f"\nlinear threshold at x{threshold['critical_multiplier']:.3f} of the "
            f"shipped drive (fit residual {threshold['relative_fit_residual']:.3f}, "
            f"{threshold['points_used']} points)"
        )
        print(
            "The nonlinear scan should bracket this: the Dimits shift puts the "
            "transport onset ABOVE it, so span roughly x0.8 to x1.6 of the "
            "critical multiplier."
        )
    else:
        print(f"\nlinear threshold NOT resolved: {threshold}")

    summary = {
        "kind": "dimits_linear_threshold",
        "claim_level": "linear_threshold_only_no_nonlinear_onset_measured",
        "case": args.toml.name,
        "ky_indices": list(ky_indices),
        "refinement_passes": args.refine,
        "threshold": threshold,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}")
    return 0 if threshold.get("resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
