"""A4, step one: where does GKX's linear ITG threshold sit?

The Dimits shift is the gap between the gradient at which the linear mode first
grows and the gradient at which nonlinear transport actually turns on. Measuring
it needs both ends, and the linear end is much the cheaper: dense eigenvalues on
a CPU, minutes, against a nonlinear point that costs a saturated run.

So this locates the linear threshold first. It sets the scan window for the
nonlinear pass and it is a check in its own right -- if GKX's linear critical
gradient disagrees with the Cyclone base case, the nonlinear number would be
uninterpretable and no GPU time should be spent on it.

Convention: the shipped Cyclone TOML carries ``tprim = a/L_T = 2.49``. This tool
scans the drive as a multiple of the shipped value so the reported threshold is
independent of whether a reader thinks in ``a/L_T`` or ``R/L_T``; both the
multiplier and the resulting ``R_over_LTi`` are recorded.

Build the case's operator, not a default one
--------------------------------------------

The single thing that had to be right, and was not for four attempts: the
operator is built with ``build_runtime_linear_terms(runtime)``. Passing
``terms=None`` takes the ``LinearTerms`` dataclass defaults, which set
``hyperdiffusion = 0.0`` while this TOML sets it to ``1.0``. Without it the
operator has unstable eigenvalues at drives far below the ITG threshold whose
growth rate *rises* with Hermite resolution -- measured at ``0.15x`` drive,
:math:`\\gamma = 1.2\\times10^{-3}` at ``Nm=8``, :math:`3.9\\times10^{-3}` at
``Nm=16``, :math:`2.2\\times10^{-2}` at ``Nm=32``, at :math:`|\\omega|` up to 23
and with no dependence on the drive at all. With the case's own terms every
wavenumber is damped at ``0.15x``. Everything below is downstream of that.

Two reductions are also deliberately not used:

**Not the most unstable eigenvalue.** Near marginality the ITG mode need not be
the most unstable one, so the branch is followed by eigenvalue continuation from
high drive downwards. This also produces the damped side of the curve, which is
what makes an independent bracket on the root available.

**Not the growth rate maximised over** ``k_y``. A critical gradient is defined per
wavenumber, and the case's value is the minimum of those. Maximising first and
fitting afterwards mixes wavenumbers by construction, because the argmax itself
moves between them as the drive changes.
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
    """Follow each wavenumber's ITG eigenvalue from high drive down through zero.

    The scan starts at the highest drive, where the ITG mode is unambiguously the
    most unstable, and steps *down*, at each drive taking the eigenvalue nearest
    the previous one in the complex plane. That is what makes a threshold
    measurable at all here: the tracked branch passes through
    :math:`\\gamma = 0` and goes damped, whereas the maximum-growth eigenvalue
    never does, because some other weakly unstable mode takes over below the ITG
    threshold.

    The distance the eigenvalue moves at each step is recorded, so a continuation
    that jumped to a neighbouring branch is visible rather than silently fitted.
    """

    import jax.numpy as jnp

    from gkx.geometry.flux_tube import sample_flux_tube_geometry
    from gkx.objectives.core import solver_linear_operator_matrix_from_geometry
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )
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
    # The case's own term switches, not the LinearTerms defaults. This is not a
    # detail: the shipped Cyclone TOML sets hyperdiffusion = 1.0 and the
    # dataclass defaults it to 0.0, and without it the operator has unstable
    # eigenvalues at drives far below the ITG threshold -- gamma ~ 1e-3 at
    # Nm = 8, rising to ~2e-2 at Nm = 32, with |omega| ~ 20 and no dependence on
    # the drive at all. Those are what argmax returns below threshold, and
    # measuring them as if they were the ITG branch is what defeated the first
    # four versions of this tool.
    case_terms = build_runtime_linear_terms(runtime)

    descending = np.sort(np.asarray(multipliers, dtype=float))[::-1]
    tracked: dict[float, dict[str, dict]] = {}
    for index in ky_indices:
        started = time.time()
        previous = None
        for multiplier in descending:
            params = dataclasses.replace(
                base, R_over_LTi=base_drive * float(multiplier)
            )
            matrix = np.asarray(
                solver_linear_operator_matrix_from_geometry(
                    geometry,
                    selected_ky_index=int(index),
                    n_laguerre=n_laguerre,
                    n_hermite=n_hermite,
                    # The case's own perpendicular box, not the objective's
                    # default: ky spacing is 2 pi / Ly, so a default Ly puts the
                    # scanned wavenumbers somewhere the ITG mode does not live.
                    ny=int(runtime.grid.Ny),
                    ly=float(runtime.grid.Ly),
                    params_linear=params,
                    terms=case_terms,
                )
            )
            spectrum = np.linalg.eigvals(matrix)
            if previous is None:
                picked, jump = spectrum[int(np.argmax(spectrum.real))], 0.0
            else:
                distance = np.abs(spectrum - previous)
                nearest = int(np.argmin(distance))
                picked, jump = spectrum[nearest], float(distance[nearest])
            previous = picked
            tracked.setdefault(float(multiplier), {})[str(int(index))] = {
                "gamma": float(picked.real),
                "omega": float(picked.imag),
                "branch_jump": jump,
            }
        traced = [tracked[m][str(int(index))] for m in descending]
        unstable = [m for m, t in zip(descending, traced) if t["gamma"] > 0.0]
        print(
            f"  ky#{index:<3} unstable down to "
            + (f"x{min(unstable):5.3f}" if unstable else "nowhere in range")
            + f", largest continuation step "
            f"{max(t['branch_jump'] for t in traced):.2e}"
            f"  [{time.time() - started:.1f}s]",
            flush=True,
        )
    return [
        {
            "multiplier": multiplier,
            "R_over_LTi": float(np.asarray(base_drive).ravel()[0] * multiplier),
            "by_ky": by_ky,
        }
        for multiplier, by_ky in sorted(tracked.items())
    ]


def _zero_crossing(x: np.ndarray, g: np.ndarray, fit_points: int) -> dict:
    """Where one wavenumber's tracked growth rate crosses zero.

    ``gamma`` is fitted **linearly** in the drive across the crossing, and this
    is not the usual near-threshold square-root law. The square-root law
    describes a bifurcation in which two roots merge; a simple eigenvalue
    crossing the imaginary axis moves analytically in the parameter, so
    :math:`\\gamma` passes through zero with a finite slope. The measured Cyclone
    curves settle it -- successive differences of ``gamma`` are constant to a few
    percent across the crossing -- and so does the control below: the linear root
    lands inside the sign-change bracket for every wavenumber (residuals 0.03 to
    0.13) while the square-root root lands outside every one of them (0.13 to
    0.21). An earlier version of this tool used the square-root law, on textbook
    reasoning that does not apply here.

    The fit is checked against a quantity it does not use. Because the branch is
    tracked rather than reduced, ``gamma`` changes sign inside the scan, and the
    two adjacent drives straddling that change bracket the root with no model at
    all. A fitted root outside the bracket is reported with ``in_bracket`` false
    rather than returned as a number, and the bracket tightens with refinement,
    so the control gets stronger exactly as the estimate does.

    Always returns a dict, carrying ``reason`` when no fit was possible, so a
    wavenumber that failed is diagnosable from the artifact.
    """

    order = np.argsort(x)
    x, g = np.asarray(x)[order], np.asarray(g)[order]
    unstable = g > 0.0
    if not unstable.any():
        return {"reason": "damped across the whole scanned range"}
    first = int(np.argmax(unstable))
    if first == 0:
        return {"reason": "unstable at the lowest drive scanned; widen the scan"}
    # Straddle the crossing: two damped points below it, the rest above. Fitting
    # only unstable points would extrapolate to the root from one side.
    start = max(first - 2, 0)
    window = slice(start, start + fit_points)
    xs, gs = x[window], g[window]
    if xs.size < 4:
        return {"reason": f"only {int(xs.size)} points straddling the crossing"}
    slope, intercept = np.polyfit(xs, gs, 1)
    if slope <= 0.0:
        return {"reason": "growth rate does not increase with the drive"}
    critical = float(-intercept / slope)
    residual = float(
        np.std(gs - np.polyval([slope, intercept], xs)) / max(np.std(gs), 1e-30)
    )
    bracket = [float(x[first - 1]), float(x[first])]
    return {
        "critical_multiplier": critical,
        "relative_fit_residual": residual,
        "points_used": int(xs.size),
        "sign_change_bracket": bracket,
        "in_bracket": bool(
            np.isfinite(critical) and bracket[0] <= critical <= bracket[1]
        ),
    }


def threshold_from_scan(
    rows: list[dict], *, fit_points: int = 7, max_residual: float = 0.15
) -> dict:
    """A threshold per wavenumber, then the minimum over ``k_y``.

    Each ``k_y`` is fitted on its own tracked ``gamma(drive)`` curve. The case's
    critical gradient is the smallest of those roots: the drive at which *some*
    mode first goes unstable. The wavenumber that achieves it is reported too,
    because the nonlinear scan needs it.

    A wavenumber counts only if its fit passes both checks -- residual under
    ``max_residual``, and root inside its own sign-change bracket. Every fit is
    kept in ``per_ky`` whether or not it passed.
    """

    x_all = np.array([r["multiplier"] for r in rows])
    per_ky: dict[str, dict] = {}
    for key in sorted({k for r in rows for k in r["by_ky"]}, key=int):
        g = np.array([r["by_ky"][key]["gamma"] for r in rows])
        jumps = [r["by_ky"][key].get("branch_jump", 0.0) for r in rows]
        per_ky[key] = {
            **_zero_crossing(x_all, g, fit_points),
            "max_branch_jump": float(max(jumps)),
        }
    brackets = [v.get("sign_change_bracket") for v in per_ky.values()]
    present = [b for b in brackets if b]
    fitted = {k: v for k, v in per_ky.items() if "critical_multiplier" in v}
    good = {
        k: v
        for k, v in fitted.items()
        if v["in_bracket"] and v["relative_fit_residual"] < max_residual
    }
    pool = good or fitted
    if not pool:
        return {
            "resolved": False,
            "reason": "no wavenumber produced a fit; widen the scan or refine",
            # Refinement still has somewhere to go even with no fit yet.
            "lowest_sign_change_bracket": min(present, key=lambda b: b[0])
            if present
            else None,
            "per_ky": per_ky,
            "total_rows": len(rows),
        }
    best = min(pool, key=lambda k: pool[k]["critical_multiplier"])
    return {
        "resolved": bool(good),
        "critical_multiplier": pool[best]["critical_multiplier"],
        "critical_ky_index": int(best),
        "fit_model": "per-k_y gamma linear across its zero crossing, "
        "checked against the sign-change bracket, minimised over k_y",
        "relative_fit_residual": pool[best]["relative_fit_residual"],
        "points_used": pool[best]["points_used"],
        "sign_change_bracket": pool[best]["sign_change_bracket"],
        # Refinement targets this, not the winner's bracket: the case threshold
        # is a minimum over k_y, so the interval where *any* mode first goes
        # unstable is where added points can still change the answer.
        "lowest_sign_change_bracket": min(present, key=lambda b: b[0])
        if present
        else None,
        "fitted_ky_count": len(fitted),
        "accepted_ky_count": len(good),
        "per_ky": per_ky,
        "total_rows": len(rows),
    }


def _literature_comparison(toml_path: Path, threshold: dict) -> dict | None:
    """Put the threshold in the units the Cyclone literature quotes it in.

    Two conversions, both stated rather than folded in. The drive that the
    operator consumes is ``tprim = a/L_T`` -- the field carrying it is *named*
    ``R_over_LTi`` but every consumer treats it as ``tprim``, and the shipped
    value 2.49 is the Cyclone ``a/L_T``. Turning that into the quoted
    :math:`R/L_T` needs ``R/a``, which the cyclone normalization contract fixes
    by setting ``a = 1``, so ``R/a`` is the TOML's own ``R0``.

    Returns ``None`` rather than guessing when the case is not on that contract.
    """

    import tomllib

    critical = threshold.get("critical_multiplier")
    if critical is None or not np.isfinite(critical):
        return None
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    if str(raw.get("normalization", {}).get("contract", "")) != "cyclone":
        return None
    r_over_a = float(raw["geometry"]["R0"])
    tprim_crit = critical * float(raw["species"][0]["tprim"])
    return {
        "tprim_crit": tprim_crit,
        "R_over_a": r_over_a,
        "R_over_LT_crit": tprim_crit * r_over_a,
        "literature_R_over_LT": 4.0,
        "literature_source": "Dimits et al., Phys. Plasmas 7, 969 (2000), "
        "Cyclone base case linear critical gradient",
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
        default=[1, 2, 3, 4, 6],
        help="k_y indices to threshold independently. Each gets its own critical "
        "drive; the case's value is the minimum over them, so this list bounds "
        "which modes can win",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=2,
        help="refinement passes adding points inside the earliest sign change; "
        "the fit is a local linearisation about the crossing, so a uniform "
        "span spends most of its budget where the fit cannot use it",
    )
    parser.add_argument("--min-multiplier", type=float, default=0.15)
    parser.add_argument("--max-multiplier", type=float, default=1.2)
    parser.add_argument("--points", type=int, default=10)
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

    # Refine inside the interval where the earliest mode changes sign. Adding
    # points there both tightens the window the linear fit is taken over and
    # tightens the bracket that checks its root, so the estimate and its control
    # improve together. Continuation needs the whole drive list
    # in order, so a pass re-runs the scan over the union rather than appending
    # to it; that is the price of a branch that stays continuous.
    for pass_index in range(args.refine):
        bracket = threshold.get("lowest_sign_change_bracket")
        if not bracket:
            break
        extra = np.linspace(bracket[0], bracket[1], 5)[1:-1]
        multipliers = np.unique(np.concatenate([multipliers, extra]))
        print(
            f"\nrefinement {pass_index + 1}: sign change between "
            f"x{bracket[0]:.3f} and x{bracket[1]:.3f}, now {multipliers.size} drives",
            flush=True,
        )
        rows = linear_growth_scan(args.toml, multipliers, ky_indices=ky_indices)
        threshold = threshold_from_scan(rows)

    reference = _literature_comparison(args.toml, threshold)
    if threshold.get("resolved"):
        print(
            f"\nlinear threshold at x{threshold['critical_multiplier']:.4f} of the "
            f"shipped drive, set by ky#{threshold['critical_ky_index']} "
            f"(fit residual {threshold['relative_fit_residual']:.3f}, "
            f"{threshold['points_used']} points; "
            f"{threshold['accepted_ky_count']} of {threshold['fitted_ky_count']} "
            "wavenumbers accepted). Sign change bracketed to "
            f"[{threshold['sign_change_bracket'][0]:.4f}, "
            f"{threshold['sign_change_bracket'][1]:.4f}]."
        )
        if reference:
            print(
                f"  tprim_crit = {reference['tprim_crit']:.3f}, and with a = 1 as "
                f"the cyclone contract fixes, R/L_T,crit = "
                f"{reference['R_over_LT_crit']:.2f} against the Cyclone linear "
                f"value of about {reference['literature_R_over_LT']:.1f}."
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
        "literature_comparison": reference,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}")
    return 0 if threshold.get("resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
