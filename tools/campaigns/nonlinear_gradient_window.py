"""Where does the nonlinear adjoint stop being a gradient? Measure it.

Derivatives of long-time averages of chaotic systems are ill-conditioned as
initial-value problems: the adjoint grows with the leading Lyapunov exponent, so
backpropagating through a long trajectory returns a number that is large,
reproducible, and meaningless. GKX's production answer is to differentiate only
the last ``N`` steps from a state already saturated -- biased, but bounded.

That only helps if ``N`` is chosen below the divergence. This tool measures the
divergence directly. Every rung starts from the *same* detached saturated state
and differentiates :func:`gkx.nonlinear_heat_flux_window` -- the shipped entry
point, not a re-implementation -- through ``N`` steps of the production map. It
reports, per rung, the reverse-mode gradient, a centered finite difference of
the same function, and their relative disagreement. The largest rung that still
agrees is the divergence knee, and it is what
``gkx.solvers.nonlinear.state_integration.DIVERGENCE_KNEE_STEPS`` records.

This regenerates the AD/FD ladder and the knee behind ``docs/nonlinear_autodiff
.rst`` and the right-hand panel of ``docs/_static/nonlinear_autodiff_validation
.png``. It replaces a longer pre-1.7 campaign that drove a private sensitivity
module deleted in 612e1311; the claims are the same, the code path is now the
one users call.

Two traps it keeps guarding. ``TermConfig`` defaults ``nonlinear=0.0``, so a
hand-assembled config runs a *linear* case whose unbounded growth would
manufacture a divergence that has nothing to do with turbulence -- the
configuration therefore comes from a shipped TOML and the nonlinear term is
asserted on. And ``LinearParams`` defaults every dissipation coefficient to
zero, so a run with the hyperdiffusion *term* enabled but no coefficient has no
dissipation at all; that is asserted against too.

Example, ~40 min on one A4000 for the shipped 16x16x16 Cyclone case::

    python tools/campaigns/nonlinear_saturated_state.py --nx 16 --ny 16 --nz 16 \\
        --state-out tools_out/cyclone_saturated.npz
    python tools/campaigns/nonlinear_gradient_window.py \\
        --saturated-state tools_out/cyclone_saturated.npz \\
        --nx 16 --ny 16 --nz 16 --max-window 2048 --fd-step 1e-5 \\
        --output docs/_static/nonlinear_gradient_window.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from nonlinear_saturated_state import (
    _campaign_source_provenance,
    _gkx_source_tree_matches,
)


def _require_compatible_state_source(archive, provenance: dict[str, object]) -> str:
    """Require a clean state produced by the same installable GKX source."""
    if "gkx_git_commit" not in archive or "gkx_git_dirty" not in archive:
        raise SystemExit(
            "state has no GKX source provenance; regenerate it with "
            "nonlinear_saturated_state.py"
        )
    recorded = str(archive["gkx_git_commit"])
    recorded_dirty = int(archive["gkx_git_dirty"])
    current = str(provenance["git_commit"] or "")
    if recorded_dirty != 0 or provenance["git_dirty"] is not False:
        raise SystemExit("state and gradient campaign must both use clean Git sources")
    if not recorded or not current:
        raise SystemExit("state or gradient campaign has no Git commit identity")
    if recorded != current and not _gkx_source_tree_matches(
        provenance["repository_root"], recorded, current
    ):
        raise SystemExit(
            f"state GKX source {recorded} differs from current source {current}"
        )
    return recorded


def build_window_case(
    toml_path: Path, grid_override: dict | None = None
) -> dict[str, Any]:
    """Load a shipped nonlinear case as arguments for the production window."""

    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import ensure_flux_tube_geometry_data
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(toml_path)
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )
    grid = build_spectral_grid(runtime.grid)
    # Sample the geometry HERE, outside any trace. Inside jit every jnp
    # operation returns a tracer even when its inputs are constants, so an
    # analytic geometry resampled under jit hands the cache builder a traced
    # s_hat -- and a linked-boundary case then refuses to build, because the
    # integer twist-shift link map has no derivative with respect to shear.
    geometry = ensure_flux_tube_geometry_data(build_runtime_geometry(runtime), grid.z)
    # build_runtime_term_config, NOT build_runtime_linear_terms: the latter is
    # the linear term set and reports nonlinear=0.0 even for a nonlinear TOML.
    term_cfg = build_runtime_term_config(runtime)
    run_section = raw.get("run", {})
    n_laguerre = int(run_section["Nl"])
    n_hermite = int(run_section["Nm"])
    params = build_runtime_linear_params(runtime, Nm=n_hermite, geom=geometry)

    if float(term_cfg.nonlinear) == 0.0:
        raise SystemExit(
            f"{toml_path.name} has the nonlinear term off; this would measure a "
            "linear run whose unbounded growth is indistinguishable from a "
            "diverging adjoint"
        )
    dissipation = {
        name: float(np.max(np.abs(np.asarray(getattr(params, name)))))
        for name in ("nu", "nu_hyper", "nu_hyper_m", "nu_hyper_l", "D_hyper")
        if hasattr(params, name)
    }
    if not any(value > 0.0 for value in dissipation.values()):
        raise SystemExit(
            f"no active dissipation in {toml_path.name}: {dissipation}. A "
            "collisionless run at finite resolution piles energy at the grid "
            "scale and diverges for reasons that are not the adjoint's."
        )
    return {
        "case": toml_path.name,
        "grid": grid,
        "geometry": geometry,
        "params": params,
        "terms": term_cfg,
        "method": str(runtime.time.method),
        "dissipation": dissipation,
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "shape": (
            len(runtime.species),
            n_laguerre,
            n_hermite,
            grid.ky.size,
            grid.kx.size,
            grid.z.size,
        ),
    }


def make_window(
    case: dict[str, Any], state, dt: float, window: int, *, checkpoint: bool = True
):
    """Return ``scale -> mean heat flux`` over ``window`` production steps.

    The differentiable design parameter is a uniform multiplier on the
    per-species ``a/L_T`` drive: a scalar an optimizer would actually perturb,
    and one whose gradient is a single number a ladder can plot.
    """

    import jax.numpy as jnp

    from gkx.solvers.nonlinear.state_integration import nonlinear_heat_flux_window

    base_drive = jnp.asarray(case["params"].tprim)

    def evaluate(scale):
        params = dataclasses.replace(case["params"], tprim=base_drive * scale)
        return nonlinear_heat_flux_window(
            state,
            case["grid"],
            case["geometry"],
            params,
            dt=dt,
            steps=window,
            method=case["method"],
            terms=case["terms"],
            checkpoint=checkpoint,
            # The whole point of the ladder is to run past the knee, so the
            # runtime guard the knee feeds is switched off here rather than
            # printing a warning on every rung it is measuring.
            divergence_knee_steps=None,
        )

    return evaluate


def measure_ladder(case, state, dt: float, windows, fd_step: float, tolerance: float):
    """Differentiate nested prefixes of one trajectory and difference each one."""

    import jax
    import jax.numpy as jnp

    rows: list[dict[str, Any]] = []
    for window in windows:
        started = time.time()
        evaluate = make_window(case, state, dt, window)
        value, gradient = jax.value_and_grad(evaluate)(jnp.asarray(1.0))
        row = {
            "window": int(window),
            "objective": float(value),
            "gradient": float(gradient),
            "abs_gradient": abs(float(gradient)),
            "seconds": time.time() - started,
        }
        if fd_step > 0.0:
            step = jnp.asarray(fd_step)
            centered = float(
                (evaluate(1.0 + step) - evaluate(1.0 - step)) / (2.0 * step)
            )
            row["centered_fd_gradient"] = centered
            row["ad_fd_relative_error"] = abs(float(gradient) - centered) / max(
                abs(centered), 1.0e-300
            )
            row["agrees"] = bool(row["ad_fd_relative_error"] <= tolerance)
        rows.append(row)
        print(
            f"  N={window:>5d}  <Q>={row['objective']:.6e}  "
            f"|dQ/dscale|={row['abs_gradient']:.6e}"
            + (
                ""
                if fd_step <= 0.0
                else f"  FD={row['centered_fd_gradient']:+.6e}"
                f"  rel={row['ad_fd_relative_error']:.2e}"
                f"  {'ok' if row['agrees'] else 'DIVERGED'}"
            )
            + f"   [{row['seconds']:.1f}s]",
            flush=True,
        )
        if not np.isfinite(row["gradient"]):
            print("  gradient became non-finite; stopping the ladder", flush=True)
            break
    return rows


def fit_tail(rows, dt: float) -> dict[str, Any]:
    """Report both growth models rather than assuming the exponential one.

    Below the knee a windowed adjoint grows as a POWER of ``N`` while the
    perturbation still propagates coherently; beyond it, EXPONENTIALLY in time.
    Fitting only the exponential returns a confident rate for data that is a
    straight line on log-log, announcing a divergence that is not there.
    """

    finite = [r for r in rows if np.isfinite(r["abs_gradient"]) and r["abs_gradient"]]
    if len(finite) < 3:
        return {}
    tail = finite[-max(3, len(finite) // 2) :]
    times = np.array([r["window"] * dt for r in tail])
    values = np.log(np.array([r["abs_gradient"] for r in tail]))
    exp_fit = np.polyfit(times, values, 1)
    pow_fit = np.polyfit(np.log(times), values, 1)
    exp_res = float(np.std(values - np.polyval(exp_fit, times)))
    pow_res = float(np.std(values - np.polyval(pow_fit, np.log(times))))
    better = "power_law" if pow_res < exp_res else "exponential"
    print(
        f"\ntail fits: power law |grad| ~ N^{float(pow_fit[0]):.3f} "
        f"(residual {pow_res:.3e}); exponential rate {float(exp_fit[0]):+.4f}/time "
        f"(residual {exp_res:.3e}) -> {better} fits better",
        flush=True,
    )
    if better == "power_law":
        print(
            "  Power law means the adjoint has NOT diverged over this range: no "
            "knee here, and the usable window extends at least this far.",
            flush=True,
        )
    return {
        "tail_power_law_exponent": float(pow_fit[0]),
        "tail_power_law_residual": pow_res,
        "tail_exponential_rate_per_time": float(exp_fit[0]),
        "tail_exponential_residual": exp_res,
        "tail_better_model": better,
    }


def locate_knee(rows) -> dict[str, Any]:
    """The knee is the last rung whose adjoint still tracks the difference."""

    graded = [r for r in rows if "agrees" in r]
    if not graded:
        return {"divergence_knee_steps": None, "knee_bracket": None}
    last_ok = None
    first_bad = None
    for row in graded:
        if row["agrees"] and first_bad is None:
            last_ok = row["window"]
        elif not row["agrees"] and first_bad is None:
            first_bad = row["window"]
    return {
        "divergence_knee_steps": last_ok,
        "knee_bracket": None if first_bad is None else [last_ok, first_bad],
    }


def build_parser() -> argparse.ArgumentParser:
    """Command line for the ladder; exposed so contracts can read the defaults."""

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
    parser.add_argument(
        "--saturated-state",
        type=Path,
        required=True,
        help="npz from nonlinear_saturated_state.py, which reaches saturation "
        "with the production CFL-adaptive stepper; a fixed-step loop cannot, "
        "because the ExB CFL tightens as the amplitude grows",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="step for the differentiated window; defaults to the adaptive step "
        "recorded in the state file, which is the step that trajectory was "
        "produced with",
    )
    parser.add_argument("--allow-unsaturated", action="store_true")
    parser.add_argument("--min-window", type=int, default=64)
    parser.add_argument("--max-window", type=int, default=2048)
    parser.add_argument(
        "--fd-step",
        type=float,
        default=1.0e-5,
        help="centered drive-scale step; 1e-5 is what the tracked ladder in "
        "docs/_static/nonlinear_heat_flux_gradient_window_rk3.json used",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,
        help="relative AD-vs-FD disagreement above which a rung counts as "
        "diverged; the knee is the last rung below it",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    import gkx
    import jax
    import jax.numpy as jnp

    source_provenance = _campaign_source_provenance(gkx.__file__)
    print(
        f"GKX source: {source_provenance['source_file']} "
        f"commit={source_provenance['git_commit']} "
        f"dirty={source_provenance['git_dirty']}",
        flush=True,
    )
    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    override = {
        key: value
        for key, value in (("Nx", args.nx), ("Ny", args.ny), ("Nz", args.nz))
        if value is not None
    }
    case = build_window_case(args.toml, override or None)
    print(
        f"case: {case['case']}  (Nl,Nm)=({case['n_laguerre']},{case['n_hermite']})  "
        f"method={case['method']}  dissipation={case['dissipation']}",
        flush=True,
    )

    archive = np.load(args.saturated_state)
    state_source_commit = _require_compatible_state_source(archive, source_provenance)
    state = jnp.asarray(archive["state"])
    saturated = bool(archive["saturated"])
    recorded_dt = float(archive["adaptive_dt"]) if "adaptive_dt" in archive else None
    recorded_method = str(archive["method"]) if "method" in archive else None
    if recorded_method is not None and recorded_method != case["method"]:
        raise SystemExit(
            f"state was produced with {recorded_method}, but {case['case']} "
            f"configures {case['method']}; refusing to differentiate a different map"
        )
    if tuple(state.shape) != tuple(case["shape"]):
        raise SystemExit(
            f"state shape {tuple(state.shape)} does not match the case built "
            f"from {case['case']} ({tuple(case['shape'])}); the grid overrides "
            "must match those used to produce the state"
        )
    if not saturated and not args.allow_unsaturated:
        raise SystemExit(
            "refusing to measure a gradient ladder on an unsaturated state: the "
            "curve would report linear growth, not a chaotic adjoint. Rerun "
            "nonlinear_saturated_state.py for longer, or pass "
            "--allow-unsaturated to record it as uninterpretable."
        )
    dt = args.dt if args.dt is not None else recorded_dt
    if dt is None:
        raise SystemExit(
            "state file records no adaptive_dt and --dt was not given; pass the "
            "step that trajectory was produced with"
        )
    if (
        args.dt is not None
        and recorded_dt is not None
        and abs(dt - recorded_dt) > 1e-12
    ):
        print(
            f"  WARNING: --dt {dt:g} differs from the state's recorded "
            f"{recorded_dt:g}; the window is integrated on a different step "
            "from the trajectory it starts on",
            flush=True,
        )
    # The production stepper pins its state dtype from the seed, so a
    # single-precision seed writes a complex64 saturated state even under
    # JAX_ENABLE_X64. Differentiating it as-is silently runs the whole ladder in
    # single precision: the adjoint still looks fine, but the centered
    # difference is roundoff at any step small enough to be a derivative, and
    # every rung reports a spurious AD/FD disagreement that reads as a knee.
    window_dtype = jnp.result_type(
        state.dtype, jnp.complex64, *jax.tree_util.tree_leaves(case["params"])
    )
    if state.dtype != window_dtype:
        print(
            f"  state dtype {state.dtype} != window dtype {window_dtype}; "
            "casting the window up",
            flush=True,
        )
        state = state.astype(window_dtype)
    print(
        f"state {args.saturated_state.name}: shape={tuple(state.shape)} "
        f"dtype={state.dtype} dt={dt:g} "
        f"-> {'SATURATED' if saturated else 'NOT SATURATED'}",
        flush=True,
    )

    if args.min_window < 1 or args.max_window < args.min_window:
        raise SystemExit("require 1 <= --min-window <= --max-window")
    windows = [
        1 << k
        for k in range(int(np.log2(args.max_window)) + 1)
        if args.min_window <= (1 << k) <= args.max_window
    ]
    rows = measure_ladder(
        case, state, dt, windows, float(args.fd_step), float(args.tolerance)
    )
    summary = {
        "kind": "nonlinear_gradient_window",
        "claim_level": (
            "production_heat_flux_windowed_discrete_adjoint_not_infinite_time_gradient"
        ),
        "objective": "post_saturation_production_heat_flux_window_mean",
        "entry_point": "gkx.nonlinear_heat_flux_window",
        "source_provenance": source_provenance,
        "saturated_state_source_commit": state_source_commit,
        "case": case["case"],
        "grid_override": override,
        "dissipation": case["dissipation"],
        "method": case["method"],
        "dt": dt,
        "dt_source": "state file" if args.dt is None else "command line",
        "saturated_state": str(args.saturated_state),
        "saturated": saturated,
        "interpretable": saturated,
        "finite_difference_step": float(args.fd_step),
        "agreement_tolerance": float(args.tolerance),
        "window_dtype": str(state.dtype),
        "rows": rows,
        **fit_tail(rows, dt),
        **locate_knee(rows),
    }
    knee = summary.get("divergence_knee_steps")
    print(f"\nmeasured divergence knee: {knee} steps", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
