"""Step N2: where does a nonlinear adjoint stop being a gradient?

Derivatives of long-time averages of chaotic systems are ill-conditioned as
initial-value problems: the adjoint grows with the leading Lyapunov exponent, so
backpropagating through a long trajectory returns a number that is large,
reproducible, and meaningless. The working alternative is to backpropagate only
the last ``N`` steps from a state already in the saturated regime, which is
biased but bounded.

That only helps if ``N`` is chosen below the divergence. This tool measures the
divergence directly: start every rung from the same detached saturated state,
differentiate the heat-flux average through ``N`` production-map steps, and
report the gradient against ``N``. The windows are nested prefixes of one
physical trajectory. A plateau followed by exponential growth locates the usable
window; the growth rate of the tail estimates the Lyapunov exponent that causes
it.

The comparison that matters is against ``tau_ac`` from
``heat_flux_autocorrelation.py``: the published result for a differentiable
flux-tube gyrokinetic code is that the two coincide -- the usable gradient window
is the dynamical memory of the turbulence, not a solver tolerance. If GKX's knee
sits far *below* its ``tau_ac``, that is a numerical problem on top of the
physical one and has to be fixed before any windowed-adjoint scheme is worth
building.

Two traps this avoids explicitly. ``TermConfig`` defaults ``nonlinear=0.0``, so a
config assembled by hand runs a *linear* case that grows without bound and would
manufacture a divergence that has nothing to do with turbulence -- the nonlinear
term is asserted on. And the common saturated state is detached with
``stop_gradient``, so every ``N`` differentiates the same trajectory prefix and
the curve is not confounded by different histories.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def build_nonlinear_case(
    toml_path: Path, grid_override: dict | None = None
) -> dict[str, Any]:
    """Load a shipped production nonlinear case, overriding only the grid size.

    Assembling ``LinearParams`` and ``TermConfig`` by hand is how this tool first
    failed: ``TermConfig`` defaults ``nonlinear=0.0`` (a silently linear run) and
    ``LinearParams`` defaults ``nu = nu_hyper = 0.0``, so switching the
    hyperdiffusion *term* on multiplies a zero coefficient and leaves the run
    with no dissipation at all -- which duly blew up during saturation. The
    shipped TOMLs carry the coefficients that actually saturate this case
    (``D_hyper``, ``p_hyper*``, ``hypercollisions*``, end damping), so the
    configuration comes from there and only the grid is reduced for affordability.
    """

    import dataclasses

    import jax.numpy as jnp

    from gkx.core.grid import build_spectral_grid
    from gkx.diagnostics import fieldline_quadrature_weights, heat_flux_total
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.nonlinear.projection import _make_hermitian_projector
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached
    from gkx.solvers.nonlinear.explicit import advance_explicit_nonlinear_state
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(toml_path)
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )

    grid = build_spectral_grid(runtime.grid)
    geometry = build_runtime_geometry(runtime)
    # build_runtime_term_config, NOT build_runtime_linear_terms: the latter is
    # the linear term set and reports nonlinear=0.0 even for a nonlinear TOML.
    term_cfg = build_runtime_term_config(runtime)

    # Nl/Nm live in the raw [run] table rather than on RuntimeConfig.
    run_section = raw.get("run", {})
    n_laguerre = int(run_section["Nl"])
    n_hermite = int(run_section["Nm"])
    params = build_runtime_linear_params(runtime, Nm=n_hermite, geom=geometry)

    if float(term_cfg.nonlinear) == 0.0:
        raise RuntimeError(
            f"{toml_path.name} has the nonlinear term off; this would measure a "
            "linear run whose unbounded growth is indistinguishable from a "
            "diverging adjoint"
        )
    # Some coefficients are per-species or per-moment arrays, so summarise by
    # maximum magnitude rather than assuming a scalar.
    dissipation = {}
    for name in ("nu", "nu_hyper", "nu_hyper_m", "nu_hyper_l", "D_hyper"):
        if hasattr(params, name):
            dissipation[name] = float(np.max(np.abs(np.asarray(getattr(params, name)))))
    if not any(v > 0.0 for v in dissipation.values()):
        raise RuntimeError(
            f"no active dissipation in {toml_path.name}: {dissipation}. A "
            "collisionless run at finite resolution piles energy at the grid "
            "scale and diverges."
        )

    cache = build_linear_cache(grid, geometry, params, Nl=n_laguerre, Nm=n_hermite)
    shape = (1, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_drive = jnp.asarray(params.tprim)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geometry, grid)
    project_state = _make_hermitian_projector(
        np.asarray(cache.ky), int(np.asarray(cache.kx).size)
    )

    def rhs(state, scale):
        """RHS differentiable in a scalar multiplier on the a/L_T drive.

        tprim is per-species, so the differentiable design parameter is a
        uniform scale on it rather than the array itself -- a scalar an
        optimizer would actually perturb, and one whose gradient is a single
        number the divergence ladder can plot.
        """

        scaled = dataclasses.replace(params, tprim=base_drive * scale)
        out, _fields = nonlinear_rhs_cached(state, cache, scaled, term_cfg)
        return out

    integration_method = str(runtime.time.method)

    def window_functions(dt: float, *, differentiable: bool = False):
        """Return the configured production map and heat-flux observable.

        ``differentiable=True`` selects the primitive field solve needed by
        forward-over-reverse methods such as multiple-shooting JVPs.  The
        default retains the faster production custom VJP for reverse mode.
        """

        def scaled_params(scale):
            return dataclasses.replace(params, tprim=base_drive * scale[0])

        def step(state, scale):
            state = project_state(state)
            current_params = scaled_params(scale)

            def bound_rhs(local_state):
                return nonlinear_rhs_cached(
                    local_state,
                    cache,
                    current_params,
                    term_cfg,
                    differentiable=differentiable,
                )

            derivative, _fields = bound_rhs(state)
            return advance_explicit_nonlinear_state(
                state,
                derivative,
                jnp.asarray(dt, dtype=jnp.real(state).dtype),
                method=integration_method,
                rhs_fn=bound_rhs,
                project_state=project_state,
                state_dtype=state.dtype,
            )

        def objective(state, scale):
            current_params = scaled_params(scale)
            _derivative, fields = nonlinear_rhs_cached(
                state,
                cache,
                current_params,
                term_cfg,
                differentiable=differentiable,
            )
            apar = jnp.zeros_like(fields.phi) if fields.apar is None else fields.apar
            bpar = jnp.zeros_like(fields.phi) if fields.bpar is None else fields.bpar
            return heat_flux_total(
                state,
                fields.phi,
                apar,
                bpar,
                cache,
                grid,
                current_params,
                flux_factor,
            )

        return step, objective

    return {
        "rhs": rhs,
        "window_functions": window_functions,
        "shape": shape,
        "drive": 1.0,  # the differentiable parameter is a multiplier
        "base_drive": [float(v) for v in np.asarray(base_drive).ravel()],
        "dissipation": dissipation,
        "case": toml_path.name,
        "method": integration_method,
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
    }


def fluctuation_energy(state):
    """Mean square state amplitude: the objective this tool differentiates.

    Deliberately NOT called a heat flux. The production heat flux needs the
    field-solve output; this is a state-only functional that shares the
    trajectory's Lyapunov behaviour, which is all a divergence curve needs. Any
    claim about dQ/dp requires the real flux, and naming this one dQ would smuggle
    that claim in.
    """

    import jax.numpy as jnp

    return jnp.real(jnp.vdot(state, state)) / state.size


def windowed_gradient(step, objective, saturated, drive: float, window: int):
    """Differentiate the physical heat-flux mean over one saturated window."""

    import jax.numpy as jnp

    from gkx.solvers.nonlinear.sensitivity import discrete_window_value_and_grad

    value, gradient = discrete_window_value_and_grad(
        step,
        objective,
        saturated,
        jnp.asarray([drive]),
        steps=window,
        checkpoint=True,
    )
    return float(value), float(gradient[0])


def windowed_value(step, objective, saturated, drive: float, window: int) -> float:
    """Evaluate the same finite heat-flux window without a reverse pass."""

    import jax
    import jax.numpy as jnp

    from gkx.solvers.nonlinear.sensitivity import integrate_discrete_observable

    _final, values = integrate_discrete_observable(
        step,
        objective,
        jax.lax.stop_gradient(saturated),
        jnp.asarray([drive]),
        steps=window,
        checkpoint=False,
    )
    return float(jnp.mean(values))


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
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="step for the differentiated window. Defaults to the adaptive step "
        "recorded in the state file, which is the step that trajectory was "
        "produced with; overriding it rescales every t/tau_ac this tool reports",
    )
    parser.add_argument(
        "--saturated-state",
        type=Path,
        required=True,
        help="npz written by nonlinear_saturated_state.py, which reaches "
        "saturation with the production CFL-adaptive stepper; a fixed-step loop "
        "cannot, because the ExB CFL tightens as the amplitude grows",
    )
    parser.add_argument(
        "--allow-unsaturated",
        action="store_true",
        help="proceed even if the state was flagged NOT SATURATED (the ladder "
        "is then uninterpretable and is labelled as such)",
    )
    parser.add_argument("--min-window", type=int, default=1)
    parser.add_argument("--max-window", type=int, default=512)
    parser.add_argument(
        "--fd-step",
        type=float,
        default=0.0,
        help="optional centered-FD drive-scale step evaluated on each exact window",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    override = {
        k: v
        for k, v in (("Nx", args.nx), ("Ny", args.ny), ("Nz", args.nz))
        if v is not None
    }
    case = build_nonlinear_case(args.toml, override or None)
    print(
        f"case: {case['case']}  (Nl,Nm)=({case['n_laguerre']},{case['n_hermite']})  "
        f"dissipation={case['dissipation']}",
        flush=True,
    )
    # The saturated state comes from the production CFL-adaptive stepper via
    # nonlinear_saturated_state.py. This tool no longer integrates to saturation
    # itself: a fixed-step loop cannot get there, because the ExB CFL tightens
    # with amplitude and the scheme destabilises exactly when the nonlinearity
    # would otherwise saturate the run.
    archive = np.load(args.saturated_state)
    saturated = jnp.asarray(archive["state"])
    state_saturated = bool(archive["saturated"])
    recorded_dt = float(archive["adaptive_dt"]) if "adaptive_dt" in archive else None
    state_tau_ac = float(archive["tau_ac"]) if "tau_ac" in archive else float("nan")
    recorded_method = str(archive["method"]) if "method" in archive else None
    if recorded_method is not None and recorded_method != case["method"]:
        raise SystemExit(
            f"state was produced with {recorded_method}, but {case['case']} "
            f"configures {case['method']}; refusing to differentiate a different map"
        )
    if args.dt is None:
        if recorded_dt is None:
            raise SystemExit(
                "state file records no adaptive_dt and --dt was not given. "
                "Regenerate it with a current nonlinear_saturated_state.py, or "
                "pass the step that trajectory was produced with."
            )
        dt = recorded_dt
    else:
        dt = args.dt
        if recorded_dt is not None and abs(dt - recorded_dt) > 1e-12:
            print(
                f"  WARNING: --dt {dt:g} differs from the state's recorded "
                f"{recorded_dt:g}; the window is being integrated on a different "
                "step from the trajectory it starts on",
                flush=True,
            )
    print(
        f"loaded state from {args.saturated_state.name}: shape={saturated.shape} "
        f"t_end={float(archive['t_end']):.1f} "
        f"-> {'SATURATED' if state_saturated else 'NOT SATURATED'}",
        flush=True,
    )
    if saturated.shape != case["shape"]:
        raise RuntimeError(
            f"state shape {saturated.shape} does not match the case built from "
            f"{case['case']} ({case['shape']}); the grid overrides must match "
            "those used to produce the state"
        )
    if not state_saturated and not args.allow_unsaturated:
        raise SystemExit(
            "refusing to measure a gradient ladder on an unsaturated state: the "
            "curve would report linear growth, not a chaotic adjoint. Rerun "
            "nonlinear_saturated_state.py for longer, or pass --allow-unsaturated "
            "to record it as uninterpretable."
        )
    # The production stepper sets its state dtype from the seed via
    # result_type(G0, complex64), so a single-precision seed pins the whole
    # trajectory to complex64 even under JAX_ENABLE_X64 -- the saved state comes
    # back single precision. The differentiated window runs in the dtype this
    # case's RHS actually produces, and the mismatch is reported rather than
    # silently promoted inside the scan carry.
    probe_dtype = case["rhs"](saturated, 1.0).dtype
    if saturated.dtype != probe_dtype:
        print(
            f"  state dtype {saturated.dtype} != RHS dtype {probe_dtype}; "
            f"casting the window to {probe_dtype}",
            flush=True,
        )
        saturated = saturated.astype(probe_dtype)

    saturated_ok = state_saturated
    energy = float(fluctuation_energy(saturated))
    print(f"  fluctuation energy at the start of the window: {energy:.6e}", flush=True)
    step_fn, objective_fn = case["window_functions"](dt)

    if int(args.min_window) < 1 or int(args.max_window) < int(args.min_window):
        raise SystemExit("require 1 <= --min-window <= --max-window")
    windows = [
        2**k
        for k in range(int(np.log2(args.max_window)) + 1)
        if int(args.min_window) <= 2**k <= int(args.max_window)
    ]
    rows = []
    for window in windows:
        started = time.time()
        value, gradient = windowed_gradient(
            step_fn, objective_fn, saturated, case["drive"], window
        )
        row = {
            "window": window,
            "objective": value,
            "gradient": gradient,
            "abs_gradient": abs(gradient),
            "seconds": time.time() - started,
        }
        if float(args.fd_step) > 0.0:
            plus = windowed_value(
                step_fn,
                objective_fn,
                saturated,
                case["drive"] + float(args.fd_step),
                window,
            )
            minus = windowed_value(
                step_fn,
                objective_fn,
                saturated,
                case["drive"] - float(args.fd_step),
                window,
            )
            centered_fd = (plus - minus) / (2.0 * float(args.fd_step))
            row["centered_fd_gradient"] = centered_fd
            row["ad_fd_relative_error"] = abs(gradient - centered_fd) / max(
                abs(centered_fd), 1.0e-30
            )
        rows.append(row)
        print(
            f"  N={window:>4d}  Q={value:.6e}  "
            f"|d<Q>/d(drive scale)|={abs(gradient):.6e}"
            f"   [{rows[-1]['seconds']:.1f}s]",
            flush=True,
        )
        if not np.isfinite(gradient):
            print("  gradient became non-finite; stopping the ladder", flush=True)
            break

    finite = [
        r for r in rows if np.isfinite(r["abs_gradient"]) and r["abs_gradient"] > 0
    ]
    growth = None
    power = None
    if len(finite) >= 3:
        # Two models, both reported. A windowed adjoint below its knee grows as a
        # POWER of N while the perturbation still propagates coherently; beyond
        # the knee it grows EXPONENTIALLY in t. Fitting only the exponential
        # returns a confident rate for data that is a straight line on log-log,
        # which is what this ladder turned out to be -- so the fit would have
        # announced a divergence that is not there.
        tail = finite[-max(3, len(finite) // 2) :]
        times = np.array([r["window"] * dt for r in tail])
        values = np.log(np.array([r["abs_gradient"] for r in tail]))
        exp_fit = np.polyfit(times, values, 1)
        pow_fit = np.polyfit(np.log(times), values, 1)
        growth = float(exp_fit[0])
        power = float(pow_fit[0])
        exp_res = float(np.std(values - np.polyval(exp_fit, times)))
        pow_res = float(np.std(values - np.polyval(pow_fit, np.log(times))))
        better = "power law" if pow_res < exp_res else "exponential"
        print("\ntail of the ladder, both models fitted:")
        print(f"  power law    |grad| ~ N^{power:.3f}   residual {pow_res:.3e}")
        print(f"  exponential  rate {growth:+.4f}/time  residual {exp_res:.3e}")
        print(f"  -> {better} fits better")
        if better == "power law":
            print(
                "  Power law means the adjoint has NOT diverged over this range:"
                " no knee, and the usable window extends at least this far."
            )

    summary = {
        "kind": "nonlinear_gradient_window",
        "claim_level": "production_heat_flux_windowed_discrete_adjoint_not_infinite_time_gradient",
        "objective": "post_saturation_production_heat_flux_window_mean",
        "case": case["case"],
        "grid_override": override,
        "dissipation": case["dissipation"],
        "dt": dt,
        "method": case["method"],
        "dt_source": "state file" if args.dt is None else "command line",
        "tau_ac_from_state": state_tau_ac,
        "saturated_energy": energy,
        "finite_difference_step": float(args.fd_step),
        "saturated_state": str(args.saturated_state),
        "saturated": saturated_ok,
        "interpretable": saturated_ok,
        "window_dtype": str(probe_dtype),
        "tail_exponential_rate_per_time": growth,
        "tail_power_law_exponent": power,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
