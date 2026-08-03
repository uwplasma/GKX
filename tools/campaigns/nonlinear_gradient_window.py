"""Step N2: where does a nonlinear adjoint stop being a gradient?

Derivatives of long-time averages of chaotic systems are ill-conditioned as
initial-value problems: the adjoint grows with the leading Lyapunov exponent, so
backpropagating through a long trajectory returns a number that is large,
reproducible, and meaningless. The working alternative is to backpropagate only
the last ``N`` steps from a state already in the saturated regime, which is
biased but bounded.

That only helps if ``N`` is chosen below the divergence. This tool measures the
divergence directly: hold the trajectory fixed up to step ``T-N``, differentiate
the windowed heat-flux average through the final ``N`` steps, and report the
gradient against ``N``. A plateau followed by exponential growth locates the
usable window; the growth rate of the tail estimates the Lyapunov exponent that
causes it.

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
term is asserted on. And the state is detached at ``T-N`` with ``stop_gradient``
rather than by re-running a shorter trajectory, so every ``N`` differentiates the
same physical trajectory and the curve is not confounded by different histories.
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
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached
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
    base_drive = jnp.asarray(params.R_over_LTi)

    def rhs(state, scale):
        """RHS differentiable in a scalar multiplier on the R/L_T drive.

        R_over_LTi is per-species, so the differentiable design parameter is a
        uniform scale on it rather than the array itself -- a scalar an
        optimizer would actually perturb, and one whose gradient is a single
        number the divergence ladder can plot.
        """

        scaled = dataclasses.replace(params, R_over_LTi=base_drive * scale)
        out, _fields = nonlinear_rhs_cached(state, cache, scaled, term_cfg)
        return out

    return {
        "rhs": rhs,
        "shape": shape,
        "drive": 1.0,  # the differentiable parameter is a multiplier
        "base_drive": [float(v) for v in np.asarray(base_drive).ravel()],
        "dissipation": dissipation,
        "case": toml_path.name,
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
    }


def heat_flux_proxy(state):
    """Fluctuation energy: a positive, state-only scalar that tracks transport.

    The production heat flux needs the field solve output; for a
    gradient-divergence curve any smooth positive functional of the saturated
    state has the same Lyapunov behaviour, and this one avoids threading the
    field state through the differentiated window.
    """

    import jax.numpy as jnp

    return jnp.real(jnp.vdot(state, state)) / state.size


def integrate(rhs, state, drive, dt: float, steps: int, *, checkpoint: bool = False):
    """Fixed-step RK2 over ``steps``, returning the final state.

    ``checkpoint`` rematerializes each step in the backward pass instead of
    storing its residuals. Reverse mode through a plain scan keeps every
    intermediate, which is the standard memory wall for windowed adjoints -- at
    a 16x16x16 grid and N=2048 it asked for 5.4 GB and exhausted a 16 GB card.
    Remat trades that for recomputing the forward step, which is the right side
    of the trade when the alternative is not running at all.
    """

    import jax
    import jax.numpy as jnp

    def body(carry, _):
        k1 = rhs(carry, drive)
        k2 = rhs(carry + dt * k1, drive)
        return carry + 0.5 * dt * (k1 + k2), None

    step = jax.checkpoint(body) if checkpoint else body
    final, _ = jax.lax.scan(step, state, jnp.arange(steps))
    return final


def windowed_gradient(case, saturated, drive: float, dt: float, window: int):
    """d(flux)/d(drive) differentiating only the last ``window`` steps."""

    import jax

    detached = jax.lax.stop_gradient(saturated)

    def objective(value):
        final = integrate(case["rhs"], detached, value, dt, window, checkpoint=True)
        return heat_flux_proxy(final)

    value, gradient = jax.value_and_grad(objective)(drive)
    return float(value), float(gradient)


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
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--saturate-steps", type=int, default=4000)
    parser.add_argument("--max-window", type=int, default=512)
    parser.add_argument(
        "--min-growth",
        type=float,
        default=100.0,
        help="minimum energy growth over the seed before a run counts as "
        "saturated rather than merely flat",
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
    generator = np.random.default_rng(0)
    seed = 1.0e-3 * (
        generator.standard_normal(case["shape"])
        + 1j * generator.standard_normal(case["shape"])
    )
    state = jnp.asarray(seed)
    seed_energy = float(heat_flux_proxy(state))

    # Saturation is sampled, not assumed: a gradient-divergence curve measured
    # on a still-growing linear phase reports the growth rate, not the Lyapunov
    # exponent, and the two look identical on a log plot.
    print(f"saturating: {args.saturate_steps} steps of dt={args.dt}", flush=True)
    chunks = 20
    per_chunk = max(1, args.saturate_steps // chunks)
    trace = []
    started = time.time()
    saturated = state
    for chunk in range(chunks):
        saturated = integrate(case["rhs"], saturated, case["drive"], args.dt, per_chunk)
        energy = float(heat_flux_proxy(saturated))
        trace.append({"step": (chunk + 1) * per_chunk, "energy": energy})
        if not np.isfinite(energy):
            raise RuntimeError("saturation diverged; reduce dt or raise hyperdiffusion")
    energy = trace[-1]["energy"]
    print(
        f"  {time.time() - started:.1f}s, fluctuation energy {energy:.6e}", flush=True
    )

    # Saturation needs BOTH conditions. Flatness alone is not enough: a trace
    # still sitting at the seed amplitude, before the instability has grown, is
    # perfectly flat and would pass a drift test on its own. So also require the
    # energy to have grown well clear of the seed.
    quarter = max(1, len(trace) // 4)
    late = np.mean([r["energy"] for r in trace[-quarter:]])
    prior = np.mean([r["energy"] for r in trace[-2 * quarter : -quarter]])
    drift = abs(late - prior) / max(abs(late), 1e-300)
    growth_factor = energy / max(seed_energy, 1e-300)
    flat = bool(drift < 0.25)
    grown = bool(growth_factor > args.min_growth)
    saturated_ok = flat and grown
    print(
        f"  energy drift over the last two quarters: {drift:.1%} "
        f"({'flat' if flat else 'still trending'})",
        flush=True,
    )
    print(
        f"  growth over the seed: {growth_factor:.3g}x "
        f"({'grown' if grown else 'STILL AT SEED LEVEL'})",
        flush=True,
    )
    print(f"  -> {'SATURATED' if saturated_ok else 'NOT SATURATED'}", flush=True)
    if not saturated_ok:
        print(
            "  WARNING: not saturated. The ladder below does not measure "
            "turbulence and must not be read as a Lyapunov result.",
            flush=True,
        )

    windows = [2**k for k in range(int(np.log2(args.max_window)) + 1)]
    rows = []
    for window in windows:
        started = time.time()
        value, gradient = windowed_gradient(
            case, saturated, case["drive"], args.dt, window
        )
        rows.append(
            {
                "window": window,
                "objective": value,
                "gradient": gradient,
                "abs_gradient": abs(gradient),
                "seconds": time.time() - started,
            }
        )
        print(
            f"  N={window:>4d}  |dQ/d(drive scale)| = {abs(gradient):.6e}"
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
    if len(finite) >= 3:
        # Exponential tail fit over the largest half of the ladder.
        tail = finite[len(finite) // 2 :]
        slope, _ = np.polyfit(
            [r["window"] * args.dt for r in tail],
            np.log([r["abs_gradient"] for r in tail]),
            1,
        )
        growth = float(slope)
        print(f"\ntail growth rate of |gradient|: {growth:+.4f} per code time unit")
        print("(compare against the leading Lyapunov exponent of this case)")

    summary = {
        "kind": "nonlinear_gradient_window",
        "claim_level": "windowed_adjoint_divergence_curve_on_the_production_nonlinear_rhs",
        "case": case["case"],
        "grid_override": override,
        "dissipation": case["dissipation"],
        "dt": args.dt,
        "saturate_steps": args.saturate_steps,
        "saturated_energy": energy,
        "saturation_trace": trace,
        "saturation_drift": drift,
        "seed_energy": seed_energy,
        "growth_over_seed": growth_factor,
        "saturated": saturated_ok,
        "tail_growth_per_time": growth,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
