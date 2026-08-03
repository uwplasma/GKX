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
    n_laguerre: int, n_hermite: int, nx: int, ny: int, nz: int
) -> dict[str, Any]:
    """Cyclone-like nonlinear flux tube with the nonlinear term verified on."""

    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams
    from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached
    from gkx.terms.config import TermConfig

    cfg = CycloneBaseCase(grid=GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=6.0, Ly=12.0))
    grid = build_spectral_grid(cfg.grid)
    geometry = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    # TermConfig() defaults nonlinear=0.0. A linear run grows forever and would
    # look exactly like a diverging adjoint, so this is asserted, not assumed.
    term_cfg = TermConfig(nonlinear=1.0, hyperdiffusion=1.0)
    if float(term_cfg.nonlinear) == 0.0:  # pragma: no cover - guard, not logic
        raise RuntimeError("nonlinear term is off; this would measure a linear run")

    cache = build_linear_cache(grid, geometry, params, Nl=n_laguerre, Nm=n_hermite)
    shape = (1, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)

    def rhs(state, drive):
        """RHS with the temperature-gradient drive as a differentiable input."""

        scaled = params.__class__(**{**vars(params), "R_over_LTi": drive})
        value, _fields = nonlinear_rhs_cached(state, cache, scaled, term_cfg)
        return value

    return {"rhs": rhs, "shape": shape, "drive": float(params.R_over_LTi)}


def heat_flux_proxy(state):
    """Fluctuation energy: a positive, state-only scalar that tracks transport.

    The production heat flux needs the field solve output; for a
    gradient-divergence curve any smooth positive functional of the saturated
    state has the same Lyapunov behaviour, and this one avoids threading the
    field state through the differentiated window.
    """

    import jax.numpy as jnp

    return jnp.real(jnp.vdot(state, state)) / state.size


def integrate(rhs, state, drive, dt: float, steps: int):
    """Fixed-step RK2 over ``steps``, returning the final state."""

    import jax
    import jax.numpy as jnp

    def body(carry, _):
        k1 = rhs(carry, drive)
        k2 = rhs(carry + dt * k1, drive)
        return carry + 0.5 * dt * (k1 + k2), None

    final, _ = jax.lax.scan(body, state, jnp.arange(steps))
    return final


def windowed_gradient(case, saturated, drive: float, dt: float, window: int):
    """d(flux)/d(drive) differentiating only the last ``window`` steps."""

    import jax

    detached = jax.lax.stop_gradient(saturated)

    def objective(value):
        final = integrate(case["rhs"], detached, value, dt, window)
        return heat_flux_proxy(final)

    value, gradient = jax.value_and_grad(objective)(drive)
    return float(value), float(gradient)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=4)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--saturate-steps", type=int, default=4000)
    parser.add_argument("--max-window", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    case = build_nonlinear_case(
        args.n_laguerre, args.n_hermite, args.nx, args.ny, args.nz
    )
    generator = np.random.default_rng(0)
    seed = 1.0e-3 * (
        generator.standard_normal(case["shape"])
        + 1j * generator.standard_normal(case["shape"])
    )
    state = jnp.asarray(seed)

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

    # Accept saturation only if the last quarter of the trace has stopped
    # trending: |mean(last quarter) - mean(previous quarter)| / mean < 25%.
    quarter = max(1, len(trace) // 4)
    late = np.mean([r["energy"] for r in trace[-quarter:]])
    prior = np.mean([r["energy"] for r in trace[-2 * quarter : -quarter]])
    drift = abs(late - prior) / max(abs(late), 1e-300)
    saturated_ok = bool(drift < 0.25)
    print(
        f"  energy drift over the last two quarters: {drift:.1%} "
        f"-> {'SATURATED' if saturated_ok else 'NOT SATURATED'}",
        flush=True,
    )
    if not saturated_ok:
        print(
            "  WARNING: still trending. The ladder below measures the growth "
            "phase, not turbulence, and must not be read as a Lyapunov result.",
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
            f"  N={window:>4d}  |dQ/d(R/LT)| = {abs(gradient):.6e}"
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
        "resolution": {
            "n_laguerre": args.n_laguerre,
            "n_hermite": args.n_hermite,
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
        },
        "dt": args.dt,
        "saturate_steps": args.saturate_steps,
        "saturated_energy": energy,
        "saturation_trace": trace,
        "saturation_drift": drift,
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
