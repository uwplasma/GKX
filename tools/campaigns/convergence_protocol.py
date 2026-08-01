"""Per-configuration convergence protocol with full provenance for publication.

Every scan point must be converged in each parameter that can change the
answer, and the evidence must be recorded rather than asserted. This module
runs a refinement ladder in one parameter at a time, holding the rest at their
current best, and reports the smallest value whose observable agrees with the
next-finer one to within tolerance.

The parameters, and why each is here:

``n_laguerre`` / ``n_hermite``
    Velocity-space resolution. Not optional: a measured growth rate shifted 33%
    between (8, 10) and (12, 16) on a QA boundary, so a value chosen for speed
    silently changes the physics. Published guidance for Cyclone-like ITG is
    (P, J) ~ (32, 16) with the optimal basis aspect ratio P ~ 2J.

``ntheta``
    Parallel resolution of the VMEC -> flux-tube mapping, which sets how well
    the drift and shear profiles along the field line are represented.

``npol`` (poloidal turns)
    Flux-tube length. Too short truncates the ballooning envelope of the mode
    and biases the growth rate; the tube must be long enough that the mode
    amplitude has decayed at the ends.

``s_index`` (flux surface)
    Radial location. Not a convergence parameter in the refinement sense -- the
    answer genuinely differs by surface -- but it must be reported, and the
    campaign has to sample it consistently across devices.

``alpha`` (field line)
    Which field line on the surface. In a stellarator, distinct field lines see
    genuinely different geometry, so a single alpha is a sample and not the
    flux-surface answer. Reported as a spread across alpha rather than reduced
    to one number.

``dt`` and the time window
    Nonlinear only. dt must satisfy the CFL condition of the SATURATED state,
    not the linear phase -- a step that is stable while the mode grows can go
    unstable once the nonlinearity bites. The window must start after the
    transient and be long enough that the running mean has stopped drifting.

Nothing here decides that a run is converged on the basis of a single
comparison: a ladder value is accepted only if it agrees with the next finer
value AND that finer value agrees with the one beyond it, so a coincidental
crossing does not pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class LadderResult:
    """Outcome of refining one parameter."""

    parameter: str
    values: list[Any]
    observables: list[float]
    relative_changes: list[float]
    converged_value: Any | None
    tolerance: float
    seconds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "values": [v if not isinstance(v, tuple) else list(v) for v in self.values],
            "observables": self.observables,
            "relative_changes": self.relative_changes,
            "converged_value": (
                self.converged_value
                if not isinstance(self.converged_value, tuple)
                else list(self.converged_value)
            ),
            "converged": self.converged_value is not None,
            "tolerance": self.tolerance,
            "seconds": self.seconds,
        }


def refine(
    parameter: str,
    ladder: Sequence[Any],
    evaluate: Callable[[Any], float],
    *,
    tolerance: float = 0.05,
    verbose: bool = True,
) -> LadderResult:
    """Evaluate a refinement ladder and pick the first converged rung.

    A rung ``i`` is accepted when ``|o[i] - o[i+1]| / |o[i+1]| < tolerance`` and
    the same holds for ``i+1`` against ``i+2``. Requiring two consecutive
    agreements is what stops a curve that happens to cross the tolerance band on
    its way somewhere else from being mistaken for a plateau.
    """

    observables: list[float] = []
    seconds: list[float] = []
    for value in ladder:
        started = time.time()
        observables.append(float(evaluate(value)))
        seconds.append(time.time() - started)
        if verbose:
            print(
                f"    {parameter}={value!r:>18}  obs={observables[-1]:+.6e}"
                f"  [{seconds[-1]:.1f}s]",
                flush=True,
            )

    changes = [
        abs(observables[i] - observables[i + 1]) / max(abs(observables[i + 1]), 1e-30)
        for i in range(len(observables) - 1)
    ]

    converged = None
    for index in range(len(changes) - 1):
        if changes[index] < tolerance and changes[index + 1] < tolerance:
            converged = ladder[index]
            break

    if verbose:
        state = (
            f"converged at {converged!r}" if converged is not None else "NOT CONVERGED"
        )
        print(f"    -> {parameter}: {state} (tol {tolerance:.0%})", flush=True)

    return LadderResult(
        parameter=parameter,
        values=list(ladder),
        observables=observables,
        relative_changes=changes,
        converged_value=converged,
        tolerance=tolerance,
        seconds=seconds,
    )


def field_line_spread(
    evaluate: Callable[[float], float], alphas: Sequence[float]
) -> dict[str, Any]:
    """Sample several field lines and report the spread.

    This is deliberately not a convergence ladder. In a stellarator the answer
    genuinely differs between field lines, so refining alpha does not converge to
    anything -- the honest summary is the distribution, plus how far a single
    alpha = 0 sample sits from the mean.
    """

    values = [float(evaluate(a)) for a in alphas]
    array = np.array(values)
    mean = float(array.mean())
    return {
        "alphas": list(alphas),
        "observables": values,
        "mean": mean,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "relative_spread": float(array.std(ddof=1) / max(abs(mean), 1e-30))
        if array.size > 1
        else 0.0,
        "alpha0_relative_bias": float((values[0] - mean) / max(abs(mean), 1e-30)),
    }


def provenance() -> dict[str, Any]:
    """Everything a reader needs to reproduce a number, recorded per run."""

    def _capture(command: list[str]) -> str | None:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return None

    import jax

    import gkx

    return {
        "gkx_version": getattr(gkx, "__version__", "unknown"),
        "git_commit": _capture(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_capture(["git", "status", "--porcelain"])),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def growth_evaluator(
    equilibrium,
    *,
    s_index: int,
    ky_index: int,
    r_over_lt: float,
    r_over_ln: float,
    eigensolver: str = "dense",
    adaptive_config: Any | None = None,
) -> Callable[..., float]:
    """Return a growth-rate callable over one equilibrium at chosen settings.

    The geometry is built through VMEX and the eigenbranch through GKX's own
    objective, rather than through ``vmex.turbulent_growth_rate``, because that
    helper hard-codes a dense ``eigvals`` reduction. Routing through
    ``solver_objective_vector_from_geometry`` is what lets one ladder run on
    either solver and be compared rung by rung.

    ``eigensolver="adaptive-propagator"`` never forms the operator, so it reaches
    truncations whose dense matrix does not fit in memory. It is opt-in: the
    dense path stays the default here so existing ladder results are unchanged.
    """

    import dataclasses

    from vmex.core import turbulence as turb

    from gkx.objectives.core import (
        _default_gradient_linear_params,
        solver_objective_vector_from_geometry,
    )

    params_linear = dataclasses.replace(
        _default_gradient_linear_params(),
        R_over_LTi=float(r_over_lt),
        R_over_Ln=float(r_over_ln),
    )

    def growth(
        *, n_laguerre: int, n_hermite: int, ntheta: int, alpha: float = 0.0
    ) -> float:
        geometry = turb.flux_tube_geometry(
            equilibrium.state,
            equilibrium.runtime,
            s_index=s_index,
            alpha=alpha,
            ntheta=ntheta,
        )
        values = solver_objective_vector_from_geometry(
            geometry,
            selected_ky_index=ky_index,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            params_linear=params_linear,
            eigensolver=eigensolver,
            adaptive_config=adaptive_config,
        )
        return float(values[0])  # SOLVER_OBJECTIVE_NAMES[0] == "gamma"

    return growth


def linear_convergence(
    equilibrium,
    *,
    s_index: int = 7,
    tolerance: float = 0.05,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    velocity_ladder: Sequence[tuple[int, int]] = (
        (2, 3),
        (4, 6),
        (8, 10),
        (12, 16),
        (16, 24),
    ),
    ntheta_ladder: Sequence[int] = (32, 48, 64, 96),
    ky_index: int = 1,
    r_over_lt: float = 6.9,
    r_over_ln: float = 2.2,
    eigensolver: str = "dense",
    adaptive_config: Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Converge the linear growth rate for one equilibrium."""

    best = {"n_laguerre": 8, "n_hermite": 10, "ntheta": 64}
    evaluate = growth_evaluator(
        equilibrium,
        s_index=s_index,
        ky_index=ky_index,
        r_over_lt=r_over_lt,
        r_over_ln=r_over_ln,
        eigensolver=eigensolver,
        adaptive_config=adaptive_config,
    )

    def growth(**overrides) -> float:
        return evaluate(**{**best, **overrides})

    if verbose:
        print("  linear: velocity-space ladder", flush=True)
    velocity = refine(
        "(n_laguerre, n_hermite)",
        velocity_ladder,
        lambda v: growth(n_laguerre=v[0], n_hermite=v[1]),
        tolerance=tolerance,
        verbose=verbose,
    )
    if velocity.converged_value is not None:
        best["n_laguerre"], best["n_hermite"] = velocity.converged_value

    if verbose:
        print("  linear: parallel-mapping ladder", flush=True)
    theta = refine(
        "ntheta",
        ntheta_ladder,
        lambda v: growth(ntheta=v),
        tolerance=tolerance,
        verbose=verbose,
    )
    if theta.converged_value is not None:
        best["ntheta"] = theta.converged_value

    if verbose:
        print("  linear: field-line spread", flush=True)
    spread = field_line_spread(lambda a: growth(alpha=a), alphas)

    return {
        "kind": "linear_convergence",
        "eigensolver": eigensolver,
        "s_index": s_index,
        "ky_index": ky_index,
        "r_over_lt": r_over_lt,
        "r_over_ln": r_over_ln,
        "ladders": {
            "velocity_space": velocity.to_dict(),
            "ntheta": theta.to_dict(),
        },
        "field_line_spread": spread,
        "converged_settings": dict(best),
        "fully_converged": (
            velocity.converged_value is not None and theta.converged_value is not None
        ),
        "provenance": provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--eigensolver",
        choices=("dense", "adaptive-propagator"),
        default="dense",
        help="dense forms the operator; adaptive-propagator does not and so "
        "reaches rungs whose matrix would not fit in memory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt

    print(f"equilibrium: {args.input}", flush=True)
    equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(args.input))
    report = linear_convergence(
        equilibrium,
        s_index=args.s_index,
        tolerance=args.tolerance,
        eigensolver=args.eigensolver,
    )
    report["input"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nconverged settings: {report['converged_settings']}")
    print(f"fully converged: {report['fully_converged']}")
    spread = report["field_line_spread"]
    print(
        f"field-line spread: {100 * spread['relative_spread']:.1f}%  "
        f"(alpha=0 bias {100 * spread['alpha0_relative_bias']:+.1f}%)"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
