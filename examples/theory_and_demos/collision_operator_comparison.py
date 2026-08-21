"""Compare every collision operator GKX ships, on one linear ITG case.

The models form a physical hierarchy, from the cheapest diagonal relaxation to
the full linearized Coulomb operator with finite-Larmor-radius effects:

``lenard_bernstein``
    Conserving diagonal Lenard-Bernstein/Dougherty relaxation.
``sugama`` / ``improved_sugama``
    Drift-kinetic Sugama models, conservative by construction
    (Frei, Ernst & Ricci 2022, Appendix C).
``coulomb``
    Drift-kinetic linearized Coulomb (Landau) operator, equations (C9a)-(C9f).
``coulomb_finite_kperp``
    Gyrokinetic Coulomb operator retaining finite perpendicular wavelength
    (Frei, Ball, Hoffmann, Jorge, Ricci & Stenger 2021, equations 3.47-3.50).
    Its ``k_perp -> 0`` limit is the drift-kinetic Coulomb operator.

Run::

    JAX_ENABLE_X64=1 python examples/theory_and_demos/collision_operator_comparison.py
    JAX_ENABLE_X64=1 python examples/theory_and_demos/collision_operator_comparison.py --nu-scan
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.plotting import set_plot_style
from gkx.config import CycloneBaseCase
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.params import COLLISION_OPERATOR_NAMES, LinearParams
from gkx.solvers.time.runners import integrate_linear_from_config

# The drift-kinetic tables are the eight-moment truncation of the reference
# papers, so the run must use Nl * Nm = 8.
HERMITE_COUNT = 4
LAGUERRE_COUNT = 2
# "none" and "lenard_bernstein" both keep the built-in diagonal term.
MODELS = tuple(name for name in COLLISION_OPERATOR_NAMES if name != "none")


def growth_rate(potential: np.ndarray, dt: float, stride: int) -> float:
    """Fit an exponential growth rate over the last 40% of the trace."""

    amplitude = np.abs(potential).reshape(potential.shape[0], -1).max(axis=1)
    times = np.arange(amplitude.size) * dt * stride
    window = slice(int(0.6 * amplitude.size), amplitude.size)
    return float(np.polyfit(times[window], np.log(amplitude[window]), 1)[0])


def run_case(model: str, collisionality: float, t_max: float = 2.0) -> dict:
    """Integrate the Cyclone base case with one collision operator."""

    config = CycloneBaseCase()
    grid = build_spectral_grid(config.grid)
    geometry = SAlphaGeometry.from_config(config.geometry)
    parameters = LinearParams(nu=collisionality)
    time_config = dataclasses.replace(
        config.time,
        # The moment operators run on the fixed-step cached integrator.
        use_diffrax=False,
        method="rk2",
        t_max=t_max,
        dt=0.002,
        progress_bar=False,
        collision_operator=model,
    )

    state = jnp.zeros(
        (HERMITE_COUNT, LAGUERRE_COUNT, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex128,
    )
    state = state.at[0, 0, 1, 0, :].set(1.0e-3)
    final_state, potential = integrate_linear_from_config(
        state, grid, geometry, parameters, time_config
    )
    return {
        "collision_operator": model,
        "nu": collisionality,
        "growth_rate": growth_rate(
            np.asarray(potential), time_config.dt, time_config.sample_stride
        ),
        "final_state_norm": float(jnp.linalg.norm(final_state)),
    }


def collisionality_scan(values: tuple[float, ...]) -> list[dict]:
    return [run_case(model, nu) for nu in values for model in MODELS]


def plot_scan(rows: list[dict], output: Path) -> None:
    """Plot growth rate against collisionality and the error against Coulomb."""

    set_plot_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    # The two Sugama variants nearly coincide, so give each model a distinct
    # dash pattern and marker rather than letting one hide the other.
    styles = {
        "lenard_bernstein": {"linestyle": "-", "marker": "o"},
        "sugama": {"linestyle": "-", "marker": "s", "linewidth": 3.0, "alpha": 0.45},
        "improved_sugama": {"linestyle": "--", "marker": "^"},
        "coulomb": {"linestyle": "-", "marker": "d"},
        "coulomb_finite_kperp": {"linestyle": "-", "marker": "*", "markersize": 11},
    }

    def series(model: str) -> tuple[np.ndarray, np.ndarray]:
        selected = sorted(
            (row for row in rows if row["collision_operator"] == model),
            key=lambda row: row["nu"],
        )
        return (
            np.array([row["nu"] for row in selected]),
            np.array([row["growth_rate"] for row in selected]),
        )

    for model in MODELS:
        nu_values, growth = series(model)
        left.plot(nu_values, growth, label=model, **styles[model])
    left.set_xscale("log")
    left.set_xlabel(r"collisionality $\nu$")
    left.set_ylabel(r"growth rate $\gamma$")
    left.set_title("Collisional damping of the Cyclone ITG mode")
    left.legend(frameon=False, fontsize="small")

    # The finite-Larmor Coulomb operator is the most complete model shipped, so
    # it is the natural reference for the reduced ones.
    reference_nu, reference = series("coulomb_finite_kperp")
    for model in MODELS:
        if model == "coulomb_finite_kperp":
            continue
        nu_values, growth = series(model)
        right.plot(nu_values, growth - reference, label=model, **styles[model])
    right.axhline(0.0, color="0.4", linewidth=0.8)
    right.set_xscale("log")
    right.set_xlabel(r"collisionality $\nu$")
    right.set_ylabel(r"$\gamma - \gamma_{\mathrm{Coulomb,\ finite\ }k_\perp}$")
    right.set_title("Reduced-model error against finite-Larmor Coulomb")
    right.legend(frameon=False, fontsize="small")

    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument(
        "--nu-scan",
        action="store_true",
        help="scan collisionality and plot the damping of each model",
    )
    parser.add_argument("--out", type=Path, default=Path("collision_comparison.png"))
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if arguments.nu_scan:
        values = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
        rows = collisionality_scan(values)
        plot_scan(rows, arguments.out)
    else:
        rows = [run_case(model, arguments.nu) for model in MODELS]
        width = max(len(model) for model in MODELS)
        print(f"Cyclone ITG, nu = {arguments.nu}, (Nl, Nm) = "
              f"({HERMITE_COUNT}, {LAGUERRE_COUNT})\n")
        print(f"{'collision_operator':<{width}}  {'growth rate':>12}  {'|G|':>14}")
        for row in rows:
            print(
                f"{row['collision_operator']:<{width}}  "
                f"{row['growth_rate']:>+12.6f}  {row['final_state_norm']:>14.6e}"
            )

    if arguments.json is not None:
        arguments.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
