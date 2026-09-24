"""Compare every collision operator GKX ships on one linear Cyclone ITG case.

The models form a hierarchy, from the cheapest diagonal relaxation to the full
linearized Coulomb operator with finite-Larmor-radius effects:

``lenard_bernstein``
    Conserving diagonal Lenard-Bernstein/Dougherty relaxation.
``sugama`` / ``improved_sugama``
    Drift-kinetic Sugama models (Frei, Ernst & Ricci 2022, Appendix C).
``coulomb``
    Drift-kinetic linearized Coulomb (Landau) operator, equations (C9a)-(C9f).
``coulomb_finite_kperp``
    Gyrokinetic Coulomb operator with finite perpendicular wavelength
    (Frei, Ball, Hoffmann, Jorge, Ricci & Stenger 2021, equations 3.47-3.50).

Loads ``case.toml``, solves the linear problem once per model at collisionality
``NU``, prints the growth rates, saves them as JSON, and plots them. Set
``NU_SCAN = True`` to repeat the table over ``NU_VALUES`` and plot the
collisional damping of each model against the finite-Larmor Coulomb reference
(the ``collision_operator_comparison.png`` figure). The single table takes
under a minute on a laptop CPU; the scan multiplies that by the number of
collisionalities. ``case_full.toml`` runs the same physics to t_max = 120.
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/07_collisions")
MODELS = (
    "lenard_bernstein",
    "sugama",
    "improved_sugama",
    "coulomb",
    "coulomb_finite_kperp",
)
NU = 0.05  # ion collisionality nu a / v_ti for the single table
NU_SCAN = False  # True: repeat the table for every value in NU_VALUES
NU_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

case = gkx.load(CASE)
rows = []
for nu in NU_VALUES if NU_SCAN else (NU,):
    for model in MODELS:
        local = case.replace(
            species=tuple(replace(species, nu=nu) for species in case.species),
            time=replace(case.time, collision_operator=model),
        )
        result = gkx.solve(
            local,
            ky_target=case.run.ky,
            Nl=case.run.Nl,
            Nm=case.run.Nm,
            solver=case.run.solver,
            mode_method="z_index",
        )
        rows.append({"collision_operator": model, "nu": nu, **result.summary()})
        print(
            f"nu = {nu:<6} {model:<22} gamma = {result.gamma:+.5f}  omega = {result.omega:+.5f}"
        )

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(
    json.dumps({"ky": case.run.ky, "rows": rows}, indent=2) + "\n"
)

if NU_SCAN:
    fig, (ax_gamma, ax_error) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    reference = np.array(
        [row["gamma"] for row in rows if row["collision_operator"] == MODELS[-1]]
    )
    for marker, model in zip("osd^*", MODELS):
        gamma = np.array(
            [row["gamma"] for row in rows if row["collision_operator"] == model]
        )
        ax_gamma.plot(NU_VALUES, gamma, marker=marker, label=model)
        if model != MODELS[-1]:
            ax_error.plot(NU_VALUES, gamma - reference, marker=marker, label=model)
    ax_gamma.set(
        xscale="log",
        xlabel=r"$\nu\,a/v_{ti}$",
        ylabel=r"$\gamma\,a/v_{ti}$",
        title="Collisional damping of the ITG mode",
    )
    ax_error.axhline(0.0, color="0.4", linewidth=0.8)
    ax_error.set(
        xscale="log",
        xlabel=r"$\nu\,a/v_{ti}$",
        ylabel=r"$\gamma-\gamma_{\mathrm{Coulomb},k_\perp}$",
        title="Error against finite-Larmor Coulomb",
    )
    axes = (ax_gamma, ax_error)
else:
    fig, ax_bar = plt.subplots(figsize=(7.5, 3.8))
    ax_bar.barh(MODELS, [row["gamma"] for row in rows], color="tab:blue")
    ax_bar.set(
        xlabel=r"$\gamma\,a/v_{ti}$",
        title=f"Cyclone ITG growth rate, $\\nu$ = {NU}, $k_y\\rho_i$ = {case.run.ky}",
    )
    axes = (ax_bar,)
for ax in axes:
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(OUTPUT / "collisions.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/collisions.png")
