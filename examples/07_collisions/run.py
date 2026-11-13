"""Compare every collision operator GKX ships on one linear Cyclone ITG case.

The models form a hierarchy, from the cheapest diagonal relaxation to the full
linearized Coulomb operator with finite-Larmor-radius effects:

``lenard_bernstein``
    Conserving diagonal Lenard-Bernstein/Dougherty relaxation.
``landau``
    Linearized Landau (Coulomb) operator, drift-kinetic limit, the default.
``sugama`` / ``improved_sugama``
    Sugama et al. (2009) model, and its 2019 improvement with the exact Landau
    heat-flux and friction coefficients. Same cost as ``landau``.
``coulomb_finite_kperp``
    Gyrokinetic Coulomb operator with finite perpendicular wavelength
    (Frei, Ball, Hoffmann, Jorge, Ricci & Stenger 2021, equations 3.47-3.50).

Loads ``case.toml`` and solves the linear problem once per model and
collisionality ``NU``. The first four models solve for the eigenvalue at
``(Nl, Nm) = (4, 8)``; the tabulated finite-k_perp operator needs its table's
``(2, 4)`` basis and the fixed-step integrator. Prints the growth rates, saves
them as JSON, and plots them. A few minutes on a laptop CPU, no GPU needed;
``case_full.toml`` runs the same physics to t_max = 120.
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
    "landau",
    "sugama",
    "improved_sugama",
    "coulomb_finite_kperp",
)
NU = (0.05, 0.5)  # ion collisionality nu a / v_ti
BASIS = (4, 8)  # (Nl, Nm) of the eigenvalue solves
TABULATED = {"coulomb_finite_kperp": (2, 4, "time")}  # (Nl, Nm, solver)

case = gkx.load(CASE)
rows = []
for nu in NU:
    for model in MODELS:
        nl, nm, solver = TABULATED.get(model, (*BASIS, "krylov"))
        local = case.replace(
            species=tuple(replace(species, nu=nu) for species in case.species),
            time=replace(case.time, collision_operator=model),
        )
        result = gkx.solve(
            local,
            ky_target=case.run.ky,
            Nl=nl,
            Nm=nm,
            solver=solver,
            mode_method="z_index",
            show_progress=True,
        )
        rows.append(
            {"collision_operator": model, "nu": nu, "Nl": nl, "Nm": nm}
            | result.summary()
        )
        print(
            f"nu = {nu:<5} {model:<22} gamma = {result.gamma:+.5f}  omega = {result.omega:+.5f}"
        )

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(
    json.dumps({"ky": case.run.ky, "rows": rows}, indent=2) + "\n"
)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
height = 0.8 / len(NU)
for i, nu in enumerate(NU):
    gamma = [row["gamma"] for row in rows if row["nu"] == nu]
    ax.barh(np.arange(len(MODELS)) + i * height, gamma, height, label=rf"$\nu$ = {nu}")
ax.set_yticks(np.arange(len(MODELS)) + 0.4 - height / 2, MODELS)
ax.set(
    xlabel=r"$\gamma\,a/v_{ti}$",
    title=f"Cyclone ITG growth rate, $k_y\\rho_i$ = {case.run.ky}",
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "collisions.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/collisions.png")
