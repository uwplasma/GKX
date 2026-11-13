"""Collision operators on one linear Cyclone ITG case.

Fast path: ``case.toml`` (Cyclone ITG with the ``landau`` operator, the
default) solved with ``gkx.solve`` and the deck's ``[run]`` block.

Research path: the operators form a hierarchy, from a diagonal relaxation to the
linearized Coulomb operator with finite Larmor radius, and ``MODELS`` solves
each at every collisionality in ``NU``.

``lenard_bernstein``
    Conserving diagonal Lenard-Bernstein/Dougherty relaxation.
``landau``
    Linearized Landau (Coulomb) operator, drift-kinetic limit.
``sugama`` and ``improved_sugama``
    The Sugama et al. (2009) model and its 2019 improvement with the exact
    Landau heat-flux and friction coefficients. Same cost as ``landau``.
``coulomb_finite_kperp``
    Gyrokinetic Coulomb operator with finite perpendicular wavelength (Frei et
    al. 2021, equations 3.47-3.50).

The first four models use the Krylov eigensolver at ``(Nl, Nm) = (4, 8)``; the
tabulated finite-k_perp operator needs its table's ``(2, 4)`` basis and the
fixed-step integrator. ``case_full.toml`` runs the same physics to t_max = 120.

Prints gamma and omega per model and collisionality. Writes ``summary.json`` and
``collisions.png`` to ``outputs/07_collisions``. A few minutes on a laptop CPU.
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

# ---- Fast path ----
case = gkx.load(CASE)
run = dict(
    ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)
default = gkx.solve(case, **run, show_progress=True)
print(f"default operator: gamma = {default.gamma:+.5f}, omega = {default.omega:+.5f}")

# ---- Research path: every operator at every collisionality ----
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
            **{**run, "Nl": nl, "Nm": nm, "solver": solver},
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

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {"ky": case.run.ky, "default": default.summary(), "rows": rows}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

gkx.set_plot_style()
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
fig.tight_layout()
fig.savefig(OUTPUT / "collisions.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/collisions.png")
