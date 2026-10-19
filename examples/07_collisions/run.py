"""Compare every collision operator GKX ships on one linear Cyclone ITG case.

The models form a hierarchy, from the cheapest diagonal relaxation to the full
linearized Coulomb operator with finite-Larmor-radius effects:

``lenard_bernstein``
    Conserving diagonal Lenard-Bernstein/Dougherty relaxation.
``sugama`` / ``improved_sugama``
    Drift-kinetic Sugama models (Frei, Ernst & Ricci 2022, Appendix C).
``landau``
    Linearized Landau (Coulomb) operator, drift-kinetic limit, the default.
``coulomb_finite_kperp``
    Gyrokinetic Coulomb operator with finite perpendicular wavelength
    (Frei, Ball, Hoffmann, Jorge, Ricci & Stenger 2021, equations 3.47-3.50).

Loads ``case.toml``, solves the linear problem once per model at collisionality
``NU``, prints the growth rates, saves them as JSON, and plots them. Under a
minute on a laptop CPU. ``case_full.toml`` runs the same physics to
t_max = 120.
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/07_collisions")
MODELS = (
    "lenard_bernstein",
    "sugama",
    "improved_sugama",
    "landau",
    "coulomb_finite_kperp",
)
NU = 0.05  # ion collisionality nu a / v_ti

case = gkx.load(CASE)
rows = []
for model in MODELS:
    local = case.replace(
        species=tuple(replace(species, nu=NU) for species in case.species),
        time=replace(case.time, collision_operator=model),
    )
    result = gkx.solve(
        local,
        ky_target=case.run.ky,
        Nl=case.run.Nl,
        Nm=case.run.Nm,
        solver=case.run.solver,
        mode_method="z_index",
        show_progress=True,
    )
    rows.append({"collision_operator": model, "nu": NU, **result.summary()})
    print(f"{model:<22} gamma = {result.gamma:+.5f}  omega = {result.omega:+.5f}")

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(
    json.dumps({"ky": case.run.ky, "rows": rows}, indent=2) + "\n"
)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(MODELS, [row["gamma"] for row in rows], color="tab:blue")
ax.set(
    xlabel=r"$\gamma\,a/v_{ti}$",
    title=f"Cyclone ITG growth rate, $\\nu$ = {NU}, $k_y\\rho_i$ = {case.run.ky}",
)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "collisions.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/collisions.png")
