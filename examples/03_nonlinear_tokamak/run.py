"""Nonlinear Cyclone ITG turbulence in s-alpha tokamak geometry.

Loads ``case.toml``, integrates the nonlinear gyrokinetic system with streamed
diagnostics, prints the final free energy and ion heat flux, saves the
diagnostic time series, and plots energy, heat flux and growth-rate history.
About ten seconds on a laptop CPU at the tutorial resolution, which does not
reach saturation. ``case_full.toml`` is the 64x64x24 production deck: minutes
on a GPU, and it should be run to a saturated window before any flux is quoted.
"""

import json
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/03_nonlinear_tokamak")
STEPS = None  # None keeps the deck's [run] steps

case = gkx.load(CASE)
grid = case.grid
print(
    f"nonlinear Cyclone ITG on {grid.Nx}x{grid.Ny}x{grid.Nz}, (Nl, Nm) = ({case.run.Nl}, {case.run.Nm})"
)

result = gkx.solve(
    case,
    ky_target=case.run.ky,
    Nl=case.run.Nl,
    Nm=case.run.Nm,
    steps=STEPS or case.run.steps,
)
diagnostics = result.diagnostics
t = np.asarray(diagnostics.t)
heat_flux = np.asarray(diagnostics.heat_flux_t)
free_energy = np.asarray(diagnostics.Wg_t)

print(f"t_final = {t[-1]:.2f}, samples = {t.size}")
print(f"free energy W_g: {free_energy[0]:.3e} -> {free_energy[-1]:.3e}")
print(f"ion heat flux Q_i at t_final: {heat_flux[-1]:.3e} (not a saturated value)")

OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
summary = {
    "case": CASE.name,
    **result.summary(),
    "free_energy_final": float(free_energy[-1]),
    "heat_flux_final": float(heat_flux[-1]),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "nonlinear_tokamak.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/nonlinear_tokamak.png")
