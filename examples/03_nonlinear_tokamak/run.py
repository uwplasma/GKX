"""Nonlinear Cyclone ITG turbulence in s-alpha tokamak geometry.

Fast path: load ``case.toml`` and call ``gkx.solve`` with the deck's
``[run]`` block and no other tuning. The deck
names no stepper, so the adaptive default applies (example 13 compares the
alternatives). There is no research path: the tutorial run stops before
saturation, so the printed flux is not a transport number. ``case_full.toml``
is the 64x64x24 production deck (minutes on a GPU); run it to a saturated window
before quoting a flux.

Prints the free energy and the ion heat flux at the final time. Writes
``summary.json``, the run files and ``nonlinear_tokamak.png`` (energy, heat flux
and growth-rate history) to ``outputs/03_nonlinear_tokamak``. About ten seconds
on a laptop CPU.
"""

import json
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/03_nonlinear_tokamak")

# ---- Fast path ----
case = gkx.load(CASE)
run = dict(
    ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm
)  # the deck's [run] block
result = gkx.solve(case, **run, show_progress=True)
d = result.diagnostics
free_energy, heat_flux = np.asarray(d.Wg_t), np.asarray(d.heat_flux_t)
print(f"t_final = {float(d.t[-1]):.2f}")
print(f"free energy W_g: {free_energy[0]:.3e} -> {free_energy[-1]:.3e}")
print(f"ion heat flux Q_i at t_final: {heat_flux[-1]:.3e} (not a saturated value)")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
summary = {
    "case": CASE.name,
    **result.summary(),
    "free_energy_final": float(free_energy[-1]),
    "heat_flux_final": float(heat_flux[-1]),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "nonlinear_tokamak.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/nonlinear_tokamak.png")
