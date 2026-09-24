"""Nonlinear ITG turbulence in a quasi-helically symmetric stellarator.

Loads ``case.toml``: the HSX-like quasi-helically symmetric equilibrium of
Nuhrenberg & Zille (1988), flux tube at s = 0.64 sampled from a VMEC wout.
If the wout is missing it is solved once with vmex (``pip install vmex``;
several minutes on a CPU) and
written to ``examples/vmec/``. The script integrates the nonlinear system with
streamed diagnostics, prints the final free energy and heat flux, saves the
diagnostics and a JSON summary, and plots the standard nonlinear panel. About
a minute on a laptop CPU once the wout exists; the tutorial run does not reach
saturation. ``case_full.toml`` is the 96x96x48 production deck (a GPU run).
"""

import json
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
VMEC_INPUT = Path(__file__).parents[1] / "vmec" / "input.NuhrenbergZille_1988_QHS"
OUTPUT = Path("outputs/04_nonlinear_stellarator")

case = gkx.load(CASE)
wout = Path(case.geometry.vmec_file)
if not wout.exists():
    import vmex
    from vmex import optimize

    print(f"solving {VMEC_INPUT.name} with vmex to create {wout.name}")
    equilibrium = optimize.solve_equilibrium(vmex.VmecInput.from_file(VMEC_INPUT))
    vmex.write_wout(str(wout), equilibrium.wout)
grid = case.grid
print(
    f"nonlinear ITG, {wout.name}, s = {case.geometry.torflux}, grid {grid.Nx}x{grid.Ny}x{grid.Nz}"
)

result = gkx.solve(
    case, ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, steps=case.run.steps
)
diagnostics = result.diagnostics
print(f"t_final = {float(diagnostics.t[-1]):.2f}")
print(
    f"free energy W_g: {float(diagnostics.Wg_t[0]):.3e} -> {float(diagnostics.Wg_t[-1]):.3e}"
)
print(
    f"ion heat flux at t_final: {float(diagnostics.heat_flux_t[-1]):.3e} (not a saturated value)"
)

OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
summary = {
    "case": CASE.name,
    "wout": wout.name,
    **result.summary(),
    "free_energy_final": float(np.asarray(diagnostics.Wg_t)[-1]),
    "heat_flux_final": float(np.asarray(diagnostics.heat_flux_t)[-1]),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "nonlinear_stellarator.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/nonlinear_stellarator.png")
