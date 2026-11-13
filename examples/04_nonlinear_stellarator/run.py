"""Nonlinear ITG turbulence in a quasi-helically symmetric stellarator.

Fast path: ``case.toml`` is the HSX-like equilibrium of example 02 (flux tube at
s = 0.64 from a VMEC wout; solved once with vmex and written to ``examples/vmec/``
if missing). ``gkx.solve`` runs it with the deck's ``[run]`` block and no other tuning. There is no research path:
the tutorial run stops before saturation. ``case_full.toml`` is the 96x96x48
production deck (a GPU run).

Prints the free energy and the ion heat flux at the final time. Writes
``summary.json``, the run files and ``nonlinear_stellarator.png`` to
``outputs/04_nonlinear_stellarator``. About a minute on a laptop CPU once the
wout exists.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
VMEC_INPUT = Path(__file__).parents[1] / "vmec" / "input.NuhrenbergZille_1988_QHS"
OUTPUT = Path("outputs/04_nonlinear_stellarator")

# ---- Fast path ----
case = gkx.load(CASE)
wout = Path(case.geometry.vmec_file)
if (
    not wout.exists()
):  # vmex exits with status 2 on a loose tolerance but still writes it
    print(f"solving {VMEC_INPUT.name} with vmex to create {wout.name}")
    subprocess.run(
        [sys.executable, "-m", "vmex", VMEC_INPUT.name],
        cwd=VMEC_INPUT.parent,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
if not wout.exists():
    raise SystemExit(f"{wout.name} is missing: pip install vmex, then rerun")

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
    "wout": wout.name,
    **result.summary(),
    "free_energy_final": float(free_energy[-1]),
    "heat_flux_final": float(heat_flux[-1]),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "nonlinear_stellarator.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/nonlinear_stellarator.png")
