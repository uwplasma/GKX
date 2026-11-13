"""Linear ITG growth rates in a quasi-helically symmetric stellarator.

Fast path: ``case.toml`` is an HSX-like equilibrium (Nuhrenberg and Zille 1988)
whose flux tube at s = 0.64 is cut from a VMEC wout. If the wout is missing it
is solved once with vmex (``pip install vmex``, several minutes on a CPU) and
written to ``examples/vmec/``. ``gkx.scan`` gives the rates; the eigenfunction
comes from a time run at the fastest-growing ``k_y``, long enough for the mode
to emerge from the initial transient. There is no research path here: the
tutorial resolution makes the rates qualitative, and ``case_full.toml`` is the
96x96x48 quasilinear-audit deck.

Prints gamma and omega for each ``k_y``. Writes ``summary.json``, the scan files
and ``linear_stellarator.png`` to ``outputs/02_linear_stellarator``. One to two
minutes on a laptop CPU once the wout exists.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
VMEC_INPUT = Path(__file__).parents[1] / "vmec" / "input.NuhrenbergZille_1988_QHS"
OUTPUT = Path("outputs/02_linear_stellarator")
KY = [0.2, 0.4, 0.6, 0.8]  # k_y rho_i
PEAK_T_MAX = 120.0  # time-run length for the eigenfunction

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
    Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)  # the deck's [run] block
scan = gkx.scan(case, KY, **run, show_progress=True)
peak_ky = float(scan.ky[int(np.argmax(scan.gamma))])
peak = gkx.solve(
    case.replace(time=replace(case.time, t_max=PEAK_T_MAX)),
    ky_target=peak_ky,
    **{**run, "solver": "time"},
    show_progress=True,
)
for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
    print(f"ky = {ky:.2f}: gamma = {gamma:+.5f}, omega = {omega:+.5f}")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
scan.save(OUTPUT / "scan")
summary = {
    "case": CASE.name,
    "wout": wout.name,
    "ky": np.asarray(scan.ky).tolist(),
    "gamma": np.asarray(scan.gamma).tolist(),
    "omega": np.asarray(scan.omega).tolist(),
    "peak": peak.summary(),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

gkx.set_plot_style()
fig, (ax_gamma, ax_mode) = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax_gamma.plot(scan.ky, scan.gamma, "o-", label=r"$\gamma$")
ax_gamma.plot(scan.ky, scan.omega, "s--", label=r"$\omega$")
ax_gamma.set(
    xlabel=r"$k_y\rho_i$", ylabel=r"rate $\times\,a/v_{ti}$", title="HSX-like QHS, ITG"
)
ax_gamma.legend(frameon=False)
mode = np.abs(np.asarray(peak.eigenfunction))
ax_mode.plot(peak.z, mode / mode.max())
ax_mode.set(
    xlabel="field-line coordinate",
    ylabel=r"$|\phi|/|\phi|_{\max}$",
    title=f"Eigenfunction, $k_y\\rho_i$={peak_ky:.1f}",
)
fig.tight_layout()
fig.savefig(OUTPUT / "linear_stellarator.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/linear_stellarator.png")
