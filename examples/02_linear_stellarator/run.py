"""Linear ITG growth-rate scan in a quasi-helically symmetric stellarator.

Loads ``case.toml``: an HSX-like quasi-helically symmetric equilibrium
(Nuhrenberg & Zille 1988) whose flux tube at s = 0.64 is sampled from a VMEC
wout through booz_xform_jax. If the wout is missing it is solved once with
vmex (``pip install vmex``; several minutes on a CPU) and written to
``examples/vmec/``. The script then scans ``k_y``, prints gamma and omega,
saves JSON and CSV, and plots the spectrum and the eigenfunction along the
field line. One to two minutes on a CPU once the wout exists; tutorial
resolution, so the rates are qualitative. ``case_full.toml`` is the
96x96x48 quasilinear-audit deck.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
VMEC_INPUT = Path(__file__).parents[1] / "vmec" / "input.NuhrenbergZille_1988_QHS"
OUTPUT = Path("outputs/02_linear_stellarator")
KY = [0.2, 0.4, 0.6, 0.8]  # k_y rho_i

case = gkx.load(CASE)
wout = Path(case.geometry.vmec_file)
if not wout.exists():
    import vmex
    from vmex import optimize

    print(f"solving {VMEC_INPUT.name} with vmex to create {wout.name}")
    equilibrium = optimize.solve_equilibrium(vmex.VmecInput.from_file(VMEC_INPUT))
    vmex.write_wout(str(wout), equilibrium.wout)
print(
    f"geometry {case.geometry.model}: {wout.name}, s = {case.geometry.torflux}, alpha = {case.geometry.alpha}"
)

scan = gkx.scan(case, KY, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver)
peak_ky = float(scan.ky[int(np.argmax(scan.gamma))])
peak = gkx.solve(
    case, ky_target=peak_ky, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)
for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
    print(f"ky = {ky:.2f}: gamma = {gamma:+.5f}, omega = {omega:+.5f}")

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
for ax in (ax_gamma, ax_mode):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "linear_stellarator.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/linear_stellarator.png")
