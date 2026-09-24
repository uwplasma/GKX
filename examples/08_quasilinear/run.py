"""Quasilinear heat-flux spectrum for the Cyclone ITG case.

Loads ``case.toml`` (Cyclone ITG with the ``[quasilinear]`` diagnostic on),
runs a linear ``k_y`` scan, and reports at each ``k_y`` the quasilinear heat-flux
weight Q/|phi|^2 and the mixing-length saturated estimate
``gamma / <k_perp^2>`` times that weight. Prints the spectrum, saves it as
JSON and CSV, and plots gamma, the weight, and the saturated flux. About ten
seconds on a laptop CPU. The mixing-length rule is uncalibrated: it ranks
spectra and designs, it does not predict absolute nonlinear flux.
``case_full.toml`` uses the Krylov eigensolver at (Nl, Nm) = (8, 8).
The companion ``implicit_sensitivity.py`` differentiates these observables.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/08_quasilinear")
KY = [0.1, 0.2, 0.3, 0.4]  # k_y rho_i

case = gkx.load(CASE)
scan = gkx.scan(case, KY, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver)
weight = np.array([point["heat_flux_weight_total"] for point in scan.quasilinear])
kperp2 = np.array([point["kperp_eff2"] for point in scan.quasilinear])
flux = np.array([point["saturated_heat_flux_total"] for point in scan.quasilinear])

print(f"{'ky':>6} {'gamma':>9} {'<kperp^2>':>10} {'Q/|phi|^2':>10} {'Q_ML':>9}")
for row in zip(scan.ky, scan.gamma, kperp2, weight, flux):
    print("{:6.2f} {:9.5f} {:10.4f} {:10.4f} {:9.4f}".format(*row))
print(f"k_y-summed mixing-length heat flux (uncalibrated): {flux.sum():.4f}")

OUTPUT.mkdir(parents=True, exist_ok=True)
scan.save(OUTPUT / "scan")
summary = {
    "case": CASE.name,
    "claim_level": scan.quasilinear[0]["metadata"]["claim_level"],
    "ky": np.asarray(scan.ky).tolist(),
    "gamma": np.asarray(scan.gamma).tolist(),
    "kperp_eff2": kperp2.tolist(),
    "heat_flux_weight": weight.tolist(),
    "saturated_heat_flux": flux.tolist(),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
panels = (
    (scan.gamma, r"$\gamma\,a/v_{ti}$", "Growth rate"),
    (weight, r"$Q_i/|\phi|^2$", "Quasilinear weight"),
    (flux, r"$Q_i^{ML}$ (uncalibrated)", "Mixing-length flux"),
)
for ax, (values, label, title) in zip(axes, panels):
    ax.plot(scan.ky, values, "o-")
    ax.set(xlabel=r"$k_y\rho_i$", ylabel=label, title=title)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "quasilinear.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/quasilinear.png")
