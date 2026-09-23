"""Linear Cyclone ITG with kinetic electrons.

Loads ``case.toml`` (two kinetic species, low beta), solves one linear
initial-value problem at ``k_y rho_i = 0.3``, and compares it with the same
case with adiabatic electrons. Prints both growth rates, saves a JSON
summary, and plots the potential traces and eigenfunctions. About ten seconds
on a laptop CPU. At the tutorial horizon (t = 10) both fits still carry the
start-up transient, so the two rates are qualitative; compare them at
``case_full.toml`` resolution (the reference-aligned deck, minutes on a GPU).
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/05_kinetic_electrons")
KY = 0.3  # k_y rho_i

kinetic = gkx.load(CASE)
ion = kinetic.species[0]
adiabatic = kinetic.replace(
    species=(ion,),
    physics=replace(
        kinetic.physics,
        adiabatic_electrons=True,
        electromagnetic=False,
        electrostatic=True,
        use_apar=False,
    ),
    init=replace(kinetic.init, init_electrons_only=False),
)

results = {}
for label, case in (("kinetic electrons", kinetic), ("adiabatic electrons", adiabatic)):
    results[label] = gkx.solve(
        case,
        ky_target=KY,
        Nl=case.run.Nl,
        Nm=case.run.Nm,
        solver=case.run.solver,
        fit_signal="phi",
    )
    print(
        f"{label:>20}: gamma = {results[label].gamma:+.4f}, omega = {results[label].omega:+.4f}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {label: result.summary() for label, result in results.items()}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, (ax_trace, ax_mode) = plt.subplots(1, 2, figsize=(10.0, 3.8))
for label, result in results.items():
    signal = np.abs(np.asarray(result.signal))
    ax_trace.semilogy(
        result.t, signal / signal[0], label=f"{label}, $\\gamma$={result.gamma:.3f}"
    )
    mode = np.asarray(result.eigenfunction)
    ax_mode.plot(result.z, np.abs(mode) / np.abs(mode).max(), label=label)
ax_trace.set(
    xlabel=r"$t\,v_{ti}/a$",
    ylabel=r"$|\phi|/|\phi(0)|$",
    title=f"Growth at $k_y\\rho_i$={KY}",
)
ax_mode.set(
    xlabel=r"$\theta$", ylabel=r"$|\phi|/|\phi|_{\max}$", title="Eigenfunction envelope"
)
for ax in (ax_trace, ax_mode):
    ax.legend(frameon=False, fontsize="small")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "kinetic_electrons.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/kinetic_electrons.png")
