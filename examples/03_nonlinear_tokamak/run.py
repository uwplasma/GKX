"""Nonlinear Cyclone ITG turbulence in s-alpha tokamak geometry.

Fast path: load ``case.toml`` and call ``gkx.solve`` with the deck's
``[run]`` block and no other tuning. The deck
names no stepper, so the adaptive default applies (example 13 compares the
alternatives). The tutorial run stops before saturation, so its flux is not a
transport number.

Research path: the same box with ``[time] run_to = "saturation"``. The run stops
once the heat-flux window passes the saturation test (``saturation_t_min``, and a
standard error from batch means below ``saturation_rel_sem`` of the mean with no
drift) or at ``t_max``, and reports the verdict with the window mean and its
standard error. A transport number is that mean when the verdict is ``saturated``.
The 16x16x16 box with (Nl, Nm) = (2, 4) saturates, but at a flux far from the
literature; ``case_full.toml`` is the 64x64x24 production deck (minutes on a GPU).

Prints the free energy and the ion heat flux at the final time, then the saturation
verdict. Writes ``summary.json``, both runs and ``nonlinear_tokamak.png`` (free
energy of the tutorial run, heat flux and averaging window of the saturated one) to
``outputs/03_nonlinear_tokamak``. About ten seconds for the fast path and two minutes
for the research path on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/03_nonlinear_tokamak")
SAT_T_MAX = 1200.0  # cap of the run to saturation, a/v_ti
SAT_T_MIN = 400.0  # no saturated verdict before this time

# ---- Fast path ----
case = gkx.load(CASE)
result = gkx.solve(case, show_progress=True)
d = result.diagnostics
free_energy, heat_flux = np.asarray(d.Wg_t), np.asarray(d.heat_flux_t)
print(f"t_final = {float(d.t[-1]):.2f}")
print(f"free energy W_g: {free_energy[0]:.3e} -> {free_energy[-1]:.3e}")
print(f"ion heat flux Q_i at t_final: {heat_flux[-1]:.3e} (not a saturated value)")

# ---- Research path: run to saturation ----
saturated_case = case.replace(
    time=replace(
        case.time, run_to="saturation", t_max=SAT_T_MAX, saturation_t_min=SAT_T_MIN
    )
)
sat = gkx.solve(saturated_case, show_progress=True)
verdict = sat.summary()
t_sat, q_sat = np.asarray(sat.diagnostics.t), np.asarray(sat.diagnostics.heat_flux_t)
mean, sem = verdict["heat_flux_mean"], verdict["heat_flux_sem"]
print(f"saturation verdict: {verdict['saturation_status']} at t = {t_sat[-1]:.0f}")
if mean is not None and sem is not None:
    window = (verdict["window_tmin"], verdict["window_tmax"])
    print(
        f"Q_i over t in [{window[0]:.0f}, {window[1]:.0f}] = {mean:.3f} +/- {sem:.3f}"
    )

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
sat.save(OUTPUT / "saturated_run")
summary = {
    "case": CASE.name,
    **result.summary(),
    "free_energy_final": float(free_energy[-1]),
    "heat_flux_final": float(heat_flux[-1]),
    "saturation": {
        key: verdict[key]
        for key in (
            "saturated",
            "saturation_status",
            "heat_flux_mean",
            "heat_flux_sem",
            "window_tmin",
            "window_tmax",
        )
    },
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, (ax_energy, ax_flux) = plt.subplots(1, 2, figsize=(10.0, 3.8))
ax_energy.semilogy(d.t, free_energy)
ax_energy.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$W_g$", title="Tutorial run")
ax_flux.plot(t_sat, q_sat, color="0.5")
if mean is not None and sem is not None:
    ax_flux.axvspan(*window, color="tab:green", alpha=0.1, label="window")
    ax_flux.axhline(mean, color="tab:green", label=f"{mean:.2f} $\\pm$ {sem:.2f}")
    ax_flux.legend(frameon=False, fontsize="small")
ax_flux.set(
    xlabel=r"$t\,v_{ti}/a$",
    ylabel=r"$Q_i$",
    title=f"Run to saturation: {verdict['saturation_status']}",
)
fig.tight_layout()
fig.savefig(OUTPUT / "nonlinear_tokamak.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/nonlinear_tokamak.png")
