"""Restart a nonlinear run from its saved state and analyse the joined trace.

Fast path: the tutorial Cyclone nonlinear deck of example 03 is run in two legs.
The first returns its final distribution, which is written as a restart file; the
second starts from that file through ``[init] init_file``. The saturation test of
``gkx.diagnostics.saturation`` then classifies the joined ion heat-flux trace and
supplies the window mean with a batch-means standard error when the trace allows one.

Research path: an uninterrupted run over the same horizon is the reference. The
restart file is complex64, so agreement is to single precision. The step is fixed
because the adaptive controller's step history is not in the restart file. The last
streamed diagnostic sample of a run repeats the previous one, which draws a small
step at the join; the restarted state itself matches the reference. The tutorial
run is still growing, so the test returns a verdict without a window and the script
prints the plain mean over the last part of the trace with no error bar: a naive
standard error over correlated, unsaturated samples would be far too small. Run
longer (example 03 runs to saturation) and the same call returns the mean and its error.

Prints the restart difference, the saturation verdict and the averaged flux.
Writes ``summary.json``, both legs and ``restart_and_analysis.png`` to
``outputs/12_restart_and_analysis``. About twenty seconds on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx
from gkx.diagnostics.saturation import assess_saturation
from gkx.artifacts.io import write_netcdf_restart_state

CASE = Path(__file__).parents[1] / "03_nonlinear_tokamak" / "case.toml"
OUTPUT = Path("outputs/12_restart_and_analysis")
STEPS_PER_LEG = 100
AVERAGE_FRACTION = 0.5  # fraction of the joined trace used for the flux average

case = gkx.load(CASE)
case = case.replace(time=replace(case.time, dt=0.05, method="rk3", fixed_dt=True))
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---- Fast path: two legs joined by a restart file ----
first = gkx.solve(case, steps=STEPS_PER_LEG, return_state=True, show_progress=True)
restart_file = write_netcdf_restart_state(
    OUTPUT / "leg1.restart.bin", first.state, ny_full=case.grid.Ny
)
restarted = case.replace(init=replace(case.init, init_file=str(restart_file.resolve())))
second = gkx.solve(restarted, steps=STEPS_PER_LEG, show_progress=True)
first.save(OUTPUT / "leg1")
second.save(OUTPUT / "leg2")

t_first = np.asarray(first.diagnostics.t)
t_joined = np.concatenate([t_first, t_first[-1] + np.asarray(second.diagnostics.t)])
heat_joined = np.concatenate(
    [first.diagnostics.heat_flux_t, second.diagnostics.heat_flux_t]
)
energy_joined = np.concatenate([first.diagnostics.Wg_t, second.diagnostics.Wg_t])
verdict = assess_saturation(t_joined, heat_joined, t_min=0.0)
mean_flux, sem_flux = verdict.get("mean"), verdict.get("sem")
if mean_flux is None:  # no saturated window: the plain mean, no error bar
    window = t_joined >= t_joined[-1] * (1.0 - AVERAGE_FRACTION)
    mean_flux = float(np.mean(heat_joined[window]))
    t_window = (float(t_joined[window][0]), float(t_joined[-1]))
else:
    t_window = (verdict["window_tmin"], verdict["window_tmax"])
error = "no error bar" if sem_flux is None else f"+/- {sem_flux:.1e} (batch means)"
print(f"leg 1 ends at t = {t_first[-1]:.2f}; leg 2 restarts from {restart_file.name}")
print(f"saturation verdict: {verdict['status']}")
print(f"Q_i over t in [{t_window[0]:.1f}, {t_window[1]:.1f}]: {mean_flux:.3e}, {error}")

# ---- Research path: uninterrupted reference ----
reference = gkx.solve(case, steps=2 * STEPS_PER_LEG, show_progress=True)
final = reference.diagnostics.Wg_t[-1]
restart_error = abs(energy_joined[-1] - final) / abs(final)
print(
    f"final free energy: restarted {energy_joined[-1]:.8e}, uninterrupted {final:.8e}"
)
print(f"relative restart difference = {restart_error:.2e}")

# ---- Outputs ----
summary = {
    "case": str(CASE.relative_to(CASE.parents[1])),
    "steps_per_leg": STEPS_PER_LEG,
    "restart_relative_difference": float(restart_error),
    "saturation_status": verdict["status"],
    "heat_flux_window": list(t_window),
    "heat_flux_mean": mean_flux,
    "heat_flux_sem": sem_flux,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

gkx.set_plot_style()
fig, (ax_energy, ax_flux) = plt.subplots(1, 2, figsize=(10.0, 3.8))
for ax, joined, full, label in (
    (ax_energy, energy_joined, reference.diagnostics.Wg_t, r"$W_g$"),
    (ax_flux, heat_joined, reference.diagnostics.heat_flux_t, r"$Q_i$"),
):
    ax.plot(
        reference.diagnostics.t, full, color="0.6", linewidth=3.0, label="uninterrupted"
    )
    ax.plot(t_joined, joined, "--", color="tab:blue", label="leg 1 + restarted leg 2")
    ax.axvline(t_first[-1], color="tab:red", linestyle=":", label="restart")
    ax.set(xlabel=r"$t\,v_{ti}/a$", ylabel=label)
ax_flux.axvspan(
    *t_window,
    color="tab:green",
    alpha=0.1,
    label="averaging window",
)
ax_energy.set_title("Free energy")
ax_flux.set_title("Ion heat flux")
ax_flux.legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(OUTPUT / "restart_and_analysis.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/restart_and_analysis.png")
