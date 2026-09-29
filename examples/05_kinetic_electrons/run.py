"""Nonlinear turbulence with kinetic electrons: a tokamak and a stellarator.

Runs ``case.toml`` (Cyclone base case, s-alpha) and ``case_stellarator.toml``
(Landreman-Paul precise QA, VMEC flux tube at s = 0.64) with kinetic ions and
electrons at the physical mass ratio and beta = 1e-4. Both decks let the CFL
controller choose dt; with kinetic electrons it is set by electron parallel
streaming, and the script prints that bound next to the ion drift rate. It
saves the per-species heat-flux traces and the step history and plots them.
The stellarator needs a wout, solved once with vmex when it is missing
(``pip install vmex``; minutes on a CPU) and skipped when neither exists.
About a minute per case on a laptop CPU at the tutorial horizon (t = 0.3, the
start of the linear phase). Set ``T_MAX`` to 150 or more to reach the
saturated state shown in the README figure; see the decks for production
resolution and for how dt and t_max were chosen.
"""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

HERE = Path(__file__).parent
CASES = {"tokamak": HERE / "case.toml", "stellarator": HERE / "case_stellarator.toml"}
VMEC_INPUT = HERE.parent / "vmec" / "input.LandremanPaul2021_QA_lowres"
OUTPUT = Path("outputs/05_kinetic_electrons")
T_MAX = None  # None keeps each deck's t_max

OUTPUT.mkdir(parents=True, exist_ok=True)
summary, traces = {}, {}
for label, path in CASES.items():
    case = gkx.load(path)
    if T_MAX is not None:
        case = case.replace(time=replace(case.time, t_max=T_MAX))
    if case.geometry.model == "vmec":
        wout = Path(case.geometry.vmec_file)
        if not wout.exists():
            if importlib.util.find_spec("vmex") is None:
                print(f"{label}: skipped, needs {wout.name} or vmex to solve it")
                continue
            import vmex
            from vmex import optimize

            print(f"solving {VMEC_INPUT.name} with vmex to create {wout.name}")
            equilibrium = optimize.solve_equilibrium(
                vmex.VmecInput.from_file(VMEC_INPUT)
            )
            vmex.write_wout(str(wout), equilibrium.wout)
    result = gkx.solve(case, ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm)
    d = result.diagnostics
    t, dt = np.asarray(d.t), np.asarray(d.dt_t)
    q = np.asarray(d.heat_flux_species_t)  # (time, species): ion, electron
    scales = np.asarray(d.cfl_scales)[:3].tolist()
    cfl = dict(zip(("drift_x", "drift_y", "streaming"), scales))
    print(
        f"{label}: t_final = {t[-1]:.3f}, mean dt = {dt.mean():.2e}, "
        f"omega_stream = {cfl['streaming']:.0f} vs omega_drift = {max(cfl['drift_x'], cfl['drift_y']):.2f}, "
        f"Q_i = {q[-1, 0]:.3e}, Q_e = {q[-1, 1]:.3e}"
    )
    traces[label] = (t, q, dt)
    summary[label] = {
        **result.summary(),
        "dt_mean": float(dt.mean()),
        "cfl_omega": cfl,
        "heat_flux_final": q[-1].tolist(),
    }

(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, (ax_q, ax_dt) = plt.subplots(1, 2, figsize=(10.0, 3.8))
for label, (t, q, dt) in traces.items():
    ax_q.plot(t, q[:, 0], label=f"{label}, ions")
    ax_q.plot(t, q[:, 1], "--", label=f"{label}, electrons")
    ax_dt.plot(t, dt, label=label)
ax_q.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$Q_s/Q_{GB}$", title="Heat flux")
ax_dt.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$\Delta t$", title="CFL-controlled step")
for ax in (ax_q, ax_dt):
    ax.legend(frameon=False, fontsize="small")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "kinetic_electrons.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/kinetic_electrons.png")
