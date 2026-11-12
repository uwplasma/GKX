"""Nonlinear turbulence with kinetic electrons: a tokamak and a stellarator.

Runs ``case.toml`` (Cyclone base case, s-alpha) and ``case_stellarator.toml``
(Landreman-Paul precise QA, VMEC flux tube at s = 0.64) with kinetic ions and
electrons at the physical mass ratio and beta = 1e-4. Each species keeps its own
Laguerre and Hermite counts (``[[species]] Nl``, ``Nm`` in the deck); the script
prints them with the cost against uniform counts and runs the tokamak with both,
whose heat fluxes agree. The decks name no stepper:
at this beta the default is the adaptive ``imex-ark3``, which treats parallel
streaming and the field response implicitly, so the step is set by the ion
drift and the bracket and not by the electron streaming rate. The tokamak is
also run with two pinned steppers (advanced options): explicit ``rk4-pi``, whose
step is set by electron streaming, which the script prints next to the ion drift
rate, and fixed-step ``imex-ars3``. It saves the per-species heat-flux traces and
the step history and plots them. The stellarator needs a wout, solved once with
vmex when it is missing (``pip install vmex``; minutes on a CPU) and skipped when
neither exists. About a minute per case on a laptop CPU at the tutorial horizon
(t = 0.3, the start of the linear phase). Set ``T_MAX`` to 150 or more to reach
the saturated state shown in the README figure; see the decks for production
resolution and for how t_max was chosen.
"""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

HERE = Path(__file__).parent
CASES = {
    "tokamak": HERE / "case.toml",
    "tokamak, uniform counts": HERE / "case.toml",
    "tokamak, explicit rk4-pi": HERE / "case.toml",
    "tokamak, imex-ars3": HERE / "case.toml",
    "stellarator": HERE / "case_stellarator.toml",
}
# Advanced: pinned steppers. ``imex-ars3`` solves the same implicit linear step at a
# fixed dt; 0.05 reproduced the explicit run's saturated heat flux on this deck at
# Nx = Ny = 16 and 32.
PINNED = {
    "tokamak, explicit rk4-pi": {"method": "rk4-pi"},
    "tokamak, imex-ars3": {
        "method": "imex-ars3",
        "fixed_dt": True,
        "dt": 0.05,
        "dt_max": 0.05,
    },
}
VMEC_INPUT = HERE.parent / "vmec" / "input.LandremanPaul2021_QA_lowres"
OUTPUT = Path("outputs/05_kinetic_electrons")
T_MAX = None  # None keeps each deck's t_max

OUTPUT.mkdir(parents=True, exist_ok=True)
summary, traces = {}, {}
for label, path in CASES.items():
    case = gkx.load(path)
    if (
        "uniform" in label
    ):  # every species at the [run] counts, the largest of any species
        case = case.replace(
            species=tuple(replace(s, Nl=None, Nm=None) for s in case.species)
        )
    if label in PINNED:
        case = case.replace(time=replace(case.time, **PINNED[label]))
    if T_MAX is not None:
        case = case.replace(time=replace(case.time, t_max=T_MAX))
    if case.geometry.model == "vmec":
        wout = Path(case.geometry.vmec_file)
        if not wout.exists():  # solve the equilibrium once (minutes on a CPU)
            if importlib.util.find_spec("vmex") is None:
                print(f"{label}: skipped, needs {wout.name} (pip install vmex)")
                continue
            print(f"solving {VMEC_INPUT.name} with vmex to create {wout.name}")
            # vmex exits with status 2 on a loose tolerance but still writes the wout
            subprocess.run(
                [sys.executable, "-m", "vmex", VMEC_INPUT.name],
                cwd=VMEC_INPUT.parent,
                env={
                    **os.environ,
                    "JAX_PLATFORMS": "cpu",
                },  # CPU: no GPU memory contention
            )
            if not wout.exists():
                raise SystemExit(f"vmex did not write {wout.name}")
    kept = {s.name: (s.Nl or case.run.Nl, s.Nm or case.run.Nm) for s in case.species}
    cost = sum(nl * nm for nl, nm in kept.values())  # moments per grid point
    uniform = len(kept) * case.run.Nl * case.run.Nm
    counts = ", ".join(f"{n} {nl} x {nm}" for n, (nl, nm) in kept.items())
    print(
        f"{label}: {counts} (Nl x Nm); {cost} moments per grid point, {uniform / cost:.2f}x fewer than uniform ({uniform})"
    )
    result = gkx.solve(
        case, ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, show_progress=True
    )
    d = result.diagnostics
    t, dt = np.asarray(d.t), np.asarray(d.dt_t)
    q = np.asarray(d.heat_flux_species_t)  # (time, species): ion, electron
    scales = np.asarray(d.cfl_scales if d.cfl_scales is not None else [], float)[:3]
    # A run without the explicit CFL controller reports no bound.
    cfl = dict(zip(("drift_x", "drift_y", "streaming"), scales.tolist()))
    cfl = {k: v for k, v in cfl.items() if np.isfinite(v)}
    bound = (
        f"omega_stream = {cfl['streaming']:.0f} vs omega_drift = "
        f"{max(cfl['drift_x'], cfl['drift_y']):.2f}"
        if len(cfl) == 3
        else "implicit linear step"
    )
    print(
        f"  t_final = {t[-1]:.3f}, mean dt = {dt.mean():.2e}, "
        f"{bound}, Q_i = {q[-1, 0]:.3e}, Q_e = {q[-1, 1]:.3e}"
    )
    traces[label] = (t, q, dt)
    summary[label] = {
        **result.summary(),
        "moments_per_species": kept,
        "moments_per_grid_point": [cost, uniform],
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
