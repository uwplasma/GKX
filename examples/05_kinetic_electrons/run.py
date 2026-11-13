"""Nonlinear turbulence with kinetic ions and electrons, tokamak and stellarator.

Fast path: ``case.toml`` (Cyclone base case, s-alpha) and
``case_stellarator.toml`` (Landreman-Paul precise QA, VMEC flux tube at s = 0.64)
at the physical mass ratio and beta = 1e-4, run with ``gkx.solve`` and the deck's ``[run]`` block.
Each species keeps its own Laguerre and Hermite counts (``[[species]] Nl, Nm``);
the script prints them with their cost against uniform counts. The decks name no
stepper, so the default is the adaptive ``imex-ark3``: parallel streaming and
the field response are implicit, and the step follows the ion drift, not the
electron streaming rate. The stellarator wout is solved once with vmex if
missing (``pip install vmex``, minutes on a CPU) and the case is skipped when
neither exists.

Research path (``RESEARCH``): the tokamak at uniform counts, whose heat flux
agrees, and with two pinned steppers, explicit ``rk4-pi`` (step set by electron
streaming, printed next to the ion drift rate) and fixed-step ``imex-ars3``.
Both pins set ``damp_ends_rate = 0.1`` so they compare steppers at one operator;
the deck's own rate is 598, which would cut the ``rk4-pi`` step to 5e-4 because
rate * dt must stay below about 0.3. That rate is not converged for electrons
(docs/guidance.rst, End damping), so these runs compare steps, not heat fluxes.

Prints per-species moment counts, step sizes and final heat fluxes. Writes
``summary.json`` and ``kinetic_electrons.png`` to ``outputs/05_kinetic_electrons``.
About a minute per case on a laptop CPU at the tutorial horizon (t = 0.3, the
start of the linear phase); ``T_MAX = 150`` reaches the saturated state of the
README figure.
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

HERE = Path(__file__).parent
VMEC_INPUT = HERE.parent / "vmec" / "input.LandremanPaul2021_QA_lowres"
OUTPUT = Path("outputs/05_kinetic_electrons")
T_MAX = None  # None keeps each deck's t_max
GRID = {}  # overrides of the deck [grid] entries, e.g. {"nperiod": 1}
# label -> (deck, [time] overrides, uniform counts)
FAST = {
    "tokamak": ("case.toml", {}, False),
    "stellarator": ("case_stellarator.toml", {}, False),
}
RESEARCH = {
    "tokamak, uniform counts": ("case.toml", {}, True),
    "tokamak, explicit rk4-pi": (
        "case.toml",
        {"method": "rk4-pi", "damp_ends_rate": 0.1},
        False,
    ),
    "tokamak, imex-ars3": (
        "case.toml",
        # dt = 0.05 reproduced the explicit run's saturated heat flux at Nx = Ny = 16, 32
        {
            "method": "imex-ars3",
            "fixed_dt": True,
            "dt": 0.05,
            "dt_max": 0.05,
            "damp_ends_rate": 0.1,
        },
        False,
    ),
}

# ---- Fast path, then research path ----
summary, traces = {}, {}
for label, (deck, pin, uniform) in {**FAST, **RESEARCH}.items():
    case = gkx.load(HERE / deck)
    if uniform:  # every species at the [run] counts, the largest of any species
        case = case.replace(
            species=tuple(replace(s, Nl=None, Nm=None) for s in case.species)
        )
    case = case.replace(time=replace(case.time, **pin))
    case = case.replace(grid=replace(case.grid, **GRID))
    if T_MAX is not None:
        case = case.replace(time=replace(case.time, t_max=T_MAX))
    if case.geometry.model == "vmec":
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
            print(f"{label}: skipped, needs {wout.name} (pip install vmex)")
            continue
    kept = {s.name: (s.Nl or case.run.Nl, s.Nm or case.run.Nm) for s in case.species}
    cost = sum(nl * nm for nl, nm in kept.values())  # moments per grid point
    uniform_cost = len(kept) * case.run.Nl * case.run.Nm
    counts = ", ".join(f"{n} {nl} x {nm}" for n, (nl, nm) in kept.items())
    print(
        f"{label}: {counts} (Nl x Nm); {cost} moments per grid point, "
        f"{uniform_cost / cost:.2f}x fewer than uniform ({uniform_cost})"
    )
    result = gkx.solve(
        case, ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, show_progress=True
    )
    d = result.diagnostics
    t, dt = np.asarray(d.t), np.asarray(d.dt_t)
    q = np.asarray(d.heat_flux_species_t)  # (time, species): ion, electron
    # A run without the explicit CFL controller reports no bound.
    scales = np.asarray(d.cfl_scales if d.cfl_scales is not None else [], float)[:3]
    cfl = dict(zip(("drift_x", "drift_y", "streaming"), scales.tolist()))
    cfl = {k: v for k, v in cfl.items() if np.isfinite(v)}
    bound = (
        f"omega_stream = {cfl['streaming']:.0f} vs omega_drift = "
        f"{max(cfl['drift_x'], cfl['drift_y']):.2f}"
        if len(cfl) == 3
        else "implicit linear step"
    )
    print(
        f"  t_final = {t[-1]:.3f}, mean dt = {dt.mean():.2e}, {bound}, "
        f"Q_i = {q[-1, 0]:.3e}, Q_e = {q[-1, 1]:.3e}"
    )
    traces[label] = (t, q, dt)
    summary[label] = {
        **result.summary(),
        "moments_per_species": kept,
        "moments_per_grid_point": [cost, uniform_cost],
        "dt_mean": float(dt.mean()),
        "cfl_omega": cfl,
        "heat_flux_final": q[-1].tolist(),
    }

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, (ax_q, ax_dt) = plt.subplots(1, 2, figsize=(10.0, 3.8))
for label, (t, q, dt) in traces.items():
    ax_q.plot(t, q[:, 0], label=f"{label}, ions")
    ax_q.plot(t, q[:, 1], "--", label=f"{label}, electrons")
    ax_dt.plot(t, dt, label=label)
ax_q.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$Q_s/Q_{GB}$", title="Heat flux")
ax_dt.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$\Delta t$", title="CFL-controlled step")
for ax in (ax_q, ax_dt):
    ax.legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(OUTPUT / "kinetic_electrons.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/kinetic_electrons.png")
