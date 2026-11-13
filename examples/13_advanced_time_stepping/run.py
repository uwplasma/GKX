"""Pin the time stepper of a nonlinear run and compare it with the default.

Fast path: the tutorial Cyclone deck of example 03 with no stepper named. A
nonlinear deck without ``dt`` runs with ``method = "auto"``: adaptive ``rk4-pi``,
or ``lawson-rk4-pi`` (the magnetic drift advanced exactly) where the drift is at
least 56% of the linear CFL bound and the step cap is at least 2.6 times the
``rk4-pi`` bound step. At ``(Nl, Nm) = (2, 4)`` the drift is a small share of the
bound and the default is ``rk4-pi``; raise ``NL`` and ``NM`` (to 8 and 16, say) to
see the rule select ``lawson-rk4-pi``.

Research path: the choices a caller can pin, in ``PINS``: either error-controlled
stepper, an explicit ``cfl_fac`` and a fixed-step ``rk2``. Pins change cost, not
the statistics of the saturated state; fix the step only to reproduce a published
run.

Prints the steps and wall time of each run. Writes ``summary.json`` and
``advanced_steppers.png`` (step histories) to ``outputs/13_advanced_time_stepping``.
About a minute on a laptop CPU.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).parents[1] / "03_nonlinear_tokamak" / "case.toml"
OUTPUT = Path("outputs/13_advanced_time_stepping")
NL, NM = 2, 4
PINS = {
    "default (auto)": {},  # the fast path
    "rk4-pi": {"method": "rk4-pi"},
    "lawson-rk4-pi": {"method": "lawson-rk4-pi"},
    "lawson-rk4-pi, cfl_fac 2.0": {"method": "lawson-rk4-pi", "cfl_fac": 2.0},
    "rk2, fixed dt 0.05": {"method": "rk2", "dt": 0.05, "fixed_dt": True},
}

# ---- Fast path, then research path ----
case = gkx.load(CASE)
summary, steps_dt = {}, {}
for label, pin in PINS.items():
    start = time.perf_counter()
    result = gkx.solve(
        case.replace(time=replace(case.time, **pin)),
        ky_target=case.run.ky,
        Nl=NL,
        Nm=NM,
        show_progress=True,
    )
    wall = time.perf_counter() - start
    d = result.diagnostics
    t, dt = np.asarray(d.t), np.asarray(d.dt_t)
    steps_dt[label] = (t, dt)
    summary[label] = {
        "wall_s": wall,
        "dt_mean": float(dt.mean()),
        "steps": int(round(t[-1] / dt.mean())),
        "free_energy_final": float(np.asarray(d.Wg_t)[-1]),
    }
    print(
        f"{label}: {summary[label]['steps']} steps, mean dt {dt.mean():.3f}, {wall:.1f} s"
    )

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for label, (t, dt) in steps_dt.items():
    ax.plot(t, dt, label=label)
ax.set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$\Delta t$", title="Step history by stepper")
ax.legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(OUTPUT / "advanced_steppers.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/advanced_steppers.png")
