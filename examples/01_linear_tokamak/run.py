"""Linear Cyclone ITG growth rates in s-alpha tokamak geometry.

Fast path: load ``case.toml`` and call ``gkx.scan`` and ``gkx.solve`` with the
resolution and solver of the deck's ``[run]`` block and no other tuning. Growth
rates and frequencies are within 2% of the GX reference at every ``k_y`` of the
scan. Research path: re-solve the most unstable mode with the time integrator and
the Krylov eigensolver at each velocity resolution in ``RESOLUTIONS``, to see how
far coarser resolutions are from the deck's and whether the two solvers agree.

Prints gamma and omega for each ``k_y`` and the convergence table. Writes
``summary.json``, the scan files and ``linear_tokamak.png`` to
``outputs/01_linear_tokamak``. A few minutes on a laptop CPU. Set ``CASE`` to
``case_full.toml`` for the literature resolution.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/01_linear_tokamak")
KY = [0.2, 0.3, 0.4]  # k_y rho_i
RESOLUTIONS = [(8, 24), (16, 48)]  # (Nl, Nm) of the research path
SOLVERS = ("time", "krylov")

# ---- Fast path ----
case = gkx.load(CASE)
run = dict(
    Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)  # the deck's [run] block
scan = gkx.scan(case, KY, **run, show_progress=True)
peak_ky = float(scan.ky[int(np.argmax(scan.gamma))])
peak = gkx.solve(case, ky_target=peak_ky, **run, show_progress=True)
for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
    print(f"ky = {ky:.2f}: gamma = {gamma:+.5f}, omega = {omega:+.5f}")

# ---- Research path: velocity resolution and eigensolver ----
convergence = []
for nl, nm in RESOLUTIONS:
    for solver in SOLVERS:
        r = gkx.solve(case, ky_target=peak_ky, Nl=nl, Nm=nm, solver=solver)
        convergence.append(
            {"Nl": nl, "Nm": nm, "solver": solver, "gamma": r.gamma, "omega": r.omega}
        )
        print(
            f"Nl, Nm = {nl}, {nm} {solver:>7}: gamma = {r.gamma:+.5f}, omega = {r.omega:+.5f}"
        )

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
scan.save(OUTPUT / "scan")
summary = {
    "case": CASE.name,
    "ky": np.asarray(scan.ky).tolist(),
    "gamma": np.asarray(scan.gamma).tolist(),
    "omega": np.asarray(scan.omega).tolist(),
    "peak": peak.summary(),
    "convergence": convergence,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

gkx.set_plot_style()
fig, (ax_gamma, ax_omega, ax_mode) = plt.subplots(1, 3, figsize=(12.0, 3.6))
ax_gamma.plot(scan.ky, scan.gamma, "o-")
ax_gamma.set(xlabel=r"$k_y\rho_i$", ylabel=r"$\gamma\,a/v_{ti}$", title="Growth rate")
ax_omega.plot(scan.ky, scan.omega, "s-", color="tab:orange")
ax_omega.set(
    xlabel=r"$k_y\rho_i$", ylabel=r"$\omega\,a/v_{ti}$", title="Real frequency"
)
mode = np.asarray(peak.eigenfunction)
mode = mode / mode[np.argmax(np.abs(mode))]
ax_mode.plot(peak.z, mode.real, label=r"Re $\phi$")
ax_mode.plot(peak.z, mode.imag, "--", label=r"Im $\phi$")
ax_mode.set(
    xlabel=r"$\theta$",
    ylabel=r"$\phi/\phi_{\max}$",
    title=f"Eigenfunction, $k_y\\rho_i$={peak_ky:.1f}",
)
ax_mode.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT / "linear_tokamak.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/linear_tokamak.png")
