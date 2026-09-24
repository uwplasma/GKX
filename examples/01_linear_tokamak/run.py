"""Linear Cyclone ITG growth-rate scan in s-alpha tokamak geometry.

Loads ``case.toml`` (Cyclone base case, adiabatic electrons), solves one
linear initial-value problem per ``k_y``, prints gamma and omega, saves the
scan as JSON and CSV, and plots the spectrum plus the eigenfunction at the
most unstable ``k_y``. About ten seconds on a laptop CPU at the tutorial
resolution. For the literature resolution set ``CASE`` to ``case_full.toml``
(minutes on a CPU; its ``[run]`` block selects the Krylov eigensolver).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/01_linear_tokamak")
KY = [0.1, 0.2, 0.3, 0.4]  # binormal wavenumbers k_y rho_i

case = gkx.load(CASE)
print(
    f"geometry {case.geometry.model}: q={case.geometry.q}, s_hat={case.geometry.s_hat}"
)

scan = gkx.scan(case, KY, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver)
peak_ky = float(scan.ky[int(np.argmax(scan.gamma))])
peak = gkx.solve(
    case, ky_target=peak_ky, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)

print(f"{'ky':>6} {'gamma':>10} {'omega':>10}")
for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
    print(f"{ky:6.2f} {gamma:10.5f} {omega:10.5f}")
print(f"most unstable ky = {peak_ky:.2f}, gamma = {peak.gamma:.5f}")

OUTPUT.mkdir(parents=True, exist_ok=True)
scan.save(OUTPUT / "scan")
summary = {
    "case": CASE.name,
    "ky": np.asarray(scan.ky).tolist(),
    "gamma": np.asarray(scan.gamma).tolist(),
    "omega": np.asarray(scan.omega).tolist(),
    "peak": peak.summary(),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

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
for ax in (ax_gamma, ax_omega, ax_mode):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "linear_tokamak.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/linear_tokamak.png")
