"""Electromagnetic ITG in Cyclone geometry with A_parallel.

Fast path: ``case.toml`` (kinetic ions, adiabatic electrons, A_parallel on,
beta = 0.015) solved with ``gkx.solve`` at the deck's ``k_y rho_i = 0.3``.

Research path: the kinetic ballooning mode (KBM). ``case_full.toml`` is the GX
comparison deck: kinetic electrons (mass ratio 0.00027), A_parallel at beta = 0.015 and the
Krylov eigensolver. It is solved at ``KBM_NZ`` points along the line and
``KBM_RESOLUTION`` = (Nl, Nm) and compared with the GX rates in
``src/gkx/data/kbm_reference.csv``. The tutorial values (16, (2, 8)) take under a
minute and are not converged: gamma is about 25% below GX and rises with
resolution. The deck's own (Nz, Nl, Nm) = (96, 16, 48) is the literature
resolution, a GPU run of a few minutes; set both constants to None to use it.

Prints gamma and omega of each run and the KBM comparison. Writes ``summary.json``,
the run files and ``electromagnetic.png`` (potential trace and eigenfunction) to
``outputs/06_electromagnetic``. A few seconds for the fast path on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
CASE_KBM = Path(__file__).with_name("case_full.toml")
KBM_NZ = 16  # points along the line of the research path; None keeps the deck's 96
KBM_RESOLUTION = (2, 8)  # (Nl, Nm); None keeps the deck's (16, 48)
OUTPUT = Path("outputs/06_electromagnetic")

# ---- Fast path ----
case = gkx.load(CASE)
print(f"beta = {case.physics.beta}, A_parallel = {case.physics.use_apar}")
result = gkx.solve(case, show_progress=True)
print(f"ky = {result.ky:.2f}: gamma = {result.gamma:+.4f}, omega = {result.omega:+.4f}")

# ---- Research path: kinetic ballooning mode ----
kbm_case = gkx.load(CASE_KBM)
if KBM_NZ is not None:
    kbm_case = kbm_case.replace(grid=replace(kbm_case.grid, Nz=KBM_NZ))
nl_nm = (
    {} if KBM_RESOLUTION is None else dict(Nl=KBM_RESOLUTION[0], Nm=KBM_RESOLUTION[1])
)
kbm = gkx.solve(kbm_case, show_progress=True, **nl_nm)
data = np.genfromtxt(
    Path(gkx.__file__).parent / "data" / "kbm_reference.csv", delimiter=",", names=True
)
ref = data[int(np.argmin(np.abs(data["ky"] - kbm.ky)))]
print(f"KBM ky = {kbm.ky:.2f}: gamma = {kbm.gamma:+.4f}, omega = {kbm.omega:+.4f}")
print(f"GX reference: gamma = {ref['gamma']:+.4f}, omega = {ref['omega']:+.4f}")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
summary = {
    "case": CASE.name,
    "beta": case.physics.beta,
    **result.summary(),
    "kbm": {
        "case": CASE_KBM.name,
        "Nz": int(kbm_case.grid.Nz),
        "gamma": kbm.gamma,
        "omega": kbm.omega,
        "gamma_gx": float(ref["gamma"]),
        "omega_gx": float(ref["omega"]),
    },
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "electromagnetic.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/electromagnetic.png")
