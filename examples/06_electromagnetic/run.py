"""Electromagnetic kinetic ballooning mode (KBM) in Cyclone geometry.

Loads ``case.toml`` (kinetic ions and electrons, A_parallel on, beta = 0.015)
and solves the linear problem at ``k_y rho_i = 0.3``. Prints gamma and omega,
saves the fit summary as JSON, and plots the potential trace and the
eigenfunction. A few seconds on a laptop CPU. The tutorial deck only exercises
the electromagnetic path: with (Nl, Nm) = (2, 4) and Nz = 16 the KBM is not
resolved and the growth rate is not physical. ``case_full.toml`` is the
GX-benchmark deck (Nz = 96, Nl = 16, Nm = 48; minutes on a GPU), which is the
one to use for any KBM number.
"""

import json
from pathlib import Path

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/06_electromagnetic")

case = gkx.load(CASE)
print(
    f"beta = {case.physics.beta}, A_parallel = {case.physics.use_apar}, species = {len(case.species)}"
)

result = gkx.solve(
    case, ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, solver="time"
)
print(f"ky = {result.ky:.2f}: gamma = {result.gamma:+.4f}, omega = {result.omega:+.4f}")
if case.run.Nm < 16:
    print("tutorial resolution: this growth rate is not converged; use case_full.toml")

OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "kbm")
summary = {"case": CASE.name, "beta": case.physics.beta, **result.summary()}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "electromagnetic.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/electromagnetic.png")
