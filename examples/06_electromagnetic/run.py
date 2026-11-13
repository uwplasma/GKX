"""Electromagnetic ITG in Cyclone geometry with A_parallel.

Fast path: ``case.toml`` (kinetic ions, adiabatic electrons, A_parallel on,
beta = 0.015) solved with ``gkx.solve`` at the deck's ``k_y rho_i = 0.3``. There
is no research path here. The kinetic ballooning mode needs kinetic electrons and
(Nl, Nm) = (16, 48); ``case_full.toml`` is the GX comparison deck for it
(Nz = 96).

Prints gamma and omega. Writes ``summary.json``, the run files and
``electromagnetic.png`` (potential trace and eigenfunction) to
``outputs/06_electromagnetic``. A few seconds on a laptop CPU.
"""

import json
from pathlib import Path

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/06_electromagnetic")

# ---- Fast path ----
case = gkx.load(CASE)
print(f"beta = {case.physics.beta}, A_parallel = {case.physics.use_apar}")
run = dict(
    ky_target=case.run.ky, Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)
result = gkx.solve(case, **run, show_progress=True)
print(f"ky = {result.ky:.2f}: gamma = {result.gamma:+.4f}, omega = {result.omega:+.4f}")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "run")
summary = {"case": CASE.name, "beta": case.physics.beta, **result.summary()}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
gkx.set_plot_style()
fig, _ = gkx.plot(result)
fig.savefig(OUTPUT / "electromagnetic.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/electromagnetic.png")
