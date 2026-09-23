"""Independent k_y scan dispatched to parallel workers.

Loads ``case.toml``, whose ``[parallel]`` block selects ``strategy = "batch"``
with two thread workers. Each ``k_y`` point is an ordinary single-``k_y``
linear solve run in its own worker, and results are gathered in input order,
so the parallel scan is identical to the serial one; the combined-``k_y``
solver layout is never used. The script runs the scan both ways, checks that
the growth rates agree, prints the worker summary, saves JSON, and plots the
spectrum. About ten seconds on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/11_parallel_scan")
KY = [0.1, 0.2, 0.3, 0.4]  # k_y rho_i points, one worker task each

case = gkx.load(CASE)
run = case.run
options = dict(Nl=run.Nl, Nm=run.Nm, solver=run.solver)
parallel = gkx.scan(case, KY, **options)
serial = gkx.scan(
    case.replace(parallel=replace(case.parallel, strategy="serial")), KY, **options
)
identical = bool(
    np.array_equal(parallel.gamma, serial.gamma)
    and np.array_equal(parallel.omega, serial.omega)
)

for ky, gamma, omega in zip(parallel.ky, parallel.gamma, parallel.omega):
    print(f"ky = {ky:.2f}: gamma = {gamma:+.5f}, omega = {omega:+.5f}")
info = parallel.parallel or {}
print(
    f"strategy = {info.get('strategy')}, executor = {info.get('executor')}, workers = {info.get('effective_workers')}"
)
print(f"parallel scan identical to serial scan: {identical}")

OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {
    "ky": np.asarray(parallel.ky).tolist(),
    "gamma": np.asarray(parallel.gamma).tolist(),
    "omega": np.asarray(parallel.omega).tolist(),
    "parallel": info,
    "identical_to_serial": identical,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
fig, _ = gkx.plot(parallel)
fig.savefig(OUTPUT / "parallel_scan.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/parallel_scan.png")
