"""Independent k_y scan dispatched to parallel workers.

Fast path: ``case.toml`` has a ``[parallel]`` block with ``strategy = "batch"``
and two thread workers, so ``gkx.scan`` runs each ``k_y`` as an ordinary
single-``k_y`` linear solve in its own worker and gathers the results in input
order. Research path: the same scan with ``strategy = "serial"``, to check that
the parallel rates are identical to the serial ones.

Prints gamma and omega per ``k_y``, the worker summary and the identity check.
Writes ``summary.json`` and ``parallel_scan.png`` to ``outputs/11_parallel_scan``.
About ten seconds on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import gkx

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/11_parallel_scan")
KY = [0.1, 0.2, 0.3, 0.4]  # k_y rho_i points, one worker task each

# ---- Fast path ----
case = gkx.load(CASE)
parallel = gkx.scan(case, KY, show_progress=True)
info = parallel.parallel or {}
for ky, gamma, omega in zip(parallel.ky, parallel.gamma, parallel.omega):
    print(f"ky = {ky:.2f}: gamma = {gamma:+.5f}, omega = {omega:+.5f}")
print(
    f"strategy = {info.get('strategy')}, executor = {info.get('executor')}, "
    f"workers = {info.get('effective_workers')}"
)

# ---- Research path: the serial scan as a reference ----
serial = gkx.scan(case.replace(parallel=replace(case.parallel, strategy="serial")), KY)
identical = bool(
    np.array_equal(parallel.gamma, serial.gamma)
    and np.array_equal(parallel.omega, serial.omega)
)
print(f"parallel scan identical to serial scan: {identical}")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {
    "ky": np.asarray(parallel.ky).tolist(),
    "gamma": np.asarray(parallel.gamma).tolist(),
    "omega": np.asarray(parallel.omega).tolist(),
    "parallel": info,
    "identical_to_serial": identical,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
gkx.set_plot_style()
fig, _ = gkx.plot(parallel)
fig.savefig(OUTPUT / "parallel_scan.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/parallel_scan.png")
