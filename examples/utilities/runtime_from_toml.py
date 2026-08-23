#!/usr/bin/env python3
"""Run a runtime-configured linear simulation (or ky scan) from a TOML file.

Point ``CONFIG`` at any linear runtime TOML.  When the file has a ``[scan]``
block with a ``ky`` list, every ky point is solved and one ``gamma``/``omega``
line is printed per point; otherwise the single ``[run]`` case is solved.
Runtime depends entirely on the TOML -- the default Cyclone config takes about
a minute per ky point on a laptop CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.runtime import run_runtime_linear, run_runtime_scan

# Path to the runtime TOML file to execute.
CONFIG = (
    Path(__file__).resolve().parents[1] / "linear" / "axisymmetric" / "cyclone.toml"
)

FIT_KEYS = {
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
}


def _status(message: str) -> None:
    print(f"runtime: {message}")


cfg, data = load_runtime_from_toml(CONFIG)

run_cfg = data.get("run", {})
scan_cfg = data.get("scan", {})
fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in FIT_KEYS}
show_progress = bool(getattr(sys.stdout, "isatty", lambda: False)())

if "ky" in scan_cfg:
    ky_values = np.asarray(scan_cfg["ky"], dtype=float)
    scan = run_runtime_scan(
        cfg,
        ky_values,
        Nl=int(scan_cfg.get("Nl", 24)),
        Nm=int(scan_cfg.get("Nm", 12)),
        solver=str(scan_cfg.get("solver", "krylov")),
        method=scan_cfg.get("method", None),
        dt=scan_cfg.get("dt", None),
        steps=scan_cfg.get("steps", None),
        show_progress=show_progress,
        **fit_cfg,
    )
    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
else:
    res = run_runtime_linear(
        cfg,
        ky_target=float(run_cfg.get("ky", 0.3)),
        Nl=int(run_cfg.get("Nl", 24)),
        Nm=int(run_cfg.get("Nm", 12)),
        solver=str(run_cfg.get("solver", "krylov")),
        method=run_cfg.get("method", None),
        dt=run_cfg.get("dt", None),
        steps=run_cfg.get("steps", None),
        show_progress=show_progress,
        status_callback=_status,
        **fit_cfg,
    )
    print(f"ky={res.ky:.4f} gamma={res.gamma:.6f} omega={res.omega:.6f}")
