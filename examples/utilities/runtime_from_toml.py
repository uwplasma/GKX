#!/usr/bin/env python3
"""Run unified runtime-configured linear simulations from TOML."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.runtime import run_runtime_linear, run_runtime_scan


def _status(message: str) -> None:
    print(f"runtime: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to runtime TOML file")
    args = parser.parse_args()

    cfg, data = load_runtime_from_toml(args.config)

    run_cfg = data.get("run", {})
    scan_cfg = data.get("scan", {})
    fit_cfg = data.get("fit", {})
    fit_keys = {
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
    fit_cfg = {k: v for k, v in fit_cfg.items() if k in fit_keys}

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
            show_progress=bool(getattr(sys.stdout, "isatty", lambda: False)()),
            **fit_cfg,
        )
        for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
            print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
        return 0

    res = run_runtime_linear(
        cfg,
        ky_target=float(run_cfg.get("ky", 0.3)),
        Nl=int(run_cfg.get("Nl", 24)),
        Nm=int(run_cfg.get("Nm", 12)),
        solver=str(run_cfg.get("solver", "krylov")),
        method=run_cfg.get("method", None),
        dt=run_cfg.get("dt", None),
        steps=run_cfg.get("steps", None),
        show_progress=bool(getattr(sys.stdout, "isatty", lambda: False)()),
        status_callback=_status,
        **fit_cfg,
    )
    print(f"ky={res.ky:.4f} gamma={res.gamma:.6f} omega={res.omega:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
