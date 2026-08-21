#!/usr/bin/env python3
"""Run a runtime-configured independent ky scan with [parallel] strategy="batch".

This example intentionally uses the independent-worker runtime path: each ky
point is solved by the normal single-ky runtime solver, and results are gathered
in input order. It does not request the combined-ky solver layout and does not
change solver defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.runtime import run_runtime_scan


_DEFAULT_CONFIG = Path(__file__).with_name("runtime_batch_ky_scan.toml")
_FIT_KEYS = {
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
    "fit_signal",
}


def run_example(
    config: str | Path = _DEFAULT_CONFIG,
    *,
    workers: int = 1,
    executor: str = "thread",
):
    """Run the configured scan and return the runtime scan result.

    Keep ``workers=1`` to let ``[parallel].num_devices`` select the worker count.
    Pass a larger value to override the TOML at the call site.
    """

    cfg, data = load_runtime_from_toml(config)
    scan_cfg = data.get("scan", {})
    fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in _FIT_KEYS}
    ky_values = np.asarray(scan_cfg.get("ky", []), dtype=float)
    if ky_values.size == 0:
        raise ValueError("[scan].ky must contain at least one value")

    return run_runtime_scan(
        cfg,
        ky_values,
        Nl=int(scan_cfg.get("Nl", 2)),
        Nm=int(scan_cfg.get("Nm", 3)),
        solver=str(scan_cfg.get("solver", "time")),
        method=scan_cfg.get("method", None),
        dt=scan_cfg.get("dt", None),
        steps=scan_cfg.get("steps", None),
        sample_stride=scan_cfg.get("sample_stride", None),
        workers=workers,
        parallel_executor=executor,
        show_progress=False,
        **fit_cfg,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Override [parallel].num_devices; default lets the TOML select workers.",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help=(
            "Fallback executor when --workers overrides the TOML; the TOML "
            "[parallel].backend still applies when --workers is left at 1."
        ),
    )
    args = parser.parse_args()

    scan = run_example(args.config, workers=args.workers, executor=args.executor)
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega, strict=True):
        print(f"ky={ky:.4f} gamma={gamma:.6f} omega={omega:.6f}")
    if scan.parallel is not None:
        print(
            "parallel "
            f"strategy=batch executor={scan.parallel['executor']} "
            f"requested_workers={scan.parallel['requested_workers']} "
            f"effective_workers={scan.parallel['effective_workers']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
