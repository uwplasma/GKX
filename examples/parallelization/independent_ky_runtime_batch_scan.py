#!/usr/bin/env python3
"""Runtime-configured independent ky scan with ``[parallel] strategy="batch"``.

Each ky point in the TOML's ``[scan].ky`` list is solved by the normal
single-ky runtime solver in an independent worker, and results are gathered in
input order; the combined-ky solver layout is never requested and solver
defaults are unchanged.  Prints one ``gamma``/``omega`` line per ky point plus
the parallel-execution summary.  The shipped config is tiny -- the scan
finishes in well under a minute.

``run_example`` is imported by the integration tests, so the run itself stays
under the ``__main__`` guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.runtime import run_runtime_scan

CONFIG = Path(__file__).with_name("runtime_batch_ky_scan.toml")
WORKERS = 1  # 1 lets the TOML [parallel].num_devices select the worker count
EXECUTOR = "thread"  # fallback executor ("thread" or "process") when WORKERS > 1

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
    config: str | Path = CONFIG,
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


if __name__ == "__main__":
    scan = run_example(CONFIG, workers=WORKERS, executor=EXECUTOR)
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega, strict=True):
        print(f"ky={ky:.4f} gamma={gamma:.6f} omega={omega:.6f}")
    if scan.parallel is not None:
        print(
            "parallel "
            f"strategy=batch executor={scan.parallel['executor']} "
            f"requested_workers={scan.parallel['requested_workers']} "
            f"effective_workers={scan.parallel['effective_workers']}"
        )
