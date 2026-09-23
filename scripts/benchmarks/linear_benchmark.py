#!/usr/bin/env python3
"""Linear benchmark scans against the tracked references: Cyclone, ETG, kinetic-electron ITG, TEM and KBM.

    python scripts/benchmark.py linear_benchmark cyclone --outdir tools_out/cyclone
    python scripts/benchmark.py linear_benchmark etg --ky 10.0

``cyclone`` and ``etg`` time-integrate a ``k_y`` scan, fit the deck's
asymptotic window, and write ``<case>_validation.png``, ``<case>_comparison.png``
and ``<case>_fit_window.png``. ``kinetic`` and ``tem`` run the solver-selected
scan and write ``<case>_scan.png``. Every case reads its shipped example deck,
so the published figures cannot drift from the examples. ``kbm`` runs no solver:
it plots the reviewed fixed-beta table ``docs/_static/comparison/
kbm_reference_candidates.csv`` (its ``selected`` rows) to
``kbm_linear_comparison.png``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gkx import (
    LinearValidationPanel,
    growth_fit_figure,
    linear_validation_figure,
    load_runtime_from_toml,
    normalize_eigenfunction,
    run_runtime_linear,
    run_runtime_scan,
    scan_comparison_figure,
)
from gkx.benchmarking_shared import (
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_tem_reference,
)

ROOT = Path(__file__).resolve().parents[2]
DECKS = ROOT / "examples" / "linear" / "axisymmetric"

# Cyclone's asymptotic fit window is the last 30% of the deck's horizon, well
# clear of the startup transient (the fitted gamma is still 24% high at t = 30
# and does not settle until t ~= 60). It tracks [time].t_max in the deck; the
# coupling is enforced by tests/validation/benchmarks/test_benchmark_contracts.py::
# test_cyclone_publication_driver_uses_asymptotic_fit_window.
CASES: dict[str, dict[str, Any]] = {
    "cyclone": {
        "config": DECKS / "cyclone.toml",
        "reference": load_cyclone_reference,
        "title": "Cyclone",
        "Nl": 16,
        "Nm": 48,
        "window": {
            "auto_window": False,
            "tmin": 105.0,
            "tmax": 150.0,
            "min_points": 80,
            "require_positive": True,
        },
    },
    "etg": {
        "config": DECKS / "etg.toml",
        "reference": load_etg_reference,
        "title": "ETG",
        "Nl": 24,
        "Nm": 8,
        # tmax None: the fit window ends at the deck's t_max.
        "window": {"auto_window": False, "tmin": 1.0, "tmax": None},
    },
    "kinetic": {
        "config": DECKS / "runtime_kinetic_electron.toml",
        "reference": load_cyclone_reference_kinetic,
        "title": "Kinetic-electron ITG",
        "x_label": r"$k_y \rho_i$",
        "Nl": 12,
        "Nm": 32,
        "t_max": 8.0,
    },
    "kbm": {
        "table": ROOT / "docs/_static/comparison/kbm_reference_candidates.csv",
        "title": r"KBM linear scan ($\beta=0.015$)",
    },
    "tem": {
        "config": DECKS / "runtime_tem.toml",
        "reference": load_tem_reference,
        "title": "TEM (s-alpha)",
        "x_label": r"$k_y \rho_s$",
        "Nl": 12,
        "Nm": 32,
    },
}


def _reference_figure(case: dict[str, Any], scan: Any, ref: Any, path: Path) -> None:
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        x_label=case["x_label"],
        title=case["title"],
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig(path, dpi=200)


def _kbm_table_figure(case: dict[str, Any], outdir: Path) -> None:
    table = pd.read_csv(case["table"]).sort_values("ky")
    if "selected" in table:
        selected = table["selected"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
        table = table[selected]
    fig, _axes = scan_comparison_figure(
        table["ky"].to_numpy(),
        table["gamma"].to_numpy(),
        table["omega"].to_numpy(),
        x_label=r"$k_y\rho_i$",
        title=case["title"],
        x_ref=table["ky"].to_numpy(),
        gamma_ref=table["gamma_gx"].to_numpy(),
        omega_ref=table["omega_gx"].to_numpy(),
        ref_label="Reference",
    )
    fig.savefig(outdir / "kbm_linear_comparison.png", dpi=200, bbox_inches="tight")


def _time_fit_case(
    name: str, case: dict[str, Any], cfg: Any, ref: Any, args: argparse.Namespace
) -> None:
    ky_values = (
        np.array([float(args.ky)]) if args.ky is not None else np.asarray(ref.ky)
    )
    window = dict(case["window"])
    if window["tmax"] is None:
        window["tmax"] = cfg.time.t_max
    # Take the integrator settings from the deck rather than repeating them, so
    # the published figures cannot drift away from the shipped example again.
    options = dict(
        dt=float(cfg.time.dt),
        steps=round(float(cfg.time.t_max) / float(cfg.time.dt)),
        method=str(cfg.time.method),
        solver="time",
        fit_signal="phi",
        mode_method="z_index",
        **window,
    )
    resolution = dict(Nl=case["Nl"], Nm=case["Nm"])
    scan = run_runtime_scan(cfg, ky_values, batch_ky=True, **resolution, **options)
    ky_selected = float(scan.ky[int(np.nanargmax(scan.gamma))])
    mode = run_runtime_linear(cfg, ky_target=ky_selected, **resolution, **options)
    if (
        mode.eigenfunction is None
        or mode.z is None
        or mode.t is None
        or mode.signal is None
    ):
        raise RuntimeError("time-integrated runtime diagnostics are required")

    title = case["title"]
    panel = LinearValidationPanel(
        name=title,
        z=mode.z,
        eigenfunction=normalize_eigenfunction(mode.eigenfunction, mode.z),
        x=scan.ky,
        gamma=scan.gamma,
        omega=scan.omega,
        x_label=r"$k_y \rho_i$",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig, _axes = linear_validation_figure([panel])
    fig.savefig(args.outdir / f"{name}_validation.png", dpi=200)
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$k_y \rho_i$",
        f"{title} comparison",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig.savefig(args.outdir / f"{name}_comparison.png", dpi=200)
    if not args.no_fit:
        fig, _axes = growth_fit_figure(
            mode.t,
            mode.signal,
            tmin=mode.fit_window_tmin,
            tmax=mode.fit_window_tmax,
            title=f"{title} fit window",
        )
        fig.savefig(args.outdir / f"{name}_fit_window.png", dpi=200)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--ky", type=float, default=None, help="Run a single ky value.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("."), help="Output directory."
    )
    parser.add_argument("--no-fit", action="store_true", help="Skip fit-window plot.")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    case = CASES[args.case]
    if "table" in case:
        _kbm_table_figure(case, args.outdir)
        return
    cfg, _raw = load_runtime_from_toml(case["config"])
    ref = case["reference"]()
    if "window" in case:
        _time_fit_case(args.case, case, cfg, ref, args)
        return
    if "t_max" in case:
        cfg = replace(cfg, time=replace(cfg.time, t_max=case["t_max"]))
    ky_values = np.array([float(args.ky)]) if args.ky is not None else ref.ky
    scan = run_runtime_scan(cfg, ky_values, Nl=case["Nl"], Nm=case["Nm"], solver="auto")
    _reference_figure(case, scan, ref, args.outdir / f"{args.case}_scan.png")


if __name__ == "__main__":
    main()
