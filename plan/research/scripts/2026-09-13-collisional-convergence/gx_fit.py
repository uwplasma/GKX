#!/usr/bin/env python3
"""Fit the growth rate and frequency of one GX linear run for Q8.

Usage: gx_fit.py OUT_NC KEY OUT_JSON

gamma is half the least-squares slope of log(Diagnostics/Phi2_t) on the late
window [0.7 T, T] (the GKX parity runner's fit_start_fraction), and on
[0.35 T, 0.5 T] for the half-time probe; "settled" uses the runner's 5% rule.
omega and GX's own gamma come from Diagnostics/omega_kxkyt (index 0 = omega,
1 = gamma), averaged over the second half of the trace exactly as
tools/comparison/build_gx_parity_matrix.py load_reference_spectrum does, and
also averaged over [0.7 T, T].
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset

KY = 0.550000011920929


def _fit(t: np.ndarray, phi2: np.ndarray, lo: float, hi: float) -> float:
    mask = (t >= lo) & (t <= hi) & np.isfinite(phi2) & (phi2 > 0.0)
    slope, _ = np.polyfit(t[mask], np.log(phi2[mask]), 1)
    return 0.5 * float(slope)


def main() -> None:
    path, key, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    with Dataset(path, "r") as root:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        inputs = root.groups["Inputs"]
        t = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        phi2 = np.asarray(diag.variables["Phi2_t"][:], dtype=float)
        series = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
        controls = inputs.groups["Controls"]
        dissipation = controls.groups["Numerical_Diss"]
        record = {
            "key": key,
            "nlaguerre": int(root.variables["nlaguerre"][...]),
            "nhermite": int(root.variables["nhermite"][...]),
            "species_nu": np.asarray(
                inputs.groups["Species"].variables["nu"][:], dtype=float
            ).tolist(),
            "collisions": int(controls.variables["collisions"][...]),
            "hypercollisions_kz": int(dissipation.variables["hypercollisions_kz"][...]),
            "hypercollisions_const": int(
                dissipation.variables["hypercollisions_const"][...]
            ),
            "nu_hyper_m": float(dissipation.variables["nu_hyper_m"][...]),
            "p_hyper_m": int(dissipation.variables["p_hyper_m"][...]),
            "dt": float(inputs.groups["Time"].variables["dt"][...]),
        }
    t_end = float(t[-1])
    iky = int(np.argmin(np.abs(ky - KY)))
    gamma = _fit(t, phi2, 0.7 * t_end, t_end)
    gamma_half = _fit(t, phi2, 0.35 * t_end, 0.5 * t_end)
    half = int(len(t) / 2)
    late = t >= 0.7 * t_end
    shift = (gamma_half - gamma) / abs(gamma)
    record.update(
        ky=float(ky[iky]),
        samples=int(len(t)),
        t_end=t_end,
        nonfinite_phi2=int(np.sum(~np.isfinite(phi2))),
        nonfinite_omega=int(np.sum(~np.isfinite(series))),
        gamma_phi2_late=gamma,
        gamma_phi2_half_probe=gamma_half,
        gamma_half_time_shift=shift,
        settled=bool(np.isfinite(gamma) and abs(shift) <= 0.05),
        omega_second_half_mean=float(np.mean(series[half:, iky, 0, 0])),
        gamma_omega_output_second_half_mean=float(np.mean(series[half:, iky, 0, 1])),
        omega_late_mean=float(np.mean(series[late, iky, 0, 0])),
        gamma_omega_output_late_mean=float(np.mean(series[late, iky, 0, 1])),
        omega_final=float(series[-1, iky, 0, 0]),
        gamma_omega_output_final=float(series[-1, iky, 0, 1]),
    )
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
