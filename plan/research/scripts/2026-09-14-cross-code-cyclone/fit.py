#!/usr/bin/env python3
"""Q20 growth-rate extraction for GS2 and stella run directories (manifest.toml [fit]).

    python fit.py <root> [case-dir-glob ...]  -> CSV on stdout

Columns: code, case, ky_code, ky_gx, t_end, gamma_code, omega_code, drift, settled,
gamma_gx, omega_gx, gamma_phi_fit (GS2 midplane-phi cross-check, code units), wall_s, np, rc.
Unit conversion: gamma_gx = sqrt(2) * gamma_code * (a / Lref); Lref = R cases (G*) use a/R = 1/2.77778.
"""

from __future__ import annotations

import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

SQRT2 = math.sqrt(2.0)
A_OVER_R = 1.0 / 2.77778


def window_stats(t: np.ndarray, om: np.ndarray, ga: np.ndarray):
    good = np.isfinite(om) & np.isfinite(ga)
    t, om, ga = t[good], om[good], ga[good]
    if t.size < 10:
        return math.nan, math.nan, math.nan, float(t[-1]) if t.size else math.nan
    T = t[-1]
    w1 = t >= 0.8 * T
    w0 = (t >= 0.6 * T) & (t < 0.8 * T)
    g1, o1 = float(ga[w1].mean()), float(om[w1].mean())
    g0 = float(ga[w0].mean()) if w0.any() else math.nan
    drift = (g1 - g0) / g1 if g1 != 0 else math.nan
    return g1, o1, drift, float(T)


def gs2_fit(d: Path):
    nc = sorted(d.glob("*.out.nc"))[0]
    ds = Dataset(nc)
    t = np.asarray(ds.variables["t"][:], dtype=float)
    ky = float(np.asarray(ds.variables["ky"][:])[-1])
    om = np.asarray(ds.variables["omega"][:], dtype=float)  # (t, kx, ky, ri)
    omega, gamma = om[:, 0, -1, 0], om[:, 0, -1, 1]
    g, o, drift, T = window_stats(t, omega, gamma)
    phi = np.asarray(ds.variables["phi_igomega_by_mode"][:], dtype=float)
    z = phi[:, 0, -1, 0] + 1j * phi[:, 0, -1, 1]
    w = (t >= 0.8 * t[-1]) & (np.abs(z) > 0)
    gfit = (
        float(np.polyfit(t[w], np.log(np.abs(z[w])), 1)[0]) if w.sum() > 3 else math.nan
    )
    return ky, T, g, o, drift, gfit


def stella_fit(d: Path):
    nc = sorted(d.glob("*.out.nc"))[0]
    ds = Dataset(nc)
    t = np.asarray(ds.variables["t"][:], dtype=float)
    ky = float(np.asarray(ds.variables["ky"][:])[0])
    om = np.asarray(ds.variables["omega"][:], dtype=float)  # (t, kx, ky, ri)
    omega, gamma = om[:, 0, 0, 0], om[:, 0, 0, 1]
    g, o, drift, T = window_stats(t, omega, gamma)
    phi2 = np.asarray(ds.variables["phi2"][:], dtype=float)
    w = (t >= 0.8 * t[-1]) & (phi2 > 0)
    gfit = (
        float(0.5 * np.polyfit(t[w], np.log(phi2[w]), 1)[0])
        if w.sum() > 3
        else math.nan
    )
    return ky, T, g, o, drift, gfit


def done_info(d: Path):
    f = d / "DONE"
    if not f.exists():
        return math.nan, math.nan, "running"
    s = f.read_text()
    wall = float(re.search(r"wall=([0-9.]+)", s).group(1))
    np_ = int(re.search(r"np=([0-9]+)", s).group(1))
    rc = re.search(r"rc=([0-9]+)", s).group(1)
    return wall, np_, rc


def main() -> None:
    root = Path(sys.argv[1])
    pats = sys.argv[2:] or ["*"]
    dirs = sorted(
        {
            Path(p)
            for pat in pats
            for p in glob.glob(str(root / "*" / pat))
            if Path(p).is_dir()
        }
    )
    print(
        "code,case,ky_code,ky_gx,t_end,gamma_code,omega_code,drift,settled,gamma_gx,omega_gx,gamma_crosscheck_code,wall_s,np,rc"
    )
    for d in dirs:
        code = d.parent.name
        if code not in {"gs2", "stella"}:
            continue
        wall, np_, rc = done_info(d)
        try:
            ky, T, g, o, drift, gfit = (gs2_fit if code == "gs2" else stella_fit)(d)
        except Exception as exc:  # missing/partial output is reported, not hidden
            print(
                f"{code},{d.name},,,,,,,error:{type(exc).__name__},,,,{wall},{np_},{rc}"
            )
            continue
        scale = SQRT2 * (A_OVER_R if d.name.startswith("G") else 1.0)
        settled = "yes" if abs(drift) <= 0.01 else "no"
        print(
            f"{code},{d.name},{ky:.5f},{ky / SQRT2:.4f},{T:.1f},{g:.6f},{o:.6f},{drift:+.2e},{settled},"
            f"{g * scale:.6f},{o * scale:.6f},{gfit:.6f},{wall:.1f},{np_},{rc}"
        )


if __name__ == "__main__":
    main()
