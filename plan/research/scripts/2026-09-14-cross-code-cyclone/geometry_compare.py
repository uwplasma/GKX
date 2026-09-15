#!/usr/bin/env python3
"""Q20 geometry cross-check: are the codes integrating the same field-line coefficients?

    python geometry_compare.py --gs2-s F --gs2-m F --stella-m F --gx-s F --gx-m F --gkx-m F

All arrays are put in the GX/GKX convention (Lref = a): GS2 gbdrift and cvdrift are divided by 2
(GS2 absorbs vref^2 = 2T/m into the drift coefficient); stella's B x gradB . grad y and B x kappa . grad y
are already in that convention (checked on the install run). For each quantity the table reports the
value at theta = 0 and theta = pi, and max |a - b| / max |b| over |theta| <= pi after linear
interpolation onto the reference theta grid (reference = GX golden for s-alpha, GS2 for Miller).
GKX s-alpha is sampled from gkx.geometry.analytic.SAlphaGeometry with the cyclone.toml parameters.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from netCDF4 import Dataset

Q = ("bmag", "gradpar", "gds2", "gbdrift", "cvdrift")


def gs2(path: str) -> dict:
    d = Dataset(path)
    th = np.asarray(d.variables["theta"][:], float)
    out = {"theta": th}
    for k in Q:
        v = np.asarray(d.variables[k][:], float)
        out[k] = v * (0.5 if k in ("gbdrift", "cvdrift") else 1.0)
    return out


def stella(path: str) -> dict:
    d = Dataset(path)
    th = np.asarray(d.variables["zed"][:], float)
    pick = {
        "bmag": "bmag",
        "gradpar": "b_dot_gradz",
        "gds2": "grady_dot_grady",
        "gbdrift": "B_times_gradB_dot_grady",
        "cvdrift": "B_times_kappa_dot_grady",
    }
    out = {"theta": th}
    for k, name in pick.items():
        v = np.asarray(d.variables[name][:], float)
        out[k] = v[:, 0] if v.ndim == 2 else v
    return out


def gx(path: str) -> dict:
    d = Dataset(path)
    th = np.asarray(d.groups["Grids"].variables["theta"][:], float)
    g = d.groups["Geometry"]
    out = {"theta": th}
    for k in Q:
        v = np.asarray(g.variables[k][:], float)
        out[k] = np.full_like(th, float(v)) if v.ndim == 0 else v
    return out


def eik(path: str) -> dict:
    d = Dataset(path)
    th = np.asarray(d.variables["theta"][:], float)
    out = {"theta": th}
    for k in Q:
        v = np.asarray(d.variables[k][:], float)
        # *.eiknc.nc files store drifts in the GS2 convention (factor 2), as GX's eik reader expects.
        out[k] = v * (0.5 if k in ("gbdrift", "cvdrift") else 1.0)
    return out


def gkx_salpha(theta: np.ndarray) -> dict:
    from gkx.geometry.analytic import SAlphaGeometry

    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    th = np.asarray(theta, float)
    out = {
        "theta": th,
        "bmag": np.asarray(geom.bmag(th), float),
        "gradpar": np.full_like(th, geom.gradpar()),
    }
    gds2, _, _ = geom.metric_coeffs(th)
    out["gds2"] = np.asarray(gds2, float)
    cv, gb, _, _ = geom.drift_coeffs(th)
    out["cvdrift"] = np.asarray(cv, float)
    out["gbdrift"] = np.asarray(gb, float)
    return out


def at(src: dict, k: str, x: float) -> float:
    return float(np.interp(x, src["theta"], src[k]))


def maxdiff(a: dict, b: dict, k: str) -> float:
    if k not in a or k not in b:
        return math.nan
    th = b["theta"][np.abs(b["theta"]) <= math.pi + 1e-9]
    va = np.interp(th, a["theta"], a[k])
    vb = np.interp(th, b["theta"], b[k])
    return float(np.max(np.abs(va - vb)) / max(np.max(np.abs(vb)), 1e-300))


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("gs2-s", "gs2-m", "stella-m", "gx-s", "gx-m", "gkx-m"):
        p.add_argument(f"--{name}", required=True)
    a = p.parse_args()
    sets = {
        "S": ("gx", {"gx": gx(a.gx_s), "gs2": gs2(a.gs2_s)}),
        "M": (
            "gs2",
            {
                "gs2": gs2(a.gs2_m),
                "stella": stella(a.stella_m),
                "gx": gx(a.gx_m),
                "gkx": eik(a.gkx_m),
            },
        ),
    }
    try:
        sets["S"][1]["gkx"] = gkx_salpha(sets["S"][1]["gx"]["theta"])
    except Exception as exc:  # reported, not hidden
        print(f"# gkx s-alpha sampling failed: {type(exc).__name__}: {exc}")
    print("geometry,code,quantity,theta0,thetapi,maxrel_vs_ref(|theta|<=pi),ref")
    for geo, (ref, codes) in sets.items():
        for code, src in codes.items():
            for k in Q:
                if k not in src:
                    continue
                print(
                    f"{geo},{code},{k},{at(src, k, 0.0):.5f},{at(src, k, math.pi):.5f},"
                    f"{maxdiff(src, codes[ref], k):.2e},{ref}"
                )


if __name__ == "__main__":
    main()
